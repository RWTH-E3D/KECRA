import { Component, ElementRef, OnInit } from '@angular/core';
import { RosService } from '../ros.service';
import { LangGraphService } from '../langgraph.service';
import { v4 as uuid } from 'uuid';

@Component({
  selector: 'ur5e-command',
  templateUrl: './ur5e-command.component.html',
  styleUrls: ['./ur5e-command.component.css']
})
export class Ur5eCommandComponent implements OnInit {

  userCommand: string = '';
  commandResponseLog: any[] = [];
  classNames: string = '';

  leftArmPoses: any[] = [];
  rightArmPoses: any[] = [];

  combinedLog: string = '';

  messages: { type: 'human' | 'ai'; content: string }[] = [];
  inputMessage: string = '';
  interrupted: boolean = false;
  waiting = false;
  threadId = uuid();
  lastInterruptType: string | null = null;


  constructor(
    private rosService: RosService,
    private langGraphService: LangGraphService
  ) { }

  ngOnInit(): void {
    // Subscribe to detection results
    this.rosService.getDetectionUpdates().subscribe(
      (message) => {
        this.handleDetectionMessage(message);
      },
      (error) => {
        console.error('Error receiving detection message:', error);
      }
    );

    this.rosService.getLeftArmPosesUpdates().subscribe(
      (message) => {
        this.leftArmPoses = this.parseArmPosesMessage(message);
        // this.handleLeftArmPosesMessage(message);
      },
      (error) => {
        console.error('Error receiving left arm poses:', error);
      }
    );

    // Subscribe to right arm poses updates
    this.rosService.getRightArmPosesUpdates().subscribe(
      (message) => {
        this.rightArmPoses = this.parseArmPosesMessage(message);
        // this.handleRightArmPosesMessage(message);
      },
      (error) => {
        console.error('Error receiving right arm poses:', error);
      }
    );
  }

  forkConversation() {
    this.threadId = uuid();
    this.messages = [];
    this.waiting = false;
    this.inputMessage = '';
    this.interrupted = false;
  }


  sendMessage() {
    const txt = this.inputMessage.trim();
    if (!txt) return;

    const fullMessage = `${this.combinedLog}\n\n${txt}`;   
    this.messages.push({ type: 'human', content: txt });
    this.inputMessage = '';

    let body: any;

    if (this.waiting) {
      let resumePayload: string = txt;

      if (
        this.lastInterruptType === 'pose_confirmation' &&
        !/^(y|yes|ok)$/i.test(txt)
      ) {
        resumePayload = fullMessage;
      }

      body = { resume: resumePayload, thread_id: this.threadId };
    } else {
      body = { message: fullMessage, thread_id: this.threadId };
    }

    this.langGraphService.callLangGraph(body).subscribe({
      next: (res: any) => {
        switch (res.status) {
          case 'waiting':
            this.waiting = true;
            this.handleInterrupt(res.interrupt_type, res.payload);
            break;

          case 'done':
            this.waiting = false;
            this.messages.push({ type: 'ai', content: res.messages });
            break;
        }
      },
      error: () => {
        this.waiting = false;
        this.messages.push({ type: 'ai', content: '[Error] backend failed' });
      },
    });
  }

  handleInterrupt(type: string, payload: any) {
    this.lastInterruptType = type;

    if (type === 'task_confirmation') {
      this.messages.push({
        type: 'ai',
        content: `${payload.tip}\n\n${payload.tasks.join('\n')}`,
      });
    } else if (type === 'pose_confirmation') {
      this.rosService.publishPoseSequence(payload.poses);

      this.messages.push({
        type: 'ai',
        content: `${payload.tip}\n\nPoses: ${JSON.stringify(payload.poses)}`,
      });
    } else {
      this.messages.push({ type: 'ai', content: '(unknown interrupt)' });
    }
  }

  sendClassNames() {
    // Check if the input is empty or contains only spaces
    if (!this.classNames || this.classNames.trim() === '') {
      this.commandResponseLog.push({ error: 'Class names cannot be empty.' });
      return;
    }

    // Check whether class names are separated by commas and there are no empty entries
    const classNamesList = this.classNames.split(',').map(name => name.trim());
    if (classNamesList.some(name => name === '')) {
      this.commandResponseLog.push({ error: 'Class names must be comma-separated and cannot contain empty values.' });
      return;
    }

    // Publish class names
    this.rosService.publishClassNames(this.classNames);
    this.commandResponseLog.push({ success: `Class names sent: ${this.classNames}` });
  }

  parseArmPosesMessage(message: string): any[] {
    const armPoses: any[] = [];
    const lines = message.split('\n');

    lines.forEach((line) => {
      const match = /Frame: (.*?), Position: \[(.*?)\]/.exec(line); // Match frame and position
      if (match) {
        const frame = match[1].trim();
        const position = match[2].split(',').map((v) => parseFloat(v.trim()));
        armPoses.push({ Frame: frame, Position: position });
      }
    });

    return armPoses;
  }

  handleDetectionMessage(message: any) {
    try {
      const parsedMessage = JSON.parse(message.data);
      let tempCombinedLog = ''; 

      const maxWaitTime = 2000; // Maximum wait time (milliseconds)
      const checkInterval = 100; // Check interval (milliseconds)
      let elapsedTime = 0;

      const waitForPoses = new Promise<void>((resolve) => {
        const checkPosesAvailable = () => {
          if (this.leftArmPoses.length > 0 || this.rightArmPoses.length > 0) {
            resolve();
          } else if (elapsedTime >= maxWaitTime) {
            console.warn('Timeout: Arm poses not available.');
            resolve();
          } else {
            elapsedTime += checkInterval;
            setTimeout(checkPosesAvailable, checkInterval);
          }
        };
        checkPosesAvailable();
      });

      waitForPoses.then(() => {
        parsedMessage.forEach((detection: any) => {
          const { class_name, real_width, real_height, object_height } = detection;

          const leftPose = this.leftArmPoses.find((pose) => pose.Frame === class_name);
          const rightPose = this.rightArmPoses.find((pose) => pose.Frame === class_name);

          if (leftPose) {
            tempCombinedLog += `${class_name} has a bounding box width of ${real_width.toFixed(
              3
            )} meters, height of ${real_height.toFixed(3)} meters, and an object height of ${object_height.toFixed(
              2
            )} meters. In the left arm coordinate frame, its position is [${leftPose.Position.join(', ')}]. `;
          }

          if (rightPose) {
            tempCombinedLog += `${class_name} has a bounding box width of ${real_width.toFixed(
              3
            )} meters, height of ${real_height.toFixed(3)} meters, and an object height of ${object_height.toFixed(
              2
            )} meters. In the right arm coordinate frame, its position is [${rightPose.Position.join(', ')}]. `;
          }
        });

        this.combinedLog = tempCombinedLog.trim();

        if (this.combinedLog) {
          this.commandResponseLog.push({ log: this.combinedLog });
        } else {
          this.commandResponseLog.push({ log: 'No matching poses found for the detected objects.' });
        }
      });
    } catch (error) {
      this.commandResponseLog.push({ error: `Failed to parse detection message: ${message.data}` });
      console.error('Error parsing detection message:', error);
    }
  }

  formatLog(response: any): string {
    if (response.log) {
      return response.log.replace(/\n/g, '<br>');
    } else {
      // Return JSON-formatted string by default
      return JSON.stringify(response, null, 2).replace(/\n/g, '<br>');
    }
  }


  handleLeftArmPosesMessage(message: any) {
    const formattedMessage = `${message}`;
    this.commandResponseLog.push({ log: formattedMessage });
    console.log('Received left arm poses message:', formattedMessage);
  }

  handleRightArmPosesMessage(message: any) {
    const formattedMessage = `${message}`;
    this.commandResponseLog.push({ log: formattedMessage });
    console.log('Received right arm poses message:', formattedMessage);
  }

}
