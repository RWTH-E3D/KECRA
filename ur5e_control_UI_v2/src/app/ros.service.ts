import { Injectable } from '@angular/core';
import * as ROSLIB from 'roslib';
import { Observable, Subject } from 'rxjs';
import { HttpClient } from '@angular/common/http';

@Injectable({
  providedIn: 'root'
})
export class RosService {
  private ros: any;
  private commandTopic: any;
  private classNamesTopic: any;
  private detectionTopic: any;
  private detectionSubject = new Subject<any>();
  private leftArmPosesTopic: any;
  private rightArmPosesTopic: any;
  private leftArmPosesSubject = new Subject<string>();
  private rightArmPosesSubject = new Subject<string>();
  private actionSequenceTopic: any;
  private poseSequenceTopic: any;

  constructor(private http: HttpClient) {
    this.ros = new ROSLIB.Ros({
      url: 'ws://localhost:9090'
    });

    this.ros.on('connection', () => {
      console.log('Connected to websocket server.');
    });

    this.ros.on('error', (error: any) => {
      console.error('Error connecting to websocket server: ', error);
    });

    this.ros.on('close', () => {
      console.log('Connection to websocket server closed.');
    });

    this.commandTopic = new ROSLIB.Topic({
      ros: this.ros,
      name: '/user_command',
      messageType: 'std_msgs/String'
    });

    this.classNamesTopic = new ROSLIB.Topic({
      ros: this.ros,
      name: '/class_names',
      messageType: 'std_msgs/String'
    });

    // Initialize detection topic subscription
    this.detectionTopic = new ROSLIB.Topic({
      ros: this.ros,
      name: '/detailed_objects',
      messageType: 'std_msgs/String'
    });

    this.detectionTopic.subscribe((message: any) => {
      this.detectionSubject.next(message);
    });

    this.leftArmPosesTopic = new ROSLIB.Topic({
      ros: this.ros,
      name: '/left_arm_poses',
      messageType: 'std_msgs/String'
    });

    this.leftArmPosesTopic.subscribe((message: any) => {
      this.leftArmPosesSubject.next(message.data);
    });

    this.rightArmPosesTopic = new ROSLIB.Topic({
      ros: this.ros,
      name: '/right_arm_poses',
      messageType: 'std_msgs/String'
    });

    this.rightArmPosesTopic.subscribe((message: any) => {
      this.rightArmPosesSubject.next(message.data);
    });

    this.actionSequenceTopic = new ROSLIB.Topic({
      ros: this.ros,
      name: '/action_sequence',
      messageType: 'std_msgs/String'
    });

    this.poseSequenceTopic = new ROSLIB.Topic({
      ros: this.ros,
      name: '/pose_sequence',
      messageType: 'std_msgs/String'
    });
  }

  publishClassNames(classNames: string): void {
    const message = new ROSLIB.Message({
      data: classNames
    });
    this.classNamesTopic.publish(message);
    console.log(`Class names published: ${classNames}`);
  }

  publishActionSequence(actionSequence: string): void {
    const message = new ROSLIB.Message({
      data: actionSequence
    });

    this.actionSequenceTopic.publish(message);
    console.log(`Published action sequence: ${actionSequence}`);
  }

  publishPoseSequence(poses: number[][]): void {
    const msg = new ROSLIB.Message({ data: JSON.stringify(poses) });
    this.poseSequenceTopic.publish(msg);
    console.log('[ROS] pose sequence ->', poses);
  }


  getDetectionUpdates(): Observable<any> {
    return this.detectionSubject.asObservable();
  }

  getLeftArmPosesUpdates(): Observable<string> {
    return this.leftArmPosesSubject.asObservable();
  }

  getRightArmPosesUpdates(): Observable<string> {
    return this.rightArmPosesSubject.asObservable();
  }
}
