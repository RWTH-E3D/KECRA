import { Component, OnInit } from '@angular/core';
import { RosService } from '../ros.service';

@Component({
  selector: 'ur5e-visual',
  templateUrl: './ur5e-visual.component.html',
  styleUrls: ['./ur5e-visual.component.css']
})
export class Ur5eVisualComponent implements OnInit {
  jointStates: any[] = [];
  joints: string[] = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'];
  jointPositions: number[] = [0, 0, 0, 0, 0, 0];

  constructor(private rosService: RosService) { }

  ngOnInit(): void {
  }

}

// sk-c00e142a75274728aacb3da731b0b5f6