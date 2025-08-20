# KECRA
KECRA is a framework for language-based control of collaborative robots (cobots) in dynamic construction environments. It combines Large Language Models (LLMs) with Visual Foundation Models (VFMs) to allow human operators to guide robotic assistance through natural language dialogues.
KECRA integrates:
  - Perception module based on SAM + CLIP, providing zero-shot recognition and real-time understanding of objects (type, location, physical dimensions, coordinate frame).
  - State-graph planning agent (LangGraph) that decomposes high-level instructions into verifiable sub-tasks and robot poses using Chain-of-Thought reasoning.
  - Human-in-the-loop feedback loops at key reasoning stages (task plan / pose plan) enabling interactive correction and safe mid-execution adjustments.
  - Memory system combining short-term interaction context and long-term knowledge graph (episodes of past tasks) to improve future reasoning via retrieval-augmented generation.

Workflow:
Instruction → Plan Sub-tasks → Confirm Tasks → Generate Poses → Confirm Poses → Execution (with rewindable checkpoints)

This repository includes code for perception (SAM/CLIP API), state-graph agent (LangGraph), human-AI interaction logic, and knowledge-graph construction/retrieval.

Please cite us if you use the codes: Tang, R., Lorenz, C. L., Frisch, J., van Treeck, C. 2025. Knowledge-Enhanced Cobot Reasoning Assistant for Robot Manipulation. Forum Bauinformatik. Aachen, Germany.

##Installation
### 1. Start ROS2 Rosbridge WebSocket
```bash
ros2 launch rosbridge_server rosbridge_websocket_launch.xml
