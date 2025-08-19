TASK_PARSE_USER_PROMPT = (
    'EnviromentInfo + Instruction: "{instruction}"\n'
    "You are an expert assistant for robotic manipulation planning. You will interpret the user's instruction and break it down into a sequence of high-level tasks for a dual-arm robot to execute.\n"
    "List the high-level tasks required to fulfill this instruction as a list array of strings (no additional commentary).\n"
    "{few_shot}"
)


POSE_GEN_USER_PROMPT = (
    "Generate a list array of 8-DOF pose [x, y, z, roll, pitch, yaw, arm, gripper] for tasks: '{action}' (arm: 0=right, 1=left, gripper: 0=open, 1=closed).\n"
    "roll, pitch, yaw default to 0.0, 3.1415159265358979323846, 0.0! when reset, xyz all default to 0.0.\n" 
    "Respond only with the float pose values, no additional commentary. Do not include intermediate operators in the result!\n"
    "If user feedback is given, adjust accordingly: '{feedback}'\n"
    "The content format can only be: [] or [[],[]...]."
)


BAD_POSE_FIX_USER_PROMPT = (
    "Wrong output: '{bad}'. Please regenerate.\n"
    "Do not include intermediate operators in the result!\n"
    "roll, pitch, yaw default to 0.0, 3.1415159265358979323846, 0.0! Respond only with the float pose values, no additional commentary!\n"
    "The content format can only be: [] or [[],[]...]."
)