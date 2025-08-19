import json
import os
from datetime import datetime, UTC
from openai import OpenAI
from typing import Dict, List, Literal, cast, Optional
import ast
from neo4j import GraphDatabase

from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from react_agent.configuration import Configuration
from react_agent.state import State
from react_agent.utils import load_chat_model
from dotenv import load_dotenv

load_dotenv()

# checkpointer = SqliteSaver.from_conn_string("sqlite:///langgraph.db")
client  = OpenAI()

print("NEO4J_URI =", os.getenv("NEO4J_URI"))
# NEO4J_URI  = os.environ["NEO4J_URI"]     
NEO4J_URI  = "bolt://localhost:7687"
NEO4J_USER = os.environ["NEO4J_USER"]     
NEO4J_PWD  = os.environ["NEO4J_PWD"]      

driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USER, NEO4J_PWD),
    encrypted=False,              
)

def embed(text: str) -> List[float]:
    """OpenAI embedding(1536 D)"""
    return client.embeddings.create(
        input=text, model="text-embedding-3-small"
    ).data[0].embedding

def retrieve_similar_instructions(instr: str, k: int = 3) -> List[Dict]:
    """
    返回 k 条 few-shot:
      {
        "env":   "envirnment text",
        "instr": "instruction text",
        "tasks": ["t1", "t2", ...]   
      }
    """
    CYPHER = """
    CALL db.index.vector.queryNodes('instr_vec', $k, $vec)
    YIELD node AS i, score

    MATCH (i)-[:GENERATED_TASKS|FEEDBACK*0..5]->(tl:TaskList)
    WITH i, tl
    ORDER BY tl.step DESC
    WITH i, head(collect(tl)) AS latest_tl    

    RETURN
      i.env             AS env,          
      i.text            AS instr,
      latest_tl.tasks   AS tasks_json
    LIMIT $k
    """

    vec = embed(instr)
    with driver.session() as sess:
        rows = sess.run(CYPHER, vec=vec, k=k)
        examples = []
        for r in rows:
            examples.append(
                dict(
                    env   = r["env"]   or "",
                    instr = r["instr"],
                    tasks = json.loads(r["tasks_json"]),
                )
            )
        return examples


def retrieve_task_feedback(fb_text: str, k=3):
    CYPHER = """
    CALL db.index.vector.queryNodes('fb_vec', $k, $vec)
    YIELD node AS f
    WHERE f.phase='task'
    MATCH (f)<-[:FEEDBACK]-(old:TaskList)
    MATCH (f)-[:GENERATED_TASKS]->(new:TaskList)
    RETURN f.text          AS feedback,
           old.tasks       AS before_json,
           new.tasks       AS after_json
    LIMIT $k
    """
    with driver.session() as sess:
        return [
            {
                "feedback": r["feedback"],
                "before":   json.loads(r["before_json"]),
                "after":    json.loads(r["after_json"]),
            }
            for r in sess.run(CYPHER, vec=embed(fb_text), k=k)
        ]
    
def retrieve_pose_feedback(fb_text: str, k=3):
    CYPHER = """
    CALL db.index.vector.queryNodes('fb_vec', $k, $vec)
    YIELD node AS f
    WHERE f.phase='pose'
    MATCH (f)<-[:FEEDBACK]-(old:PoseList)
    MATCH (f)-[:GENERATED_POSES]->(new:PoseList)
    RETURN f.text        AS feedback,
           old.poses     AS before_json,
           new.poses     AS after_json
    LIMIT $k
    """
    with driver.session() as sess:
        return [
            {
                "feedback": r["feedback"],
                "before":   json.loads(r["before_json"]),
                "after":    json.loads(r["after_json"]),
            }
            for r in sess.run(CYPHER, vec=embed(fb_text), k=k)
        ]

def call_model(state: State, user_prompt: str) -> AIMessage:
    """Call the LLM powering our "agent".

    This function prepares the prompt, initializes the model, and processes the response.

    Args:
        state (State): The current state of the conversation.
        config (RunnableConfig): Configuration for the model run.

    Returns:
        dict: A dictionary containing the model's response message.
    """
    configuration = Configuration.from_context()

    model = load_chat_model(configuration.model)

    user_msg = HumanMessage(content=user_prompt)

    # Get the model's response
    response = cast(
        AIMessage,
        model.invoke(
            [user_msg, *state.messages]
        ),
    )

    # Return the model's response as a list to be added to existing messages
    # return {"messages": [response]}
    # state.messages.append(user_msg)
    state.messages.append(response)
    return response.content.strip()  

# def plan_tasks(state: State):
#     cfg = Configuration.from_context()
#     instr = state.messages[-1].content.strip()           
#     user_prompt = cfg.task_parse_user_prompt.format(instruction=instr)

#     tasks_str = call_model(state, user_prompt)  
#     # tasks_str = tasks_str.strip().strip("```").replace("json", "").strip()
#     tasks_str = tasks_str.strip("```").replace("json", "").strip()

#     try:
#         tasks = json.loads(tasks_str)
#         if not isinstance(tasks, list) or not all(isinstance(t, str) for t in tasks):
#             raise ValueError("Parsed result is not a list of strings.")
#     except Exception as e:
#         raise ValueError(f"Failed to parse tasks from LLM output: {tasks_str}\nError: {e}")
    
#     state.tasks = tasks  
#     # print(f"Parsed tasks: {state.tasks}")       
#     # state.task_idx = 0          
#     return state
#     # return {"tasks": tasks}

def plan_tasks(state: State):
    cfg    = Configuration.from_context()
    instr  = state.messages[-1].content.strip()

    examples = retrieve_similar_instructions(instr, k=3)
    fs_txt   = "\n\n".join(
        f"### Example {idx+1}\n"
        f"Environment:\n{ex['env']}\n"
        f"Instruction:\n{ex['instr']}\n"
        f"Tasks:{json.dumps(ex['tasks'], ensure_ascii=False)}"
        for idx, ex in enumerate(examples)
    )

    print(f"Few-shot examples:\n{fs_txt}")

    prompt = cfg.task_parse_user_prompt.format(
        instruction=instr,
        few_shot=fs_txt,
    )

    tasks_str = call_model(state, prompt).strip("```").replace("json", "").strip()
    try:
        tasks = json.loads(tasks_str)
        if not isinstance(tasks, list) or not all(isinstance(t, str) for t in tasks):
            raise ValueError("Parsed result is not a list of strings.")
    except Exception as e:
        raise ValueError(f"Failed to parse tasks from LLM output: {tasks_str}\nError: {e}")

    state.tasks = tasks
    return state


def confirm_tasks(state: State) -> Command[Literal["process_action", "plan_tasks"]]:
    """Task confirmation node"""
    # Send interrupt request (metadata contains list of tasks requiring confirmation)
    feedback : Optional[str] = interrupt(
        {
            "type": "task_confirmation",
            "tasks": state.tasks,
            # "message": "Please confirm whether the task decomposition is reasonable. Reply 'Y' to approve or provide suggestions."
            "tip": "Please confirm the task list. Reply 'Y' to approve or send your corrections."
        },
    )
    
    
    # Process user feedback
    if feedback.strip().lower() in ["y", "yes", "ok"]:
        # Confirm approval and proceed to action breakdown.
        return Command(goto="process_action",
                        update={
                            "messages": [HumanMessage(content=feedback)],
                            "feedback": feedback,  
                        })
    else:
        # Users provide feedback, and the feedback is used as new information to trigger the regeneration of tasks.
        print("again")
        return Command(
            goto="plan_tasks",  # Jump back to task generation node
            update={
                "messages": [HumanMessage(content=feedback)],  # Add user feedback to message history
                "tasks": None,  # Empty old tasks
                # "task_idx": None
                "feedback": feedback,  
            }
        )
    
# def confirm_tasks(state: State) -> Command[Literal["process_action", "plan_tasks"]]:
#     fb: Optional[str] = interrupt({
#         "type": "task_confirmation",
#         "tasks": state.tasks,
#         "tip" : "Please confirm the task list. Reply 'Y' to approve or send your corrections."
#     })

#     if fb and fb.strip().lower() not in {"y","yes","ok"}:
#         # few-shot：相似反馈
#         examples = retrieve_task_feedback(fb, k=1)
#         shots    = "\n\n".join(
#             f"- User:\n{ex['feedback']}\n  Before:{ex['before']}\n  After:{ex['after']}"
#             for ex in examples
#         )
#         # 把 few-shot 和用户纠正插回 messages，触发 plan_tasks 重写
#         return Command(
#             goto="plan_tasks",
#             update={
#                 "messages": [HumanMessage(content=f"Reference to similar amendments:\n{shots}\n\n{fb}")],
#                 "feedback": fb,
#                 "tasks": None,
#             },
#         )

#     return Command(goto="process_action",
#                         update={
#                             "messages": [HumanMessage(content=fb)],
#                             "feedback": fb,  
#                         })

def process_action(state: State):
    cfg = Configuration.from_context()
    # task = state.tasks[state.task_idx]
    tasks = state.tasks
    # print(f"Processing tasks: {tasks}")
    feedback = state.feedback or ""
    user_prompt = cfg.pose_gen_user_prompt.format(action=tasks, feedback=feedback)
    pose_text = call_model(state, user_prompt).strip("```").replace("json", "").strip()
    print(f"Generated state text: {state.messages}")
    try:
        parsed = ast.literal_eval(pose_text)
        print(f"Parsed pose: {parsed}")
        # case 1: single pose
        if isinstance(parsed, list) and all(isinstance(x, (int, float)) for x in parsed) and len(parsed) == 8:
            state.poses = parsed
        # case 2: multi pose
        elif isinstance(parsed, list) and all(isinstance(p, list) and len(p) == 8 for p in parsed):
            state.poses = parsed
    except Exception:
        # raise ValueError(f"Invalid pose format: {pose_text}")
        return Command(
            goto="fix_pose",
            update={
                "bad_poses": pose_text,
                "poses": None,
            })
    return state

# def process_action(state: State):
#     cfg       = Configuration.from_context()
#     tasks_str = state.tasks
#     prompt    = cfg.pose_gen_user_prompt.format(action=tasks_str, feedback=state.feedback or "")
#     pose_out  = call_model(state, prompt)

#     try:
#         parsed = ast.literal_eval(pose_out)
#         if isinstance(parsed, list) and all(isinstance(p, (int,float)) for p in parsed) and len(parsed) == 8:
#             state.poses = parsed
#         elif isinstance(parsed, list) and all(isinstance(p, list) and len(p) == 8 for p in parsed):
#             state.poses = parsed
#     except Exception:
#         return Command(goto="fix_pose", 
#                        update={
#                            "bad_poses": pose_out, 
#                            "poses": None,
#                         })

def confirm_pose(state: State) -> Command[Literal["process_action", "__end__"]]:
    fb = interrupt({
        "type": "pose_confirmation",
        "poses": state.poses,
        # "task": state.tasks[state.task_idx],
        "tip": "Confirm the poses [x, y, z, roll, pitch, yaw, arm, gripper]. Reply 'Y' to end task or send your corrections."
    })

    if fb is None or (isinstance(fb, str) and fb.strip().lower() in ["y", "yes", "ok"]):
        # state.poses.append(state.pose)
        state.feedback = "end"
        # state.task_idx += 1
        # return Command(goto="process_action", update={"task_idx": state.task_idx}) if state.task_idx < len(state.tasks) else Command(goto="__end__")
        Command(goto="__end__",
                update={
                    "messages": [HumanMessage(content=fb)],
                    "feedback": state.feedback,  
                })
        
    else:
        state.feedback = fb if isinstance(fb, str) else str(fb)
        state.poses = None
        return Command(
            goto="process_action",
            update={
                "messages": [HumanMessage(content=fb)],  # Add user feedback to message history
                "feedback": fb,  
            })
    

# def confirm_pose(state: State) -> Command[Literal["process_action", "__end__"]]:
#     fb = interrupt({
#         "type": "pose_confirmation",
#         "poses": state.poses,
#         "tip" : "Confirm the poses [x, y, z, roll, pitch, yaw, arm, gripper]. Reply 'Y' to end task or send your corrections."
#     })

#     if fb and fb.strip().lower() not in {"y","yes","ok"}:
#         # pose few-shot
#         exs  = retrieve_pose_feedback(fb, k=1)
#         shot = "\n\n".join(
#             f"- User:\n{ex['feedback']}\n  Before:{ex['before']}\n  After:{ex['after']}"
#             for ex in exs
#         )
#         return Command(
#             goto="process_action",
#             update={
#                 "messages": [HumanMessage(content=f"Reference to similar amendments:\n{shot}\n\n{fb}")],
#                 "feedback": fb,
#                 "poses"   : None,
#             },
#         )

#     return Command(goto="__end__")
    
def fix_pose(state: State) -> Command[Literal["confirm_pose", "fix_pose"]]:
    cfg = Configuration.from_context()
    bad = state.bad_poses or ""

    user_prompt = cfg.bad_pose_fix_user_prompt.format(bad=bad)

    fixed_text = call_model(state, user_prompt)

    try:
        parsed = ast.literal_eval(fixed_text)
        if isinstance(parsed, list) and all(isinstance(x, (int, float)) for x in parsed) and len(parsed) == 8:
            state.poses = parsed
        elif isinstance(parsed, list) and all(isinstance(p, list) and len(p) == 8 for p in parsed):
            state.poses = parsed
        return Command(goto="confirm_pose",
                       update={
                            "poses"   : state.poses,
            },
                       )
    
    except Exception:
        return Command(goto="fix_pose")


async def load_graph(checkpointer: AsyncSqliteSaver):
    builder = StateGraph(State, config_schema=Configuration)

    builder.add_node(plan_tasks, "plan_tasks")
    builder.add_node(confirm_tasks, "confirm_tasks")
    builder.add_node(process_action, "process_action")
    builder.add_node(confirm_pose, "confirm_pose")
    builder.add_node(fix_pose, "fix_pose")

    builder.add_edge("__start__", "plan_tasks")
    builder.add_edge("plan_tasks", "confirm_tasks")
    builder.add_edge("process_action", "confirm_pose")

    graph = builder.compile(checkpointer=checkpointer)
    graph.name = "DualArmPlanner" 
    return graph






