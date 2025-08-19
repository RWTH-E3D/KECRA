#!/usr/bin/env python3
"""
sqlite → Neo4j ： thread_id = 8f1e6004-94e2-4a93-820f-0ef18b69acc7
graph structure:
  (:Thread)-[:HAS_INSTRUCTION]->(:Instruction)
  —(GENERATED_TASKS)→(:TaskList)-[:FEEDBACK {phase:'task'}]→(:Feedback)
  —(GENERATED_POSES)→(:PoseList)-[:FEEDBACK {phase:'pose'}]→(:Feedback)
TaskList and PoseList are both connected in a feedback chain to form a single chain.
"""

import json, sqlite3, uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
from neo4j import GraphDatabase
from openai import OpenAI

# setup 
SQLITE_PATH = Path("/home/trb/dual_arms_manipulation_agent/chat_history.db") # replace with your SQLite path
THREAD_ID   = "********"    # replace with your thread_id, e.g. "8f1e6004-94e2-4a93-820f-0ef18b69acc7"
NEO4J_URI   = "bolt://localhost:7687"
NEO4J_USER  = "neo4j"
NEO4J_PWD   = "*******" # replace with your Neo4j password


client = OpenAI(api_key="sk-********")  # replace with your OpenAI API key

def embed_text(text: str) -> list[float]:
    """Returns a 1536-dimensional vector (list[float])"""
    return client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    ).data[0].embedding


def merge_node(tx, label, _id, **props):
    if "name" not in props:
        if label == "TaskList":
            props["name"] = f"Tasks (step {props.get('step', '?')})"
        elif label == "PoseList":
            props["name"] = f"Poses (step {props.get('step', '?')})"
        elif label == "Instruction":
            short = props.get("text", "")[:30].replace("\n", " ")
            props["name"] = f"Instr: {short}…"
        elif label == "Feedback":
            short = props.get("text", "")[:30].replace("\n", " ")
            props["name"] = f"⚠ {props.get('phase','?')}: {short}…"

    if label in {"Instruction", "Feedback"} and "embedding" not in props:
        raw_text = props.get("text", "")
        if raw_text:
            props["embedding"] = embed_text(raw_text)

    if props:
        p = ", ".join(f"{k}: ${k}" for k in props)
        tx.run(f"MERGE (n:{label} {{id:$id}}) SET n += {{{p}}}", id=_id, **props)
    else:
        tx.run(f"MERGE (n:{label} {{id:$id}})", id=_id)


def merge_rel(tx, a_id, a_label, rel, b_id, b_label):
    tx.run(
        f"MATCH (a:{a_label} {{id:$a}}),(b:{b_label} {{id:$b}}) "
        f"MERGE (a)-[:{rel}]->(b)",
        a=a_id, b=b_id,
    )


def fetch_rows() -> List[Dict[str, Any]]:
    with sqlite3.connect(str(SQLITE_PATH)) as conn:
        cur = conn.cursor()
        cur.execute("SELECT metadata FROM checkpoints WHERE thread_id=?", (THREAD_ID,))
        rows = [json.loads(j) for (j,) in cur.fetchall() if j]
    rows.sort(key=lambda r: r.get("step", 0))
    return rows

def yes(fb: str) -> bool:
    return fb.strip().lower() in {"", "y", "yes", "ok"}

def main():
    rows = fetch_rows()
    if not rows:
        print("no record")
        return

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PWD), encrypted=False)

    anchor_task:  Optional[str] = None   # Where should the next TaskList be hung
    anchor_pose:  Optional[str] = None   # Where should the next PoseList be hung
    instr_id:     Optional[str] = None   # Unique Instruction

    stage = "plan"   # plan → confirm_tasks → action → confirm_pose

    with driver.session() as sess:
        thread_node = f"thread-{THREAD_ID}"
        sess.execute_write(merge_node, "Thread", thread_node, id_short=THREAD_ID)

        for meta in rows:
            w, step = meta.get("writes") or {}, meta.get("step", -1)

            # 1. plan_tasks 
            if "plan_tasks" in w and stage in {"plan", "confirm_tasks"}:
                node = w["plan_tasks"]
                raw  = node["messages"][0]["kwargs"]["content"]
                env, instr = raw.split("\n\n", 1) if "\n\n" in raw else ("", raw)
                tasks_js   = json.dumps(node.get("tasks", []), ensure_ascii=False)

                # First build Instruction
                if instr_id is None:
                    instr_id = f"instr-{uuid.uuid4()}"
                    sess.execute_write(
                        merge_node, "Instruction", instr_id,
                        text=instr.strip(), env=env.strip(), step=step
                    )
                    sess.execute_write(
                        merge_rel, thread_node, "Thread",
                        "HAS_INSTRUCTION", instr_id, "Instruction"
                    )
                    anchor_task = instr_id   # Anchor points to Instruction

                # Generate TaskList and connect it to anchor_task
                tlist_id = f"tasks-{uuid.uuid4()}"
                sess.execute_write(merge_node, "TaskList", tlist_id, tasks=tasks_js, step=step)
                sess.execute_write(
                    merge_rel, anchor_task, "Instruction" if anchor_task==instr_id else "Feedback",
                    "GENERATED_TASKS", tlist_id, "TaskList"
                )
                anchor_task = tlist_id        
                anchor_pose = tlist_id        
                stage = "confirm_tasks"
                continue

            # 2. confirm_tasks 
            if "confirm_tasks" in w and stage == "confirm_tasks":
                fb = (w["confirm_tasks"] or {}).get("feedback", "").strip()
                if yes(fb):
                    stage = "action"
                else:
                    fb_id = f"fb-{uuid.uuid4()}"
                    sess.execute_write(merge_node, "Feedback", fb_id, text=fb, phase="task", step=step)
                    sess.execute_write(
                        merge_rel, anchor_task, "TaskList", "FEEDBACK", fb_id, "Feedback"
                    )
                    anchor_task = fb_id       
                    stage = "plan"
                continue

            # 3. process_action 
            if "process_action" in w and stage == "action":
                poses_js = json.dumps((w["process_action"] or {}).get("poses", []), ensure_ascii=False)
                plist_id = f"poses-{uuid.uuid4()}"
                sess.execute_write(merge_node, "PoseList", plist_id, poses=poses_js, step=step)
                sess.execute_write(
                    merge_rel, anchor_pose, "TaskList" if anchor_pose==anchor_task else "Feedback",
                    "GENERATED_POSES", plist_id, "PoseList"
                )
                anchor_pose = plist_id        
                stage = "confirm_pose"
                continue

            # 4. confirm_pose 
            if "confirm_pose" in w and stage == "confirm_pose":
                raw_fb = (w["confirm_pose"] or {}).get("feedback", "").strip()

                if yes(raw_fb):                         # Positive confirmation — Process completed
                    anchor_task = instr_id
                    stage = "plan"
                else:                                   # Negative feedback — poses need to be regenerated
                    # Split env and feedback text
                    if "\n\n" in raw_fb:
                        env_text, fb_text = raw_fb.split("\n\n", 1)
                    else:
                        env_text, fb_text = "", raw_fb

                    # Create a Feedback node with env + text
                    fb_id = f"fb-{uuid.uuid4()}"
                    sess.execute_write(
                        merge_node, "Feedback", fb_id,
                        text=fb_text.strip(), env=env_text.strip(),
                        phase="pose", step=step
                    )
                    sess.execute_write(
                        merge_rel, anchor_pose, "PoseList",
                        "FEEDBACK", fb_id, "Feedback"
                    )

                    # Subsequent PoseList concatenates from this Feedback
                    anchor_pose = fb_id
                    stage = "action"
                continue
        # for rows
    driver.close()
    print("Written to Neo4j: Chain structure completed")

if __name__ == "__main__":
    main()