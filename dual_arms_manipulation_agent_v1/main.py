from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager
from langchain_core.messages import HumanMessage
from langgraph.types import Command, Interrupt
from langgraph.graph import END
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from react_agent.graph import load_graph   
from fastapi.middleware.cors import CORSMiddleware

# Life cycle: Loading LangGraph
@asynccontextmanager
async def lifespan(app: FastAPI):
    async with AsyncSqliteSaver.from_conn_string("./chat_history.db") as saver:
        app.state.graph = await load_graph(saver)
        yield

app = FastAPI(lifespan=lifespan)

app = FastAPI(lifespan=lifespan)  
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RunInput(BaseModel):
    message  : str | None = None   
    resume   : str | None = None   
    thread_id: str        = "default"

@app.post("/run")
async def run_graph(inp: RunInput):
    g = app.state.graph

    # Construct graph input
    if inp.resume:
        input_obj = Command(resume=inp.resume)
    elif inp.message:
        input_obj = {"messages": [HumanMessage(content=inp.message)]}
    else:
        return {"error": "Either 'message' or 'resume' is required."}

    result = await g.ainvoke(
        input=input_obj,
        config={"configurable": {"thread_id": inp.thread_id}},
    )

    # print(f"Graph run result: {result}")

    # Process Interrupt
    intr_list = result.get("__interrupt__")
    if intr_list:
        intr: Interrupt = intr_list[0]               
        intr_val = intr.value                        
        intr_type = intr_val.pop("type", "unknown")  

        return {
            "status":         "waiting",
            "interrupt_type": intr_type,   
            "payload":        intr_val,    
        }

    # if END in result:
    #     last_ai = result[END]["messages"][-1]
    #     return {"status": "done", "message": last_ai.content}
 
    return {
        "status": "done",
        "messages": "Bingo!",
    }



