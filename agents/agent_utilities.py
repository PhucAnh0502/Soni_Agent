import time
from langgraph.graph import StateGraph, MessagesState, START, END
from pydantic import BaseModel, Field
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.types import Command
from typing import Literal, List
from langchain_google_vertexai import ChatVertexAI
from dotenv import load_dotenv
from typing import TypedDict
import os


load_dotenv()

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "credentials/vertexai.json"

llm = ChatVertexAI(model="gemini-1.5-pro")

class State(MessagesState):
    next: str

workers = ["finance_info", "extract_news", "sentiment_analysis", "chart"]
options = workers + ["FINISH"]

system_promp = {
        "You are a supervisor tasked with managing a conversation between the"
        f" following workers: {workers}. Given the following user request,"
        " respond with the worker to act next. Each worker will perform a"
        " task and respond with their results and status. When finished,"
        " respond with FINISH."
}

class Router(TypedDict):
    """Worker to route to next. If no workers needed, route to FINISH."""
    next: Literal[*options]

def supervisor_node(state: State) -> Command[Literal[*workers, "__end__"]]:
    messages = [{"role":"system", "content":system_promp},] + state["messages"]

    response = llm.with_structured_output(Router).invoke(messages)
    goto = response["next"]
    if goto == "FINISH":
        goto = END
    
    return Command(goto=goto, update={"next":goto})

