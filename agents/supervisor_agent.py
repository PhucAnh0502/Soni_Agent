from agent_utilities import supervisor_node, State
from langgraph.graph import StateGraph, START
from agent_utilities import llm 
from financial_agent import *
from news_search_agent import *

builder = StateGraph(State)
builder.add_edge(START, "supervisor")
builder.add_node("supervisor", supervisor_node)
builder.add_node("finance_info", finance_info_agent_node)
builder.add_edge("supervisor", "search")
builder.add_node("extract_news", extract_news_agent_node)
builder.add_node("search", search_agent_node)
builder.add_node("sentiment_analysis", sentiment_analysis_agent_node)
builder.add_node("chart", chart_agent_node)
graph = builder.compile()


from IPython.display import display, Image


img = graph.get_graph().draw_mermaid_png()
with open("graph.png", "wb") as f:
    f.write(img)
