from langchain_core.messages import HumanMessage
from langchain_google_vertexai import ChatVertexAI
from typing import List, Optional, Literal
from langgraph.types import Command
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, START
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from agents.agent_utilities import State
from agents.agent_utilities import llm 
from tools.finance_tools import *


chart_agent = create_react_agent(llm, tools=[plot_volume_chart,plot_candlestick, plot_monthly_returns_heatmap, plot_shareholders_piechart,plot_volume_and_closed_price,plot_line_chart])


def chart_agent_node(state: State) -> Command[Literal["supervisor"]]:
    """Invoke the chart agent to draw financial data and return the result."""
    result = chart_agent.invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="chart")
            ]
        },
        goto="supervisor",
    )

finance_agent = create_react_agent(llm, tools=[get_internal_reports, get_stock_data])

def finance_info_agent_node(state: State) -> Command[Literal["supervisor"]]:
    """Invoke the finance info agent and return the result."""
    result = finance_agent.invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="finance_info")
            ]
        },
        goto="supervisor",
    )





