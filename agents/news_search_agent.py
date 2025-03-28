import requests
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from tools.finance_tools import semantic_search_news_db
from tools.web_tools import tavily_tool, extract_info_tool
from langgraph.graph import START, END
from agents.agent_utilities import State
from langgraph.prebuilt import create_react_agent
from agents.agent_utilities import llm
from typing import Literal
from dotenv import load_dotenv
import os
load_dotenv()

HF_API_URL = "https://api-inference.huggingface.co/models/ProsusAI/finbert"
HF_API_KEY = os.getenv("HF_TOKEN")

HEADERS = {"Authorization": f"Bearer {HF_API_KEY}"}

search_agent = create_react_agent(llm, tools=[tavily_tool, semantic_search_news_db])

extract_news_agent = create_react_agent(llm, tools=[extract_info_tool])


def search_agent_node(state: State) -> Command[Literal["supervisor"]]:
    """Agent tìm kiếm bài viết tài chính"""
    result = search_agent.invoke(state)
    return Command(
        update={"messages": [HumanMessage(content=result["messages"][-1].content, name="search")]},
        goto="supervisor",
    )


def extract_news_agent_node(state: State) -> Command[Literal["sentiment_analysis"]]:
    """Agent trích xuất nội dung bài viết"""
    result = extract_news_agent.invoke(state)
    return Command(
        update={"messages": [HumanMessage(content=result["messages"][-1].content, name="extract_news")]},
        goto="sentiment_analysis",
    )


def analyze_sentiment_huggingface(text: str) -> str:
    """Phân tích cảm xúc bằng API từ Hugging Face"""
    try:
        response = requests.post(HF_API_URL, headers=HEADERS, json={"inputs": text}, timeout=10)
        response.raise_for_status()
        predictions = response.json()

        if isinstance(predictions, list) and predictions:
            sentiment = max(predictions[0], key=lambda x: x["score"])["label"]
        else:
            sentiment = "neutral"  

    except requests.RequestException as e:
        print(f"⚠️ Error calling Hugging Face API: {e}")
        sentiment = "neutral"

    mapping = {"positive": "tích cực", "neutral": "trung bình", "negative": "tiêu cực"}
    return mapping.get(sentiment, "trung bình")


def sentiment_analysis_agent_node(state: State) -> Command[Literal["supervisor"]]:
    """Agent phân tích cảm xúc bài viết tài chính"""
    last_message = state["messages"][-1].content  

    try:
        sentiment = analyze_sentiment_huggingface(last_message)  
    except Exception:
        prompt = f"""
        Đánh giá cảm xúc của bài báo tài chính dưới đây. 
        Trả lời bằng một từ: "tích cực", "tiêu cực" hoặc "trung bình".
        
        Nội dung: {last_message}
        """
        result = llm.invoke([HumanMessage(content=prompt)])
        sentiment = result["messages"][-1].content.strip().lower()

    return Command(
        update={"messages": [HumanMessage(content=sentiment, name="sentiment_analysis")]},
        goto="supervisor",
    )
