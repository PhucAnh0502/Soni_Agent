from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL
from typing import Annotated
import os
import time
import requests
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer


####### INIT ##########
MONGO_URI = "mongodb+srv://uytbvn:13022005@nludatabase.leaih.mongodb.net/?retryWrites=true&w=majority&appName=NLUDatabase"
client = MongoClient(MONGO_URI)
db = client["Soni_Agent"]
collection = db["stock_news"]

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

load_dotenv()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")





####### TOOLS ##########

tavily_tool = TavilySearchResults(max_results=5)

repl = PythonREPL()

@tool
def python_repl_tool(
    code: Annotated[str, "The Python code to execute for calculations."]
):
    """Executes Python code and returns the result."""
    try:
        result = repl.run(code)
        return f"Executed successfully:\n```python\n{code}\n```\nOutput: {result}"
    except Exception as e:
        return f"Execution failed. Error: {repr(e)}"


def clean_html(html_content):
    """Removes scripts, styles, and extracts visible text."""
    soup = BeautifulSoup(html_content, "html.parser")
    for script_or_style in soup(["script", "style"]):
        script_or_style.decompose()
    return soup.get_text(separator=" ", strip=True)


def get_facebook_content(url, headless=True):
    """Extracts content from Facebook using Selenium."""
    options = Options()
    options.headless = headless
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    try:
        driver = webdriver.Chrome(options=options)
        driver.get(url)
        time.sleep(5)

        page_source = driver.page_source
        driver.quit()

        soup = BeautifulSoup(page_source, "html.parser")
        texts = [div.get_text(separator=" ", strip=True) for div in soup.find_all('div')]
        return " ".join(texts)
    except Exception as e:
        return f"Failed to fetch Facebook content: {e}"


def get_web_content(url):
    """Fetches and cleans webpage content."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return clean_html(response.text)
    except Exception as e:
        return f"Failed to fetch webpage content: {e}"


@tool
def extract_info_tool(url: Annotated[str, "The URL to extract information from."]):
    """Extracts text content from a given URL."""
    if "facebook.com" in url or "m.facebook.com" in url:
        return get_facebook_content(url)
    return get_web_content(url)

@tool
def semantic_search_news_db(
    query: str,
    score_threshold: float = 0.7,
    limit: int = 4
) -> list[str]:
    """
    Perform semantic search in MongoDB with score filtering.    

    Args:
        query (str): Search query string.
        score_threshold (float, optional): Minimum similarity score threshold. Default is 0.7.
        limit (int, optional): Maximum number of results. Default is 4.

    Returns:
        list[str]: List of result URLs.
    """
    try:
        query_vector = model.encode(query).tolist()
        
        results = collection.aggregate([
            {"$vectorSearch": {
                "queryVector": query_vector,
                "path": "embedding",
                "numCandidates": 100,
                "limit": limit,
                "index": "PlotSemanticSearch",
                "scoreDetails": "similarity"
            }},
            {"$addFields": {
                "score": {"$meta": "vectorSearchScore"}
            }},
            {"$match": {  
                "score": {"$gte": score_threshold}
            }},
            {"$project": {  
                "_id": 0,
                "full_url": 1,
                "score": 1
            }}
        ])
        
        sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)
        
        urls = [doc["full_url"] for doc in sorted_results]
        
        
        return urls
    
    except Exception as e:
        return []
    
print(semantic_search_news_db("Tp Bank hôm nay"))