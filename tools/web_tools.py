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

