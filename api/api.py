from dotenv import load_dotenv
from fastapi import FastAPI
from sentence_transformers import SentenceTransformer
from pymongo import MongoClient
from pymongo.collection import Collection
from typing import Annotated
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import traceback
import os
import time

app = FastAPI()

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
MONGO_URI = os.getenv("MONGODB_URI")
client = MongoClient(MONGO_URI)
db = client["Soni_Agent"]
collection: Collection = db["stock_news"]

import requests
from bs4 import BeautifulSoup

load_dotenv()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

def clean_html(html_content):
    """Removes scripts, styles, and extracts visible text."""
    soup = BeautifulSoup(html_content, "html.parser")
    for script_or_style in soup(["script", "style"]):
        script_or_style.decompose()
    return soup.get_text(separator=" ", strip=True)

def get_web_content(url: str) -> str:
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        return soup.get_text()
    except Exception as e:
        return f"Error fetching content: {e}"
    return f"Mock content from {url}"

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
    
def extract_info_tool(url: Annotated[str, "The URL to extract information from."]):
    """Extracts text content from a given URL."""
    if "facebook.com" in url or "m.facebook.com" in url:
        return get_facebook_content(url)
    return get_web_content(url)

@app.get("/extract_content/")
def extract_content(url: str):
    """Extracts content from a given webpage."""
    return {"content": extract_info_tool(url)}

@app.get("/")
def home():
    return {"message": "Welcome to the Semantic Search API!"}