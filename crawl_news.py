import requests
from bs4 import BeautifulSoup
import time
import urllib.parse
import re
import logging
from datetime import datetime, timedelta
from pymongo import MongoClient
import pymongo
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_full_url(base_url, relative_url):
    if not relative_url:
        return ""
    return urllib.parse.urljoin(base_url, relative_url.lstrip('/'))

def parse_relative_time(relative_time):
    current_time = datetime.now()
    match = re.search(r"(\d+)\s*(giờ|phút)", relative_time)
    if match:
        amount, unit = int(match.group(1)), match.group(2)
        if unit == "giờ":
            return (current_time - timedelta(hours=amount)).timestamp()
        elif unit == "phút":
            return (current_time - timedelta(minutes=amount)).timestamp()
    return current_time.timestamp()

def get_article_details(article_url, headers, model):
    try:
        article_response = requests.get(article_url, headers=headers)
        article_response.encoding = 'utf-8'
        article_soup = BeautifulSoup(article_response.text, 'html.parser')

        description = "Không có mô tả"
        desc_tag = (
            article_soup.find("p", class_="sapo") or 
            article_soup.find("div", class_="sapo") or 
            article_soup.find("meta", {"name": "description"}) or 
            article_soup.find("meta", {"property": "og:description"})
        )
        if desc_tag:
            description = desc_tag.get("content", "").strip() if desc_tag.name == "meta" else desc_tag.get_text(strip=True)
        if description == "Không có mô tả":
            first_paragraph = article_soup.select_one('div.detail-content p')
            if first_paragraph:
                description = first_paragraph.get_text(strip=True)


        embedding = model.encode(description).tolist() if description else []

        post_time = None
        time_tag = article_soup.find("span", class_="time") or article_soup.find("div", class_="time")
        time_ago_tag = article_soup.find("span", class_="time-ago")  
        meta_time = article_soup.find("meta", {"property": "article:published_time"})
        
        if meta_time:
            post_time = datetime.strptime(meta_time.get("content", "").strip(), "%Y-%m-%dT%H:%M:%S").timestamp()
        elif time_ago_tag:
            post_time = parse_relative_time(time_ago_tag.get_text(strip=True))
        elif time_tag:
            post_time = parse_relative_time(time_tag.get_text(strip=True))
        else:
            post_time = time.time()

        return description, post_time, embedding
    except Exception as e:
        logging.error(f"Lỗi khi lấy bài viết {article_url}: {e}")
        return "Không có mô tả", time.time(), []

def crawl_news_urls(sites, model):
    headers = {"User-Agent": "Mozilla/5.0"}
    news_urls = []
    
    for site in sites:
        try:
            response = requests.get(site["url"], headers=headers)
            response.encoding = 'utf-8'
            soup = BeautifulSoup(response.text, 'html.parser')
            
            for selector in site["selectors"]:
                for link in soup.select(selector):
                    href = link.get('href', '').strip()
                    title = link.get_text(strip=True)  
                    if not href or not title:
                        continue
                    full_url = get_full_url(site["url"], href)
                    description, post_timestamp, embedding = get_article_details(full_url, headers, model)
                    
                    news_urls.append({
                        "title": title,
                        "full_url": full_url,
                        "description": description,
                        "post_time": post_timestamp,
                        "crawl_timestamp": time.time(),
                        "embedding": embedding
                    })
        except Exception as e:
            logging.error(f"Lỗi crawl {site['url']}: {e}")
    
    return news_urls

def main():
    load_dotenv()
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    mongodb_url = os.getenv("MONGODB_URI")
    client = MongoClient(mongodb_url)
    db = client["Soni_Agent"]
    collection = db["stock_news"]
    config_collection = db["configs"]

    sites = [
        {"url": "https://cafef.vn/thi-truong-chung-khoan.chn", "selectors": ['h3.title a', 'div.box-category-item a', 'article a']},
        {"url": "https://vnexpress.net/kinh-doanh/chung-khoan", "selectors": ['h3.title-news a', 'article a']},
        {"url": "https://tuoitre.vn/kinh-doanh.htm", "selectors": ['h3.title-news a', 'article a']}
    ]

    while True:
        last_crawl = config_collection.find_one({"name": "last_crawl_timestamp"})
        last_crawl_timestamp = last_crawl["timestamp"] if last_crawl else 0
        
        news_data = crawl_news_urls(sites, model)
        if news_data:
            for news in news_data:
                if news["post_time"] > last_crawl_timestamp and not collection.find_one({"full_url": news["full_url"]}):
                    collection.insert_one(news)
                    logging.info(f"Đã thêm: {news['title']}")
                else:
                    logging.info(f"Bỏ qua bài viết đã tồn tại hoặc cũ: {news['title']}")
            
            config_collection.update_one(
                {"name": "last_crawl_timestamp"}, 
                {"$set": {"timestamp": time.time()}}, 
                upsert=True
            )
        else:
            logging.info("Không tìm thấy bài viết nào mới.")
        
        logging.info("Chờ 1h trước lần crawl tiếp theo...")
        time.sleep(3600*60)

if __name__ == "__main__":
    main()