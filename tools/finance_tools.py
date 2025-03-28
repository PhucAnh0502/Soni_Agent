from langchain_core.tools import tool
from typing import Annotated
from vnstock import Vnstock
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns
import os
from dotenv import load_dotenv
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer

############## INIT ##############
load_dotenv()
MONGO_URI = os.getenv("MONGODB_URI")
client = MongoClient(MONGO_URI)
db = client["Soni_Agent"]
collection = db["stock_news"]
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")




############## GET STOCK DATA ################
@tool
def get_stock_data(
    symbol_and_dates: Annotated[str, "Combination of stock symbol, start date, end date, and interval separated by '|'"
    "Example: 'VNM|2025-01-01|2025-03-27|1D'"]
):
    """Fetches historical stock data."""
    parts = symbol_and_dates.split('|')
    if len(parts) != 4:
        return f"Error: Invalid input format. Expected 'symbol|start_date|end_date|interval'"
    
    symbol, start_date, end_date, interval = parts
    
    stock = Vnstock().stock(symbol=symbol, source="VCI")
    df = stock.quote.history(start=start_date, end=end_date, interval=interval)
    return df

@tool 
def get_internal_reports(symbol: Annotated[str, "The stock symbol to get internal reports for."]):
    """Fetches internal reports for a given stock symbol."""
    from vnstock.explorer.vci import Company
    company = Company(symbol)
    data_report = company.reports()
    return data_report

@tool
def semantic_search_news_db(
    query: str,
    score_threshold: float = 0.7,
    limit: int = 3
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


############## PLOTTING TOOLS ################

@tool
def plot_volume_chart(
    symbol_and_dates: Annotated[str, "Combination of stock symbol, start date, end date, and interval separated by '|'"]
):
    """Plots the volume chart for a given stock symbol."""
    parts = symbol_and_dates.split('|')
    if len(parts) != 4:
        return f"Error: Invalid input format. Expected 'symbol|start_date|end_date|interval'"
    
    symbol, start_date, end_date, interval = parts
    
    df = get_stock_data.run(symbol_and_dates)
    
    plt.figure(figsize=(10, 5))
    plt.bar(df['time'], df['volume'], color='g', alpha=0.7)
    plt.title(f'Volume Chart - {symbol}')
    plt.xlabel('Date')
    plt.ylabel('Volume')
    plt.grid()
    plt.savefig(f"{symbol}_volume_chart.png")
    plt.close() 
    return f"Volume chart saved as {symbol}_volume_chart.png"

@tool
def plot_line_chart(
    symbol_and_dates: Annotated[str, "Combination of stock symbol, start date, end date, and interval separated by '|'"]
):
    """Plots the line chart for a given stock symbol."""
    parts = symbol_and_dates.split('|')
    if len(parts) != 4:
        return f"Error: Invalid input format. Expected 'symbol|start_date|end_date|interval'"
    
    symbol, start_date, end_date, interval = parts
    
    df = get_stock_data.run(symbol_and_dates)
    
    plt.figure(figsize=(10, 5))
    plt.plot(df['time'], df['close'], label=symbol, color='b')
    plt.title(f'Line Chart - {symbol}')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    plt.grid()
    plt.savefig(f"{symbol}_line_chart.png")
    plt.close()  
    return f"Line chart saved as {symbol}_line_chart.png"

@tool
def plot_candlestick(
    symbol_and_dates: Annotated[str, "Combination of stock symbol, start date, end date, and interval separated by '|'"
    "Example: 'VNM|2025-01-01|2025-03-27|1D'"]
):
    """Plots the candlestick chart for a given stock symbol."""
    parts = symbol_and_dates.split('|')
    if len(parts) != 4:
        return f"Error: Invalid input format. Expected 'symbol|start_date|end_date|interval'"
    
    symbol, start_date, end_date, interval = parts
    
    df = get_stock_data.run(symbol_and_dates)
    
    fig = go.Figure(data=[
        go.Candlestick(x=df['time'],
                       open=df['open'],
                       high=df['high'],
                       low=df['low'],
                       close=df['close'],
                       name=symbol)
    ])
    fig.update_layout(title=f'Candlestick Chart - {symbol}', xaxis_title='Date', yaxis_title='Price')
    fig.write_image(f"{symbol}_candlestick.png")
    return f"Candlestick chart saved as {symbol}_candlestick.png"

@tool
def plot_volume_and_closed_price(
    symbol_and_dates: Annotated[str, "Combination of stock symbol, start date, end date, and interval separated by '|'"
    "Example: 'VNM|2025-01-01|2025-03-27|1D'"]
):
    """Plots a combo chart with volume as bars and close price as a line."""
    parts = symbol_and_dates.split('|')
    if len(parts) != 4:
        return f"Error: Invalid input format. Expected 'symbol|start_date|end_date|interval'"
    
    symbol, start_date, end_date, interval = parts
    
    df = get_stock_data.run(symbol_and_dates)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df['time'],
        y=df['volume'],
        name="Volume",
        marker_color='blue',
        yaxis='y1'  
    ))

    fig.add_trace(go.Scatter(
        x=df['time'],
        y=df['close'],
        name="Close Price",
        mode='lines',
        line=dict(color='red', width=2),
        yaxis='y2'  
    ))

    fig.update_layout(
        title=f'Giá đóng cửa và khối lượng giao dịch - {symbol}',
        xaxis=dict(title='Date'),
        yaxis=dict(title='Volume (M)', side='left', showgrid=False),
        yaxis2=dict(title='Price (K)', overlaying='y', side='right', showgrid=False),
        legend=dict(x=0.01, y=0.99),
        bargap=0.2,
        width=900,
        height=500
    )

    fig.write_image(f"{symbol}_volume_price.png")
    return f"Volume and price chart saved as {symbol}_volume_price.png"


@tool
def plot_shareholders_piechart(symbol: Annotated[str, "The stock symbol to plot shareholders pie chart for."]):
    """Plots a pie chart of shareholders for a given stock symbol."""
    company = Vnstock().stock(symbol=symbol, source="VCI").company
    shareholders_df = company.shareholders()
    
    threshold = 0.03  

    total_quantity = shareholders_df['quantity'].sum()

    shareholders_df['share_own_percent'] = shareholders_df['quantity'] / total_quantity

    major_shareholders = shareholders_df[shareholders_df['share_own_percent'] >= threshold].copy()

    other_share = shareholders_df[shareholders_df['share_own_percent'] < threshold]['quantity'].sum()

    if other_share > 0 and not major_shareholders.empty:
        other_row = pd.DataFrame({'share_holder': ['Others'], 'quantity': [other_share]})
        major_shareholders = pd.concat([major_shareholders, other_row], ignore_index=True)
    elif major_shareholders.empty:
        major_shareholders = shareholders_df.copy()

    major_shareholders['share_own_percent'] = (major_shareholders['quantity'] / major_shareholders['quantity'].sum()) * 100

    fig, ax = plt.subplots(figsize=(10, 6))
    explode = [0.1 if label == 'Others' else 0 for label in major_shareholders['share_holder']]

    ax.pie(
        major_shareholders['share_own_percent'],
        labels=major_shareholders['share_holder'],
        autopct='%1.1f%%',
        colors=plt.cm.Paired.colors,
        startangle=140,
        pctdistance=0.8,
        labeldistance=1.1,
        explode=explode
    )

    ax.set_title(f"Cổ đông lớn {symbol} ")
    plt.savefig(f"shareholders_{symbol}_pie.png", dpi=300, bbox_inches="tight")
    plt.close()



@tool
def plot_monthly_returns_heatmap(
    symbol_and_dates: Annotated[str, "Combination of stock symbol, start date, end date, and interval separated by '|'"]
):
    """
    Creates a heatmap of monthly average returns for a given stock symbol.
    
    Input format: 'symbol|start_date|end_date|interval'
    Returns a saved heatmap image.
    """
    parts = symbol_and_dates.split('|')
    if len(parts) != 4:
        return f"Error: Invalid input format. Expected 'symbol|start_date|end_date|interval'"
    
    symbol, start_date, end_date, interval = parts
    
    try:
        df = get_stock_data.run(symbol_and_dates)
        
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
        
        df['returns'] = df['close'].pct_change() * 100
        
  
        return_pivot = pd.pivot_table(
            df, 
            index=df.index.year, 
            columns=df.index.month, 
            values='returns', 
            aggfunc='mean'
        )
        
        # Set up the plot
        plt.figure(figsize=(12, 8))
        
        # Create heatmap
        sns.heatmap(
            return_pivot, 
            annot=True, 
            cmap='RdYlGn', 
            center=0, 
            fmt='.2f'
        )
        
        plt.title(f'Monthly Average Returns - {symbol} ({start_date} to {end_date})', fontsize=15)
        plt.xlabel('Month', fontsize=12)
        plt.ylabel('Year', fontsize=12)
        
        filename = f"{symbol}_returns_heatmap.png"
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
        
        return f"Returns heatmap saved as {filename}"
    
    except Exception as e:
        return f"Error generating heatmap: {str(e)}"
