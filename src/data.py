import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import yfinance as yf
from config import SP500_URL, horizon, time, threshold, method

def get_sp500_tickers():
    
    headers = {"User-Agent": "Mozzila/5.0"}
    response = requests.get(SP500_URL, headers = headers)
    soup = BeautifulSoup(response.text, "html.parser")
    table = soup.find("table", {"id": "constituents"})
    rows = table.find_all("tr")[1:]
    tickers = [row.find("td").get_text(strip = True) for row in rows if row.find("td")]
    
    return tickers

def get_sectors():
    
    headers = {"User-Agent": "Mozzila/5.0"}
    response = requests.get(SP500_URL, headers = headers)
    soup = BeautifulSoup(response.text, "html.parser")
    table = soup.find("table", {"id": "constituents"})
    rows = table.find_all("tr")[1:]
    sectors = [row.find_all("td")[2].get_text(strip = True) for row in rows if row.find("td")]
    
    return sectors

def get_data(tickers: list):
    
    tickers = [asset.replace(".", "-") for asset in tickers]
    n = len(tickers)
    step = 50
    chunks = []
    
    for i in range(0, n, step):
        
        data: pd.DataFrame = yf.download(tickers[i: i+step], period = horizon, interval = time, auto_adjust = False)
        
        if "Adj Close" in data:
            group: pd.DataFrame = data["Adj Close"]
        else:
            group: pd.DataFrame = data["Close"]
            
        group.index = pd.to_datetime(group.index).astype("datetime64[ns]")
        
        if group.isna().sum().sum() > 0:
            mask = group.isna().mean(axis = 0) <= (1 - threshold)
            group = group.loc[:, mask]
            group.ffill(inplace = True)
            group.bfill(inplace = True)
            
        chunks.append(group)
        
    prices: pd.DataFrame = pd.concat(chunks, axis = 1)
    
    return prices

def compute_returns(prices: pd.DataFrame):
    
    if method == "log":
        returns: pd.DataFrame = np.log(prices).diff().dropna()
    elif method == "simple":
        returns: pd.DataFrame = prices.pct_change().dropna()
    else:
        raise ValueError("method must be log or simple")
    
    return returns

def get_capitalization(prices: pd.DataFrame):
    
    market_caps = []
    
    for asset in prices.columns:
        
        try:
            capitalization = yf.Ticker(asset).fast_info["market_cap"]
        except Exception:
            capitalization = None
            
        market_caps.append(capitalization)
        
    market_caps = pd.Series(market_caps, index = prices.columns)
    market_caps.rename("Capitalization", inplace = True)
    market_caps.fillna(0, inplace = True)
    
    return market_caps

def get_market_returns(returns: pd.DataFrame, capitalization: pd.Series):
    
    weights = capitalization.values / capitalization.values.sum()
    market_returns = returns.values @ weights
    market_returns = pd.Series(market_returns, index = returns.index, name = "Returns")
    
    return market_returns, weights