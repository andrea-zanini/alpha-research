import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import pandas as pd
from src.data import get_sp500_tickers, get_data, get_capitalization, compute_returns, get_market_returns
from src.factors import factors, build_panel

BASE_DIR= Path(__file__).resolve().parent.parent
price_path = BASE_DIR / "results" / "prices"
capitalization_path = BASE_DIR / "results" / "capitalization"
factors_path = BASE_DIR / "results" / "factors"
price_path.mkdir(parents = True, exist_ok = True)
capitalization_path.mkdir(parents = True, exist_ok = True)
factors_path.mkdir(parents = True, exist_ok = True)

tickers = get_sp500_tickers()
prices = get_data(tickers)
capitalization = get_capitalization(prices)
log_returns_daily = compute_returns(prices)
log_returns_daily.index = log_returns_daily.index.astype("datetime64[ns]")
market_portfolio_returns, market_portfolio_weights = get_market_returns(log_returns_daily, capitalization)

prices.to_parquet(price_path / f"daily_prices.parquet")
log_returns_daily.to_parquet(price_path / f"log_returns_daily.parquet")
capitalization.to_frame().to_parquet(capitalization_path / f"capitalization.parquet")
market_portfolio_returns.to_frame().to_parquet(price_path / f"market_portfolio.parquet")
pd.Series(market_portfolio_weights, index=capitalization.index, name = "Weight").to_frame().to_parquet(capitalization_path / "market_portfolio_weights.parquet")

signal = factors()
panel = build_panel(signal)
panel.to_parquet(factors_path / f"factors.parquet")