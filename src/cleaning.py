import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import pandas as pd
from src.data import get_sp500_tickers, get_sectors, get_data, compute_returns, get_capitalization, get_market_returns
from src.factors import factors, build_panel

BASE_DIR = Path(__file__).resolve().parent.parent
price_path = BASE_DIR / "results" / "prices"
info_path = BASE_DIR / "results" / "info"
factors_path = BASE_DIR / "results" / "factors"
price_path.mkdir(parents = True, exist_ok = True)
info_path.mkdir(parents = True, exist_ok = True)
factors_path.mkdir(parents = True, exist_ok = True)

# tickers = get_sp500_tickers()
# gics_sectors = get_sectors()
# prices = get_data(tickers)
# log_returns_daily = compute_returns(prices)
# capitalization = get_capitalization(prices)
# market_portfolio_returns, market_portfolio_weights = get_market_returns(log_returns_daily, capitalization)

# log_returns_daily.index = log_returns_daily.index.astype("datetime64[ns]")
# maps = dict(zip(tickers, gics_sectors))
# sectors = prices.columns.to_series().map(maps).fillna("Others").rename("Sectors")

# prices.to_parquet(price_path / f"daily_prices.parquet")
# log_returns_daily.to_parquet(price_path / f"log_returns_daily.parquet")
# capitalization.to_frame().to_parquet(info_path / f"capitalization.parquet")
# sectors.to_frame().to_parquet(info_path / "sectors.parquet")
# market_portfolio_returns.to_frame().to_parquet(price_path / f"market_portfolio.parquet")
# pd.Series(market_portfolio_weights, index = capitalization.index, name = "Weight").to_frame().to_parquet(info_path / "market_portfolio_weights.parquet")

y, x, volatility = factors()
targets, features, standardizzator = build_panel(y, x, volatility)

targets.to_frame().to_parquet(factors_path / f"target.parquet")
features.to_parquet(factors_path / f"factors.parquet")
standardizzator.to_frame().to_parquet(factors_path / f"standardizzator.parquet")