import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import pandas as pd
import numpy as np
from src.diagnostic import feature_correlation, feature_significant, feature_distribution, signal_significant
from src.regression import linear_signal, non_linear_signal, signal
from src.backtest import rebalancing
from src.sanity_check import sum_weights

BASE_DIR = Path(__file__).resolve().parent

factors_path = BASE_DIR / "results" / "factors"
price_path = BASE_DIR / "results" / "prices"
portfolio_path = BASE_DIR / "results" / "portfolio"
portfolio_path.mkdir(parents = True, exist_ok = True)

target = pd.read_parquet(factors_path / f"target.parquet").squeeze()
features = pd.read_parquet(factors_path / f"factors.parquet")
standardizzator = pd.read_parquet(factors_path / f"standardizzator.parquet").squeeze()
returns = pd.read_parquet(price_path / f"log_returns_daily.parquet")
market_returns = pd.read_parquet(price_path / f"market_portfolio.parquet").squeeze()

feature_correlation(features)
feature_significant(target, features)
feature_distribution(features)

predictions = linear_signal()
residual = non_linear_signal()
complete_signal = signal()

signal_significant(complete_signal, target)
portfolio_weights, portfolio_returns = rebalancing(complete_signal, returns, market_returns)

portfolio_returns.to_frame().to_parquet(portfolio_path / "portfolio_returns.parquet")
portfolio_weights.to_parquet(portfolio_path / "portfolio_weights.parquet")
sum_weights(portfolio_weights)