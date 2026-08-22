import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import pandas as pd
import numpy as np
from config import gamma, method
from src.regression import compute_signal
from src.backtest import rebalancing, equal_weight
from src.metrics import portfolio_metrics, equal_weights_metrics, benchmark_metrics
from src.sanity_check import sum_weights

BASE_DIR = Path(__file__).resolve().parent
price_path = BASE_DIR / "results" / "prices"
capitalization_path = BASE_DIR / "results" / "capitalization"
factors_path = BASE_DIR / "results" / "factors"
portfolio_path = BASE_DIR / "results" / "portfolio"
metrics_path = BASE_DIR / "results" / "metrics"
portfolio_path.mkdir(parents = True, exist_ok = True)
metrics_path.mkdir(parents = True, exist_ok = True)

log_returns_daily = pd.read_parquet(price_path / "log_returns_daily.parquet")
capitalization = pd.read_parquet(capitalization_path / "capitalization.parquet").squeeze()
market_portfolio_returns = pd.read_parquet(price_path / "market_portfolio.parquet").squeeze()
market_portfolio_weights = pd.read_parquet(capitalization_path / "market_portfolio_weights.parquet").squeeze()

signal, metrics = compute_signal()

log_returns_daily = log_returns_daily.loc[signal.index.unique()]

portfolio_weights, portfolio_returns = rebalancing(signal, log_returns_daily, market_portfolio_returns, market_portfolio_weights)
equal_weight_weights, equal_weight_returns = equal_weight(signal, log_returns_daily)

market_portfolio_returns = market_portfolio_returns.loc[portfolio_returns[1].index]

if method == "log":
    
    market_portfolio_returns = np.exp(market_portfolio_returns) - 1

sum_weights(portfolio_weights)

portfolio_net_return, portfolio_turnover, portfolio_sharpe_ratio_annual, portfolio_sharpe_ratio_rolling, portfolio_drawdown, portfolio_informatio_ratio, portfolio_effective_number = portfolio_metrics(portfolio_returns, portfolio_weights, market_portfolio_returns)
equal_weight_net_return, equal_weight_turnover, equal_weight_sharpe_ratio_annual, equal_weight_sharpe_ratio_rolling, equal_weight_drawdown, equal_weight_informatio_ratio, equal_weight_effective_number = equal_weights_metrics(equal_weight_returns, equal_weight_weights, market_portfolio_returns)
market_portfolio_sharpe_ratio_annual, market_portfolio_sharpe_ratio_rolling, market_portfolio_drawdown = benchmark_metrics(market_portfolio_returns, market_portfolio_weights)

for parameter in gamma:
    portfolio_net_return[parameter].to_frame().to_parquet(portfolio_path / f"portfolio_net_returns_{parameter}.parquet")
    portfolio_weights[parameter].to_parquet(portfolio_path / f"portfolio_weights_{parameter}.parquet")
    portfolio_turnover[parameter].to_frame().to_parquet(metrics_path / f"portfolio_turnover_{parameter}.parquet")
    pd.DataFrame([portfolio_sharpe_ratio_annual[parameter]], columns = ["Annual Sharpe"], index = [parameter]).to_parquet(metrics_path / f"portfolio_sharpe_ratio_{parameter}_annual.parquet")
    portfolio_sharpe_ratio_rolling[parameter].to_frame().to_parquet(metrics_path / f"portfolio_sharpe_ratio_{parameter}_rolling.parquet")
    portfolio_drawdown[parameter].to_frame().to_parquet(metrics_path / f"portfolio_drawdown_{parameter}.parquet")
    pd.DataFrame([portfolio_informatio_ratio[parameter]], columns = ["Information Ratio"], index = [parameter]).to_parquet(metrics_path / f"portfolio_information_ratio_{parameter}.parquet")
    portfolio_effective_number[parameter].to_frame().to_parquet(metrics_path / f"portfolio_effective_number_{parameter}.parquet")
    
equal_weight_net_return.to_frame().to_parquet(portfolio_path / f"equal_weight_returns.parquet")
equal_weight_weights.to_parquet(portfolio_path / f"equal_weight_weights.parquet")
equal_weight_turnover.to_frame().to_parquet(metrics_path / f"equal_weight_turnover.parquet")
pd.DataFrame([equal_weight_sharpe_ratio_annual], columns = ["Annual Sharpe"]).to_parquet(metrics_path / f"equal_weight_sharpe_ratio_annual.parquet")
equal_weight_sharpe_ratio_rolling.to_frame().to_parquet(metrics_path / f"equal_weight_sharpe_ratio_rolling.parquet")
equal_weight_drawdown.to_frame().to_parquet(metrics_path / f"equal_weight_drawdown.parquet")
pd.DataFrame([equal_weight_informatio_ratio], columns = ["Information Ratio"]).to_parquet(metrics_path / f"equal_weight_informatio_ratio.parquet")
equal_weight_effective_number.to_frame().to_parquet(metrics_path / f"equal_weight_effective_number.parquet")

pd.DataFrame([market_portfolio_sharpe_ratio_annual], columns = ["Annual Sharpe"]).to_parquet(metrics_path / f"market_portfolio_sharpe_ratio_annual.parquet")
market_portfolio_sharpe_ratio_rolling.to_frame().to_parquet(metrics_path / f"market_portfolio_sharpe_ratio_rolling.parquet")
market_portfolio_drawdown.to_frame().to_parquet(metrics_path / f"market_portfolio_drawdown.parquet")