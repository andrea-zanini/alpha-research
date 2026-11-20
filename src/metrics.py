import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import pandas as pd
import numpy as np
from config import rf_daily, cost_rate

def portfolio_metrics(return_portfolio: pd.DataFrame, weights_portfolio: pd.DataFrame, market_portfolio_returns: pd.Series):

    turnover = 0.5 * weights_portfolio.diff().abs().sum(axis = 1).fillna(0)
    turnover.name = "Turnover"
    net_return = return_portfolio - (turnover * cost_rate)
    net_return.name = "Net Return"
    sharpe_ratio = (net_return - rf_daily).mean() / net_return.std() * np.sqrt(252)
    wealth = (1 + net_return).cumprod()
    drawdown = (wealth - wealth.cummax()) / wealth.cummax()
    drawdown.name = "Drawdown"
    active_return = net_return - market_portfolio_returns
    informatio_ratio = (active_return.mean() / active_return.std() * np.sqrt(252))
    effective_number = 1 / (weights_portfolio**2).sum(axis = 1)
    effective_number.name = "Effective Number"
    
    return net_return, turnover, sharpe_ratio, drawdown, informatio_ratio, effective_number

def benchmark_metrics(market_portfolio_returns: pd.Series):

    sharpe_ratio = (market_portfolio_returns - rf_daily).mean() / market_portfolio_returns.std() * np.sqrt(252)
    wealth = (1 + market_portfolio_returns).cumprod()
    drawdown = (wealth - wealth.cummax()) / wealth.cummax()
    drawdown.name = "Drawdown"

    return sharpe_ratio, drawdown