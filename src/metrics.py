import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import pandas as pd
import numpy as np
from config import window, rf, cost_rate, gamma

def portfolio_metrics(return_portfolio: dict, weights_portfolio: dict, market_portfolio_returns: pd.Series):
    
    net_return = {}
    turnover = {}
    sharpe_annual = {}
    sharpe_rolling = {}
    drawdown = {}
    informatio_ratio = {}
    effective_number = {}

    for parameter in gamma:
        
        turnover[parameter] = 0.5 * weights_portfolio[parameter].diff().abs().sum(axis = 1).fillna(0)
        turnover[parameter].name = "Turnover"
        net_return[parameter] = return_portfolio[parameter] - (turnover[parameter] * cost_rate)
        net_return[parameter].name = "Net Return"
        sharpe_rolling[parameter] = (net_return[parameter] - rf).rolling(window).mean() / net_return[parameter].rolling(window).std() * np.sqrt(252)
        sharpe_rolling[parameter].name = "Rolling Sharpe"
        sharpe_annual[parameter] = (net_return[parameter] - rf).mean() / net_return[parameter].std() * np.sqrt(252)
        wealth = (1 + net_return[parameter]).cumprod()
        drawdown[parameter] = (wealth - wealth.cummax()) / wealth.cummax()
        drawdown[parameter].name = "Drawdown"
        active_return = net_return[parameter] - market_portfolio_returns
        informatio_ratio[parameter] = (active_return.mean() / active_return.std() * np.sqrt(252))
        effective_number[parameter] = 1 / (weights_portfolio[parameter]**2).sum(axis = 1)
        effective_number[parameter].name = "Effective Number"

    return net_return, turnover, sharpe_annual, sharpe_rolling, drawdown, informatio_ratio, effective_number

def equal_weights_metrics(return_portfolio: dict, weights_portfolio: dict, market_portfolio_returns: pd.Series):
        
    turnover = 0.5 * weights_portfolio.diff().abs().sum(axis = 1).fillna(0)
    turnover.name = "Turnover"
    net_return = return_portfolio - (turnover * cost_rate)
    net_return.name = "Net Return"
    sharpe_rolling = (net_return - rf).rolling(window).mean() / net_return.rolling(window).std() * np.sqrt(252)
    sharpe_rolling.name = "Rolling Sharpe"
    sharpe_annual = (net_return - rf).mean() / net_return.std() * np.sqrt(252)
    wealth = (1 + net_return).cumprod()
    drawdown = (wealth - wealth.cummax()) / wealth.cummax()
    drawdown.name = "Drawdown"
    active_return = net_return - market_portfolio_returns
    informatio_ratio = (active_return.mean() / active_return.std() * np.sqrt(252))
    effective_number = 1 / (weights_portfolio**2).sum(axis = 1)
    effective_number.name = "Effective Number"

    return net_return, turnover, sharpe_annual, sharpe_rolling, drawdown, informatio_ratio, effective_number

def benchmark_metrics(market_portfolio_returns: pd.Series, market_portfolio_weights: pd.Series):
        
    sharpe_rolling = (market_portfolio_returns - rf).rolling(window).mean() / market_portfolio_returns.rolling(window).std() * np.sqrt(252)
    sharpe_rolling.name = "Rolling Sharpe"
    sharpe_annual = (market_portfolio_returns - rf).mean() / market_portfolio_returns.std() * np.sqrt(252)
    wealth = (1 + market_portfolio_returns).cumprod()
    drawdown = (wealth - wealth.cummax()) / wealth.cummax()
    drawdown.name = "Drawdown"

    return sharpe_annual, sharpe_rolling, drawdown