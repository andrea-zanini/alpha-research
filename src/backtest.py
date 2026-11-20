import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import pandas as pd
import numpy as np
from src.portfolio import portfolio
from config import window_portofolio, rebalance, method

def rebalancing(signal: pd.Series, returns: pd.DataFrame, market_portfolio_returns: pd.Series):

    common_index = signal.index.get_level_values(level = "Date").unique().intersection(returns.index).intersection(market_portfolio_returns.index)
    signal = signal.loc[common_index]
    returns = returns.loc[common_index]
    market_portfolio_returns = market_portfolio_returns.loc[common_index]
    
    T = len(returns)
    date = returns.index
    next_rebalance_date = date[window_portofolio]

    portfolio_weights = []
    portfolio_returns = []
    portfolio_date = []
    
    previous_weights = np.zeros(returns.shape[1], dtype = float)

    for t in range(window_portofolio, T):

        current_date = date[t]
        signal_date = date[t - 1]
        
        current_signal = signal.loc[signal_date].values.reshape(-1)
        train = returns.iloc[t - window_portofolio : t].values
        market_portfolio_returns_window = market_portfolio_returns.iloc[t - window_portofolio : t].values

        if t == window_portofolio or current_date >= next_rebalance_date:
            
            next_rebalance_date = next_rebalance_date + pd.DateOffset(months = rebalance)
            weights, previous_weights = portfolio(current_signal, train, previous_weights, market_portfolio_returns_window)

        test = returns.iloc[t].values

        if method == "log":
            
            test = np.exp(test) - 1
            
        portfolio_weights.append(weights)
        portfolio_returns.append(np.dot(weights, test))
        portfolio_date.append(current_date)

    portfolio_weights = pd.DataFrame(portfolio_weights, index = portfolio_date, columns = returns.columns).fillna(0)
    portfolio_returns = pd.Series(portfolio_returns, index = portfolio_date, name = "Returns")

    return portfolio_weights, portfolio_returns