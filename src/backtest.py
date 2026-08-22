import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import pandas as pd
import numpy as np
from src.portfolio import portfolio
from config import n_asset, window, rebalance, gamma, method

def rebalancing(signal: pd.DataFrame, returns: pd.DataFrame, market_portfolio_returns: pd.Series, market_portfolio_weights: pd.Series):

    T = len(returns)
    date = returns.index
    next_rebalance_date = date[window]
    signal = signal.drop(columns = "realized_alpha")

    portfolio_weights = {parameter: [] for parameter in gamma}
    portfolio_returns = {parameter: [] for parameter in gamma}
    rebalance_date = []

    top_assets = None
    weights = {parameter: [] for parameter in gamma}

    for t in range(window, T):

        current_date = date[t]
        signal_date = date[t - 1]

        if t == window or current_date >= next_rebalance_date:
            
            current_signal = signal.loc[signal_date].set_index("asset")
            top_assets = current_signal.sort_values("predicted_alpha", ascending = False).head(n_asset).index
            bottom_assets = current_signal.sort_values("predicted_alpha", ascending = True).head(n_asset).index
            selected_asset = top_assets.union(bottom_assets)
            benchmark_weights = market_portfolio_weights.loc[selected_asset]
            benchmark_weights = benchmark_weights / benchmark_weights.sum()
            train = returns.iloc[t - window : t].loc[:, selected_asset].values
            signal_selected = current_signal.loc[selected_asset].values.reshape(-1)
            market_portfolio_returns_window = market_portfolio_returns.iloc[t - window : t].values
            next_rebalance_date = next_rebalance_date + pd.DateOffset(months = rebalance)

            for parameter in gamma:

                weights[parameter]= portfolio(signal_selected, train, benchmark_weights, parameter, market_portfolio_returns_window)

        test = returns.iloc[t].loc[selected_asset].values

        if method == "log":
            
            test = np.exp(test) - 1
            
        for parameter in gamma:
            
            portfolio_weights[parameter].append(pd.Series(weights[parameter], index = selected_asset, name = current_date))
            portfolio_returns[parameter].append(np.dot(weights[parameter], test))

        rebalance_date.append(current_date)

    portfolio_weights = {parameter: pd.DataFrame(portfolio_weights[parameter]).fillna(0) for parameter in gamma}
    portfolio_returns = {parameter: pd.Series(portfolio_returns[parameter], index = rebalance_date, name = "Returns") for parameter in gamma}

    return portfolio_weights, portfolio_returns

def equal_weight(signal: pd.DataFrame, returns: pd.DataFrame):

    T = len(returns)
    date = returns.index
    next_rebalance_date = date[window]
    signal = signal.drop(columns = "realized_alpha")

    portfolio_weights = []
    portfolio_returns = []
    rebalance_date = []

    top_assets = None

    for t in range(window, T):

        current_date = date[t]
        signal_date = date[t - 1]
        
        if t == window or current_date >= next_rebalance_date:
            
            current_signal = signal.loc[signal_date].set_index("asset")
            top_assets = current_signal.sort_values("predicted_alpha", ascending = False).head(n_asset).index
            bottom_assets = current_signal.sort_values("predicted_alpha", ascending = True).head(n_asset).index
            selected_asset = top_assets.union(bottom_assets)
            next_rebalance_date = next_rebalance_date + pd.DateOffset(months = rebalance)

        test = returns.iloc[t].loc[selected_asset].values
        n_long = len(top_assets)
        n_short = len(bottom_assets)
        weights_long = np.full(n_long, 1 / n_long)
        weights_shorts = np.full(n_short, - 1 / n_short)
        weights = np.concatenate((weights_long, weights_shorts), axis = 0)

        if method == "log":
            test = np.exp(test) - 1

        portfolio_weights.append(pd.Series(weights, index = selected_asset, name = current_date))
        portfolio_returns.append(np.dot(weights, test))
        rebalance_date.append(current_date)

    portfolio_weights = pd.DataFrame(portfolio_weights).fillna(0)
    portfolio_returns = pd.Series(portfolio_returns, index = rebalance_date, name = "Returns")

    return portfolio_weights, portfolio_returns