import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import pandas as pd
import numpy as np
from config import method, window, minimum_returns

BASE_DIR= Path(__file__).resolve().parent.parent
price_path = BASE_DIR / "results" / "prices"

def factors():

    returns_daily = pd.read_parquet(price_path / "log_returns_daily.parquet")
    market_portfolio_return = pd.read_parquet(price_path / "market_portfolio.parquet").squeeze()
    prices = pd.read_parquet(price_path / "daily_prices.parquet")

    x = {}

    for asset in prices:

        y = returns_daily[asset].rolling(126).sum().shift(-126)

        moving_average_200 = prices[asset].rolling(200).mean()
        moving_average_distance = prices[asset] / moving_average_200 - 1
        maximum_distance = prices[asset] / prices[asset].rolling(window).max() - 1

        if method == "log":
            momentum_signal_1m = returns_daily[asset].rolling(22).sum()
            momentum_signal_3m = returns_daily[asset].rolling(66).sum()
            momentum_signal_6m = returns_daily[asset].rolling(132).sum()
            momentum_signal_12m = returns_daily[asset].shift(22).rolling(window).sum()

        else:
            momentum_signal_1m = (1 + returns_daily[asset]).rolling(22).apply(np.prod, raw = True) - 1
            momentum_signal_3m = (1 + returns_daily[asset]).rolling(66).apply(np.prod, raw = True) - 1
            momentum_signal_6m = (1 + returns_daily[asset]).rolling(132).apply(np.prod, raw = True) - 1
            momentum_signal_12m = (1 + returns_daily[asset].shift(22)).rolling(252).apply(np.prod, raw = True) - 1

        volatility_1m = returns_daily[asset].rolling(22).std()
        volatility_3m = returns_daily[asset].rolling(66).std()
        volatility_12m = returns_daily[asset].rolling(window).std()
        downside_volatility = np.minimum(0, returns_daily[asset] - minimum_returns).pow(2).rolling(window).mean().pow(0.5)

        covariance = returns_daily[asset].rolling(window).cov(market_portfolio_return)
        market_variance = market_portfolio_return.rolling(window).var()
        beta_rolling = covariance / market_variance

        correlation_rolling = returns_daily[asset].rolling(window).corr(market_portfolio_return)

        def compute_rsi(returns_series, period):
            delta = returns_series.diff()
            gain = delta.clip(lower = 0)
            loss = - delta.clip(upper = 0)
            avg_gain = gain.rolling(period).mean()
            avg_loss = loss.rolling(period).mean()
            rs = avg_gain / avg_loss
            rsi: pd.Series = 100 - (100 / (1 + rs))
            return rsi
        
        rsi_14 = compute_rsi(prices[asset], 14)
        rsi_30 = compute_rsi(prices[asset], 30)

        common_index = (y.dropna().index.intersection(moving_average_distance.dropna().index)
                        .intersection(maximum_distance.dropna().index).intersection(momentum_signal_1m.dropna().index)
                        .intersection(momentum_signal_3m.dropna().index).intersection(momentum_signal_6m.dropna().index)
                        .intersection(momentum_signal_12m.dropna().index).intersection(volatility_1m.dropna().index)
                        .intersection(volatility_3m.dropna().index).intersection(volatility_12m.dropna().index)
                        .intersection(downside_volatility.dropna().index).intersection(beta_rolling.dropna().index)
                        .intersection(correlation_rolling.dropna().index).intersection(rsi_14.dropna().index)
                        .intersection(rsi_30.dropna().index))
        y = y.loc[common_index].rename("future_returns")
        moving_average_distance = moving_average_distance.loc[common_index].rename("moving_average_200_distance")
        maximum_distance = maximum_distance.loc[common_index].rename("maximum_distance")
        momentum_signal_1m = momentum_signal_1m.loc[common_index].rename("momentum_1m")
        momentum_signal_3m = momentum_signal_3m.loc[common_index].rename("momentum_3m")
        momentum_signal_6m = momentum_signal_6m.loc[common_index].rename("momentum_6m")
        momentum_signal_12m = momentum_signal_12m.loc[common_index].rename("momentum_12m") 
        volatility_1m = volatility_1m.loc[common_index].rename("volatility_1m")
        volatility_3m = volatility_3m.loc[common_index].rename("volatility_3m")
        volatility_12m = volatility_12m.loc[common_index].rename("volatility_12m")
        downside_volatility = downside_volatility.loc[common_index].rename("downside_volatility")
        beta_rolling = beta_rolling.loc[common_index].rename("beta_rolling")
        correlation_rolling = correlation_rolling.loc[common_index].rename("correlation_rolling")
        rsi_14 = rsi_14.loc[common_index].rename("rsi_14")
        rsi_30 = rsi_30.loc[common_index].rename("rsi_30")

        x[asset] = pd.concat([y, moving_average_distance, maximum_distance, momentum_signal_1m, momentum_signal_3m, 
                             momentum_signal_6m, momentum_signal_12m, volatility_1m, volatility_3m, volatility_12m, 
                             downside_volatility, beta_rolling, correlation_rolling, rsi_14, rsi_30], axis = 1)
    
    return x

def build_panel(x: dict):
    
    panel = []

    for asset, df in x.items():
        temp = df.copy()
        temp["asset"] = asset
        panel.append(temp)

    panel = pd.concat(panel)
    panel.index.name = "date"
    panel: pd.DataFrame = panel.reset_index()
    return panel