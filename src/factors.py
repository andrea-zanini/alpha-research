import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import pandas as pd
import numpy as np
from config import window_mean_reversion, window_momentum, window_portofolio, features_standardization

BASE_DIR= Path(__file__).resolve().parent.parent
price_path = BASE_DIR / "results" / "prices"

def factors():

    returns_daily = pd.read_parquet(price_path / "log_returns_daily.parquet")
    market_portfolio_return = pd.read_parquet(price_path / "market_portfolio.parquet").squeeze()
    prices = pd.read_parquet(price_path / "daily_prices.parquet")

    standardizzators = {}
    x = {}
    y = {}

    for asset in prices:

        standardizzator = returns_daily[asset].rolling(window_portofolio).std()
        target = returns_daily[asset].rolling(window_portofolio).sum().shift(-window_portofolio) / standardizzator
        
        def ehlers_roofing(returns: pd.Series, low = window_mean_reversion):
            
            x = returns.values
            n = len(x)
                
            a_2 = np.exp(-(1.414 * np.pi / low))
            b_2 = 2 * a_2 * np.cos(1.414 * np.pi / low)
            c_2 = b_2
            c_3 = -a_2**2
            c_4 = 1 - c_2 - c_3
            
            wave = np.zeros(n)
            
            for i in range(2, n):
                
                wave[i] = c_4 * ((x[i] + x[i - 1]) / 2) + c_2 * wave[i - 1] + c_3 * wave[i - 2]
                
            return pd.Series(wave, index = returns.index)
        
        mean_distance = (returns_daily[asset] - returns_daily[asset].rolling(window_momentum).mean()) / returns_daily[asset].rolling(window_momentum).std() / standardizzator * -1
        volatility = returns_daily[asset].rolling(window_mean_reversion).std() / standardizzator
        cycle = (ehlers_roofing(returns_daily[asset])) / standardizzator * -1
        
        covariance = returns_daily[asset].rolling(window_momentum).cov(market_portfolio_return)
        market_variance = market_portfolio_return.rolling(window_momentum).var()
        beta = covariance / market_variance
        asset_return = returns_daily[asset].rolling(window_momentum).sum()
        market_return = market_portfolio_return.rolling(window_momentum).sum()
        residual_momentum = (asset_return - (beta * market_return)) / standardizzator
        
        correlation_raw = returns_daily[asset].rolling(window_momentum).corr(market_portfolio_return)
        correlation = np.arctanh(np.clip(correlation_raw, -0.99, 0.99))
        
        market_mom_mean = market_portfolio_return.rolling(window_momentum).sum().rolling(window_momentum).mean()
        market_mom_std = market_portfolio_return.rolling(window_momentum).sum().rolling(window_momentum).std()
        market_momentum = ((market_portfolio_return.rolling(window_momentum).sum() - market_mom_mean) / market_mom_std).clip(-3, 3)
        market_volatility_mean = market_portfolio_return.rolling(window_momentum).std().rolling(window_momentum).mean()
        market_volatility_std = market_portfolio_return.rolling(window_momentum).std().rolling(window_momentum).std()
        market_volatility = ((market_portfolio_return.rolling(window_momentum).std() - market_volatility_mean) / market_volatility_std).clip(-3, 3)
        
        common_index = (target.dropna().index.intersection(mean_distance.dropna().index).intersection(volatility.dropna().index)
                        .intersection(cycle.dropna().index).intersection(correlation.dropna().index)
                        .intersection(residual_momentum.dropna().index).intersection(standardizzator.dropna().index)
                        .intersection(market_momentum.dropna().index).intersection(market_volatility.dropna().index))
        
        target = target.loc[common_index].rename("Target")
        mean_distance = mean_distance.loc[common_index].rename("Mean Distance")
        volatility = volatility.loc[common_index].rename("Volatility")
        cycle = cycle.loc[common_index].rename("Cycle")
        correlation = correlation.loc[common_index].rename("Correlation")
        residual_momentum = residual_momentum.loc[common_index].rename("Residual Momentum")
        market_momentum = market_momentum.loc[common_index].rename("Market Momentum")
        market_volatility = market_volatility.loc[common_index].rename("Market Volatility")

        standardizzators[asset] = standardizzator.loc[common_index]
        x[asset] = pd.concat([mean_distance, volatility, cycle, correlation, residual_momentum,
                              market_momentum, market_volatility], axis = 1)
        y[asset] = target.loc[common_index]
    
    return y, x, standardizzators

def build_panel(y: dict, x: dict, standardizzator: dict):
    
    vol = []
    x_feature = []
    y_target = []

    for asset, feature in standardizzator.items():
        standardizzators = feature.copy().to_frame(name = "Volatility")
        standardizzators["Asset"] = asset
        vol.append(standardizzators)

    volatility = pd.concat(vol, axis = 0)
    volatility.index.name = "Date"
    volatility.set_index("Asset", inplace = True, append = True)
    volatility.sort_index(level = ["Date", "Asset"], inplace = True)
    
    for asset, feature in x.items():
        factors = feature.copy()
        factors["Asset"] = asset
        x_feature.append(factors)

    features = pd.concat(x_feature, axis = 0)
    features.index.name = "Date"
    features.set_index("Asset", inplace = True, append = True)
    features.sort_index(level = ["Date", "Asset"], inplace = True)
    features = z_score(features)
    
    for asset, target in y.items():
        objective = target.copy().to_frame(name = "Target")
        objective["Asset"] = asset
        y_target.append(objective)

    targets = pd.concat(y_target, axis = 0)
    targets.index.name = "Date"
    targets.set_index("Asset", inplace = True, append = True)
    targets.sort_index(level = ["Date", "Asset"], inplace = True)
        
    return targets.squeeze(), features, volatility.squeeze()

def z_score(features: pd.DataFrame):
    
    features["Volatility"] = np.log(1 + features["Volatility"])
    
    for variable in features_standardization:
        features[variable] = features.groupby(level = "Date")[variable].transform(lambda x: (x - x.mean()) / x.std()).clip(-3, 3)

    return features