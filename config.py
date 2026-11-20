import numpy as np

SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
horizon = "10y"
time = "1d"
threshold = 0.95
method = "log"
minimum_returns = 0
rf_daily = (1.0465) ** (1 / 252) - 1
features_standardization = ["Mean Distance", "Volatility", "Cycle", "Correlation", "Residual Momentum"]
features_drop = ["Volatility", "Cycle"]
features_linear_model = ["Mean Distance", "Correlation", "Residual Momentum"]
features_non_linear_model = ["Mean Distance", "Correlation", "Residual Momentum", "Market Momentum", "Market Volatility"]
window_mean_reversion = 22
window_momentum = 126
window_portofolio = 126
rebalance = 1
max_weight = 0.02
min_weight = -0.02
max_turnover = 0.3
gamma = 3
volatility_annual = 0.1
cost_rate = 0.003