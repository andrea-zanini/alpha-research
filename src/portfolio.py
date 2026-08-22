import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import pandas as pd
import numpy as np
import cvxpy as cp
from sklearn.covariance import ledoit_wolf
from config import method, max_weight, min_weight

def portfolio(signal: np.ndarray, returns: np.ndarray, market_portfolio_weights: np.ndarray, gamma: int, market_portfolio_returns: pd.Series):

    if method == 'log':
        returns = np.exp(returns) - 1
        market_portfolio_returns = np.exp(market_portfolio_returns) - 1
        
    returns_std = (returns - returns.mean(axis = 0)) / returns.std(axis = 0)
    market_returns_aligned = (market_portfolio_returns - market_portfolio_returns.mean()).reshape(- 1, 1)
    returns_aligned = (returns - returns.mean(axis = 0))
    covarianza = np.mean(returns_aligned * market_returns_aligned, axis = 0)
    variance = np.var(market_portfolio_returns)
    beta = covarianza / variance
    
    def covariance():
        correlation = np.corrcoef(returns_std.T)
        eigenvalues, eigenvectors = np.linalg.eigh(correlation)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        T, N = returns.shape
        q = N / T
        variance = 1
        lambda_max = variance * (1 + np.sqrt(q))**2
        n_components = (eigenvalues > lambda_max).sum()

        loadings = eigenvectors[:, :n_components]
        factors = returns_std @ loadings
        residual = returns_std - factors @ loadings.T
        sigma_idio, shrink = ledoit_wolf(residual)

        sigma_factor = loadings @ np.diag(eigenvalues[:n_components]) @ loadings.T

        d = np.diag(returns.std(axis = 0, ddof = 1))

        sigma = d @ (sigma_idio + sigma_factor) @ d
        return sigma
    
    n = returns.shape[1]
    w = cp.Variable(n)
    sigma = covariance() * 252
    signal = (signal - signal.mean()) / signal.std(ddof = 1)
    score = signal @ w
    risk = cp.quad_form(w, sigma)
    
    objective = cp.Maximize(score - gamma * risk)
    constraints = [cp.sum(w) == 1, cp.sum(cp.abs(w)) <= 2, w >= min_weight, w <= max_weight, w @ beta == 0]

    problem = cp.Problem(objective, constraints)
    problem.solve()

    if problem.status not in ["optimal", "optimal_inaccurate"] or w.value is None:
        print("solution not found")
        return market_portfolio_weights
    
    solution = np.asarray(w.value).flatten()

    return solution