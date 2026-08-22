import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from config import gamma

BASE_DIR = Path(__file__).resolve().parent.parent
plot_path = BASE_DIR / "results" / "plots"
plot_path.mkdir(parents=True, exist_ok=True)

STYLE = {
    "style": "ticks",
    "palette": "muted",
    "font": "serif",
    "rc": {
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.figsize": (12, 6),
        "axes.grid": True,
        "grid.alpha": 0.4
    }
}

colors = {
    1: "#0072B2",
    5: "#56B4E9",
    10:   "#009E73",
    25:   "#E69F00",
    50:   "#D55E00",
    100:  "#CC79A7", 
}

def label_lines_at_end(ax):
    
    for line in ax.lines:
        label = line.get_label()
        
        if label.startswith("_"):
            continue

        x = line.get_xdata()
        y = line.get_ydata()

        if len(x) == 0 or len(y) == 0:
            continue
        
        xmin, xmax = ax.get_xlim()
        dx = 0.01 * (xmax - xmin)
        
        ax.text(x[-1] + dx, y[-1], label, color = line.get_color(), 
                va = "center", fontsize = 10, clip_on = False)

    xmin, xmax = ax.get_xlim()
    ax.set_xlim(xmin, xmax + (xmax - xmin) * 0.08)

def cumulative_returns(portfolio_returns: dict, market_portfolio_returns: pd.Series, equal_weights_returns: pd.Series):
    
    sns.set_theme(**STYLE)
    fig, ax = plt.subplots()

    cum_market_portfolio_returns = (1 + market_portfolio_returns).cumprod()
    cum_benchmark_returns = (1 + equal_weights_returns).cumprod()
    
    for parameter in gamma:

        cum_portfolio = (1 + portfolio_returns[parameter]).cumprod()
        sns.lineplot(x = cum_portfolio.index, y = cum_portfolio,
                    color = colors[parameter], linewidth = 1.5, label = fr"$\gamma={parameter}$", ax = ax)
        
    sns.lineplot(x = market_portfolio_returns.index, y = cum_market_portfolio_returns,
                color = "#2a2a2a", linewidth = 2, linestyle = "--", label = "S&P 500", ax = ax)
    sns.lineplot(x = equal_weights_returns.index, y = cum_benchmark_returns, color = "#6A3D9A", linewidth = 2, 
                 label = "Equal Weights", ax = ax)
    ax.set_xlabel("Date")
    ax.set_ylabel("Wealth index (base = 1)")
    ax.set_title("Cumulative Returns")
    label_lines_at_end(ax)   
    plt.tight_layout()
    plt.savefig(plot_path / "cumulative_returns.png", dpi = 150, bbox_inches = "tight")
    plt.close()
    
def rolling_sharpe(portfolio_rolling_sharpe_ratio: dict, equal_weight_rolling_sharpe_ratio: pd.Series, benchmark_rolling_sharpe_ratio: pd.Series):
    
    sns.set_theme(**STYLE)
    fig, ax = plt.subplots()
    
    for parameter in gamma:

        sns.lineplot(x = portfolio_rolling_sharpe_ratio[parameter].index, y = portfolio_rolling_sharpe_ratio[parameter],
                    color = colors[parameter], linewidth = 1.5, label = fr"$\gamma={parameter}$", ax = ax)
        
    sns.lineplot(x = equal_weight_rolling_sharpe_ratio.index, y = equal_weight_rolling_sharpe_ratio,
                color = "#6A3D9A", linewidth = 2, linestyle = "--", label = "Equal Weights", ax = ax)
    sns.lineplot(x = benchmark_rolling_sharpe_ratio.index, y = benchmark_rolling_sharpe_ratio, color = "#2a2a2a", linewidth = 2, 
                 label = "S&P 500", ax = ax)
    ax.set_xlabel("Date")
    ax.set_ylabel("Sharpe Ratio")
    ax.set_title("Rolling Sharpe Ratio")
    label_lines_at_end(ax)   
    plt.tight_layout()
    plt.savefig(plot_path / "rolling_sharpe.png", dpi = 150, bbox_inches = "tight")
    plt.close()