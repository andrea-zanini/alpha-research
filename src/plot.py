import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from config import gamma

BASE_DIR = Path(__file__).resolve().parent.parent
plot_path = BASE_DIR / "results" / "plots"
plot_path.mkdir(parents = True, exist_ok = True)

STYLE = {"style": "ticks",
         "palette": "muted",
         "font": "serif",
         "rc": {"axes.spines.top": False,
                "axes.spines.right": False,
                "figure.figsize": (12, 6),
                "axes.grid": True,
                "grid.alpha": 0.4}
         }

def cumulative_returns(portfolio_returns: pd.Series, market_portfolio_returns: pd.Series):
    
    sns.set_theme(**STYLE)
    fig, ax = plt.subplots()

    cum_market_portfolio_returns = (1 + market_portfolio_returns).cumprod()
    cum_portfolio_returns = (1 + portfolio_returns).cumprod()
    
    sns.lineplot(x = cum_portfolio_returns.index, y = cum_portfolio_returns.values ,color = "#b876f4", linewidth = 1.5, label = f"Portoflio", ax = ax)
        
    sns.lineplot(x = market_portfolio_returns.index, y = cum_market_portfolio_returns, color = "#261818", linewidth = 2, linestyle = "--", 
                 label = "S&P 500", ax = ax)
    ax.set_xlabel("Date")
    ax.set_ylabel("Wealth")
    ax.set_title("Cumulative Returns")
    plt.tight_layout()
    plt.savefig(plot_path / "cumulative_returns.png", dpi = 150, bbox_inches = "tight")
    plt.show()
    
def drawdown(drawdown_portfolio: pd.Series, drawdown_market_portfolio: pd.Series):
    
    sns.set_theme(**STYLE)
    fig, ax = plt.subplots()
    
    sns.lineplot(x = drawdown_portfolio.index, y = drawdown_portfolio.values * 100, color = "#b876f4", linewidth = 1.5, label = f"Portoflio", ax = ax)
    sns.lineplot(x = drawdown_market_portfolio.index, y = drawdown_market_portfolio.values * 100, color = "#261818", linewidth = 2, linestyle = "--", 
                     label = "S&P 500", ax = ax)
    ax.set_xlabel("Date")
    ax.set_ylabel("Wealth Loss")
    ax.set_title("Drawdown")
    plt.tight_layout()
    plt.savefig(plot_path / "drawdown.png", dpi = 150, bbox_inches = "tight")
    plt.show()