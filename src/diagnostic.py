import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from config import features_drop, features_standardization

BASE_DIR = Path(__file__).parent.parent
stats_test_path = BASE_DIR / "results" / "stats"
stats_test_path.mkdir(parents = True, exist_ok = True)

STYLE = {"style": "ticks",
         "palette": "muted",
         "font": "serif",
         "rc": {"axes.spines.top": False,
                "axes.spines.right": False,
                "figure.figsize": (12, 6),
                "axes.grid": True,
                "grid.alpha": 0.4}
}

def feature_correlation(factors: pd.DataFrame):
    
    correlation_matrix = factors.corr(method = "pearson")
    correlation_matrix.to_excel(stats_test_path / "Feature_Correlation.xlsx")
    
    sns.set_theme(**STYLE)
    sns.heatmap(correlation_matrix, cmap = "magma", center = 0, annot = True, cbar = True, xticklabels = True, yticklabels = True)
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig(stats_test_path / "Feature_Correlation.png", dpi = 150, bbox_inches = "tight")
    plt.close()

def feature_significant(target: pd.Series, factors: pd.DataFrame):
    
    dataframe = pd.concat([factors, target], axis = 1)
    ic_daily = dataframe.groupby(level = "Date").apply(lambda x: x.corr(method = "spearman")["Target"],).drop(columns = ["Target", "Market Momentum", "Market Volatility"])
    
    stats_list = []
    for column in ic_daily.columns:
        
        ic = ic_daily[column].dropna()
        mu = ic.mean()
        sigma = ic.std()
        n = len(ic)
        t_stat = mu / sigma * np.sqrt(n)
        
        x = np.ones(n)
        regression = sm.OLS(ic, x).fit(cov_type = 'HAC', cov_kwds = {'maxlags': 22})
        t_stat_robust = regression.tvalues.iloc[0]
        stats_list.append({"Feature": column,
                           "Mean IC": mu,
                           "Std IC": sigma,
                           "T Stat": t_stat,
                           "T Stat Robust": t_stat_robust})
                
    stats = pd.DataFrame(stats_list).set_index("Feature")
    stats = stats.reindex(stats["T Stat Robust"].abs().sort_values(ascending = False).index)
    stats.to_excel(stats_test_path / "Feature_Stats.xlsx")
        
def feature_distribution(factors: pd.DataFrame):
    
    factors = factors[features_standardization]
    factors.drop(columns = features_drop, inplace = True)
    sns.set_theme(**STYLE)
    
    for feature in factors.columns:
        
        magma = sns.color_palette("magma", n_colors = 6)[3]
        fig, ax = plt.subplots()
        
        stats.probplot(factors[feature].dropna(), dist = "norm", plot = ax)  
        
        ax.get_lines()[0].set_markerfacecolor(magma)
        ax.get_lines()[0].set_markeredgecolor(magma) 
        ax.get_lines()[0].set_alpha(0.5)
        ax.get_lines()[0].set_markersize(2)
        ax.get_lines()[0].set_label("Real Distribution")
        
        ax.get_lines()[1].set_linestyle('--')
        ax.get_lines()[1].set_color('black')
        ax.get_lines()[1].set_linewidth(2)
        ax.get_lines()[1].set_label("Theoretical Distribution")
        
        ax.set_xlabel("Theoretical Quantili")
        ax.set_ylabel("Real Quantili")
        ax.set_title(f"Quantile Distribution: Real VS Theoretical {feature}")
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(stats_test_path / f"Quantile_Distribution_{feature}.png", dpi = 150, bbox_inches = "tight")
        plt.close()
        
def signal_significant(signal: pd.Series, target: pd.Series):
    
    signal.sort_index(level = ["Date", "Asset"], inplace = True)
    target.sort_index(level = ["Date", "Asset"], inplace = True)
    
    common_index = signal.index.get_level_values(level = "Date").unique().intersection(target.index.get_level_values(level = "Date").unique())
    signal = signal.loc[common_index]
    target = target.loc[common_index]
    dataframe = pd.concat([signal, target], axis = 1)
    dataframe.columns = ["Predictions", "Target"]
    
    ic_daily = dataframe.groupby(level = "Date").apply(lambda x: x["Predictions"].corr(x["Target"], method = "spearman"))
    rolling_ic_daily = ic_daily.rolling(window = 22).mean()
    ic_monthly = dataframe.groupby(pd.Grouper(level = "Date", freq = "ME")).apply(lambda x: x["Predictions"].corr(x["Target"], method = "spearman"))
    
    mu = ic_daily.mean()
    sigma = ic_daily.std()
    n = len(ic_daily)
    t_stat = mu / sigma * np.sqrt(n)
    
    centili = dataframe.groupby(level = "Date")["Predictions"].transform(lambda x: pd.qcut(x, q = 100, labels = False)) 
    decili = dataframe.groupby(level = "Date")["Predictions"].transform(lambda x: pd.qcut(x, q = 10, labels = False))
    target_centile = dataframe.groupby(centili)["Target"].mean()
    target_decile = dataframe.groupby(decili)["Target"].mean()
    spread = (target_centile.loc[99] - target_centile.loc[0]) * 252 / 22
    
    signal_unstacked = dataframe['Predictions'].unstack(level = "Asset")
    autocorrelation_asset = signal_unstacked.corrwith(signal_unstacked.shift(1))
    autocorrelation = autocorrelation_asset.mean()
    
    hit_rate = pd.concat([decili, target], axis = 1)
    hit_rate.columns = ["Decile", "Target"]
    hit_rate_long = (hit_rate[hit_rate['Decile'] == 9]['Target'] > 0).mean()
    hit_rate_short = (hit_rate[hit_rate['Decile'] == 0]['Target'] < 0).mean()
    
    signal_stats = pd.DataFrame({"Metrics": ["Mean IC", "IC Std Dev", "IC T-Stat", "Daily Turnover Proxy", "Signal Autocorrelation", 
                                             "Spread (D99 - D0)", "Hit Rate Long", "Hit Rate Short"],
                                 "Value": [mu, sigma, t_stat, 1 - autocorrelation, autocorrelation, spread, hit_rate_long, hit_rate_short]}).set_index("Metrics")
    
    fig, ax = plt.subplots()
    sns.barplot(x = target_decile.index, y = target_decile.values, palette = "magma", ax = ax, legend = False, hue = target_decile.index,)
    ax.set_title("Mean Target Return by Prediction Decile")
    ax.set_xlabel("Decile")
    ax.set_ylabel("Mean Return")
    plt.tight_layout()
    plt.savefig(stats_test_path / f"Signal Monoticity.png", dpi = 150, bbox_inches = "tight")
    plt.close()
    
    with pd.ExcelWriter(stats_test_path / "Signal_Analysis.xlsx", engine = "openpyxl") as writer:
        
        signal_stats.to_excel(writer, sheet_name = "General Metrics")
        ic_daily.to_excel(writer, sheet_name = "IC Daily")
        rolling_ic_daily.to_excel(writer, sheet_name = "IC Daily Rolling")
        ic_monthly.to_excel(writer, sheet_name = "IC Monthly")