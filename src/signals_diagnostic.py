import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import pandas as pd
import numpy as np
from src.regression import compute_signal

signal, regression_metrics = compute_signal()
correlation = []
top_20 = []
bottom_20 = []
spread_20 = []

for date in signal.index.unique():
    
    current_date = signal.loc[date]
    ic = current_date["predicted_alpha"].corr(current_date["realized_alpha"], method = "spearman")
    correlation.append({"date": date, "ic": ic})

for date in signal.index.unique(): 
    
    current_date = signal.loc[date]
    top = current_date[current_date["predicted_alpha"] >= current_date["predicted_alpha"].quantile(0.8)]
    bottom = current_date[current_date["predicted_alpha"] <= current_date["predicted_alpha"].quantile(0.2)]
    top_return = top["realized_alpha"].mean()
    bottom_return = bottom["realized_alpha"].mean()
    spread = top_return - bottom_return
    top_20.append({"date": date, "top return": top_return})
    bottom_20.append({"date": date, "bottom return": bottom_return})
    spread_20.append({"date": date, "spread": spread})

correlation = pd.DataFrame(correlation)
correlation["date"] = pd.to_datetime(correlation["date"])
correlation["year"] = correlation["date"].dt.year
top_20 = pd.DataFrame(top_20)
top_20["date"] = pd.to_datetime(top_20["date"])
top_20["year"] = top_20["date"].dt.year
bottom_20 = pd.DataFrame(bottom_20)
bottom_20["date"] = pd.to_datetime(bottom_20["date"])
bottom_20["year"] = bottom_20["date"].dt.year
spread_20 = pd.DataFrame(spread_20)
spread_20["date"] = pd.to_datetime(spread_20["date"])
spread_20["year"] = spread_20["date"].dt.year

print(f'le metriche della regressione Ridge sono:{regression_metrics}')
print(f'l information coefficient è:{correlation.groupby("year")["ic"].mean()}')
print(f'le statistiche della correlazione tra il segnale predetto ed effetivo sono:{correlation.groupby("year")["ic"].describe()}')
print(f'la media dei titoli nel miglior 20 percentile è:{top_20.groupby("year")["top return"].mean()}')
print(f'le statistiche dei titoli nel miglior 20 percentile sono:{top_20.groupby("year")["top return"].describe()}')
print(f'la media dei titoli nel peggior 20 percentile è:{bottom_20.groupby("year")["bottom return"].mean()}')
print(f'le statistiche dei titoli nel peggior 20 percentile sono:{bottom_20.groupby("year")["bottom return"].describe()}')
print(f'la media dello spread tra i titoli nel miglior 20 percentile e nel peggior 20 percentile è:{spread_20.groupby("year")["spread"].mean()}')
print(f'le statistiche dello spread tra i titoli nel miglior 20 percentile e nel peggior 20 percentile sono:{spread_20.groupby("year")["spread"].describe()}')
print(f'il t-test è:{spread_20.groupby("year")["spread"].mean() / (spread_20.groupby("year")["spread"].std() / np.sqrt(spread_20.groupby("year")["spread"].count()))}')