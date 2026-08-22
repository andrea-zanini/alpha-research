import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

BASE_DIR = Path(__file__).resolve().parent.parent
factors_path = BASE_DIR / "results" / "factors"
factors = pd.read_parquet(factors_path / "factors.parquet")

def compute_signal():

    factors_sorted = factors.sort_values(["date", "asset"])
    dates = sorted(factors_sorted["date"].unique())
    
    tscv = TimeSeriesSplit(n_splits = 5, gap = 126)
    
    metrics = []
    predictions = []
    
    for train_idx, test_idx in tscv.split(dates):
        
        train_dates = [dates[i] for i in train_idx]
        test_dates = [dates[i] for i in test_idx]
        
        train = factors_sorted[factors_sorted["date"].isin(train_dates)]
        test = factors_sorted[factors_sorted["date"].isin(test_dates)]
        train_x = train.drop(columns = ["date", "future_returns", "asset"])
        train_y = train["future_returns"]
        test_x = test.drop(columns = ["date","future_returns", "asset"])
        test_y = test["future_returns"]
        
        regression = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(train_x, train_y)
        y_pred = regression.predict(test_x)
        
        data = test[["date", "asset"]].copy()
        data["predicted_alpha"] = y_pred
        data["realized_alpha"] = test_y.values
       
        rmse = root_mean_squared_error(test_y, y_pred)
        mae = mean_absolute_error(test_y, y_pred)
        r2 = r2_score(test_y, y_pred)
        metrics.append({"rmse": rmse,
                        "mae": mae,
                        "r2": r2})
        
        predictions.append(data)
    
    metrics = pd.DataFrame(metrics)
    predictions = pd.concat(predictions)
    predictions.index = predictions["date"]
    predictions.drop(columns = ["date"], inplace = True)
    
    return predictions, metrics