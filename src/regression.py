import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import pandas as pd
import xgboost as xg
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import GridSearchCV
from config import features_drop, features_linear_model, features_non_linear_model

BASE_DIR = Path(__file__).resolve().parent.parent
factors_path = BASE_DIR / "results" / "factors"
signal_path = BASE_DIR / "results" / "signal"
signal_path.mkdir(parents = True, exist_ok = True)

def linear_signal():
    
    target = pd.read_parquet(factors_path / "target.parquet").squeeze()
    features = pd.read_parquet(factors_path / "factors.parquet")
    features.drop(columns = features_drop, inplace = True)
    factors = features[features_linear_model]

    target.sort_index(level = ["Date", "Asset"], inplace = True)
    factors.sort_index(level = ["Date", "Asset"], inplace = True)
    
    dates = factors.index.get_level_values(level = "Date").unique()
    grid_train_length = int(len(dates) * 0.4)
    
    step_size = 22
    results = []
    
    for time in range(grid_train_length, len(dates), step_size):
        
        train = dates[:time]
        test = dates[time: time + step_size]
        
        x_train = factors.loc[train]
        y_train = target.loc[train]
        
        x_test = factors.loc[test]
        y_test = target.loc[test]
        
        best_model = SGDRegressor(loss = "huber", penalty = "l2", alpha = 0.01, max_iter = 2000, tol = 1e-3, verbose = 0, epsilon = 1.35, random_state = 73, 
                                  learning_rate = "adaptive", eta0 = 0.01, early_stopping = True, validation_fraction = 0.15, n_iter_no_change = 20).fit(x_train, y_train)
        predictions = best_model.predict(x_test)
        
        result = pd.DataFrame({"Predictions": predictions,
                               "Target": y_test.values - predictions},
                               index = y_test.index)
        results.append(result)
    
    signal = pd.concat(results, axis = 0)
    signal.to_parquet(signal_path / f"linear_signal.parquet")
    
    return signal

def non_linear_signal():
    
    target = pd.read_parquet(signal_path / "linear_signal.parquet")["Target"]
    features = pd.read_parquet(factors_path / "factors.parquet")
    features.drop(columns = features_drop, inplace = True)
    factors = features[features_non_linear_model]

    target.sort_index(level = ["Date", "Asset"], inplace = True)
    factors.sort_index(level = ["Date", "Asset"], inplace = True)
    
    common_index = factors.index.get_level_values(level = "Date").unique().intersection(target.index.get_level_values(level = "Date").unique())
    target = target.loc[common_index]
    factors = factors.loc[common_index]
    
    dates = factors.index.get_level_values(level = "Date").unique()
    grid_train_length = int(len(dates) * 0.4)
    grid_train = dates[:grid_train_length]
    
    x_grid_train = factors.loc[grid_train]
    y_grid_train = target.loc[grid_train]
    
    tscv = TimeSeriesSplit(n_splits = 3, gap = 22)
    step_size = 22
    
    param_grid = {"learning_rate": [0.01, 0.02, 0.03, 0.04, 0.05],
                  "max_depth": [3, 4],
                  "min_child_weight": [4236, 6354, 8472, 10590, 12708, 14826, 16944, 19062, 21180],
                  "subsample": [0.6],
                  "colsample_bytree": [0.8],
                  "reg_lambda": [1, 2, 3, 4, 5],
                  "reg_alpha": [0]}
    model = xg.XGBRegressor(verbosity = 0, booster = "gbtree", tree_method = "hist", max_bin = 256, objective = "reg:squarederror", random_state = 73, n_jobs = -1)
    grid_search = GridSearchCV(estimator = model, param_grid = param_grid, cv = tscv, scoring = 'neg_mean_squared_error', n_jobs = -1)
    grid_search.fit(x_grid_train, y_grid_train)
    
    best_params = grid_search.best_params_
    best_params["n_estimators"] = 2000
    results = []
    
    for time in range(grid_train_length, len(dates), step_size):
        
        data_train = dates[:time]
        test = dates[time: time + step_size]
        
        len_validation = int(len(data_train) * 0.15)
        train = data_train[:-len_validation]
        validation = data_train[-len_validation:]
        
        x_train = factors.loc[train]
        y_train = target.loc[train]
        
        x_validation = factors.loc[validation]
        y_validation = target.loc[validation]
        
        x_test = factors.loc[test]
        y_test = target.loc[test]
        
        best_model = xg.XGBRegressor(**best_params, verbosity = 0, booster = "gbtree", tree_method = "hist", max_bin = 256, objective = "reg:squarederror", 
                                     eval_metric = "rmse", early_stopping_rounds = 20, random_state = 73, n_jobs = -1)
        best_model.fit(x_train, y_train, eval_set = [(x_validation, y_validation)], verbose = False)
        predictions = best_model.predict(x_test)
        
        result = pd.Series(predictions, name = "Residual", index = y_test.index)
        results.append(result)
    
    signal = pd.concat(results, axis = 0)
    signal.to_frame().to_parquet(signal_path / f"non_linear_signal.parquet")
    
    return signal
    
def signal():
    
    prediction = pd.read_parquet(signal_path / "linear_signal.parquet")["Predictions"]
    residual = pd.read_parquet(signal_path / "non_linear_signal.parquet").squeeze()
    standardizzator = pd.read_parquet(factors_path / "standardizzator.parquet").squeeze()
    
    complete_signal = (prediction + residual) * standardizzator.reindex(residual.index)
    complete_signal = pd.Series(complete_signal, name = "Signal", index = residual.index)
    complete_signal.sort_index(level = ["Date", "Asset"], inplace = True)
    complete_signal.dropna().to_frame().to_parquet(signal_path / f"signal.parquet")
    
    return complete_signal