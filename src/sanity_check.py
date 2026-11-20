import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import pandas as pd

def sum_weights(weights_portfolio: pd.Dataframe):
    assert weights_portfolio.sum(axis = 1).abs().max() <= 1.e-7