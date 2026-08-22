import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import gamma

def sum_weights(weights_portfolio: dict):
    for parameter in gamma:
        assert (weights_portfolio[parameter].sum(axis = 1) - 1).abs().max() <= 1.e-7