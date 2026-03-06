import numpy as np

def compute_log_returns(series):
    return np.log(series / series.shift(1))