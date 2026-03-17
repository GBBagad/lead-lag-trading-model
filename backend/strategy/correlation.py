# backend/strategy/correlation.py

import numpy as np
import pandas as pd

def compute_cross_correlation(leader_returns, lagger_returns, max_lag=5):
    correlations = {}

    for k in range(1, max_lag + 1):
        shifted_leader = leader_returns.shift(k)
        corr = shifted_leader.corr(lagger_returns)
        correlations[k] = corr

    return correlations


def find_optimal_lag(correlations):
    # absolute max correlation lag
    optimal_lag = max(correlations, key=lambda k: abs(correlations[k]))
    return optimal_lag, correlations[optimal_lag]

