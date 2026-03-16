# backend/strategy/granger.py

from statsmodels.tsa.stattools import grangercausalitytests

def run_granger_test(data, max_lag, verbose=False):
    """
    dataframe should contain:
    column1 = lagger returns
    column2 = leader returns
    """

    results = grangercausalitytests(data, max_lag, verbose=False)

    p_values = {}

    for lag in range(1, max_lag + 1):
        p_value = results[lag][0]['ssr_ftest'][1]
        p_values[lag] = p_value

    return p_values
