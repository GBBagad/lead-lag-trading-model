import pandas as pd
from strategy.correlation import compute_cross_correlation, find_optimal_lag
from strategy.granger import run_granger_test
from strategy.validation import validate_lag
from strategy.parameters import MAX_LAG

def rolling_lag_detection(leader_returns, lagger_returns, window_size):

    detected_lags = []

    for i in range(window_size, len(leader_returns)):

        leader_window = leader_returns.iloc[i-window_size:i]
        lagger_window = lagger_returns.iloc[i-window_size:i]

        corrs = compute_cross_correlation(
            leader_window,
            lagger_window,
            MAX_LAG
        )

        optimal_lag, _ = find_optimal_lag(corrs)

        df = pd.DataFrame({
            "lagger": lagger_window,
            "leader": leader_window
        }).dropna()

        pvals = run_granger_test(df, MAX_LAG)

        is_valid = validate_lag(optimal_lag, corrs, pvals)

        if is_valid:
            detected_lags.append(optimal_lag)
        else:
            detected_lags.append(None)

    return detected_lags