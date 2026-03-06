# backend/strategy/signal.py

import numpy as np

def generate_signals(leader_returns, lagger_returns,
                     optimal_lag,
                     leader_threshold,
                     lagger_threshold):

    signals = []

    for t in range(optimal_lag, len(leader_returns)):

        leader_move = leader_returns.iloc[t - optimal_lag]
        lagger_move = lagger_returns.iloc[t]

        # Long
        if leader_move > leader_threshold and abs(lagger_move) < lagger_threshold:
            signals.append(1)

        # Short
        elif leader_move < -leader_threshold and abs(lagger_move) < lagger_threshold:
            signals.append(-1)

        else:
            signals.append(0)

    return signals