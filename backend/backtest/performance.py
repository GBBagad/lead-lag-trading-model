import numpy as np

def calculate_win_rate(pnl_list):
    wins = [p for p in pnl_list if p > 0]
    return len(wins) / len(pnl_list) if pnl_list else 0


def calculate_sharpe_ratio(pnl_list):
    if len(pnl_list) < 2:
        return 0
    mean_return = np.mean(pnl_list)
    std_return = np.std(pnl_list)
    if std_return == 0:
        return 0
    return (mean_return / std_return) * np.sqrt(252)


def calculate_max_drawdown(pnl_list):
    cumulative = np.cumsum(pnl_list)
    peak = cumulative[0]
    max_dd = 0

    for value in cumulative:
        if value > peak:
            peak = value
        drawdown = peak - value
        if drawdown > max_dd:
            max_dd = drawdown

    return max_dd

def avg_return(pnl):
    if len(pnl) == 0:
        return 0
    return sum(pnl) / len(pnl)

def cumulative_return(pnl):
    return sum(pnl)

def best_trade(pnl):
    if len(pnl) == 0:
        return 0
    return max(pnl)

def worst_trade(pnl):
    if len(pnl) == 0:
        return 0
    return min(pnl)

def total_pnl(pnl):
    return sum(pnl)

def calculate_cagr(pnl, initial_capital=10000, years=1):
    final_capital = initial_capital + sum(pnl)
    return (final_capital / initial_capital) ** (1 / years) - 1



import numpy as np

def calculate_sortino_ratio(pnl):
    pnl = np.array(pnl)

    downside = pnl[pnl < 0]

    if len(downside) == 0:
        return 0

    downside_std = downside.std()

    if downside_std == 0:
        return 0

    return pnl.mean() / downside_std

def calculate_volatility(pnl):
    pnl = np.array(pnl)
    return pnl.std()

def win_loss_ratio(pnl):

    wins = [x for x in pnl if x > 0]
    losses = [x for x in pnl if x < 0]

    if len(losses) == 0:
        return len(wins)

    return len(wins) / len(losses)

def average_holding_period(trades):

    if len(trades) == 0:
        return 0

    total = sum(t["holding_period"] for t in trades)

    return total / len(trades)

def accuracy(trades):

    if len(trades) == 0:
        return 0

    correct = sum(1 for t in trades if t["pnl"] > 0)

    return correct / len(trades)

def precision(trades):

    predicted = len(trades)

    if predicted == 0:
        return 0

    true_positive = sum(1 for t in trades if t["pnl"] > 0)

    return true_positive / predicted

import numpy as np
import pandas as pd


def calculate_metrics(returns):

    avg_return = returns.mean()
    cumulative = (1 + returns).prod() - 1
    best_trade = returns.max()
    worst_trade = returns.min()

    return {
        "Avg Return": avg_return,
        "Cumulative": cumulative,
        "Best Trade": best_trade,
        "Worst Trade": worst_trade
    }
def avg_win(pnl):

    wins = [x for x in pnl if x > 0]

    if len(wins) == 0:
        return 0

    return sum(wins) / len(wins)


def avg_loss(pnl):

    losses = [x for x in pnl if x < 0]

    if len(losses) == 0:
        return 0

    return sum(losses) / len(losses)


def avg_time_to_mfe(trades):

    if len(trades) == 0:
        return 0

    mfe_times = []

    for t in trades:

        # profitable trade = favorable move
        if t["pnl"] > 0:
            mfe_times.append(t["holding_period"])

    if len(mfe_times) == 0:
        return 0

    return sum(mfe_times) / len(mfe_times)


def avg_time_to_mae(trades):

    if len(trades) == 0:
        return 0

    mae_times = []

    for t in trades:

        # losing trade = adverse move
        if t["pnl"] < 0:
            mae_times.append(t["holding_period"])

    if len(mae_times) == 0:
        return 0

    return sum(mae_times) / len(mae_times)