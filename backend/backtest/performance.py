import numpy as np
import pandas as pd


def calculate_win_rate(pnl_list):
    wins = [p for p in pnl_list if p > 0]
    return len(wins) / len(pnl_list) if pnl_list else 0

def sortino_ratio(returns, risk_free=0):
    excess_returns = returns - risk_free

    downside = excess_returns[excess_returns < 0]
    downside_dev = np.sqrt(np.mean(downside**2))

    if downside_dev == 0:
        return 0

    sortino = np.mean(excess_returns) / downside_dev
    return sortino


def calculate_max_drawdown(returns):

    returns = np.array(returns)

    equity = np.cumprod(1 + returns)

    peak = np.maximum.accumulate(equity)

    drawdown = (equity - peak) / peak

    return drawdown.min()


def calculate_volatility(returns):

    returns = np.array(returns)

    return returns.std(ddof=1) * np.sqrt(252)


def calculate_cagr(returns, years=1):

    returns = np.array(returns)

    total_return = np.prod(1 + returns)

    return total_return**(1/years) - 1


def avg_return(pnl):
    return np.mean(pnl) if len(pnl) else 0


def cumulative_return(pnl):
    return sum(pnl)


def best_trade(pnl):
    return max(pnl) if len(pnl) else 0


def worst_trade(pnl):
    return min(pnl) if len(pnl) else 0


def total_pnl(pnl):
    return sum(pnl)


def avg_win(pnl):

    wins = [x for x in pnl if x > 0]

    return np.mean(wins) if wins else 0


def avg_loss(pnl):

    losses = [x for x in pnl if x < 0]

    return np.mean(losses) if losses else 0


def expectancy(returns):

    returns = np.array(returns)

    wins = returns[returns > 0]
    losses = returns[returns < 0]

    win_rate = len(wins) / len(returns)
    loss_rate = len(losses) / len(returns)

    avg_win = wins.mean() if len(wins) else 0
    avg_loss = losses.mean() if len(losses) else 0

    return (win_rate * avg_win) + (loss_rate * avg_loss)


def average_holding_period(trades):

    if len(trades) == 0:
        return 0

    return np.mean([t["holding_period"] for t in trades])


def avg_time_to_mfe(trades):

    mfe_times = [t["holding_period"] for t in trades if t["pnl"] > 0]

    return np.mean(mfe_times) if mfe_times else 0


def avg_time_to_mae(trades):

    mae_times = [t["holding_period"] for t in trades if t["pnl"] < 0]

    return np.mean(mae_times) if mae_times else 0

def calculate_sharpe_ratio(returns, risk_free=0):

    returns = np.array(returns)

    excess = returns - risk_free

    std = np.std(excess, ddof=1)

    if std == 0:
        return 0

    return np.mean(excess) / std * np.sqrt(252)

def profit_factor(pnl):

    pnl = np.array(pnl)

    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]

    if len(losses) == 0:
        return 0

    return wins.sum() / abs(losses.sum())

def win_loss_ratio(pnl):

    wins = [x for x in pnl if x > 0]
    losses = [x for x in pnl if x < 0]

    if len(losses) == 0:
        return 0

    return len(wins) / len(losses)