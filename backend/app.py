import pandas as pd
from data.data_cleaning import compute_log_returns
from strategy.correlation import compute_cross_correlation, find_optimal_lag
from strategy.granger import run_granger_test
from strategy.validation import validate_lag
from strategy.signal import generate_signals
from backtest.engine import run_backtest
from backtest.performance import *
from strategy.parameters import *


def main():

    # STEP 1 : Load Data
    leader_df = pd.read_csv("data/leader_data.csv")
    lagger_df = pd.read_csv("data/lagger_data.csv")

    leader_close = leader_df["close"]
    lagger_close = lagger_df["close"]

    # STEP 2 : Returns
    leader_returns = compute_log_returns(leader_close)
    lagger_returns = compute_log_returns(lagger_close)

    # STEP 3 : Correlation
    corrs = compute_cross_correlation(
        leader_returns,
        lagger_returns,
        MAX_LAG
    )

    optimal_lag, corr_value = find_optimal_lag(corrs)

    print("\nOptimal Lag:", optimal_lag)

    # STEP 4 : Granger Test
    df = pd.DataFrame({
        "lagger": lagger_returns,
        "leader": leader_returns
    }).dropna()

    pvals = run_granger_test(df, MAX_LAG)

    # STEP 5 : Lag Validation
    is_valid = validate_lag(optimal_lag, corrs, pvals)

    if not is_valid:
        print("Lag invalid. Trading disabled.")
        return

    # STEP 6 : Signals
    signals = generate_signals(
        leader_returns,
        lagger_returns,
        optimal_lag,
        LEADER_THRESHOLD,
        LAGGER_THRESHOLD
    )

    # STEP 7 : Backtest
    trades, strategy_pnl = run_backtest(
        lagger_close,
        signals,
        HOLDING_PERIOD
    )

    print("\nTotal Trades:", len(trades))
    print("Total Strategy PnL:", sum(strategy_pnl))

    # Benchmark returns
    benchmark_pnl = lagger_returns.dropna().tolist()

    # Strategy before cost
    gross_pnl = [t["gross_pnl"] for t in trades]

    # Final after cost
    final_pnl = [t["pnl"] for t in trades]

    print_report(gross_pnl, benchmark_pnl, final_pnl, trades)


# ===========================
# REPORT
# ===========================

def print_report(strategy_pnl, benchmark_pnl, final_pnl, trades):

    print("\n================ RETURN METRICS ================\n")

    print(f"{'Metric':25} {'Benchmark':>12} {'Strategy':>12} {'Final':>12}")
    print("-"*65)

    print(f"{'Avg Return':25} {avg_return(benchmark_pnl):>12.4f} {avg_return(strategy_pnl):>12.4f} {avg_return(final_pnl):>12.4f}")
    print(f"{'Cumulative Return':25} {cumulative_return(benchmark_pnl):>12.4f} {cumulative_return(strategy_pnl):>12.4f} {cumulative_return(final_pnl):>12.4f}")
    print(f"{'Best Trade':25} {best_trade(benchmark_pnl):>12.4f} {best_trade(strategy_pnl):>12.4f} {best_trade(final_pnl):>12.4f}")
    print(f"{'Worst Trade':25} {worst_trade(benchmark_pnl):>12.4f} {worst_trade(strategy_pnl):>12.4f} {worst_trade(final_pnl):>12.4f}")
    print(f"{'Total PnL':25} {total_pnl(benchmark_pnl):>12.4f} {total_pnl(strategy_pnl):>12.4f} {total_pnl(final_pnl):>12.4f}")
    print(f"{'CAGR':25} {calculate_cagr(benchmark_pnl):>12.4f} {calculate_cagr(strategy_pnl):>12.4f} {calculate_cagr(final_pnl):>12.4f}")

    print("\n================ RISK METRICS =================\n")

    print(f"{'Sharpe Ratio':25} {calculate_sharpe_ratio(benchmark_pnl):>12.4f} {calculate_sharpe_ratio(strategy_pnl):>12.4f} {calculate_sharpe_ratio(final_pnl):>12.4f}")
    print(f"{'Sortino Ratio':25} {calculate_sortino_ratio(benchmark_pnl):>12.4f} {calculate_sortino_ratio(strategy_pnl):>12.4f} {calculate_sortino_ratio(final_pnl):>12.4f}")
    print(f"{'Max Drawdown':25} {calculate_max_drawdown(benchmark_pnl):>12.4f} {calculate_max_drawdown(strategy_pnl):>12.4f} {calculate_max_drawdown(final_pnl):>12.4f}")
    print(f"{'Volatility':25} {calculate_volatility(benchmark_pnl):>12.4f} {calculate_volatility(strategy_pnl):>12.4f} {calculate_volatility(final_pnl):>12.4f}")
    print(f"{'Win Rate':25} {calculate_win_rate(benchmark_pnl):>12.4f} {calculate_win_rate(strategy_pnl):>12.4f} {calculate_win_rate(final_pnl):>12.4f}")
    print(f"{'Win/Loss Ratio':25} {win_loss_ratio(benchmark_pnl):>12.4f} {win_loss_ratio(strategy_pnl):>12.4f} {win_loss_ratio(final_pnl):>12.4f}")
    print(f"{'Avg Win':25} {avg_win(benchmark_pnl):>12.4f} {avg_win(strategy_pnl):>12.4f} {avg_win(final_pnl):>12.4f}")
    print(f"{'Avg Loss':25} {avg_loss(benchmark_pnl):>12.4f} {avg_loss(strategy_pnl):>12.4f} {avg_loss(final_pnl):>12.4f}")

    print("\n================ TIMING METRICS ================\n")

    print(f"{'Avg Time to MFE':25} {avg_time_to_mfe(trades):>12.2f} {avg_time_to_mfe(trades):>12.2f} {avg_time_to_mfe(trades):>12.2f}")
    print(f"{'Avg Time to MAE':25} {avg_time_to_mae(trades):>12.2f} {avg_time_to_mae(trades):>12.2f} {avg_time_to_mae(trades):>12.2f}")

    print("\n================ MODEL METRICS =================\n")

    print(f"{'Accuracy':25} {'N/A':>12} {accuracy(trades):>12.4f} {accuracy(trades):>12.4f}")
    print(f"{'Precision':25} {'N/A':>12} {precision(trades):>12.4f} {precision(trades):>12.4f}")
    print(f"{'Average Holding Period':25} {'N/A':>12} {average_holding_period(trades):>12.2f} {average_holding_period(trades):>12.2f}")

if __name__ == "__main__":
    main()