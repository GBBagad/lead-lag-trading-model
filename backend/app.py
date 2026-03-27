from backtest.performance import *
import pandas as pd
import numpy as np

from data.data_cleaning import compute_log_returns
from strategy.correlation import compute_cross_correlation, find_optimal_lag
from strategy.granger import run_granger_test
from strategy.signal import generate_signals
from backtest.engine import run_backtest
from strategy.parameters import *

from backtest.performance import calculate_sharpe_ratio, calculate_sortino_ratio

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
from datetime import datetime

# ===========================
# SETTINGS
# ===========================
TIMEFRAME = '15T'

# ===========================
# PATH SETUP (FIXED)
# ===========================
base_dir = os.path.dirname(os.path.abspath(__file__))

log_dir = os.path.join(base_dir, "results", "logs")
os.makedirs(log_dir, exist_ok=True)

log_filename = os.path.join(log_dir, "FINAL_REPORT.txt")

def write_log(text):
    print(text)
    with open(log_filename, "a", encoding="utf-8") as f:
        f.write(text + "\n")

# ===========================
# SAVE RESULTS CSV
# ===========================
def save_results(benchmark, strategy, final):
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    pd.DataFrame({"Benchmark": benchmark}).to_csv(os.path.join(results_dir, "benchmark.csv"), index=False)
    pd.DataFrame({"Strategy": strategy}).to_csv(os.path.join(results_dir, "strategy.csv"), index=False)
    pd.DataFrame({"Final": final}).to_csv(os.path.join(results_dir, "final.csv"), index=False)

# ===========================
# MAIN
# ===========================
def main():

    # reset report
    open(log_filename, "w").close()

    print(f"\n📁 Report saved at:\n{log_filename}\n")

    # HEADER
    write_log("="*80)
    write_log("FULL PIPELINE REPORT: LEAD-LAG TRADING MODEL")
    write_log("Includes: Data → Correlation → Granger → Signals → Backtest → Metrics")
    write_log(f"Session Started: {datetime.now()}")
    write_log("="*80)

    # ===========================
    # STEP 1: DATA LOADING
    # ===========================
    write_log("\nSTEP 1: DATA LOADING")

    leader_df = pd.read_csv(os.path.join(base_dir, "data", "leader_data.csv"))
    lagger_df = pd.read_csv(os.path.join(base_dir, "data", "lagger_data.csv"))

    time_col = leader_df.columns[0]

    leader_df[time_col] = pd.to_datetime(leader_df[time_col], errors='coerce')
    lagger_df[time_col] = pd.to_datetime(lagger_df[time_col], errors='coerce')

    leader_df = leader_df.dropna(subset=[time_col])
    lagger_df = lagger_df.dropna(subset=[time_col])

    leader_df = leader_df.sort_values(by=time_col)
    lagger_df = lagger_df.sort_values(by=time_col)

    leader_df.set_index(time_col, inplace=True)
    lagger_df.set_index(time_col, inplace=True)

    # ===========================
    # STEP 2: RESAMPLING
    # ===========================
    write_log("\nSTEP 2: RESAMPLING")

    leader_df = leader_df.resample(TIMEFRAME).agg({"close": "last"}).dropna()
    lagger_df = lagger_df.resample(TIMEFRAME).agg({"close": "last"}).dropna()

    write_log(f"Rows: {len(leader_df)}")
    write_log(f"First timestamps: {leader_df.index[:5]}")

    leader_close = leader_df["close"]
    lagger_close = lagger_df["close"]

    # ===========================
    # STEP 3: PREPROCESSING
    # ===========================
    write_log("\nSTEP 3: PREPROCESSING")

    leader_returns = compute_log_returns(leader_close)
    lagger_returns = compute_log_returns(lagger_close)

    write_log(f"Leader Returns Sample: {leader_returns.head().tolist()}")
    write_log(f"Lagger Returns Sample: {lagger_returns.head().tolist()}")

    # ===========================
    # STEP 4: CORRELATION
    # ===========================
    write_log("\nSTEP 4: CORRELATION & LAG")

    corrs = compute_cross_correlation(leader_returns, lagger_returns, MAX_LAG)
    optimal_lag, corr_value = find_optimal_lag(corrs)

    write_log(f"Optimal Lag: {optimal_lag}")
    write_log(f"Correlation: {corr_value}")

    # ===========================
    # STEP 5: GRANGER
    # ===========================
    write_log("\nSTEP 5: GRANGER CAUSALITY")

    df = pd.DataFrame({
        "lagger": lagger_returns,
        "leader": leader_returns
    }).dropna()

    pvals = run_granger_test(df, MAX_LAG)
    write_log(f"Granger p-values: {pvals}")

    # SIGNIFICANCE FILTER
    ALPHA = 0.05
    significant_lags = {k: v for k, v in pvals.items() if v < ALPHA}

    write_log(f"Significant Lags (p < {ALPHA}): {significant_lags}")

    # ===========================
    # STEP 6: SIGNALS
    # ===========================
    write_log("\nSTEP 6: SIGNAL GENERATION")

    if significant_lags:
        optimal_lag = min(significant_lags, key=significant_lags.get)
        write_log(f"Using statistically significant lag: {optimal_lag}")
    else:
        write_log("No significant lag found, using correlation-based lag")

    signals = generate_signals(
        leader_returns,
        lagger_returns,
        optimal_lag,
        LEADER_THRESHOLD,
        LAGGER_THRESHOLD
    )

    write_log(f"Unique signals: {set(signals)}")
    write_log(f"Total signals: {len(signals)}")

    # ===========================
    # STEP 7: BACKTEST
    # ===========================
    write_log("\nSTEP 7: BACKTEST")

    trades, strategy_pnl = run_backtest(
        lagger_close,
        signals,
        HOLDING_PERIOD
    )

    write_log(f"Total Trades: {len(trades)}")
    write_log(f"Total Strategy PnL: {sum(strategy_pnl)}")

    # ===========================
    # STEP 8: RESULTS
    # ===========================
    benchmark_pnl = lagger_returns.dropna().tolist()
    final_pnl = [t["pnl"] for t in trades]

    print_report(strategy_pnl, benchmark_pnl, final_pnl, trades)
    save_results(benchmark_pnl, strategy_pnl, final_pnl)

    # ===========================
    # FINAL SUMMARY
    # ===========================
    write_log("\n" + "="*80)
    write_log("FINAL SUMMARY")
    write_log("="*80)

    write_log(f"Optimal Lag Used: {optimal_lag}")
    write_log(f"Correlation Strength: {corr_value}")
    write_log("Full pipeline executed successfully.")

    write_log("\n" + "="*80)
    write_log("BACKTESTING SESSION COMPLETED")
    write_log(f"Session Ended: {datetime.now()}")
    write_log("="*80)

# ===========================
# REPORT
# ===========================
def print_report(strategy_pnl, benchmark_pnl, final_pnl, trades):

    report = ""

    report += "\n================ RETURN METRICS ================\n\n"
    report += f"{'Metric':25} {'Benchmark':>12} {'Strategy':>12} {'Final':>12}\n"
    report += "-"*65 + "\n"

    report += f"{'Avg Return':25} {avg_return(benchmark_pnl):>12.4f} {avg_return(strategy_pnl):>12.4f} {avg_return(final_pnl):>12.4f}\n"
    report += f"{'Cumulative Return':25} {cumulative_return(benchmark_pnl):>12.4f} {cumulative_return(strategy_pnl):>12.4f} {cumulative_return(final_pnl):>12.4f}\n"
    report += f"{'Best Trade':25} {best_trade(benchmark_pnl):>12.4f} {best_trade(strategy_pnl):>12.4f} {best_trade(final_pnl):>12.4f}\n"
    report += f"{'Worst Trade':25} {worst_trade(benchmark_pnl):>12.4f} {worst_trade(strategy_pnl):>12.4f} {worst_trade(final_pnl):>12.4f}\n"
    report += f"{'Total PnL':25} {total_pnl(benchmark_pnl):>12.4f} {total_pnl(strategy_pnl):>12.4f} {total_pnl(final_pnl):>12.4f}\n"
    report += f"{'CAGR':25} {calculate_cagr(benchmark_pnl):>12.4f} {calculate_cagr(strategy_pnl):>12.4f} {calculate_cagr(final_pnl):>12.4f}\n"

    report += "\n================ RISK METRICS =================\n\n"
    report += f"{'Sharpe Ratio':25} {calculate_sharpe_ratio(benchmark_pnl):>12.4f} {calculate_sharpe_ratio(strategy_pnl):>12.4f} {calculate_sharpe_ratio(final_pnl):>12.4f}\n"
    report += f"{'Sortino Ratio':25} {calculate_sortino_ratio(benchmark_pnl):>12.4f} {calculate_sortino_ratio(strategy_pnl):>12.4f} {calculate_sortino_ratio(final_pnl):>12.4f}\n"
    report += f"{'Max Drawdown':25} {calculate_max_drawdown(benchmark_pnl):>12.4f} {calculate_max_drawdown(strategy_pnl):>12.4f} {calculate_max_drawdown(final_pnl):>12.4f}\n"

    report += "\n================ MODEL METRICS =================\n\n"
    report += f"{'Average Holding Period':25} {average_holding_period(trades):>12.2f}\n"
    report += f"{'Profit Factor':25} {profit_factor(strategy_pnl):>12.4f}\n"

    write_log(report)

# ===========================
# RUN
# ===========================
if __name__ == "__main__":
    main()