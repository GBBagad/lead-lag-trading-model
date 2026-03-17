from backtest.performance import *
import pandas as pd
from data.data_cleaning import compute_log_returns
from strategy.correlation import compute_cross_correlation, find_optimal_lag
from strategy.granger import run_granger_test
from strategy.validation import validate_lag
from strategy.signal import generate_signals
from backtest.engine import run_backtest
from strategy.parameters import *
from backtest.performance import calculate_sharpe_ratio, calculate_sortino_ratio

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import logging
import os

# ensure results folder exists
os.makedirs("results", exist_ok=True)

logging.basicConfig(
    filename='results/backtest.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

import os

BASE_DIR = os.path.dirname(__file__)

leader_path = os.path.join(BASE_DIR, "data", "leader_data.csv")
lagger_path = os.path.join(BASE_DIR, "data", "lagger_data.csv")

leader_df = pd.read_csv(leader_path)
lagger_df = pd.read_csv(lagger_path)

# ===========================
# SAVE RESULTS FUNCTION
# ===========================
def save_results(benchmark, strategy, final):

    min_len = min(len(benchmark), len(strategy), len(final))

    benchmark = benchmark[:min_len]
    strategy = strategy[:min_len]
    final = final[:min_len]

    df = pd.DataFrame({
        "Benchmark": benchmark,
        "Strategy": strategy,
        "Final": final
    })

    df.to_csv("results/backtest_results.csv", index=False)

def save_results(benchmark, strategy, final):

    pd.DataFrame({"Benchmark": benchmark}).to_csv("results/benchmark.csv", index=False)
    pd.DataFrame({"Strategy": strategy}).to_csv("results/strategy.csv", index=False)
    pd.DataFrame({"Final": final}).to_csv("results/final.csv", index=False)
# ===========================
# MAIN PIPELINE
# ===========================
def main():

    logging.info("===== PIPELINE START =====")

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
    print("Correlation at optimal lag:", corr_value)

    # STEP 4 : Granger
    df = pd.DataFrame({
        "lagger": lagger_returns,
        "leader": leader_returns
    }).dropna()

    pvals = run_granger_test(df, MAX_LAG)
    print("Granger p-values:", pvals)

    # TEMP (demo sathi)
    is_valid = True

    # STEP 5 : Signals
    signals = generate_signals(
        leader_returns,
        lagger_returns,
        optimal_lag,
        LEADER_THRESHOLD,
        LAGGER_THRESHOLD
    )

    # STEP 6 : Backtest
    trades, strategy_pnl = run_backtest(
        lagger_close,
        signals,
        HOLDING_PERIOD
    )

    print("\nTotal Trades:", len(trades))
    print("Total Strategy PnL:", sum(strategy_pnl))

    benchmark_pnl = lagger_returns.dropna().tolist()
    final_pnl = [t["pnl"] for t in trades]

    print_report(strategy_pnl, benchmark_pnl, final_pnl, trades)

    save_results(benchmark_pnl, strategy_pnl, final_pnl)
# REPORT FUNCTION
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

    print(f"{'Average Holding Period':25} {'N/A':>12} {average_holding_period(trades):>12.2f} {average_holding_period(trades):>12.2f}")
    print(f"{'Profit Factor':25} {profit_factor(benchmark_pnl):>12.4f} {profit_factor(strategy_pnl):>12.4f} {profit_factor(final_pnl):>12.4f}")
    
  
if __name__ == "__main__":
    main()