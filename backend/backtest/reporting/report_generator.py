from reporting.report_generator import generate_report

def generate_report(data, lag, corr, granger_result, sharpe, sortino):
    
    report = ""

    report += "===== LEAD-LAG TRADING MODEL REPORT =====\n\n"

    # Data Info
    report += "DATA PREPROCESSING:\n"
    report += f"Total Rows: {len(data)}\n"
    report += f"Columns: {list(data.columns)}\n\n"

    # Correlation
    report += "CORRELATION ANALYSIS:\n"
    report += f"Optimal Lag: {lag}\n"
    report += f"Correlation Value: {corr}\n\n"

    # Granger
    report += "GRANGER CAUSALITY:\n"
    for k, v in granger_result.items():
        report += f"Lag {k}: p-value = {v}\n"
    report += "\n"

    # Performance
    report += "BACKTEST PERFORMANCE:\n"
    report += f"Sharpe Ratio: {sharpe}\n"
    report += f"Sortino Ratio: {sortino}\n\n"

    # Save file
    with open("final_report.txt", "w") as f:
        f.write(report)

    print("✅ Report Generated: final_report.txt")