# lead-lag-trading-model
# Lead-Lag Statistical Arbitrage Model

This project implements an intraday lead–lag cross-asset statistical arbitrage strategy.

## Steps in the Model
1. Collect historical price data using Polygon API
2. Compute log returns for leader and lagger assets
3. Identify lead-lag relationship using cross-correlation
4. Validate using Granger causality test
5. Generate buy/sell signals
6. Run backtesting engine
7. Evaluate performance using metrics like Sharpe ratio, drawdown, win rate, etc.

## Project Structure
backend/
data/
strategy/
backtest/
app.py

## Output
The model produces a performance report including:
- Return metrics
- Risk metrics
- Timing metrics
- Model metrics

lead-lag-trading-model
│
├── backend
│   ├── strategy
│   ├── backtest
│   └── data
│
├── data
│
├── app.py
├── README.md
├── .gitignore
