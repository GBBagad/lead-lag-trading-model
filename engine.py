from strategy.parameters import STOP_LOSS, TRANSACTION_COST

def run_backtest(lagger_prices, signals, holding_period=3):

    trades = []
    net_pnl = []

    for i in range(len(signals)):

        if signals[i] == 0:
            continue

        entry_price = lagger_prices.iloc[i]
        exit_price = None
        exit_index = i

        for j in range(1, holding_period + 1):

            if i + j >= len(lagger_prices):
                break

            current_price = lagger_prices.iloc[i + j]

            # LONG trade
            if signals[i] == 1:
                move = (current_price - entry_price) / entry_price
                if move <= -STOP_LOSS:
                    exit_price = current_price
                    exit_index = i + j
                    break

            # SHORT trade
            elif signals[i] == -1:
                move = (entry_price - current_price) / entry_price
                if move <= -STOP_LOSS:
                    exit_price = current_price
                    exit_index = i + j
                    break

        # Stop loss nahi trigger jhala tar normal exit
        if exit_price is None:
            exit_index = min(i + holding_period, len(lagger_prices) - 1)
            exit_price = lagger_prices.iloc[exit_index]

        # Gross PnL (cost before)
        if signals[i] == 1:
            gross_pnl = (exit_price - entry_price) / entry_price
        else:
            gross_pnl = (entry_price - exit_price) / entry_price

        # Net PnL (cost after)
        trade_pnl = gross_pnl - (2 * TRANSACTION_COST)

        net_pnl.append(trade_pnl)

        holding_period_value = exit_index - i

        trades.append({
            "entry": entry_price,
            "exit": exit_price,
            "gross_pnl": gross_pnl,
            "pnl": trade_pnl,
            "holding_period": holding_period_value
        })

    return trades, net_pnl