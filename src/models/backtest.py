import pandas as pd


def run_backtest(
    df: pd.DataFrame,
    initial_capital: float = 1000.0,
    stop_loss: float = 0.02,
    take_profit: float = 0.04,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Backtest long-only simple.
    - Entra al cierre de la vela con enter_long
    - Sale por exit_long, stop-loss o take-profit
    """
    df = df.copy()

    in_position = False
    entry_price = 0.0
    entry_time = None
    capital = initial_capital
    btc_amount = 0.0

    trades = []
    equity_curve = []

    for _, row in df.iterrows():
        ts = row["timestamp"]
        close = row["close"]

        if not in_position and row["enter_long"]:
            entry_price = close
            entry_time = ts
            btc_amount = capital / close
            in_position = True

        elif in_position:
            pnl_pct = (close - entry_price) / entry_price

            should_exit = (
                row["exit_long"] or
                pnl_pct <= -stop_loss or
                pnl_pct >= take_profit
            )

            if should_exit:
                exit_price = close
                exit_time = ts
                capital = btc_amount * exit_price

                trades.append({
                    "entry_time": entry_time,
                    "exit_time": exit_time,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "return_pct": (exit_price - entry_price) / entry_price,
                    "capital_after_trade": capital,
                })

                in_position = False
                entry_price = 0.0
                entry_time = None
                btc_amount = 0.0

        current_equity = capital if not in_position else btc_amount * close
        equity_curve.append({
            "timestamp": ts,
            "equity": current_equity,
            "in_position": int(in_position),
        })

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_curve)

    return trades_df, equity_df
