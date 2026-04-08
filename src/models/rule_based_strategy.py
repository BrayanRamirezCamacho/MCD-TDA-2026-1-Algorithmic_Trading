import pandas as pd


def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estrategia básica:
    Entrada long:
    - EMA20 > EMA50
    - cierre > EMA20
    - RSI < 70

    Salida:
    - EMA20 < EMA50
    - o RSI > 75
    """
    df = df.copy()

    df["enter_long"] = (
        (df["ema20"] > df["ema50"]) &
        (df["close"] > df["ema20"]) &
        (df["rsi14"] < 70)
    )

    df["exit_long"] = (
        (df["ema20"] < df["ema50"]) |
        (df["rsi14"] > 75)
    )

    return df
