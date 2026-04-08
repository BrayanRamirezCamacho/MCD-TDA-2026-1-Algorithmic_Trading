from pathlib import Path
import pandas as pd


def load_ohlcv_csv(filepath: str | Path) -> pd.DataFrame:
    """
    Carga un CSV OHLCV con columnas:
    timestamp, open, high, low, close, volume
    """
    filepath = Path(filepath)
    df = pd.read_csv(filepath)

    expected_cols = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {missing}")

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["timestamp", "open", "high", "low", "close", "volume"])
    return df
