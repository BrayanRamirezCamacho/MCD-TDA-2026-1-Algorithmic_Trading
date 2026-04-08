import pandas as pd
from src.features.build_features import add_features


def test_add_features_creates_columns():
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=60, freq="h"),
        "open": range(60),
        "high": [x + 1 for x in range(60)],
        "low": [x - 1 for x in range(60)],
        "close": range(60),
        "volume": [100] * 60,
    })

    out = add_features(df)

    assert "ema20" in out.columns
    assert "ema50" in out.columns
    assert "rsi14" in out.columns
    assert "volatility_10" in out.columns
