import pandas as pd
from src.features.build_features import add_features
from src.models.rule_based_strategy import generate_signals


def test_generate_signals_creates_columns():
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=100, freq="h"),
        "open": [100 + i for i in range(100)],
        "high": [101 + i for i in range(100)],
        "low": [99 + i for i in range(100)],
        "close": [100 + i for i in range(100)],
        "volume": [1000] * 100,
    })

    df = add_features(df)
    df = generate_signals(df)

    assert "enter_long" in df.columns
    assert "exit_long" in df.columns
