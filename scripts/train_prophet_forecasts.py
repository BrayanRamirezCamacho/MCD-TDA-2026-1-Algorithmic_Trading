from pathlib import Path
import pandas as pd

try:
    from prophet import Prophet
except ImportError:
    raise ImportError(
        "No tienes prophet instalado. Instálalo con: pip install prophet"
    )


# =========================
# Configuración
# =========================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "freqtrade" / "user_data" / "data" / "binance"
OUTPUT_DIR = BASE_DIR / "freqtrade" / "user_data" / "prophet_predictions"

TIMEFRAME = "15m"

PAIRS = [
    "BTC_USDT",
    "ETH_USDT",
    "SOL_USDT",
    "BNB_USDT",
    "XRP_USDT",
]

FORECAST_HORIZON_CANDLES = 4  # 4 velas de 15m = 1 hora

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_freqtrade_data(pair: str) -> pd.DataFrame:
    """
    Lee datos OHLCV descargados por Freqtrade en formato feather o json.
    """

    feather_path = DATA_DIR / f"{pair}-{TIMEFRAME}.feather"
    json_path = DATA_DIR / f"{pair}-{TIMEFRAME}.json"

    if feather_path.exists():
        df = pd.read_feather(feather_path)
    elif json_path.exists():
        df = pd.read_json(json_path)
    else:
        raise FileNotFoundError(
            f"No encontré datos para {pair}. Busqué:\n"
            f"- {feather_path}\n"
            f"- {json_path}"
        )

    if "date" not in df.columns:
        raise ValueError(f"El archivo de {pair} no contiene columna 'date'.")

    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.sort_values("date").reset_index(drop=True)

    return df


def prepare_prophet_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prophet necesita columnas:
    ds = fecha
    y = valor a pronosticar
    """

    prophet_df = df[["date", "close"]].copy()
    prophet_df = prophet_df.rename(columns={"date": "ds", "close": "y"})

    # Prophet trabaja mejor con fechas sin timezone
    prophet_df["ds"] = prophet_df["ds"].dt.tz_localize(None)

    prophet_df = prophet_df.dropna()

    return prophet_df


def train_and_predict(prophet_df: pd.DataFrame) -> pd.DataFrame:
    """
    Entrena Prophet y genera predicción dentro de la misma serie histórica.

    Para MVP usamos predicción in-sample como feature inicial.
    Después se debe mejorar a walk-forward para validación rigurosa.
    """

    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=False,
        changepoint_prior_scale=0.05,
    )

    model.fit(prophet_df)

    forecast = model.predict(prophet_df[["ds"]])

    result = prophet_df.copy()
    result["yhat"] = forecast["yhat"].values
    result["yhat_lower"] = forecast["yhat_lower"].values
    result["yhat_upper"] = forecast["yhat_upper"].values

    return result


def add_trading_signal_features(result: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte el pronóstico de Prophet en una feature útil para Freqtrade.
    """

    result["future_yhat"] = result["yhat"].shift(-FORECAST_HORIZON_CANDLES)

    result["forecast_return"] = (
        result["future_yhat"] - result["y"]
    ) / result["y"]

    result = result.rename(columns={"ds": "date", "y": "close"})

    result["date"] = pd.to_datetime(result["date"], utc=True)

    return result


def process_pair(pair: str) -> None:
    print(f"Procesando {pair}...")

    df = load_freqtrade_data(pair)
    prophet_df = prepare_prophet_dataframe(df)
    result = train_and_predict(prophet_df)
    result = add_trading_signal_features(result)

    output_path = OUTPUT_DIR / f"{pair}_{TIMEFRAME}_predictions.csv"

    result[
        [
            "date",
            "close",
            "yhat",
            "yhat_lower",
            "yhat_upper",
            "future_yhat",
            "forecast_return",
        ]
    ].to_csv(output_path, index=False)

    print(f"Guardado: {output_path}")


def main():
    for pair in PAIRS:
        try:
            process_pair(pair)
        except Exception as e:
            print(f"Error procesando {pair}: {e}")


if __name__ == "__main__":
    main()
