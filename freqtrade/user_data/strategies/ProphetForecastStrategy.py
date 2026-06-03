from pathlib import Path

import pandas as pd
from freqtrade.strategy import IStrategy


class ProphetForecastStrategy(IStrategy):
    """
    Baseline de pronóstico clásico usando Prophet.

    Lee predicciones precomputadas desde:
    user_data/prophet_predictions/<PAIR>_15m_predictions.csv
    """

    timeframe = "15m"
    can_short = False
    process_only_new_candles = True
    startup_candle_count = 50

    minimal_roi = {
        "0": 0.03,
        "60": 0.015,
        "180": 0.005,
    }

    stoploss = -0.03

    trailing_stop = False
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    buy_threshold = 0.005
    sell_threshold = -0.002

    @property
    def protections(self):
        return [
            {
                "method": "CooldownPeriod",
                "stop_duration_candles": 4,
            },
            {
                "method": "StoplossGuard",
                "lookback_period_candles": 48,
                "trade_limit": 3,
                "stop_duration_candles": 12,
                "only_per_pair": False,
            },
        ]

    def _pair_to_filename(self, pair: str) -> str:
        return pair.replace("/", "_").replace(":", "_")

    def _load_prophet_predictions(self, pair: str) -> pd.DataFrame:
        pair_name = self._pair_to_filename(pair)

        predictions_path = (
            Path(__file__).resolve().parents[1]
            / "prophet_predictions"
            / f"{pair_name}_{self.timeframe}_predictions.csv"
        )

        if not predictions_path.exists():
            raise FileNotFoundError(f"No existe el archivo: {predictions_path}")

        pred = pd.read_csv(predictions_path)

        pred["date"] = pd.to_datetime(pred["date"], utc=True)

        required_cols = ["date", "forecast_return"]
        missing_cols = [col for col in required_cols if col not in pred.columns]

        if missing_cols:
            raise ValueError(
                f"El archivo {predictions_path} no contiene columnas: {missing_cols}"
            )

        pred = pred[["date", "forecast_return", "yhat", "future_yhat"]].copy()
        pred = pred.sort_values("date")

        return pred

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        pair = metadata["pair"]

        dataframe["date"] = pd.to_datetime(dataframe["date"], utc=True)

        try:
            pred = self._load_prophet_predictions(pair)

            dataframe = dataframe.merge(
                pred,
                on="date",
                how="left",
            )

        except Exception as e:
            dataframe["forecast_return"] = 0.0
            dataframe["yhat"] = None
            dataframe["future_yhat"] = None
            dataframe["prophet_error"] = str(e)

        dataframe["forecast_return"] = dataframe["forecast_return"].fillna(0.0)

        return dataframe

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        entry_condition = (
            (dataframe["forecast_return"] > self.buy_threshold)
            & (dataframe["volume"] > 0)
        )

        dataframe.loc[entry_condition, "enter_long"] = 1
        dataframe.loc[entry_condition, "enter_tag"] = "prophet_bullish_forecast"

        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        exit_condition = (
            (dataframe["forecast_return"] < self.sell_threshold)
            & (dataframe["volume"] > 0)
        )

        dataframe.loc[exit_condition, "exit_long"] = 1
        dataframe.loc[exit_condition, "exit_tag"] = "prophet_bearish_forecast"

        return dataframe
