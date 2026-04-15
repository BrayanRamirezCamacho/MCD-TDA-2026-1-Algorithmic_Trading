# pragma pylint: disable=missing-docstring, invalid-name
# flake8: noqa

from pandas import DataFrame
from freqtrade.strategy import IStrategy
import talib.abstract as ta


class MVPStrategy(IStrategy):
    """
    MVP Baseline Strategy - Proyecto TDA Trading

    Basada en:
    - EMA20 / EMA50 crossover
    - RSI14 filtro de sobrecompra

    Entrada:
        EMA20 > EMA50
        close > EMA20
        RSI < 70

    Salida:
        EMA20 < EMA50
        RSI > 75

    Risk:
        TP: 4%
        SL: -2%
    """

    INTERFACE_VERSION = 3
    can_short: bool = False

    timeframe = "1h"
    startup_candle_count: int = 50

    minimal_roi = {
        "0": 0.04
    }

    stoploss = -0.02
    trailing_stop = False

    process_only_new_candles = True
    use_exit_signal = True

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe["ema20"] = ta.EMA(dataframe, timeperiod=20)
        dataframe["ema50"] = ta.EMA(dataframe, timeperiod=50)
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (
                (dataframe["ema20"] > dataframe["ema50"]) &
                (dataframe["close"] > dataframe["ema20"]) &
                (dataframe["rsi"] < 70) &
                (dataframe["volume"] > 0)
            ),
            "enter_long"
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (
                (
                    (dataframe["ema20"] < dataframe["ema50"]) |
                    (dataframe["rsi"] > 75)
                ) &
                (dataframe["volume"] > 0)
            ),
            "exit_long"
        ] = 1

        return dataframe