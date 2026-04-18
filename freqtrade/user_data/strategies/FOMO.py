from freqtrade.strategy import IStrategy, DecimalParameter, IntParameter

import pandas as pd
import numpy as np
import talib.abstract as ta
# talib viene con el docker


class FOMO(IStrategy):
    """
    Simula a una persona inexperta que compre cuando el precio
    lleva subienro un rato (por eso el FOMO). Tambien tiene paciencia
    para no vender inmediatamente cuando ve una bajada pues aún puede subir.
    Usa un trailing_stop para no perder lo ganado. Además, acepta
    perdidas pequeñas antes de que se vuelvan grandes perdidas.
    """

    timeframe = "5m"

    buy_rsi_max = IntParameter(60, 80, default=70, space="buy")
    buy_momentum = DecimalParameter(0.002, 0.015, default=0.005, space="buy")
    buy_ruido = DecimalParameter(0.1, 0.5, default=0.3, space="buy")

    stoploss = -0.04

    trailing_stop = True
    trailing_stop_positive = 0.02

    trailing_stop_positive_offset = 0.03
    # Si cae 2% desde el máximo despues de ganar 3%, vende.

    trailing_only_offset_is_reached = True
    # Antes de ganar 3%, solo el stoploss de -4% lo saca.

    minimal_roi = {
        "0": 0.06,
        "120": 0.04,
        "240": 0.02,
        "480": 0.01,
        "720": 0,
    }

    def populate_indicators(
        self, dataframe: pd.DataFrame, metadata: dict
    ) -> pd.DataFrame:

        dataframe["rsi"] = ta.RSI(dataframe["close"], timeperiod=14)

        # Retorno de los últimos 3 periodos — ¿está subiendo?
        dataframe["retorno_reciente"] = dataframe["close"].pct_change(3)

        return dataframe

    def populate_entry_trend(self, dataframe, metadata):
        np.random.seed(42)
        ruido = np.random.random(len(dataframe))

        dataframe.loc[
            (dataframe["retorno_reciente"] > self.buy_momentum.value)
            & (dataframe["rsi"] < self.buy_rsi_max.value)
            & (ruido < self.buy_ruido.value),
            "enter_long",
        ] = 1
        return dataframe

    def populate_exit_trend(
        self, dataframe: pd.DataFrame, metadata: dict
    ) -> pd.DataFrame:
        # El trailing stop y minimal_roi manejan la salida.
        # Se espera a que "se vea bien" o hasta que ya esperó mucho.
        dataframe["exit_long"] = 0
        return dataframe
