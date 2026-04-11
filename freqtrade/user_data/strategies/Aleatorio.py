from freqtrade.strategy import IStrategy
import pandas as pd
import numpy as np


class Aleatorio(IStrategy):
    """
    Estrategia que tiene como objetivo ser un baseline
    para las demás estrategias.
    Su objetivo es comprar y vender en momentos aleatorios
    con una semilla fija para asegurar reproducibilidad
    en el backtesting.
    """

    timeframe = "5m"
    stoploss = -0.05  # Cierra si la perdida llega a 5% .
    minimal_roi = {"0": 0.03, "30": 0.02, "60": 0.01, "120": 0}

    def populate_indicators(
        self, dataframe: pd.DataFrame, metadata: dict
    ) -> pd.DataFrame:
        # No se usa ningun indicador. Todo es totalmente aleatorio.
        return dataframe

    def populate_entry_trend(
        self, dataframe: pd.DataFrame, metadata: dict
    ) -> pd.DataFrame:
        np.random.seed(42)

        dataframe["enter_long"] = (
            np.random.random(len(dataframe)) < 0.1
            # Genera un numero aleatorio entre 0 y 1 para cada vela.
            # Solo si el numero es mayor a 0.1 genera una señal de compra.
        ).astype(int)
        return dataframe

    def populate_exit_trend(
        self, dataframe: pd.DataFrame, metadata: dict
    ) -> pd.DataFrame:
        np.random.seed(99)
        dataframe["exit_long"] = (
            np.random.random(len(dataframe)) < 0.1
            # Misma logica que de entry. 10% de prbabilidad de
            # de vender por vela.
            # Si no llega a vernder aqui, el minmal_roi o el stoploss cerraran la op.
        ).astype(int)
        return dataframe


3
