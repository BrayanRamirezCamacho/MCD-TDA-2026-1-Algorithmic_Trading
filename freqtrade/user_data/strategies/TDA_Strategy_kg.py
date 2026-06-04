
# ============================================================
# TDA STRATEGY – SEÑALES TOPOLOGICAS PURAS
# Kevin Galvan Lara – Proyecto TDA Trading
# ============================================================
#
# Lógica de señales:
#   COMPRAR  → topo_z < -1.0  (topología simplificándose = tendencia formándose)
#   VENDER   → topo_z >  1.5  (topología complejizándose = posible reversión)
#
# Features TDA utilizados:
#   - Entropía de persistencia H1 (regularidad topológica)
#   - Distancia Wasserstein entre diagramas consecutivos (cambio topológico)
#   - Z-score del señal combinado (normalización adaptiva)
#
# NOTA: populate_indicators es costoso (~10s/símbolo por el ripser).
#       Usar startup_candle_count alto y timeframe 1h o mayor.
# ============================================================

import numpy as np
import pandas as pd
from pandas import DataFrame
from freqtrade.strategy import IStrategy

try:
    from ripser import ripser
    from persim import wasserstein
except ImportError:
    raise ImportError("Instalar: pip install ripser persim")


class TDAStrategy(IStrategy):
    """
    Estrategia basada en Análisis Topológico de Datos (TDA).

    Compra cuando la topología del mercado se simplifica (señal de tendencia).
    Vende cuando la topología se vuelve compleja (señal de reversión/ruido).
    """

    INTERFACE_VERSION = 3
    can_short: bool = False

    timeframe = "1h"
    startup_candle_count: int = 90

    stoploss = -0.025
    minimal_roi = {
        "0":   0.05,
        "60":  0.03,
        "120": 0.02,
    }
    trailing_stop = False
    process_only_new_candles = True
    use_exit_signal = True

    # --------------------------------------------------------
    # Parámetros TDA (se pueden optimizar con hyperopt)
    # --------------------------------------------------------
    WINDOW   = 40    # Ventana de embedding
    TAU      = 1     # Delay de Takens
    Z_WINDOW = 50    # Ventana para normalizar z-score
    BUY_Z    = -1.0  # Umbral compra
    SELL_Z   =  1.5  # Umbral venta

    # --------------------------------------------------------
    # Helpers TDA
    # --------------------------------------------------------

    def _takens_embedding(self, x: np.ndarray, tau: int = 1, dim: int = 2) -> np.ndarray:
        n = len(x)
        m = n - (dim - 1) * tau
        emb = np.zeros((m, dim))
        for i in range(dim):
            emb[:, i] = x[i * tau: i * tau + m]
        return emb

    def _persistence_entropy(self, dgm: np.ndarray) -> float:
        if len(dgm) == 0:
            return 0.0
        pers = dgm[:, 1] - dgm[:, 0]
        pers = pers[pers > 0]
        if len(pers) == 0:
            return 0.0
        p = pers / pers.sum()
        return float(-np.sum(p * np.log(p + 1e-12)))

    def _build_cloud(
        self,
        ret: np.ndarray,
        vol: np.ndarray,
        hlr: np.ndarray,
        acc: np.ndarray,
    ) -> np.ndarray:
        if len(ret) < 3:
            return np.zeros((2, 2))
        tk = self._takens_embedding(ret, tau=self.TAU, dim=2)
        m = len(tk)
        cloud = np.column_stack([
            tk,
            vol[self.TAU: self.TAU + m],
            hlr[self.TAU: self.TAU + m],
            acc[self.TAU: self.TAU + m],
        ])
        mu  = cloud.mean(axis=0)
        std = cloud.std(axis=0) + 1e-8
        return (cloud - mu) / std

    # --------------------------------------------------------
    # Cálculo de features TDA sobre el dataframe completo
    # --------------------------------------------------------

    def _compute_tda_features(self, dataframe: DataFrame) -> tuple:
        n     = len(dataframe)
        close = dataframe["close"].values
        vol   = dataframe["volume"].values

        log_ret = np.concatenate([[0.0], np.log(close[1:] / (close[:-1] + 1e-12))])
        dlog_vol = np.concatenate([[0.0], np.diff(np.log(vol + 1.0))])
        hl_range = (dataframe["high"].values - dataframe["low"].values) / (close + 1e-8)
        d_hl  = np.concatenate([[0.0], np.diff(hl_range)])
        ret_acc = np.concatenate([[0.0], np.diff(log_ret)])

        ent_h1  = np.zeros(n)
        wass_h1 = np.zeros(n)
        prev_dgm = None

        for i in range(self.startup_candle_count, n):
            w = self.WINDOW
            cloud = self._build_cloud(
                log_ret[i - w: i],
                dlog_vol[i - w: i],
                d_hl[i - w: i],
                ret_acc[i - w: i],
            )
            dgms = ripser(cloud, maxdim=1)["dgms"]
            h1   = dgms[1]

            ent_h1[i] = self._persistence_entropy(h1)

            if prev_dgm is not None and len(prev_dgm) > 0 and len(h1) > 0:
                wass_h1[i] = wasserstein(prev_dgm, h1)

            prev_dgm = h1 if len(h1) > 0 else prev_dgm

        return ent_h1, wass_h1

    # --------------------------------------------------------
    # Freqtrade API
    # --------------------------------------------------------

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        ent_h1, wass_h1 = self._compute_tda_features(dataframe)

        dataframe["tda_entropy"]   = ent_h1
        dataframe["tda_wass"]      = wass_h1
        dataframe["delta_entropy"] = pd.Series(ent_h1, index=dataframe.index).diff().fillna(0)
        dataframe["delta_wass"]    = pd.Series(wass_h1, index=dataframe.index).diff().fillna(0)

        topo_signal = dataframe["delta_entropy"] + dataframe["delta_wass"]

        rolling_mean = topo_signal.rolling(self.Z_WINDOW, min_periods=10).mean()
        rolling_std  = topo_signal.rolling(self.Z_WINDOW, min_periods=10).std()
        dataframe["topo_z"] = (topo_signal - rolling_mean) / (rolling_std + 1e-8)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (
                (dataframe["topo_z"] < self.BUY_Z) &
                (dataframe["volume"] > 0)
            ),
            "enter_long",
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (
                (dataframe["topo_z"] > self.SELL_Z) &
                (dataframe["volume"] > 0)
            ),
            "exit_long",
        ] = 1

        return dataframe
