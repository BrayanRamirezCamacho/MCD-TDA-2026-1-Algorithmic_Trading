
# ============================================================
# TDA ML STRATEGY – PREDICCIÓN CON CLASIFICADOR ENTRENADO
# Kevin Galvan Lara – Proyecto TDA Trading
# ============================================================
#
# Requiere ejecutar primero:
#   notebooks/TDA_Label_and_Train.py
#
# El modelo en user_data/models/tda_model.pkl predice:
#   -1 → SELL  |  0 → HOLD  |  1 → BUY
#
# Se combinan dos capas de decisión:
#   1. Predicción ML (probabilidades del clasificador)
#   2. Confirmación RSI (filtro anti-señales falsas)
# ============================================================

import numpy as np
import pandas as pd
from pathlib import Path
from pandas import DataFrame
from freqtrade.strategy import IStrategy
import talib.abstract as ta

try:
    from ripser import ripser
    from persim import wasserstein
    import joblib
except ImportError as e:
    raise ImportError(f"Instalar dependencias: pip install ripser persim joblib\n{e}")


class TDAMLStrategy(IStrategy):
    """
    Estrategia de trading que usa un clasificador ML entrenado
    sobre features TDA (entropía de persistencia + Wasserstein).

    El modelo predice BUY / HOLD / SELL para cada vela.
    Se aplica un filtro RSI para reducir falsas señales.
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
    # Parámetros TDA (deben coincidir con los del entrenamiento)
    # --------------------------------------------------------
    WINDOW   = 40
    TAU      = 1
    Z_WINDOW = 50
    SMOOTH   = 20

    # Umbral de probabilidad para activar señal
    BUY_PROB_THR  = 0.40
    SELL_PROB_THR = 0.40

    # Filtro RSI
    RSI_BUY_MAX  = 65   # No comprar si RSI ya está alto
    RSI_SELL_MIN = 40   # No vender si RSI todavía está bajo

    # --------------------------------------------------------
    # Cargar modelo en la primera instancia
    # --------------------------------------------------------

    _model        = None
    _feature_cols = None

    def _load_model(self):
        if self._model is not None:
            return

        model_path = (
            Path(__file__).resolve().parent.parent / "models" / "tda_model.pkl"
        )

        if not model_path.exists():
            raise FileNotFoundError(
                f"Modelo no encontrado: {model_path}\n"
                "Ejecuta primero: notebooks/TDA_Label_and_Train.py"
            )

        payload = joblib.load(model_path)
        TDAMLStrategy._model        = payload["model"]
        TDAMLStrategy._feature_cols = payload["feature_cols"]

        print(f"[TDAMLStrategy] Modelo cargado: {payload.get('model_name', 'desconocido')}")

    # --------------------------------------------------------
    # Helpers TDA (idénticos a TDAStrategy)
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
        m  = len(tk)
        cloud = np.column_stack([
            tk,
            vol[self.TAU: self.TAU + m],
            hlr[self.TAU: self.TAU + m],
            acc[self.TAU: self.TAU + m],
        ])
        mu  = cloud.mean(axis=0)
        std = cloud.std(axis=0) + 1e-8
        return (cloud - mu) / std

    def _compute_tda_features(self, dataframe: DataFrame) -> pd.DataFrame:
        n     = len(dataframe)
        close = dataframe["close"].values
        vol   = dataframe["volume"].values

        log_ret  = np.concatenate([[0.0], np.log(close[1:] / (close[:-1] + 1e-12))])
        dlog_vol = np.concatenate([[0.0], np.diff(np.log(vol + 1.0))])
        hl_range = (dataframe["high"].values - dataframe["low"].values) / (close + 1e-8)
        d_hl     = np.concatenate([[0.0], np.diff(hl_range)])
        ret_acc  = np.concatenate([[0.0], np.diff(log_ret)])

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

        delta_ent  = np.concatenate([[0.0], np.diff(ent_h1)])
        delta_wass = np.concatenate([[0.0], np.diff(wass_h1)])

        topo_signal = delta_ent + delta_wass
        roll_mean = (
            pd.Series(topo_signal)
            .rolling(self.Z_WINDOW, min_periods=10)
            .mean()
            .values
        )
        roll_std = (
            pd.Series(topo_signal)
            .rolling(self.Z_WINDOW, min_periods=10)
            .std()
            .values
        )
        topo_z = (topo_signal - roll_mean) / (roll_std + 1e-8)

        ent_smooth  = pd.Series(ent_h1).rolling(self.SMOOTH, min_periods=5).mean().values
        wass_smooth = pd.Series(wass_h1).rolling(self.SMOOTH, min_periods=5).mean().values

        return pd.DataFrame({
            "entropy_h1":    ent_h1,
            "delta_entropy": delta_ent,
            "wass_h1":       wass_h1,
            "delta_wass":    delta_wass,
            "topo_z":        topo_z,
            "entropy_smooth": ent_smooth,
            "wass_smooth":   wass_smooth,
        })

    # --------------------------------------------------------
    # Freqtrade API
    # --------------------------------------------------------

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        self._load_model()

        # ── RSI como filtro ──────────────────────────────────
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)

        # ── Features TDA ─────────────────────────────────────
        tda_df = self._compute_tda_features(dataframe)
        for col in tda_df.columns:
            dataframe[col] = tda_df[col].values

        # ── Predicción ML ────────────────────────────────────
        X = dataframe[self._feature_cols].fillna(0).values

        proba = self._model.predict_proba(X)

        # Orden de clases: depende del entrenamiento (-1, 0, 1)
        classes = list(self._model.classes_) if hasattr(self._model, "classes_") else [-1, 0, 1]
        try:
            clf = self._model.named_steps["clf"]
            classes = list(clf.classes_)
        except Exception:
            pass

        sell_idx = classes.index(-1) if -1 in classes else 0
        hold_idx = classes.index(0)  if  0 in classes else 1
        buy_idx  = classes.index(1)  if  1 in classes else 2

        dataframe["prob_sell"] = proba[:, sell_idx]
        dataframe["prob_hold"] = proba[:, hold_idx]
        dataframe["prob_buy"]  = proba[:, buy_idx]
        dataframe["ml_signal"] = np.argmax(proba, axis=1) - 1  # -1/0/1

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (
                (dataframe["prob_buy"] >= self.BUY_PROB_THR) &
                (dataframe["rsi"] < self.RSI_BUY_MAX) &
                (dataframe["volume"] > 0)
            ),
            "enter_long",
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (
                (dataframe["prob_sell"] >= self.SELL_PROB_THR) &
                (dataframe["rsi"] > self.RSI_SELL_MIN) &
                (dataframe["volume"] > 0)
            ),
            "exit_long",
        ] = 1

        return dataframe
