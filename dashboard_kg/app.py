
# ============================================================
# TDA CRYPTO TRADING DASHBOARD
# Kevin Galvan Lara – Proyecto TDA Trading
# ============================================================
# Ejecutar:
#   cd trading_tda
#   .venv\Scripts\streamlit run dashboard\app.py
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
import joblib
from pathlib import Path

from ripser import ripser
from persim import wasserstein

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


class _XGBEncoded(BaseEstimator, ClassifierMixin):
    """Wrapper para XGBClassifier que acepta etiquetas [-1, 0, 1] vía LabelEncoder interno."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def fit(self, X, y):
        self._le = LabelEncoder()
        y_enc = self._le.fit_transform(y)
        self.classes_ = self._le.classes_
        self._clf = XGBClassifier(**self.kwargs)
        self._clf.fit(X, y_enc)
        return self

    def predict(self, X):
        return self._le.inverse_transform(self._clf.predict(X))

    def predict_proba(self, X):
        return self._clf.predict_proba(X)

    @property
    def feature_importances_(self):
        return self._clf.feature_importances_

import streamlit as st


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="TDA Crypto Trading",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="stMetricValue"] { font-size: 1.3rem; }
.section-title { font-size: 1.1rem; font-weight: 600; margin-top: 1rem; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# CONSTANTES
# ============================================================

FEATURE_COLS = [
    "entropy_h1", "delta_entropy",
    "wass_h1",    "delta_wass",
    "topo_z",     "entropy_smooth", "wass_smooth",
]

LABEL_NAMES  = {-1: "SELL", 0: "HOLD", 1: "BUY"}
LABEL_COLORS = {-1: "#ef4444", 0: "#6b7280", 1: "#22c55e"}


# ============================================================
# FUNCIONES TDA
# ============================================================

def takens_embedding(x, tau=1, dim=2):
    n = len(x)
    m = n - (dim - 1) * tau
    emb = np.zeros((m, dim))
    for i in range(dim):
        emb[:, i] = x[i * tau: i * tau + m]
    return emb


def persistence_entropy(dgm):
    if len(dgm) == 0:
        return 0.0
    pers = dgm[:, 1] - dgm[:, 0]
    pers = pers[pers > 0]
    if len(pers) == 0:
        return 0.0
    p = pers / pers.sum()
    return float(-np.sum(p * np.log(p + 1e-12)))


def build_cloud(ret, vol, hlr, acc, tau=1):
    if len(ret) < 3:
        return np.zeros((2, 2))
    tk = takens_embedding(ret, tau=tau, dim=2)
    m  = len(tk)
    cloud = np.column_stack([
        tk,
        vol[tau: tau + m],
        hlr[tau: tau + m],
        acc[tau: tau + m],
    ])
    mu  = cloud.mean(axis=0)
    std = cloud.std(axis=0) + 1e-8
    return (cloud - mu) / std


def compute_tda_pipeline(df_feat, window, tau, step, progress_bar=None):
    close    = df_feat["Close"].values
    vol      = df_feat["Volume"].values
    log_ret  = np.concatenate([[0.0], np.log(close[1:] / (close[:-1] + 1e-12))])
    dlog_vol = np.concatenate([[0.0], np.diff(np.log(vol + 1.0))])
    hl_range = (df_feat["High"].values - df_feat["Low"].values) / (close + 1e-8)
    d_hl     = np.concatenate([[0.0], np.diff(hl_range)])
    ret_acc  = np.concatenate([[0.0], np.diff(log_ret)])

    n_rows   = len(df_feat)
    idx_list = list(range(window, n_rows, step))
    total    = len(idx_list)

    diagrams  = []
    ent_list  = []
    wass_list = [0.0]
    prev_dgm  = None

    for k, i in enumerate(idx_list):
        cloud = build_cloud(
            log_ret[i - window: i], dlog_vol[i - window: i],
            d_hl[i - window: i],    ret_acc[i - window: i],
            tau=tau,
        )
        dgms = ripser(cloud, maxdim=1)["dgms"]
        h1   = dgms[1]

        diagrams.append(dgms)
        ent_list.append(persistence_entropy(h1))

        if prev_dgm is not None and len(prev_dgm) > 0 and len(h1) > 0:
            wass_list.append(float(wasserstein(prev_dgm, h1)))
        elif len(wass_list) <= k:
            wass_list.append(0.0)

        prev_dgm = h1 if len(h1) > 0 else prev_dgm

        if progress_bar and k % 5 == 0:
            progress_bar.progress(min((k + 1) / total, 0.99))

    ent_h1  = np.array(ent_list)
    wass_h1 = np.array(wass_list[: len(ent_h1)])

    delta_ent  = np.concatenate([[0.0], np.diff(ent_h1)])
    delta_wass = np.concatenate([[0.0], np.diff(wass_h1)])
    topo_sig   = delta_ent + delta_wass

    roll_m = pd.Series(topo_sig).rolling(50, min_periods=10).mean().values
    roll_s = pd.Series(topo_sig).rolling(50, min_periods=10).std().values
    topo_z = (topo_sig - roll_m) / (roll_s + 1e-8)

    ent_smooth  = pd.Series(ent_h1).rolling(20, min_periods=5).mean().values
    wass_smooth = pd.Series(wass_h1).rolling(20, min_periods=5).mean().values

    close_aln = df_feat["Close"].iloc[idx_list].values
    date_col  = "Date" if "Date" in df_feat.columns else df_feat.columns[0]
    date_aln  = df_feat[date_col].iloc[idx_list].values

    df_tda = pd.DataFrame({
        "date":           date_aln,
        "close":          close_aln,
        "entropy_h1":     ent_h1,
        "delta_entropy":  delta_ent,
        "wass_h1":        wass_h1,
        "delta_wass":     delta_wass,
        "topo_signal":    topo_sig,
        "topo_z":         topo_z,
        "entropy_smooth": ent_smooth,
        "wass_smooth":    wass_smooth,
    }).dropna().reset_index(drop=True)

    return df_tda, diagrams, idx_list


# ============================================================
# FUNCIONES ML
# ============================================================

def generate_labels(df_tda, horizon, buy_thr, sell_thr):
    df = df_tda.copy()
    df["future_return"] = df["close"].pct_change(horizon).shift(-horizon)
    df["label"] = 0
    df.loc[df["future_return"] >  buy_thr,  "label"] =  1
    df.loc[df["future_return"] < -sell_thr, "label"] = -1
    return df.dropna(subset=["future_return"]).reset_index(drop=True)


def train_models(df_labeled):
    X = df_labeled[FEATURE_COLS].fillna(0).values
    y = df_labeled["label"].values

    n_train = int(len(X) * 0.80)
    X_tr, X_te = X[:n_train], X[n_train:]
    y_tr, y_te = y[:n_train], y[n_train:]

    classes_train = np.unique(y_tr)
    if len(classes_train) < 2:
        raise ValueError(
            f"El conjunto de entrenamiento solo tiene la clase {classes_train.tolist()}. "
            "Reduce los umbrales de retorno (compra/venta) o amplía el rango de fechas."
        )
    if len(np.unique(y)) < 3:
        missing = [l for l in [-1, 0, 1] if l not in np.unique(y)]
        names   = {-1: "SELL", 0: "HOLD", 1: "BUY"}
        raise ValueError(
            f"No hay suficientes ejemplos de {[names[m] for m in missing]}. "
            "Reduce los umbrales de retorno o amplía el rango de fechas."
        )

    candidates = {
        "RandomForest": Pipeline([
            ("sc", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=200, max_depth=6,
                class_weight="balanced", random_state=42, n_jobs=-1,
            )),
        ]),
        "GradientBoosting": Pipeline([
            ("sc", StandardScaler()),
            ("clf", GradientBoostingClassifier(
                n_estimators=150, max_depth=4,
                learning_rate=0.05, random_state=42,
            )),
        ]),
        "LogisticReg": Pipeline([
            ("sc", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=500, class_weight="balanced",
                multi_class="multinomial", random_state=42,
            )),
        ]),
    }
    if HAS_XGB:
        candidates["XGBoost"] = Pipeline([
            ("sc", StandardScaler()),
            ("clf", _XGBEncoded(
                n_estimators=200, max_depth=5, learning_rate=0.05,
                eval_metric="mlogloss",
                random_state=42, verbosity=0,
            )),
        ])

    results = {}
    for name, model in candidates.items():
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        report = classification_report(
            y_te, y_pred,
            target_names=["SELL", "HOLD", "BUY"],
            labels=[-1, 0, 1],
            output_dict=True,
            zero_division=0,
        )
        results[name] = {
            "model":    model,
            "y_pred":   y_pred,
            "y_test":   y_te,
            "macro_f1": report["macro avg"]["f1-score"],
            "report":   report,
            "cm":       confusion_matrix(y_te, y_pred, labels=[-1, 0, 1]),
        }

    best = max(results, key=lambda k: results[k]["macro_f1"])
    return results, best, (X_tr, X_te, y_tr, y_te)


def predict_signals(model, df_labeled):
    X = df_labeled[FEATURE_COLS].fillna(0).values
    proba = model.predict_proba(X)
    try:
        classes = list(model.named_steps["clf"].classes_)
    except Exception:
        classes = [-1, 0, 1]

    sidx = classes.index(-1) if -1 in classes else 0
    hidx = classes.index(0)  if  0 in classes else 1
    bidx = classes.index(1)  if  1 in classes else 2

    df = df_labeled.copy()
    df["prob_sell"] = proba[:, sidx]
    df["prob_hold"] = proba[:, hidx]
    df["prob_buy"]  = proba[:, bidx]
    df["ml_signal"] = np.argmax(proba, axis=1) - 1
    return df


# ============================================================
# BACKTESTING
# ============================================================

def run_backtest(df_signals, initial_capital=10_000, fee=0.001, buy_z=-1.0, sell_z=1.5):

    rows = list(df_signals.itertuples(index=False))
    n    = len(rows)

    # ── ML strategy ──────────────────────────────────────────
    eq_ml, cap_ml, pos_ml, trades_ml = np.zeros(n), float(initial_capital), 0.0, []
    for k, row in enumerate(rows):
        price, sig = float(row.close), int(row.ml_signal)
        if sig == 1 and pos_ml == 0 and cap_ml > 0:
            pos_ml = cap_ml * (1 - fee) / price
            cap_ml = 0.0
            trades_ml.append({"type": "BUY",  "precio": price,
                               "fecha": getattr(row, "date", k)})
        elif sig == -1 and pos_ml > 0:
            cap_ml = pos_ml * price * (1 - fee)
            pos_ml = 0.0
            trades_ml.append({"type": "SELL", "precio": price,
                               "fecha": getattr(row, "date", k)})
        eq_ml[k] = cap_ml + pos_ml * price

    # ── TDA reglas ───────────────────────────────────────────
    eq_tda, cap_tda, pos_tda, trades_tda = np.zeros(n), float(initial_capital), 0.0, []
    for k, row in enumerate(rows):
        price, tz = float(row.close), float(getattr(row, "topo_z", 0))
        if tz < buy_z and pos_tda == 0 and cap_tda > 0:
            pos_tda = cap_tda * (1 - fee) / price
            cap_tda = 0.0
            trades_tda.append({"type": "BUY",  "precio": price,
                                "fecha": getattr(row, "date", k)})
        elif tz > sell_z and pos_tda > 0:
            cap_tda = pos_tda * price * (1 - fee)
            pos_tda = 0.0
            trades_tda.append({"type": "SELL", "precio": price,
                                "fecha": getattr(row, "date", k)})
        eq_tda[k] = cap_tda + pos_tda * price

    # ── Buy & Hold ───────────────────────────────────────────
    p0     = float(rows[0].close)
    eq_bah = initial_capital * np.array([float(r.close) for r in rows]) / p0

    def _stats(equity, trades):
        ret  = (equity[-1] / initial_capital - 1) * 100
        peak = np.maximum.accumulate(equity)
        dd   = (equity - peak) / (peak + 1e-8)
        mdd  = float(dd.min()) * 100
        buys = [t for t in trades if t["type"] == "BUY"]
        wins = sum(
            1 for j in range(len(trades) - 1)
            if trades[j]["type"] == "BUY" and trades[j + 1]["type"] == "SELL"
            and trades[j + 1]["precio"] > trades[j]["precio"]
        )
        wr = (wins / max(len(buys), 1)) * 100
        return {"return_pct": ret, "max_dd": mdd,
                "n_trades": len(buys), "win_rate": wr, "dd_series": dd}

    return {
        "eq_ml":   eq_ml,  "eq_tda":  eq_tda,  "eq_bah": eq_bah,
        "s_ml":    _stats(eq_ml,  trades_ml),
        "s_tda":   _stats(eq_tda, trades_tda),
        "bah_ret": (eq_bah[-1] / initial_capital - 1) * 100,
        "trades_ml":  trades_ml,
        "trades_tda": trades_tda,
        "dates": df_signals["date"].values if "date" in df_signals.columns else np.arange(n),
    }


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.title("📈 TDA Trading")
    st.caption("Análisis Topológico para Crypto")
    st.divider()

    st.subheader("📥 Datos")
    symbol     = st.selectbox("Símbolo", ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD"])
    start_date = st.date_input("Inicio", value=pd.Timestamp("2021-01-01"))
    end_date   = st.date_input("Fin",    value=pd.Timestamp("today"))

    st.subheader("🔵 TDA")
    window = st.slider("Ventana embedding",   20, 80, 40, 5)
    tau    = st.slider("Tau (delay)",          1,  5,  1)
    step   = st.slider("Paso entre ventanas",  1, 10,  5)

    st.subheader("📏 Umbrales (Reglas)")
    buy_z  = st.slider("Z compra (negativo)",  -3.0, 0.0, -1.0, 0.1)
    sell_z = st.slider("Z venta (positivo)",    0.0, 4.0,  1.5, 0.1)

    st.subheader("🤖 ML")
    horizon  = st.slider("Horizonte (velas)", 3, 30, 10)
    buy_thr  = st.slider("Umbral compra (%)", 0.5, 5.0, 2.0, 0.5) / 100
    sell_thr = st.slider("Umbral venta (%)",  0.5, 5.0, 2.0, 0.5) / 100

    st.divider()
    run_btn = st.button("▶ Ejecutar Pipeline", type="primary", use_container_width=True)


# ============================================================
# HEADER
# ============================================================

st.title("📈 TDA Crypto Trading Dashboard")
st.markdown(
    "Pipeline completo de **Análisis Topológico de Datos (TDA)** para predicción "
    "de señales **COMPRAR / RETENER / VENDER** en criptomonedas."
)


# ============================================================
# EJECUCIÓN DEL PIPELINE
# ============================================================

if run_btn:
    ph_step  = st.empty()
    ph_prog  = st.empty()
    ph_msg   = st.empty()

    # Paso 1 – Datos
    ph_step.info("**Paso 1/4** — Descargando datos...")
    df_raw = yf.download(symbol, start=str(start_date), end=str(end_date), progress=False)
    df_raw = df_raw.dropna().reset_index()
    df_raw.columns = [c[0] if isinstance(c, tuple) else c for c in df_raw.columns]

    # yfinance puede llamar la columna "Datetime" según la versión
    if "Date" not in df_raw.columns:
        date_candidates = [c for c in df_raw.columns if "date" in str(c).lower() or "time" in str(c).lower()]
        if date_candidates:
            df_raw = df_raw.rename(columns={date_candidates[0]: "Date"})
        else:
            df_raw = df_raw.rename(columns={df_raw.columns[0]: "Date"})

    if len(df_raw) < 120:
        st.error("Pocos datos: amplía el rango de fechas (mínimo ~120 velas).")
        st.stop()
    ph_msg.caption(f"✓ {len(df_raw)} velas  |  {str(df_raw.iloc[0,0])[:10]} → {str(df_raw.iloc[-1,0])[:10]}")

    # Paso 2 – TDA
    ph_step.info("**Paso 2/4** — Computando homología persistente (puede tardar 1-3 min)...")
    prog_bar = ph_prog.progress(0)

    df_tda, diagrams, idx_list = compute_tda_pipeline(
        df_raw, window=window, tau=tau, step=step, progress_bar=prog_bar
    )
    prog_bar.progress(1.0)
    ph_msg.caption(f"✓ {len(df_tda)} ventanas TDA completadas")

    # Paso 3 – ML
    ph_step.info("**Paso 3/4** — Etiquetando y entrenando clasificadores...")
    df_labeled = generate_labels(df_tda, horizon=horizon, buy_thr=buy_thr, sell_thr=sell_thr)

    label_counts = df_labeled["label"].value_counts()
    ph_step.empty(); ph_prog.empty(); ph_msg.empty()

    missing_cls = [n for lbl, n in {-1: "SELL", 0: "HOLD", 1: "BUY"}.items()
                   if lbl not in label_counts.index or label_counts[lbl] < 3]
    if missing_cls:
        st.error(
            f"No hay suficientes ejemplos de **{', '.join(missing_cls)}** con los umbrales actuales.  \n"
            f"Distribución actual: SELL={label_counts.get(-1,0)}, "
            f"HOLD={label_counts.get(0,0)}, BUY={label_counts.get(1,0)}  \n"
            "**Solución:** reduce el umbral de retorno (compra/venta) en el panel izquierdo, "
            "o amplía el rango de fechas."
        )
        st.stop()

    try:
        ml_results, best_name, splits = train_models(df_labeled)
    except ValueError as e:
        st.error(f"Error al entrenar: {e}")
        st.stop()

    df_signals = predict_signals(ml_results[best_name]["model"], df_labeled)
    ph_msg.caption(f"✓ Mejor modelo: {best_name}  |  Macro F1 = {ml_results[best_name]['macro_f1']:.4f}")

    # Paso 4 – Backtest
    ph_step.info("**Paso 4/4** — Simulando backtesting...")
    bt = run_backtest(df_signals, buy_z=buy_z, sell_z=sell_z)

    ph_step.empty(); ph_prog.empty(); ph_msg.empty()

    st.session_state["R"] = {
        "df_raw":     df_raw,
        "df_tda":     df_tda,
        "diagrams":   diagrams,
        "df_labeled": df_labeled,
        "ml":         ml_results,
        "best":       best_name,
        "splits":     splits,
        "df_signals": df_signals,
        "bt":         bt,
        "p": {"symbol": symbol, "window": window, "tau": tau, "step": step,
              "buy_z": buy_z, "sell_z": sell_z,
              "horizon": horizon, "buy_thr": buy_thr, "sell_thr": sell_thr},
    }
    st.success(
        f"✅ Pipeline completado  |  {symbol}  |  {len(df_raw)} velas  |  "
        f"Mejor modelo: **{best_name}** (Macro F1={ml_results[best_name]['macro_f1']:.4f})"
    )


# ============================================================
# PANTALLA DE BIENVENIDA
# ============================================================

if "R" not in st.session_state:
    col1, col2, col3 = st.columns(3)
    col1.info("**1.** Elige símbolo y fechas en el panel izquierdo")
    col2.info("**2.** Ajusta los parámetros TDA y ML")
    col3.info("**3.** Pulsa **▶ Ejecutar Pipeline**")

    with st.expander("📖 ¿Cómo funciona el pipeline?", expanded=True):
        st.markdown("### Flujo general")
        st.code(
            "Datos OHLCV\n"
            "      │\n"
            "      ▼\n"
            "Feature engineering ── log_ret · dlog_vol · ΔH-L · ret_acc\n"
            "      │\n"
            "      ▼\n"
            "Embedding de Takens ── nube de puntos en ℝ⁵\n"
            "      │\n"
            "      ▼\n"
            "Homología Persistente (ripser) ── diagrama H₁\n"
            "      │\n"
            "      ├── Entropía de Persistencia H₁   (regularidad topológica)\n"
            "      └── Distancia Wasserstein H₁       (cambio topológico)\n"
            "      │\n"
            "      ▼\n"
            "Señal Topológica Z-score\n"
            "      │\n"
            "      ├── Reglas:  topo_z < umbral → COMPRA · topo_z > umbral → VENTA\n"
            "      └── ML:      RandomForest / XGBoost  →  BUY / HOLD / SELL\n"
            "      │\n"
            "      ▼\n"
            "Backtesting  ── curva de equity vs Buy-and-Hold",
            language="text",
        )

        st.divider()
        st.markdown("### Explicación de cada fase")

        st.markdown("#### Fase 1 — Datos OHLCV")
        st.markdown("""
        Se descargan datos históricos de precio desde Yahoo Finance.
        Cada fila es una **vela** (periodo de tiempo) con:

        | Columna | Qué es |
        |---------|--------|
        | **Open** | Precio al abrir el periodo |
        | **High** | Precio más alto del periodo |
        | **Low** | Precio más bajo del periodo |
        | **Close** | Precio al cerrar el periodo |
        | **Volume** | Cantidad de moneda transaccionada |
        """)

        st.markdown("#### Fase 2 — Feature Engineering")
        st.markdown("""
        Se calculan 4 indicadores derivados que capturan distintos aspectos del mercado:

        | Feature | Fórmula | Qué mide |
        |---------|---------|---------|
        | `log_ret` | log(Closeₜ / Closeₜ₋₁) | Velocidad del precio (retorno logarítmico) |
        | `dlog_vol` | Δ log(Volumen) | Aceleración del volumen de trading |
        | `d_hl` | Δ (High - Low) / Close | Cambio en la volatilidad intradía |
        | `ret_acc` | Δ log_ret | Aceleración del retorno (¿está acelerando o frenando?) |

        Estos 4 valores por cada momento forman la materia prima para el análisis topológico.
        """)

        st.markdown("#### Fase 3 — Embedding de Takens y el espacio ℝ⁵")
        st.markdown("""
        **¿Qué es ℝ⁵?**
        Es simplemente un espacio de 5 dimensiones. Cada punto en ese espacio tiene
        5 coordenadas: (x₁, x₂, x₃, x₄, x₅).
        En lugar de ver el precio como una línea en el tiempo, lo convertimos en
        una **nube de puntos** en 5 dimensiones para revelar su forma geométrica.

        **¿Cómo se construye la nube?**
        Se usa el **Teorema de Takens** (1981): dado que los precios son un sistema
        dinámico, podemos reconstruir su geometría tomando un valor y su versión
        retrasada τ pasos en el tiempo.
        Para `log_ret` con τ=1 y dim=2 se forma:

        ```
        punto i = (log_ret[i],  log_ret[i-1])   ← embedding 2D de Takens
        ```

        Luego se agregan las otras 3 features para obtener un punto en ℝ⁵:
        ```
        punto i = (log_ret[i], log_ret[i-1], dlog_vol[i], d_hl[i], ret_acc[i])
        ```

        Esto se repite para una **ventana deslizante** de N velas, generando
        una nube de ~40 puntos en ℝ⁵ por cada momento del tiempo.
        """)

        st.markdown("#### Fase 4 — Homología Persistente")
        st.markdown("""
        **¿Qué es la homología persistente?**
        Es una técnica matemática que analiza la **forma** de una nube de puntos.
        Imagina inflar burbujas alrededor de cada punto. Conforme las burbujas crecen:
        - Primero los puntos se conectan (componentes conexas → **H₀**)
        - Luego se forman anillos o ciclos cerrados (**H₁**)
        - Cada ciclo "nace" en un radio y "muere" cuando se llena

        **El Diagrama de Persistencia** registra cuándo nace y muere cada ciclo.
        Un punto (b, d) en el diagrama significa: "existió un ciclo desde radio b hasta radio d".
        - **Puntos lejos de la diagonal** → ciclos que vivieron mucho = estructura real del mercado
        - **Puntos cerca de la diagonal** → ciclos que vivieron poco = ruido topológico

        **¿Por qué importa en trading?**
        - Un mercado **en tendencia** tiene una forma simple (pocos ciclos, poca entropía)
        - Un mercado **caótico o lateral** tiene muchos ciclos entrelazados (alta entropía)
        - Un **cambio brusco** en la forma entre dos ventanas = posible cambio de régimen
        """)

        st.markdown("#### Fase 5 — Entropía y Wasserstein")
        st.markdown("""
        De cada diagrama de persistencia se extraen dos números resumen:

        **Entropía de Persistencia H₁**
        Mide qué tan "desordenados" o "iguales" son los ciclos del diagrama.
        - Entropía **alta** → muchos ciclos con duraciones similares → mercado complejo/turbulento
        - Entropía **baja** → pocos ciclos dominantes → mercado con estructura clara o tendencia
        - Una **caída brusca** de entropía puede indicar que el mercado está ordenándose
          (formación de tendencia)

        **Distancia Wasserstein H₁**
        Mide cuánto "cambió la forma" del mercado entre dos ventanas consecutivas.
        Se interpreta como el costo mínimo de mover los puntos del diagrama anterior
        hasta que coincidan con el diagrama nuevo.
        - Wasserstein **alto** → la topología del mercado cambió mucho → posible cambio de régimen
        - Wasserstein **bajo** → el mercado sigue teniendo la misma forma → estabilidad

        La **Señal Topológica** combina ambos:
        ```
        topo_signal = Δ(entropía) + Δ(Wasserstein)
        topo_z      = z-score adaptivo de topo_signal
        ```
        """)

        st.markdown("#### Fase 6 — Señales por Reglas")
        st.markdown("""
        La señal `topo_z` se normaliza con un z-score sobre una ventana móvil,
        lo que permite comparar momentos con distinta volatilidad:

        | Condición | Señal | Interpretación |
        |-----------|-------|----------------|
        | `topo_z < umbral_compra` (ej. -1.0) | 🟢 **COMPRAR** | La topología se simplifica → tendencia formándose |
        | `topo_z > umbral_venta` (ej. +1.5) | 🔴 **VENDER** | La topología se complejiza → posible reversión o ruptura |
        | Ninguno | ⚪ **RETENER** | El mercado está en estado neutro |
        """)

        st.markdown("#### Fase 7 — Modelo ML")
        st.markdown("""
        Se entrena un clasificador supervisado usando los features TDA como entrada:

        **¿Cómo se generan las etiquetas?**
        Para cada ventana, se mira el precio N velas hacia adelante:
        ```
        Si retorno futuro >  umbral_compra  → etiqueta = BUY  ( 1)
        Si retorno futuro <  -umbral_venta  → etiqueta = SELL (-1)
        Si retorno futuro está en el medio  → etiqueta = HOLD ( 0)
        ```

        **Features que recibe el modelo:**
        `entropy_h1`, `delta_entropy`, `wass_h1`, `delta_wass`, `topo_z`,
        `entropy_smooth`, `wass_smooth`

        **Modelos entrenados** (se elige el de mayor Macro F1 en el 20% de test):
        - **RandomForest** — 200 árboles, robusto, da importancia de features
        - **GradientBoosting** — optimización secuencial, más preciso en datos no lineales
        - **LogisticRegression** — modelo lineal, rápido, sirve como baseline
        - **XGBoost** — versión acelerada de Gradient Boosting, suele ser el mejor

        **¿Qué es Macro F1?**
        Mide qué tan bien el modelo predice las 3 clases por igual (BUY, HOLD, SELL).
        Un F1 = 1.0 sería perfecto; F1 = 0.33 equivale a adivinar al azar en 3 clases.

        **División temporal:**
        El 80% más antiguo de los datos se usa para entrenar y el 20% más reciente
        para evaluar — nunca se entrena con datos del futuro.
        """)

        st.markdown("#### Fase 8 — Backtesting")
        st.markdown("""
        Se simula qué habría pasado si hubieras seguido las señales desde el inicio:
        - Se empieza con el capital que eliges (ej. $10,000)
        - Cada señal **BUY** invierte todo el capital disponible
        - Cada señal **SELL** cierra la posición y devuelve efectivo
        - Se cobra una comisión del **0.1%** por operación
        - Se compara contra **Buy & Hold** (comprar y no hacer nada)

        Las métricas clave:
        | Métrica | Qué mide |
        |---------|---------|
        | **Retorno (%)** | Ganancia o pérdida total del período |
        | **Max Drawdown** | La mayor caída desde un pico — mide el peor momento |
        | **Win Rate** | % de trades que cerraron en positivo |
        | **# Trades** | Cuántas veces se compró |
        """)
    st.stop()


# ============================================================
# TABS PRINCIPALES
# ============================================================

R = st.session_state["R"]
p = R["p"]

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Datos",
    "🔵 Análisis TDA",
    "🤖 Modelo ML",
    "🎯 Señales",
    "📊 Backtesting",
])


# ─────────────────────────────────────────────────────────────
# TAB 1 – DATOS
# ─────────────────────────────────────────────────────────────

with tab1:
    df = R["df_raw"]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Símbolo",       p["symbol"])
    c2.metric("Velas",         f"{len(df):,}")
    c3.metric("Precio máximo", f"${float(df['High'].max()):,.0f}")
    c4.metric("Precio mínimo", f"${float(df['Low'].min()):,.0f}")
    c5.metric("Retorno total",
              f"{(float(df['Close'].iloc[-1]) / float(df['Close'].iloc[0]) - 1) * 100:.1f}%")

    # Candlestick + volumen
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.75, 0.25], vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(
        x=df["Date"], open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        name=p["symbol"],
        increasing_line_color="#22c55e",
        decreasing_line_color="#ef4444",
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        x=df["Date"], y=df["Volume"],
        name="Volumen", marker_color="#6366f1", opacity=0.5,
    ), row=2, col=1)
    fig.update_layout(height=480, template="plotly_dark",
                      title=f"{p['symbol']} – OHLCV",
                      xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    # Features derivados
    st.subheader("Features de entrada al embedding de Takens")
    close    = df["Close"].values
    log_ret  = np.concatenate([[0.0], np.log(close[1:] / (close[:-1] + 1e-12))])
    dlog_vol = np.concatenate([[0.0], np.diff(np.log(df["Volume"].values + 1.0))])
    hl_range = (df["High"].values - df["Low"].values) / (close + 1e-8)
    d_hl     = np.concatenate([[0.0], np.diff(hl_range)])

    fig2 = make_subplots(rows=3, cols=1, shared_xaxes=True,
                         subplot_titles=["Retorno Logarítmico (log_ret)",
                                         "Δ Log Volumen (dlog_vol)",
                                         "Δ Rango High-Low / Close (d_hl)"],
                         vertical_spacing=0.06)
    for row, y, color in [
        (1, log_ret,  "#22c55e"),
        (2, dlog_vol, "#6366f1"),
        (3, d_hl,     "#f59e0b"),
    ]:
        fig2.add_trace(go.Scatter(x=df["Date"], y=y, mode="lines",
                                  line=dict(color=color, width=0.7)), row=row, col=1)
    fig2.update_layout(height=400, template="plotly_dark", showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)


# ─────────────────────────────────────────────────────────────
# TAB 2 – ANÁLISIS TDA
# ─────────────────────────────────────────────────────────────

with tab2:
    df_tda   = R["df_tda"]
    diagrams = R["diagrams"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Ventanas TDA",         f"{len(df_tda):,}")
    c2.metric("Entropía H₁ media",    f"{df_tda['entropy_h1'].mean():.4f}")
    c3.metric("Wasserstein H₁ medio", f"{df_tda['wass_h1'].mean():.4f}")
    c4.metric("Señales activas",
              f"{int((df_tda['topo_z'] < p['buy_z']).sum() + (df_tda['topo_z'] > p['sell_z']).sum())}")

    # Precio + topo_z + zonas
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        subplot_titles=[
            "Precio + Zonas de Señal Topológica",
            "Entropía de Persistencia H₁ (suavizada)",
            "Distancia Wasserstein H₁ (suavizada)",
        ],
        row_heights=[0.5, 0.25, 0.25],
        vertical_spacing=0.05,
    )

    # Precio
    fig.add_trace(go.Scatter(x=df_tda["date"], y=df_tda["close"],
                              line=dict(color="white", width=1), name="Precio"), row=1, col=1)

    # Zonas compra/venta sobre precio
    for sig_mask, color, name, sym in [
        (df_tda["topo_z"] < p["buy_z"],  "#22c55e", "Señal BUY",  "triangle-up"),
        (df_tda["topo_z"] > p["sell_z"], "#ef4444", "Señal SELL", "triangle-down"),
    ]:
        fig.add_trace(go.Scatter(
            x=df_tda["date"][sig_mask], y=df_tda["close"][sig_mask],
            mode="markers",
            marker=dict(color=color, size=6, symbol=sym, opacity=0.8),
            name=name,
        ), row=1, col=1)

    # topo_z como línea secundaria (eje y2 con normalización visual)
    tz_norm = (df_tda["topo_z"] - df_tda["topo_z"].mean()) / (df_tda["topo_z"].std() + 1e-8)
    price_range = df_tda["close"].max() - df_tda["close"].min()
    price_mid   = (df_tda["close"].max() + df_tda["close"].min()) / 2
    tz_overlay  = price_mid + tz_norm * price_range * 0.15
    fig.add_trace(go.Scatter(x=df_tda["date"], y=tz_overlay,
                              line=dict(color="#e879f9", width=1, dash="dot"),
                              name="topo_z (escalado)", opacity=0.7), row=1, col=1)

    # Entropía
    fig.add_trace(go.Scatter(x=df_tda["date"], y=df_tda["entropy_smooth"],
                              line=dict(color="#06b6d4", width=1.3), name="Entropía H₁"), row=2, col=1)

    # Wasserstein
    fig.add_trace(go.Scatter(x=df_tda["date"], y=df_tda["wass_smooth"],
                              line=dict(color="#f97316", width=1.3), name="Wasserstein H₁"), row=3, col=1)

    fig.update_layout(height=600, template="plotly_dark",
                      title="Señales Topológicas – Pipeline TDA Completo",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(fig, use_container_width=True)

    # Diagramas de persistencia
    st.subheader("Diagramas de Persistencia – Ejemplos")
    st.caption("Cada punto (birth, death) representa una característica topológica. "
               "Puntos lejos de la diagonal = estructuras persistentes y significativas.")

    nd = len(diagrams)
    samples = [nd // 5, nd // 2, 4 * nd // 5]
    labels  = ["Ventana temprana (20%)", "Ventana media (50%)", "Ventana tardía (80%)"]
    cols    = st.columns(3)

    for col, sidx, slabel in zip(cols, samples, labels):
        with col:
            dgm = diagrams[sidx]
            h0, h1 = dgm[0], dgm[1]

            all_vals = []
            if len(h0) > 0:
                finite = h0[h0[:, 1] < 1e9]
                if len(finite):
                    all_vals.extend(finite[:, 1].tolist())
            if len(h1) > 0:
                all_vals.extend(h1[:, 0].tolist())
                all_vals.extend(h1[:, 1].tolist())
            max_v = max(all_vals) * 1.1 if all_vals else 1.0

            fig_d = go.Figure()
            fig_d.add_trace(go.Scatter(
                x=[0, max_v], y=[0, max_v], mode="lines",
                line=dict(color="#4b5563", dash="dash", width=1),
                showlegend=False,
            ))
            if len(h0) > 0:
                fin_h0 = h0[h0[:, 1] < 1e9]
                if len(fin_h0):
                    fig_d.add_trace(go.Scatter(
                        x=fin_h0[:, 0], y=fin_h0[:, 1], mode="markers",
                        marker=dict(color="#22c55e", size=7, opacity=0.8),
                        name="H₀ (comp.)",
                    ))
            if len(h1) > 0:
                fig_d.add_trace(go.Scatter(
                    x=h1[:, 0], y=h1[:, 1], mode="markers",
                    marker=dict(color="#818cf8", size=9, opacity=0.9,
                                symbol="diamond"),
                    name="H₁ (ciclos)",
                ))
            fig_d.update_layout(
                title=slabel, height=290, template="plotly_dark",
                xaxis_title="Nacimiento (birth)",
                yaxis_title="Muerte (death)",
                legend=dict(x=0.01, y=0.99, font=dict(size=10)),
                margin=dict(l=45, r=10, t=45, b=40),
            )
            st.plotly_chart(fig_d, use_container_width=True)

    with st.expander("📖 Interpretación de los diagramas"):
        st.markdown("""
        | Elemento | Significado en trading |
        |---|---|
        | **H₀ verde** | Componentes conexas – clusters de comportamiento |
        | **H₁ morado** | Ciclos/loops – movimientos circulares o periódicos |
        | **Puntos cerca de la diagonal** | Ruido topológico (corta vida) |
        | **Puntos lejos de la diagonal** | Estructuras robustas y significativas |
        | **Entropía alta** | Mercado turbulento/complejo |
        | **Entropía baja** | Mercado en tendencia o quieto |
        | **Salto en Wasserstein** | Cambio de régimen topológico (posible reversión) |
        """)


# ─────────────────────────────────────────────────────────────
# TAB 3 – MODELO ML
# ─────────────────────────────────────────────────────────────

with tab3:
    ml         = R["ml"]
    best       = R["best"]
    df_labeled = R["df_labeled"]
    X_tr, X_te, y_tr, y_te = R["splits"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mejor modelo",     best)
    c2.metric("Macro F1 (test)",  f"{ml[best]['macro_f1']:.4f}")
    c3.metric("Train samples",    f"{len(X_tr):,}")
    c4.metric("Test samples",     f"{len(X_te):,}")

    col_pie, col_bar = st.columns([1, 2])

    with col_pie:
        st.subheader("Distribución de etiquetas")
        counts = df_labeled["label"].value_counts().sort_index()
        fig_pie = px.pie(
            values=counts.values,
            names=[LABEL_NAMES[k] for k in counts.index],
            color=[LABEL_NAMES[k] for k in counts.index],
            color_discrete_map={"BUY": "#22c55e", "HOLD": "#6b7280", "SELL": "#ef4444"},
            template="plotly_dark",
        )
        fig_pie.update_layout(height=300, margin=dict(t=10, b=10))
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_bar:
        st.subheader("Comparación de modelos – Macro F1")
        names_m  = list(ml.keys())
        f1_vals  = [ml[k]["macro_f1"] for k in names_m]
        colors_m = ["#22c55e" if k == best else "#6366f1" for k in names_m]
        fig_bar  = go.Figure(go.Bar(
            x=names_m, y=f1_vals,
            marker_color=colors_m,
            text=[f"{v:.4f}" for v in f1_vals],
            textposition="outside",
        ))
        fig_bar.update_layout(
            yaxis_range=[0, max(f1_vals) * 1.15 + 0.05],
            template="plotly_dark", height=300,
            margin=dict(t=10, b=10),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    col_cm, col_fi = st.columns(2)

    with col_cm:
        st.subheader(f"Matriz de Confusión – {best}")
        cm = ml[best]["cm"]
        fig_cm = px.imshow(
            cm,
            x=["SELL", "HOLD", "BUY"],
            y=["SELL", "HOLD", "BUY"],
            color_continuous_scale="Blues",
            text_auto=True,
            template="plotly_dark",
            labels=dict(x="Predicho", y="Real", color="N"),
        )
        fig_cm.update_layout(height=350, margin=dict(t=10, b=10))
        st.plotly_chart(fig_cm, use_container_width=True)

    with col_fi:
        st.subheader(f"Importancia de Features – {best}")
        try:
            clf = ml[best]["model"].named_steps["clf"]
            if hasattr(clf, "feature_importances_"):
                imps = clf.feature_importances_
                sort_idx = np.argsort(imps)
                fig_fi = go.Figure(go.Bar(
                    x=imps[sort_idx],
                    y=[FEATURE_COLS[i] for i in sort_idx],
                    orientation="h",
                    marker_color="#6366f1",
                ))
                fig_fi.update_layout(
                    template="plotly_dark", height=350,
                    margin=dict(t=10, b=10, l=130),
                    xaxis_title="Importancia (Gini)",
                )
                st.plotly_chart(fig_fi, use_container_width=True)
            else:
                st.info("Feature importance no disponible para este modelo.")
        except Exception:
            st.info("No se pudo calcular la importancia de features.")

    # Reporte de clasificación
    st.subheader("Reporte de Clasificación Detallado")
    rep = ml[best]["report"]
    rows_rep = []
    for lname in ["SELL", "HOLD", "BUY"]:
        r_d = rep.get(lname, {})
        rows_rep.append({
            "Clase":     lname,
            "Precision": round(r_d.get("precision", 0), 4),
            "Recall":    round(r_d.get("recall",    0), 4),
            "F1-Score":  round(r_d.get("f1-score",  0), 4),
            "Support":   int(r_d.get("support",     0)),
        })
    rows_rep.append({
        "Clase":     "Macro Avg",
        "Precision": round(rep["macro avg"]["precision"], 4),
        "Recall":    round(rep["macro avg"]["recall"],    4),
        "F1-Score":  round(rep["macro avg"]["f1-score"],  4),
        "Support":   "",
    })
    st.dataframe(pd.DataFrame(rows_rep), use_container_width=True, hide_index=True)

    st.divider()
    if st.button("💾 Guardar modelo para Freqtrade (TDA_ML_Strategy)", type="secondary"):
        save_dir  = Path(__file__).resolve().parents[1] / "freqtrade" / "user_data" / "models"
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / "tda_model.pkl"
        joblib.dump({
            "model":        ml[best]["model"],
            "feature_cols": FEATURE_COLS,
            "params":       p,
            "model_name":   best,
        }, save_path)
        st.success(f"✅ Guardado en `{save_path}`")
        st.info("Ahora puedes correr Freqtrade con `TDA_ML_Strategy.py`.")


# ─────────────────────────────────────────────────────────────
# TAB 4 – SEÑALES
# ─────────────────────────────────────────────────────────────

with tab4:
    df_s = R["df_signals"]

    counts_s = df_s["ml_signal"].value_counts()
    total_s  = len(df_s)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🟢 BUY",  counts_s.get(1, 0),
              delta=f"{100*counts_s.get(1,0)/total_s:.1f}%")
    c2.metric("⚪ HOLD", counts_s.get(0, 0),
              delta=f"{100*counts_s.get(0,0)/total_s:.1f}%")
    c3.metric("🔴 SELL", counts_s.get(-1, 0),
              delta=f"{100*counts_s.get(-1,0)/total_s:.1f}%")
    c4.metric("Señales activas (BUY+SELL)",
              counts_s.get(1, 0) + counts_s.get(-1, 0))

    # Precio + señales
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.65, 0.35],
        subplot_titles=["Precio + Señales ML (triángulos) y TDA Reglas (puntos)",
                        "Probabilidades por clase (ML)"],
        vertical_spacing=0.05,
    )

    fig.add_trace(go.Scatter(
        x=df_s["date"], y=df_s["close"],
        line=dict(color="white", width=1), name="Precio",
    ), row=1, col=1)

    # Señales ML (triángulos grandes)
    for sig, color, name, sym in [
        ( 1, "#22c55e", "ML BUY",  "triangle-up"),
        (-1, "#ef4444", "ML SELL", "triangle-down"),
    ]:
        mask = df_s["ml_signal"] == sig
        fig.add_trace(go.Scatter(
            x=df_s["date"][mask], y=df_s["close"][mask],
            mode="markers",
            marker=dict(color=color, size=9, symbol=sym),
            name=name,
        ), row=1, col=1)

    # Señales TDA reglas (círculos pequeños, semitransparentes)
    tda_buy  = df_s["topo_z"] < p["buy_z"]
    tda_sell = df_s["topo_z"] > p["sell_z"]
    for mask, color, name in [
        (tda_buy,  "#86efac", "TDA BUY (reglas)"),
        (tda_sell, "#fca5a5", "TDA SELL (reglas)"),
    ]:
        fig.add_trace(go.Scatter(
            x=df_s["date"][mask], y=df_s["close"][mask],
            mode="markers",
            marker=dict(color=color, size=5, opacity=0.5),
            name=name,
        ), row=1, col=1)

    # Probabilidades apiladas
    for col_n, color, name in [
        ("prob_buy",  "#22c55e", "P(BUY)"),
        ("prob_hold", "#6b7280", "P(HOLD)"),
        ("prob_sell", "#ef4444", "P(SELL)"),
    ]:
        fig.add_trace(go.Scatter(
            x=df_s["date"], y=df_s[col_n],
            mode="lines", line=dict(color=color, width=1.2),
            name=name, stackgroup="proba",
        ), row=2, col=1)

    fig.update_layout(
        height=620, template="plotly_dark",
        title=f"Señales de Trading – {p['symbol']}  |  Modelo: {R['best']}",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, font=dict(size=10)),
    )
    st.plotly_chart(fig, use_container_width=True)

    # topo_z + umbrales
    st.subheader("Señal Topológica Z-score")
    fig_z = go.Figure()
    fig_z.add_trace(go.Scatter(
        x=df_s["date"], y=df_s["topo_z"],
        line=dict(color="#e879f9", width=1), name="topo_z",
        fill="tozeroy", fillcolor="rgba(232,121,249,0.1)",
    ))
    fig_z.add_hline(y=p["buy_z"],  line_dash="dash", line_color="#22c55e",
                    annotation_text=f"Umbral compra ({p['buy_z']})")
    fig_z.add_hline(y=p["sell_z"], line_dash="dash", line_color="#ef4444",
                    annotation_text=f"Umbral venta ({p['sell_z']})")
    fig_z.update_layout(height=280, template="plotly_dark",
                         yaxis_title="topo_z", showlegend=False)
    st.plotly_chart(fig_z, use_container_width=True)


# ─────────────────────────────────────────────────────────────
# TAB 5 – BACKTESTING
# ─────────────────────────────────────────────────────────────

with tab5:
    bt = R["bt"]

    # Tarjetas resumen
    st.subheader("Resumen de rendimiento")
    col1, col2, col3 = st.columns(3)

    for col, label, stats, is_bah in [
        (col1, f"ML  ({R['best']})", bt["s_ml"],  False),
        (col2, "TDA Reglas",         bt["s_tda"], False),
        (col3, "Buy & Hold",         None,        True),
    ]:
        with col:
            st.markdown(f"**{label}**")
            if not is_bah and stats:
                m1, m2 = st.columns(2)
                delta_vs_bah = f"{stats['return_pct'] - bt['bah_ret']:.1f}% vs B&H"
                m1.metric("Retorno",   f"{stats['return_pct']:.1f}%", delta=delta_vs_bah)
                m2.metric("Max DD",    f"{stats['max_dd']:.1f}%")
                m3, m4 = st.columns(2)
                m3.metric("# Trades",  stats["n_trades"])
                m4.metric("Win Rate",  f"{stats['win_rate']:.1f}%")
            else:
                st.metric("Retorno",   f"{bt['bah_ret']:.1f}%")
                st.caption("Referencia: mantener durante todo el período.")

    st.divider()

    # ── Simulador de inversión ────────────────────────────────
    st.subheader("💰 Simulador de Inversión")
    st.caption("Ingresa el capital que habrías invertido al inicio del período para ver cuánto ganarías o perderías.")

    inv_col, _ = st.columns([1, 2])
    with inv_col:
        capital_sim = st.number_input(
            "Capital inicial ($)",
            min_value=1,
            max_value=10_000_000,
            value=10_000,
            step=500,
            format="%d",
        )

    # Los equity arrays están normalizados a $10,000; escalar al capital ingresado
    scale = capital_sim / 10_000

    sim_rows = []
    for label, eq, stats, is_bah in [
        (f"ML  ({R['best']})", bt["eq_ml"],  bt["s_ml"],  False),
        ("TDA Reglas",         bt["eq_tda"], bt["s_tda"], False),
        ("Buy & Hold",         bt["eq_bah"], None,        True),
    ]:
        final_val   = float(eq[-1])  * scale
        max_val     = float(eq.max()) * scale
        min_val     = float(eq.min()) * scale
        profit      = final_val - capital_sim
        ret_pct     = profit / capital_sim * 100
        max_dd_pct  = stats["max_dd"] if not is_bah else (bt["eq_bah"].min() / bt["eq_bah"][0] - 1) * 100
        max_loss    = capital_sim * max_dd_pct / 100
        sim_rows.append({
            "Estrategia":      label,
            "Capital inicial": f"${capital_sim:,.0f}",
            "Valor final":     f"${final_val:,.0f}",
            "Ganancia / Pérdida ($)": f"{'▲' if profit >= 0 else '▼'} ${abs(profit):,.0f}",
            "Retorno (%)":     f"{ret_pct:+.2f}%",
            "Máx. pérdida ($)":f"${abs(max_loss):,.0f}  ({max_dd_pct:.1f}%)",
            "# Trades":        stats["n_trades"] if not is_bah else "—",
        })

    df_sim = pd.DataFrame(sim_rows)
    st.dataframe(df_sim, use_container_width=True, hide_index=True)

    # Tarjetas visuales por estrategia
    sc1, sc2, sc3 = st.columns(3)
    for col_s, row_s, color in zip(
        [sc1, sc2, sc3],
        sim_rows,
        ["#22c55e", "#818cf8", "#f59e0b"],
    ):
        with col_s:
            profit_val = float(row_s["Valor final"].replace("$","").replace(",","")) - capital_sim
            sign       = "+" if profit_val >= 0 else "-"
            emoji      = "🟢" if profit_val >= 0 else "🔴"
            st.markdown(
                f"""
                <div style="background:#1e2130;border-radius:10px;padding:18px;border-left:4px solid {color}">
                    <div style="font-size:0.85rem;color:#9ca3af">{row_s['Estrategia']}</div>
                    <div style="font-size:1.6rem;font-weight:700;color:{color}">
                        {sign}${abs(profit_val):,.0f}
                    </div>
                    <div style="font-size:0.9rem;color:#d1d5db">
                        {emoji} {row_s['Retorno (%)']} sobre <b>${capital_sim:,}</b>
                    </div>
                    <div style="font-size:0.8rem;color:#6b7280;margin-top:6px">
                        Valor final: {row_s['Valor final']}  |  Max pérdida: {row_s['Máx. pérdida ($)']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.divider()

    # Equity curves + drawdown
    dates_bt = bt["dates"]

    fig_bt = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.68, 0.32],
        subplot_titles=[f"Curva de Equity (capital inicial: ${capital_sim:,})",
                        "Drawdown (%)"],
        vertical_spacing=0.05,
    )

    for eq, name, color in [
        (bt["eq_ml"]  * scale, f"ML ({R['best']})", "#22c55e"),
        (bt["eq_tda"] * scale, "TDA Reglas",        "#818cf8"),
        (bt["eq_bah"] * scale, "Buy & Hold",        "#f59e0b"),
    ]:
        fig_bt.add_trace(go.Scatter(
            x=dates_bt, y=eq, mode="lines",
            line=dict(color=color, width=1.8), name=name,
        ), row=1, col=1)

    fig_bt.add_hline(y=capital_sim, line_dash="dot", line_color="#6b7280",
                     opacity=0.5, row=1, col=1)

    for dd_arr, name, color in [
        (bt["s_ml"]["dd_series"],  f"ML DD",      "#22c55e"),
        (bt["s_tda"]["dd_series"], "TDA Rules DD","#818cf8"),
    ]:
        fig_bt.add_trace(go.Scatter(
            x=dates_bt, y=dd_arr * 100,
            mode="lines", line=dict(color=color, width=1),
            name=name,
        ), row=2, col=1)

    fig_bt.update_yaxes(title_text="Capital ($)",    row=1, col=1)
    fig_bt.update_yaxes(title_text="Drawdown (%)",   row=2, col=1)
    fig_bt.update_layout(
        height=580, template="plotly_dark",
        title="Backtesting – Comparativa de Estrategias",
        legend=dict(orientation="h", yanchor="bottom", y=1.01),
    )
    st.plotly_chart(fig_bt, use_container_width=True)

    # Tablas de operaciones
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        n_ml = len([t for t in bt["trades_ml"] if t["type"] == "BUY"])
        st.subheader(f"Operaciones ML – {n_ml} trades")
        if bt["trades_ml"]:
            df_tm = pd.DataFrame(bt["trades_ml"])
            df_tm["precio"] = df_tm["precio"].round(2)
            st.dataframe(df_tm.tail(25), use_container_width=True, hide_index=True)
        else:
            st.info("Sin operaciones generadas.")

    with col_t2:
        n_tda = len([t for t in bt["trades_tda"] if t["type"] == "BUY"])
        st.subheader(f"Operaciones TDA Reglas – {n_tda} trades")
        if bt["trades_tda"]:
            df_tt = pd.DataFrame(bt["trades_tda"])
            df_tt["precio"] = df_tt["precio"].round(2)
            st.dataframe(df_tt.tail(25), use_container_width=True, hide_index=True)
        else:
            st.info("Sin operaciones generadas.")

    with st.expander("⚠️ Limitaciones del backtesting"):
        st.markdown("""
        - Ejecución al **precio de cierre** (sin slippage ni spread).
        - Comisión del **0.1%** por operación (estimación conservadora).
        - Sin gestión de riesgo avanzada (position sizing, trailing stop).
        - Los resultados históricos **no garantizan rendimientos futuros**.
        - Para backtesting riguroso usar **Freqtrade** con `TDA_Strategy.py`
          o `TDA_ML_Strategy.py` (datos de exchange con timeframe real).
        """)
