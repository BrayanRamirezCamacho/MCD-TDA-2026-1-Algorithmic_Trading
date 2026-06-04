
# ============================================================
# TDA LABEL GENERATION + ML TRAINING PIPELINE
# Kevin Galvan Lara – Proyecto TDA Trading
# ============================================================
#
# Este script:
#   1. Descarga datos BTC y computa features TDA
#   2. Genera etiquetas: BUY=1 / HOLD=0 / SELL=-1
#      según el retorno futuro a N velas
#   3. Entrena RandomForest + XGBoost + LogReg
#   4. Evalúa con classification report y matriz de confusión
#   5. Guarda el mejor modelo en user_data/models/tda_model.pkl
# ============================================================

import sys
import subprocess


# ============================================================
# AUTO-INSTALL DEPENDENCIES
# ============================================================

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package, "-q"])


required = [
    "numpy", "pandas", "matplotlib", "seaborn",
    "scikit-learn", "ripser", "persim", "yfinance",
    "xgboost", "joblib"
]

for pkg in required:
    try:
        __import__(pkg.replace("-", "_"))
    except ImportError:
        print(f"Instalando {pkg}...")
        install(pkg)


# ============================================================
# IMPORTS
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from pathlib import Path

from ripser import ripser
from persim import wasserstein

import yfinance as yf

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

try:
    from xgboost import XGBClassifier
    USE_XGB = True
except ImportError:
    USE_XGB = False
    print("XGBoost no disponible, se omite.")


# ============================================================
# PARÁMETROS
# ============================================================

SYMBOL   = "BTC-USD"
START    = "2020-01-01"

WINDOW   = 40      # Ventana TDA
TAU      = 1
STEP     = 5       # Salto entre ventanas (reducir para más datos)
SMOOTH   = 20      # Rolling suavizado

HORIZON  = 10      # Velas hacia adelante para predecir
BUY_THR  = 0.02    # Retorno futuro > 2% → BUY
SELL_THR = -0.02   # Retorno futuro < -2% → SELL

MODEL_PATH = Path(__file__).resolve().parents[1] / "freqtrade" / "user_data" / "models"
MODEL_PATH.mkdir(parents=True, exist_ok=True)


# ============================================================
# DESCARGAR DATOS
# ============================================================

print("Descargando datos BTC...")

df = yf.download(SYMBOL, start=START, progress=False)
df = df.dropna().reset_index()
df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

print(f"  {len(df)} velas descargadas ({df['Date'].min().date()} – {df['Date'].max().date()})")


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["log_ret"]  = np.log(d["Close"] / d["Close"].shift(1))
    d["dlog_vol"] = np.log(d["Volume"] + 1).diff()
    d["d_hl"]     = ((d["High"] - d["Low"]) / d["Close"]).diff()
    d["ret_acc"]  = d["log_ret"].diff()
    return d.dropna().reset_index(drop=True)


df_feat = build_features(df)


# ============================================================
# FUNCIONES TDA
# ============================================================

def takens_embedding(x: np.ndarray, tau: int = 1, dim: int = 2) -> np.ndarray:
    n = len(x)
    m = n - (dim - 1) * tau
    emb = np.zeros((m, dim))
    for i in range(dim):
        emb[:, i] = x[i * tau: i * tau + m]
    return emb


def persistence_entropy(dgm: np.ndarray) -> float:
    if len(dgm) == 0:
        return 0.0
    pers = dgm[:, 1] - dgm[:, 0]
    pers = pers[pers > 0]
    if len(pers) == 0:
        return 0.0
    p = pers / pers.sum()
    return float(-np.sum(p * np.log(p + 1e-12)))


def build_cloud_multivariate(df_window: pd.DataFrame, tau: int = 1) -> np.ndarray:
    ret = df_window["log_ret"].values
    vol = df_window["dlog_vol"].values
    hlr = df_window["d_hl"].values
    acc = df_window["ret_acc"].values

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


# ============================================================
# COMPUTAR FEATURES TDA
# ============================================================

print("Computando diagramas de persistencia...")

diagrams = []
idx_list = []

for i in range(WINDOW, len(df_feat), STEP):
    window = df_feat.iloc[i - WINDOW: i]
    cloud  = build_cloud_multivariate(window, tau=TAU)
    dgms   = ripser(cloud, maxdim=1)["dgms"]
    diagrams.append(dgms)
    idx_list.append(i)

    if len(idx_list) % 50 == 0:
        print(f"  {len(idx_list)} ventanas procesadas...")

print(f"  Total: {len(diagrams)} diagramas")


# ============================================================
# WASSERSTEIN DISTANCES
# ============================================================

print("Computando distancias Wasserstein...")

wass_h1 = [0.0]

for i in range(1, len(diagrams)):
    d = wasserstein(diagrams[i - 1][1], diagrams[i][1])
    wass_h1.append(float(d))

wass_h1 = np.array(wass_h1)


# ============================================================
# ENTROPÍA DE PERSISTENCIA
# ============================================================

ent_h1 = np.array([
    persistence_entropy(dgm[1]) for dgm in diagrams
])


# ============================================================
# CONSTRUIR DATAFRAME TDA
# ============================================================

df_tda = pd.DataFrame({
    "idx":           idx_list,
    "entropy_h1":    ent_h1,
    "delta_entropy": np.concatenate([[0.0], np.diff(ent_h1)]),
    "wass_h1":       wass_h1,
    "delta_wass":    np.concatenate([[0.0], np.diff(wass_h1)]),
})

df_tda["topo_signal"] = df_tda["delta_entropy"] + df_tda["delta_wass"]

df_tda["topo_z"] = (
    (df_tda["topo_signal"] - df_tda["topo_signal"].rolling(50, min_periods=10).mean()) /
    (df_tda["topo_signal"].rolling(50, min_periods=10).std() + 1e-8)
)

df_tda["entropy_smooth"] = df_tda["entropy_h1"].rolling(SMOOTH, min_periods=5).mean()
df_tda["wass_smooth"]    = df_tda["wass_h1"].rolling(SMOOTH, min_periods=5).mean()

# Alinear precio de cierre
df_tda["close"] = df_feat["Close"].iloc[idx_list].values
df_tda["date"]  = df_feat["Date"].iloc[idx_list].values if "Date" in df_feat.columns else None

df_tda = df_tda.dropna().reset_index(drop=True)
print(f"  Dataframe TDA: {len(df_tda)} filas")


# ============================================================
# GENERAR ETIQUETAS BUY / HOLD / SELL
# ============================================================

print("Generando etiquetas...")

df_tda["future_return"] = (
    df_tda["close"]
    .pct_change(HORIZON)
    .shift(-HORIZON)
)

df_tda["label"] = 0  # HOLD

df_tda.loc[df_tda["future_return"] >  BUY_THR,  "label"] =  1   # BUY
df_tda.loc[df_tda["future_return"] < SELL_THR,  "label"] = -1   # SELL

df_tda = df_tda.dropna(subset=["future_return", "label"]).reset_index(drop=True)

counts = df_tda["label"].value_counts().sort_index()
label_names = {-1: "SELL", 0: "HOLD", 1: "BUY"}
print("  Distribución de etiquetas:")
for k, v in counts.items():
    pct = 100 * v / len(df_tda)
    print(f"    {label_names[k]:4s} ({k:+d}): {v:4d}  ({pct:.1f}%)")


# ============================================================
# FEATURES Y TARGET
# ============================================================

FEATURE_COLS = [
    "entropy_h1",
    "delta_entropy",
    "wass_h1",
    "delta_wass",
    "topo_z",
    "entropy_smooth",
    "wass_smooth",
]

X = df_tda[FEATURE_COLS].values
y = df_tda["label"].values

# TimeSeriesSplit para no filtrar datos futuros
tscv    = TimeSeriesSplit(n_splits=5)
n_train = int(len(X) * 0.80)
X_train, X_test = X[:n_train], X[n_train:]
y_train, y_test = y[:n_train], y[n_train:]

print(f"  Train: {len(X_train)}  |  Test: {len(X_test)}")


# ============================================================
# MODELOS
# ============================================================

models = {
    "RandomForest": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )),
    ]),
    "GradientBoosting": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.05,
            random_state=42,
        )),
    ]),
    "LogisticRegression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=500,
            class_weight="balanced",
            multi_class="multinomial",
            random_state=42,
        )),
    ]),
}

if USE_XGB:
    models["XGBoost"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            use_label_encoder=False,
            eval_metric="mlogloss",
            random_state=42,
        )),
    ])


# ============================================================
# ENTRENAMIENTO Y EVALUACIÓN
# ============================================================

print("\n" + "=" * 60)
print("RESULTADOS EN TEST SET")
print("=" * 60)

results     = {}
best_name   = None
best_score  = -1

for name, model in models.items():
    print(f"\n--- {name} ---")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    report = classification_report(
        y_test, y_pred,
        target_names=["SELL", "HOLD", "BUY"],
        labels=[-1, 0, 1],
        output_dict=True,
        zero_division=0,
    )
    print(classification_report(
        y_test, y_pred,
        target_names=["SELL", "HOLD", "BUY"],
        labels=[-1, 0, 1],
        zero_division=0,
    ))

    macro_f1 = report["macro avg"]["f1-score"]
    results[name] = {"model": model, "macro_f1": macro_f1, "y_pred": y_pred}

    if macro_f1 > best_score:
        best_score = macro_f1
        best_name  = name


# ============================================================
# GRÁFICAS DE EVALUACIÓN
# ============================================================

fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4))
if len(models) == 1:
    axes = [axes]

for ax, (name, res) in zip(axes, results.items()):
    cm = confusion_matrix(y_test, res["y_pred"], labels=[-1, 0, 1])
    disp = ConfusionMatrixDisplay(cm, display_labels=["SELL", "HOLD", "BUY"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"{name}\nMacro-F1={res['macro_f1']:.3f}")

plt.tight_layout()
plt.suptitle("Matrices de Confusión – TDA Classifiers", y=1.02, fontsize=12)
plt.savefig("confusion_matrices_tda.png", dpi=120, bbox_inches="tight")
plt.show()
print("Guardada: confusion_matrices_tda.png")


# ============================================================
# VISUALIZAR SEÑALES VS PRECIO
# ============================================================

label_colors = {1: "green", 0: "gray", -1: "red"}
label_labels = {1: "BUY", 0: "HOLD", -1: "SELL"}

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

ax1.plot(df_tda.index, df_tda["close"], color="black", lw=0.8, label="BTC")
best_preds = results[best_name]["y_pred"]
preds_full = results[best_name]["model"].predict(X)

for lbl, color in label_colors.items():
    mask = preds_full == lbl
    ax1.scatter(
        df_tda.index[mask],
        df_tda["close"].values[mask],
        color=color, s=8, alpha=0.5, label=label_labels[lbl]
    )
ax1.set_title(f"Señales ML ({best_name}) sobre precio BTC")
ax1.legend(loc="upper left", fontsize=8)

ax2.plot(df_tda["topo_z"].values, color="purple", lw=0.8, label="topo_z")
ax2.axhline(y=-1.0, color="green", ls="--", lw=0.7, label="Umbral compra")
ax2.axhline(y= 1.5, color="red",   ls="--", lw=0.7, label="Umbral venta")
ax2.set_title("Señal Topológica Z-score")
ax2.legend(loc="upper right", fontsize=8)

plt.tight_layout()
plt.savefig("tda_ml_signals.png", dpi=120, bbox_inches="tight")
plt.show()
print("Guardada: tda_ml_signals.png")


# ============================================================
# GUARDAR MODELO
# ============================================================

print(f"\nMejor modelo: {best_name}  (Macro-F1={best_score:.4f})")

save_path = MODEL_PATH / "tda_model.pkl"

joblib.dump({
    "model":        results[best_name]["model"],
    "feature_cols": FEATURE_COLS,
    "params": {
        "window":    WINDOW,
        "tau":       TAU,
        "step":      STEP,
        "horizon":   HORIZON,
        "buy_thr":   BUY_THR,
        "sell_thr":  SELL_THR,
    },
    "model_name": best_name,
}, save_path)

print(f"Modelo guardado en: {save_path}")
print("\nPipeline completado.")
