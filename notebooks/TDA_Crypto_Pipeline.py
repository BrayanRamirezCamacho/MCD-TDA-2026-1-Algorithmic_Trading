
# ============================================================
# TDA CRYPTO PIPELINE – VERSION DINAMICA (READY TO RUN)
# Kevin Galvan Lara – Biosciences PhD
# ============================================================

import sys
import subprocess


# ============================================================
# AUTO-INSTALL DEPENDENCIES
# ============================================================

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


required = [
    "numpy",
    "pandas",
    "matplotlib",
    "scikit-learn",
    "ripser",
    "persim",
    "yfinance"
]

for pkg in required:
    try:
        __import__(pkg)
    except:
        install(pkg)


# ============================================================
# IMPORTS
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ripser import ripser
from persim import wasserstein
import yfinance as yf


# ============================================================
# PARAMETERS
# ============================================================

SYMBOL = "BTC-USD"

WINDOW = 40
TAU = 1

STEP = 5


# ============================================================
# DOWNLOAD DATA
# ============================================================

print("Downloading BTC data...")

df = yf.download(
    SYMBOL,
    start="2018-01-01",
    progress=False
)

df = df.dropna().reset_index()


# ============================================================
# FEATURE ENGINEERING (DYNAMIC VERSION)
# ============================================================

def build_features(df):

    d = df.copy()

    d['log_ret'] = np.log(
        d['Close'] / d['Close'].shift(1)
    )

    d['dlog_vol'] = np.log(
        d['Volume'] + 1
    ).diff()

    d['d_hl_range'] = (
        (d['High'] - d['Low']) /
        d['Close']
    ).diff()

    d['ret_acc'] = d['log_ret'].diff()

    d = d.dropna().reset_index(drop=True)

    return d


df_feat = build_features(df)


# ============================================================
# TAKENS EMBEDDING
# ============================================================

def takens_embedding(x, tau=1, dim=2):

    n = len(x)

    m = n - (dim - 1) * tau

    emb = np.zeros((m, dim))

    for i in range(dim):
        emb[:, i] = x[i*tau:i*tau+m]

    return emb


# ============================================================
# MULTIVARIATE CLOUD
# ============================================================

def build_cloud_multivariate(df_window):

    ret = df_window['log_ret'].values
    vol = df_window['dlog_vol'].values
    hlr = df_window['d_hl_range'].values
    acc = df_window['ret_acc'].values

    if len(ret) < 3:
        return np.zeros((1,5))

    tk = takens_embedding(
        ret,
        tau=TAU,
        dim=2
    )

    m = len(tk)

    cloud = np.column_stack([
        tk,
        vol[TAU:TAU+m],
        hlr[TAU:TAU+m],
        acc[TAU:TAU+m]
    ])

    mu = cloud.mean(axis=0)
    std = cloud.std(axis=0) + 1e-8

    return (cloud - mu)/std


# ============================================================
# COMPUTE PERSISTENCE DIAGRAMS
# ============================================================

print("Computing persistence diagrams...")

diagrams = []

for i in range(
        WINDOW,
        len(df_feat),
        STEP
):

    window = df_feat.iloc[
        i-WINDOW:i
    ]

    cloud = build_cloud_multivariate(
        window
    )

    dgms = ripser(
        cloud,
        maxdim=1
    )['dgms']

    diagrams.append(dgms)


# ============================================================
# WASSERSTEIN DISTANCES
# ============================================================

print("Computing Wasserstein distances...")

wass_h1 = []

for i in range(1, len(diagrams)):

    d = wasserstein(
        diagrams[i-1][1],
        diagrams[i][1]
    )

    wass_h1.append(d)


wass_h1 = np.array(wass_h1)

delta_wass = np.diff(
    wass_h1,
    prepend=wass_h1[0]
)


# ============================================================
# PERSISTENCE ENTROPY
# ============================================================

def persistence_entropy(dgm):

    pers = dgm[:,1] - dgm[:,0]

    pers = pers[pers > 0]

    if len(pers) == 0:
        return 0

    p = pers / pers.sum()

    return -np.sum(
        p * np.log(p)
    )


print("Computing persistence entropy...")

ent_h1 = []

for dgm in diagrams:

    ent_h1.append(
        persistence_entropy(
            dgm[1]
        )
    )


ent_h1 = np.array(ent_h1)

delta_ent = np.diff(
    ent_h1,
    prepend=ent_h1[0]
)


# ============================================================
# BUILD DATAFRAME
# ============================================================

df_tda = pd.DataFrame({

    "entropy_h1": ent_h1[1:],

    "delta_entropy": delta_ent[1:],

    "wass_h1": wass_h1,

    "delta_wass": delta_wass

})


# ============================================================
# TOPOLOGICAL SIGNAL
# ============================================================

roll = 20

df_tda["delta_wass_smooth"] = (

    df_tda["delta_wass"]

    .rolling(roll)

    .mean()

)

df_tda["delta_entropy_smooth"] = (

    df_tda["delta_entropy"]

    .rolling(roll)

    .mean()

)


df_tda["topo_signal"] = (

    df_tda["delta_wass_smooth"]

    +

    df_tda["delta_entropy_smooth"]

)


df_tda = df_tda.dropna()


# ============================================================
# ALIGN PRICE
# ============================================================

close_idx = df_feat["Close"].iloc[
    WINDOW+STEP:WINDOW+STEP+len(df_tda)
]


# ============================================================
# PLOT SIGNAL
# ============================================================

print("Plotting results...")

fig, ax1 = plt.subplots(
    figsize=(14,5)
)

ax1.plot(
    close_idx.values,
    color="black",
    label="BTC"
)

ax2 = ax1.twinx()

ax2.plot(
    df_tda["topo_signal"].values,
    color="red",
    label="Topo signal"
)

ax1.set_title(
    "Topological signal vs BTC price"
)

ax1.legend(loc="upper left")

ax2.legend(loc="upper right")

plt.show()


# ============================================================
# SAVE OUTPUT
# ============================================================

df_tda.to_csv(
    "tda_dynamic_signal.csv",
    index=False
)

print("Pipeline completed successfully.")