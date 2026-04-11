# 📈 Predicción de Precios de Bitcoin y Trading Algorítmico
### 🚀 Proyecto de Ciencia de Datos | Maestría en Ciencia de Datos

Proyecto colaborativo desarrollado en el contexto de la Maestría en Ciencia de Datos de la Universidad de Sonora, enfocado en la construcción, evaluación y comparación de modelos para la predicción de precios de Bitcoin utilizando técnicas de análisis de series de tiempo, Machine Learning, Deep Learning y Análisis Topológico de Datos.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Status](https://img.shields.io/badge/status-in%20progress-yellow)
![License](https://img.shields.io/badge/license-MIT-green)
![Made with](https://img.shields.io/badge/Made%20with-ML%20%26%20TDA-purple)

---

## 🧠 Overview

Este proyecto desarrolla un sistema completo de **predicción de precios de Bitcoin** combinando:

- 📊 Series de tiempo
- 🤖 Machine Learning
- 🧬 Deep Learning
- 🧩 Análisis Topológico de Datos (TDA)

El objetivo no es solo predecir precios, sino **construir una base para estrategias de trading algorítmico con TDA**.

---

##🎯 Objetivo del Proyecto

Desarrollar un pipeline reproducible que permita:

-Analizar el comportamiento histórico del precio de Bitcoin
-Identificar patrones y características relevantes en series de tiempo
-Construir modelos predictivos robustos
-Evaluar el desempeño de distintos enfoques
-Explorar estrategias aplicables al trading algorítmico


##🧠 Enfoques Metodológicos

Este proyecto integra múltiples perspectivas analíticas:

###📊 Análisis de Series de Tiempo
-Modelos clásicos: ARIMA, SARIMA, ARIMAX
-Descomposición de tendencias y estacionalidad
-Análisis de autocorrelación (ACF, PACF)


###🤖 Machine Learning
-Regresión (Linear, Ridge, Lasso)
-Árboles de decisión y Random Forest
-Gradient Boosting (XGBoost, LightGBM)


###🧬 Deep Learning
-Redes neuronales recurrentes (RNN, LSTM, GRU)
-Redes convolucionales aplicadas a series de tiempo
-Modelos híbridos


###📡 Ingeniería de Características
-Indicadores técnicos (RSI, MACD, Bollinger Bands)
-Variables exógenas (volumen, noticias, sentimiento)
-Transformaciones temporales


###🧩 Análisis Topológico de Datos (TDA)
-Embedding de series de tiempo (Takens Embedding)
-Diagramas de persistencia (Homología persistente)
-Extracción de características topológicas (Top-K persistence)
-Integración con modelos de Machine Learning y Deep Learning


##🧪 MVP: Sistema Inicial de Trading Algorítmico

Como parte del proyecto, se ha implementado un Producto Mínimo Viable (MVP) enfocado en validar ideas básicas de trading algorítmico.


###🔍 Objetivo del MVP

Construir un pipeline funcional que permita:

1. Cargar datos históricos de Bitcoin (OHLCV)
2. Generar features básicas (EMA, RSI, volatilidad)
3. Aplicar una estrategia rule-based simple
4. Ejecutar un backtest básico
5. Visualizar resultados (equity curve)


###⚙️ Estrategia Implementada (Baseline)

Estrategia inicial basada en:

#### Entrada (Compra):

	EMA20 > EMA50 (tendencia alcista)
	Precio por encima de EMA20
	RSI < 70

#### Salida (Venta):

	EMA20 < EMA50
	RSI > 75
	Stop-loss (-2%)
	Take-profit (+4%)


###📂 Flujo del MVP
	Datos → Features → Señales → Backtest → Resultados → Visualización


#### Ejecución del MVP
	make install
	make run

O directamente:

	python -m src.run_mvp


###📊 Outputs generados

	data/processed/trades.csv → historial de operaciones
	data/processed/equity_curve.csv → evolución del capital
	reports/figures/equity_curve.png → gráfica del rendimiento


###🧠 Propósito dentro del proyecto

El MVP sirve como:

-Base experimental para estrategias más avanzadas
-Punto de integración con modelos de ML/DL
-Plataforma inicial para futura integración con Freqtrade
-Validación de pipelines de datos y features


## 🏗️ Estructura del Proyecto

	├── data/
	│   ├── raw/        # Datos originales
	│   ├── interim/    # Datos intermedios
	│   └── processed/  # Datos listos para modelado
	├── notebooks/      # Análisis exploratorio y experimentos
	├── src/
	│   ├── data/           # Scripts de carga y procesamiento
	│   ├── features/       # Generación de variables
	│   ├── models/         # Estrategias, entrenamiento y backtesting
	│   └── visualization/  # Gráficas y reportes
	├── reports/        # Resultados y documentación
	├── requirements.txt
	├── README.md
	└── .gitignore


## ⚙️ Instalación y Configuración

### Clonar el repositorio:
	git clone https://github.com/usuario/repositorio.git
	cd repositorio

### Crear entorno virtual:
	python -m venv venv
	source venv/bin/activate  # Linux/Mac
	venv\Scripts\activate     # Windows

### Instalar dependencias:
	pip install -r requirements.txt


## 🚀 Uso del Proyecto

1. Preparación de datos
	python src/data/make_dataset.py
2. Generación de features
	python src/features/build_features.py
3. Entrenamiento de modelos
	python src/models/train_model.py
4. Evaluación

### Preparación de datos:

	python src/data/make_dataset.py


### Generación de features:

	python src/features/build_features.py


### Entrenamiento de modelos:

	python src/models/train_model.py


### Evaluación:

	python src/models/evaluate_model.py


## 📊 Métricas de Evaluación

Se utilizan diversas métricas para evaluar el desempeño de los modelos:

-MAE (Mean Absolute Error)
-RMSE (Root Mean Squared Error)
-MAPE (Mean Absolute Percentage Error)
-Backtesting financiero (cuando aplica)


## 🤝 Colaboración

Este es un proyecto colaborativo. Para contribuir:

### 1. Crear una rama
	git checkout -b feature/nueva-funcionalidad

### 2. Realizar cambios
	git add .
	git commit -m "Descripción del cambio"

### 3. Sincronizar con main
	git pull origin main --rebase

### 4. Subir cambios
	git push origin feature/nueva-funcionalidad

### 5. Crear Pull Request
-Ir al repositorio en GitHub

-Clic en "Compare & pull request"

-Verificar:
	base: main
	compare: tu rama

-Escribir:
	título claro
	descripción de cambios

-Clic en "Create pull request"


## 📌 Buenas Prácticas
Mantener notebooks limpios y documentados
Escribir código modular en src/
Documentar funciones y clases
Usar control de versiones adecuadamente
Evitar subir datos sensibles o muy pesados


## 🔮 Líneas Futuras
Integración con API en tiempo real (Binance) ✅
Implementación de bots de trading (ej. Freqtrade) ✅
Integración TDA + Machine Learning / Deep Learning
Modelos basados en Transformers 
Análisis de sentimiento (NLP + noticias)
Optimización de portafolios


## 👥 Equipo

Proyecto desarrollado por estudiantes de la Maestría en Ciencia de Datos (en orden alfabético):

-Brayan Ramírez
-Isaac Espinoza
-Isaul Tirado 
-Kevin Galván
-Kevin García
-Luis Noriega


## 📄 Licencia

Este proyecto se distribuye bajo la licencia especificada en el archivo LICENSE.


## ⚠️ Disclaimer

Este proyecto es únicamente con fines educativos y de investigación.
No constituye asesoría financiera.
No nos hacemos responsables si usted pierde dinero utilizando nuestros métodos.
