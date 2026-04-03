📈 Predicción de Precios de Bitcoin con Ciencia de Datos

Proyecto colaborativo desarrollado en el contexto de la Maestría en Ciencia de Datos de la Universidad de Sonora, enfocado en la construcción, evaluación y comparación de modelos para la predicción de precios de Bitcoin utilizando técnicas de análisis de series de tiempo, Machine Learning, Deep Learning y Análisis Topológico de Datos.


🎯 Objetivo del Proyecto

Desarrollar un pipeline reproducible que permita:

Analizar el comportamiento histórico del precio de Bitcoin
Identificar patrones y características relevantes en series de tiempo
Construir modelos predictivos robustos
Evaluar el desempeño de distintos enfoques
Explorar estrategias aplicables al trading algorítmico


🧠 Enfoques Metodológicos

Este proyecto integra múltiples perspectivas analíticas:

📊 Análisis de Series de Tiempo
Modelos clásicos: ARIMA, SARIMA, ARIMAX
Descomposición de tendencias y estacionalidad
Análisis de autocorrelación (ACF, PACF)


🤖 Machine Learning
Regresión (Linear, Ridge, Lasso)
Árboles de decisión y Random Forest
Gradient Boosting (XGBoost, LightGBM)


🧬 Deep Learning
Redes neuronales recurrentes (RNN, LSTM, GRU)
Redes convolucionales aplicadas a series de tiempo
Modelos híbridos


📡 Feature Engineering
Indicadores técnicos (RSI, MACD, Bollinger Bands)
Variables exógenas (volumen, noticias, sentimiento)
Transformaciones temporales

🧩 Análisis Topológico de Datos (TDA)
Embedding de series de tiempo (Takens Embedding)
Diagramas de persistencia (Homología persistente)
Extracción de características topológicas (Top-K persistence)
Integración con modelos de Machine Learning y Deep Learning


🏗️ Estructura del Proyecto

├── data/
│   ├── raw/            # Datos originales
│   ├── interim/        # Datos intermedios
│   └── processed/      # Datos listos para modelado
│
├── notebooks/          # Análisis exploratorio y experimentos
│
├── src/
│   ├── data/           # Scripts de carga y procesamiento
│   ├── features/       # Generación de variables
│   ├── models/         # Entrenamiento y evaluación
│   └── visualization/  # Gráficas y reportes
│
├── reports/            # Resultados y documentación
├── requirements.txt    # Dependencias
├── README.md
└── .gitignore


⚙️ Instalación y Configuración

Clonar el repositorio:
git clone https://github.com/usuario/repositorio.git
cd repositorio

Crear entorno virtual:
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

Instalar dependencias:
pip install -r requirements.txt


🚀 Uso del Proyecto
1. Preparación de datos
python src/data/make_dataset.py
2. Generación de features
python src/features/build_features.py
3. Entrenamiento de modelos
python src/models/train_model.py
4. Evaluación
python src/models/evaluate_model.py


📊 Métricas de Evaluación

Se utilizan diversas métricas para evaluar el desempeño de los modelos:

MAE (Mean Absolute Error)
RMSE (Root Mean Squared Error)
MAPE (Mean Absolute Percentage Error)
Backtesting financiero (cuando aplica)


🤝 Colaboración

Este es un proyecto colaborativo. Para contribuir:

Crear una rama:
git checkout -b feature/nueva-funcionalidad

Realizar cambios y commit:
git commit -m "Descripción del cambio"

Hacer push:
git push origin feature/nueva-funcionalidad

Crear un Pull Request
Ir al repositorio en GitHub
Verás un botón "Compare & pull request" → hacer clic
Verificar que:
base: main (o develop)
compare: tu rama (feature/...)
Escribir:
Título claro
Descripción de los cambios realizados
(Opcional) Referencia a issues
Hacer clic en "Create pull request"

📌 Buenas Prácticas
Mantener notebooks limpios y documentados
Escribir código modular en src/
Documentar funciones y clases
Usar control de versiones adecuadamente
Evitar subir datos sensibles o muy pesados


🔮 Líneas Futuras
Integración con APIs en tiempo real
Implementación de bots de trading (ej. Freqtrade)
Modelos basados en Transformers
Análisis de sentimiento (NLP + noticias)
Optimización de portafolios


👥 Equipo

Proyecto desarrollado por estudiantes de la Maestría en Ciencia de Datos.
Brayan Ramírez
Isaul Tirado
Kevin Galván
Kevin García
Luis Noriega


📄 Licencia

Este proyecto se distribuye bajo la licencia especificada en el archivo LICENSE.


⚠️ Disclaimer

Este proyecto es únicamente con fines educativos y de investigación.
No constituye asesoría financiera.
No nos hacemos responsables si usted pierde dinero utilizando nuestros métodos.
