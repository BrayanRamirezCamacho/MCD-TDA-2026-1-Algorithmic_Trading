# TDA Crypto Trading Dashboard

## Archivo principal

`dashboard/app.py`

## Para ejecutarlo

**Opción 1 — doble clic en Windows**

```bat
trading_tda\dashboard\run.bat
```

**Opción 2 — terminal**

```bash
cd trading_tda
.venv\Scripts\python.exe -m streamlit run dashboard\app.py
```

Abre automáticamente `http://localhost:8501`

---

## Estructura del dashboard

### Panel lateral

Todos los parámetros son ajustables antes de correr el pipeline:

- Símbolo y fechas
- Parámetros TDA (ventana, tau, paso)
- Umbrales de señal por reglas (Z-score)
- Parámetros ML (horizonte, umbral de retorno)
- Botón **▶ Ejecutar Pipeline**

### 5 pestañas

| Pestaña | Contenido |
|---------|-----------|
| 📈 **Datos** | Candlestick + volumen, features derivados (`log_ret`, `dlog_vol`, `d_hl`) |
| 🔵 **Análisis TDA** | Precio + zonas de señal, entropía H₁, Wasserstein, 3 diagramas de persistencia con interpretación |
| 🤖 **Modelo ML** | Distribución de etiquetas, comparación de modelos (F1), matriz de confusión, importancia de features, reporte completo + botón 💾 Guardar modelo para Freqtrade |
| 🎯 **Señales** | Precio con triángulos BUY/SELL (ML vs Reglas TDA), probabilidades apiladas, `topo_z` con umbrales |
| 📊 **Backtesting** | Curva de equity (ML vs TDA Reglas vs Buy & Hold), drawdown, métricas de rendimiento, tabla de operaciones |
