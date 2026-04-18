# Estrategia aleatoria.

## Objetivo
El objetivo de esta estrategia es implementar un Baseline un poco más realista que la estrategia `Aleatorio`. Intenta simular a un trader que actúa por FOMO (Fear Of Missing Out), es decir, entra al mercado cuando el precio ya lleva subiendo hace rato y es paciente para no vender a la primera bajada. Se usan mecanismos básicos para proteger las ganancias.

## Funcionamiento

La estrategia combina tres elementos para recrear el sentimiento de FOMO: un indicador de momentum (RSI), una medida de tendencia reciente, y un componente aleatorio para simular la indecisión.

### **Señal de compra**:
```python
dataframe.loc[
    (dataframe["retorno_reciente"] > 0.005)  # el precio subió 0.5% en 3 velas
    & (dataframe["rsi"] < 70) # aún no está sobrecomprado
    & (ruido < 0.3), # solo actúa en el 30% de señales (la indecisión)
    "enter_long",
] = 1
```
Se agrega una señal de compra cuando el precio ya subió. No se anticipa ni hace ningún tipo de predicción, solo "reacciona". El ruido del 30% intenta simular el factor humano de duda. Es decir, que no siempre tendra la oportunidad de participar en la vela o no sabe si es una buen decisión.

### Señal de venta** :
Este método no tiene una señal de venta explicita. La forma en que se maneja es:

* **Stoploss del -4%**: Si el precio cae un 4% desde la compra, vende sin pensarlo. Esto por "miedo" a que pueda seguir cayendo pero aguantando un poco la perdida.

* **Trailing stop**: Una vez que ya gano un 3%, persigue el precio hacia arriba. Si el precio cae un 2% del máximo alcanzado, vende para asegurar así una ganancia.

* **ROI por tiempo**: Si ninguno de los dos se activó, vende por tiempo (ver tabla de abajo)


## Parámetros

| Parámetro                        | Valor | Descripción                                        |
|----------------------------------|-------|----------------------------------------------------|
| timeframe                        | 5m    | Velas de 5 minutos                                 |
| stoploss                         | -4%   | Vende si la pérdida llega a 4%                     |
| trailing_stop_positive           | 2%    | Activa trailing una vez ganado el 2%               |
| trailing_stop_positive_offset    | 3%    | El trailing protege ganancias a partir del 3%      |
| trailing_only_offset_is_reached  | True  | Antes del 3%, solo el stoploss plano protege       |
| ROI 0min                         | 6%    | Vende si gana 6% en cualquier momento              |
| ROI 120min                       | 4%    | Después de 2h, vende con 4%                        |
| ROI 240min                       | 2%    | Después de 4h, vende con 2%                        |
| ROI 480min                       | 1%    | Después de 8h, vende con 1%                        |
| ROI 720min                       | 0%    | Después de 12h, vende con lo que sea               |
| retorno_reciente umbral          | 0.5%  | Compra si el precio subió 0.5% en las últimas 3 velas |
| RSI máximo                       | 70    | No compra si ya está sobrecomprado                 |
| Probabilidad de entrada          | 30%   | Solo actúa en el 30% de las señales válidas        |


## Resultados del Backtesting

Periodo evaluado: 13/10/2025 — 01/01/2026 (80 días)
Pares: BTC/USDT, ETH/USDT, SOL/USDT

Una vez ejecutado un backtesting con datos desde 13/10/2025 a 01/01/2026 (80 días), se obtuvieron los siguientes resultados:

| Métrica       | Valor |
|--------------|-------|
| Win Rate     | 26.2%     |
| Total Profit | -854.94 USDT (-85.49%)     |
| Max Drawdown | 855.02 USDT (85.50%)     |
| Total Trades | 7,442     |


| Métrica            | Valor                  |
|--------------------|------------------------|
| Win Rate           | 59.0%                  |
| Total Profit %     | -6.51%                 |
| Total Profit USDT  | -65.14 USDT            |
| Max Drawdown       | 68.90 USDT (6.88%)     |
| Total Trades       | 261                    |
| Market Change      | -29.48%                |
| Saldo inicial      | 1,000 USDT             |
| Saldo final        | 934.86 USDT            |

| Razón de salida     | Trades | Win% | Profit promedio |
|---------------------|--------|------|-----------------|
| ROI                 | 178    | 86%  | +1.19%          |
| Trailing stop loss  | 1      | 100% | +1.16%          |
| Stop loss           | 82     | 0%   | -4.19%          |


## Conclusión
Como vemos, es una mejor estrategia que `Aleatorio`, sin embargo, aún hay margen para mejoras. La estrategia perdió un `6.51%` en un periodo donde el mercado cayó un 29.48%, es decir, que perdio muy significativamente que el mercado. En este caso, si se hubiera comprado y sostendio por 80 dias, habria terminado perdiendo más o menos 5 veces más.

Un **problema** principal de la estrategia son las razones de salida. Cuando el ROI o el trailing stop can la operación, la estrategia ganó un 86% de los casos con un promedio del `+1.19%`. El problema lo encontramos en el stoploss, con unicamente 82 trades (poco menos de la mitad de las salidas por ROI), se cerraron con un promedio del `-4.19%` cada uno, esto iba provocando que se perdieran las ganancias que ya se habian conseguido. 

En esta estrategia enconramos un winrate engañoso del `59%`, pues más de la mitad de los trades fueron ganadores, pero el desbalance entre el `+1.19%` de promedio de ganancias con el `-4.19%` de promdeio de perdidas, provocó que el balance final sea negativo.

* Saldo inicial: 1,000 USDT
* Saldo final: 934.86 USDT
