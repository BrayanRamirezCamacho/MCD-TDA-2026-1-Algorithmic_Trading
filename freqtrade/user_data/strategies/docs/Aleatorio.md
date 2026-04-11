# Estrategia aleatoria.

## Objetivo
Esta estrategia no tiene ninguna lógica en el mercado real. Es por esta misma razón que su proposito es servir como una **base**, es decir, que si se diseña una estrategia sus métricas deben superar las de esta para considerarse útil. Una buena estrategia deberá ganarle a una que hace cosas aleatorias.

## Funcionamiento

Las señales de compra y venta se generan con números aleatorios usando semillas fijas para garantizar reproducibilidad.

**Señal de compra** — 10% de probabilidad por vela:
```python
np.random.seed(42)
dataframe['enter_long'] = (
    np.random.random(len(dataframe)) < 0.1
).astype(int)
```
**Señal de venta** — 10% de probabilidad por vela:
```python
np.random.seed(99)
dataframe['exit_long'] = (
    np.random.random(len(dataframe)) < 0.1
).astype(int)
```
## Parámetros
| Parámetro    | Valor  | Descripción                        |
|-------------|--------|------------------------------------|
| timeframe   | 5m     | Velas de 5 minutos                 |
| stoploss    | -5%    | Cierre automático si pierde 5%     |
| ROI 0min    | 3%     | Vende si gana 3% en cualquier momento |
| ROI 30min   | 2%     | Vende si gana 2% después de 30min  |
| ROI 60min   | 1%     | Vende si gana 1% después de 60min  |
| ROI 120min  | 0%     | Vende sin importar ganancia a 2h   |




## Resultados del Backtesting
Una vez ejecutado un backtesting con datos desde 13/10/2025 a 11/04/2026
se obtuvieron lo siguientes resultados:

| Métrica       | Valor |
|--------------|-------|
| Win Rate     | 26.2%     |
| Total Profit | -854.94 USDT (-85.49%)     |
| Max Drawdown | 855.02 USDT (85.50%)     |
| Total Trades | 7,442     |

## Conclusión
Esta estrategia establece un piso mínimo del rendimiento de otras estrategias.
Algo importante a mencionar es que el mercado obtuvo un -46.67% de **Market Change**. Es decir que si hubieramos comprado y no hubieramos hecho nada, hubiermos perdido un 46.67%. Con esta estrategia se perdio el doble.

* Saldo inicial: 1,000 USDT
* Saldo final: 145.061 USDT


