from pathlib import Path

from src.data.load_data import load_ohlcv_csv
from src.features.build_features import add_features
from src.models.rule_based_strategy import generate_signals
from src.models.backtest import run_backtest
from src.visualization.plot_results import plot_equity_curve


def main() -> None:
    base_dir = Path(__file__).resolve().parents[1]

    input_file = base_dir / "data" / "raw" / "btcusdt_1h.csv"
    trades_file = base_dir / "data" / "processed" / "trades.csv"
    equity_file = base_dir / "data" / "processed" / "equity_curve.csv"
    plot_file = base_dir / "reports" / "figures" / "equity_curve.png"

    df = load_ohlcv_csv(input_file)
    df = add_features(df)
    df = generate_signals(df)

    trades_df, equity_df = run_backtest(
        df,
        initial_capital=1000.0,
        stop_loss=0.02,
        take_profit=0.04,
    )

    trades_file.parent.mkdir(parents=True, exist_ok=True)
    plot_file.parent.mkdir(parents=True, exist_ok=True)

    trades_df.to_csv(trades_file, index=False)
    equity_df.to_csv(equity_file, index=False)
    plot_equity_curve(equity_df, plot_file)

    print("MVP ejecutado correctamente.")
    print(f"Trades guardados en: {trades_file}")
    print(f"Equity curve guardada en: {equity_file}")
    print(f"Figura guardada en: {plot_file}")

    if not trades_df.empty:
        total_return = (equity_df['equity'].iloc[-1] / equity_df['equity'].iloc[0]) - 1
        print(f"Retorno total: {total_return:.2%}")
        print(f"Número de trades: {len(trades_df)}")
    else:
        print("No hubo trades con las reglas actuales.")


if __name__ == "__main__":
    main()
