from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def plot_equity_curve(equity_df: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)

    plt.figure(figsize=(12, 6))
    plt.plot(equity_df["timestamp"], equity_df["equity"])
    plt.title("Equity Curve")
    plt.xlabel("Tiempo")
    plt.ylabel("Capital")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
