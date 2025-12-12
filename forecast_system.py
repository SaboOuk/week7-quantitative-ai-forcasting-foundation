# forecast_system.py
# Quantitative AI Statistical Forecasting System - Part 1

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings("ignore")


class VolumeForecaster:
    """
    Quantitative AI forecasting system using statistical mathematics.
    Implements:
      - Pattern analysis (trend, moments, weekly/monthly patterns)
      - Moving average forecast
      - Exponential smoothing
      - Time series decomposition (trend/seasonal/residual)
      - Visualizations saved to results/
    """

    def __init__(self, data_path: str):
        self.data_path = data_path
        self.df = pd.read_csv(data_path)
        self.df["date"] = pd.to_datetime(self.df["date"])
        self.df.set_index("date", inplace=True)

        os.makedirs("results", exist_ok=True)

        print("=" * 60)
        print("📊 STATISTICAL AI FORECASTING SYSTEM (WEEK 7)")
        print("=" * 60)
        print(f"Loaded {len(self.df)} numerical data points")
        print(f"Data dimensions: {self.df.shape}")
        print("=" * 60)

    def analyze_patterns(self) -> dict:
        """Perform comprehensive quantitative analysis and identify patterns."""
        print("\n🔎 QUANTITATIVE PATTERN ANALYSIS...")

        x = np.arange(len(self.df))
        y = self.df["volume"].values

        # linear trend slope (units per day)
        slope, intercept = np.polyfit(x, y, 1)

        stats_dict = {
            "mean": float(np.mean(y)),
            "variance": float(np.var(y)),
            "std_dev": float(np.std(y, ddof=1)),
            "skewness": float(stats.skew(y)),
            "kurtosis": float(stats.kurtosis(y)),
            "cv": float(np.std(y, ddof=1) / np.mean(y)),
            "trend_slope": float(slope),
            "trend_intercept": float(intercept),
        }

        print("Statistical Moments:")
        print(f"  Mean (μ): {stats_dict['mean']:.2f}")
        print(f"  Variance (σ²): {stats_dict['variance']:.2f}")
        print(f"  Std Dev (σ): {stats_dict['std_dev']:.2f}")
        print(f"  Skewness: {stats_dict['skewness']:.3f}")
        print(f"  Kurtosis: {stats_dict['kurtosis']:.3f}")
        print(f"  Coefficient of Variation: {stats_dict['cv']:.3f}")
        print(f"  Linear Trend (slope): {stats_dict['trend_slope']:.3f} units/day")

        # Weekly pattern summary
        print("\n📅 Numerical Weekly Patterns (mean ± std):")
        weekly_stats = self.df.groupby(self.df.index.dayofweek)["volume"].agg(["mean", "std"])
        day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        weekly_summary = {}
        for i, day in enumerate(day_names):
            mean_val = weekly_stats.loc[i, "mean"]
            std_val = weekly_stats.loc[i, "std"]
            weekly_summary[day] = (float(mean_val), float(std_val))
            bar = "█" * int(mean_val / 50)
            print(f"  {day}: {bar} μ={mean_val:.0f}, σ={std_val:.0f}")

        # Monthly pattern
        monthly_avg = self.df.groupby(self.df.index.month)["volume"].mean()
        monthly_summary = {int(m): float(v) for m, v in monthly_avg.items()}

        return {
            "moments": stats_dict,
            "weekly_summary": weekly_summary,
            "monthly_summary": monthly_summary,
        }

    def moving_average_forecast(self, window: int = 7):
        """Simple moving average forecast and MAE on historical fit."""
        print(f"\n📈 MOVING AVERAGE FORECAST (window={window})")

        col = f"MA_{window}"
        self.df[col] = self.df["volume"].rolling(window=window).mean()

        # Compare MA shifted to predict each day (y_hat[t] = MA[t-1])
        actual = self.df["volume"].iloc[window:]
        pred = self.df[col].shift(1).iloc[window:]
        mae = mean_absolute_error(actual, pred)

        next_forecast = float(self.df[col].iloc[-1])
        print(f"Next period forecast: {next_forecast:.0f} units")
        print(f"Historical MAE: {mae:.2f} units")

        return next_forecast, mae

    def exponential_smoothing(self, alpha: float = 0.3):
        """Single exponential smoothing and MAE."""
        print(f"\n📉 EXPONENTIAL SMOOTHING (alpha={alpha})")

        values = self.df["volume"].values
        smoothed = [values[0]]
        for t in range(1, len(values)):
            smoothed.append(alpha * values[t] + (1 - alpha) * smoothed[-1])

        self.df["exp_smooth"] = smoothed

        actual = self.df["volume"].iloc[1:]
        pred = self.df["exp_smooth"].shift(1).iloc[1:]
        mae = mean_absolute_error(actual, pred)

        next_forecast = float(self.df["exp_smooth"].iloc[-1])
        print(f"Next period forecast: {next_forecast:.0f} units")
        print(f"Historical MAE: {mae:.2f} units")

        return next_forecast, mae

    def decompose_time_series(self):
        """Decompose time series into trend/seasonal/residual components."""
        print("\n🧩 DECOMPOSING TIME SERIES...")
        decomposition = seasonal_decompose(self.df["volume"], model="additive", period=365)

        self.df["trend"] = decomposition.trend
        self.df["seasonal"] = decomposition.seasonal
        self.df["residual"] = decomposition.resid

        print("Components extracted:")
        print("  ✓ Trend (long-term direction)")
        print("  ✓ Seasonal (recurring patterns)")
        print("  ✓ Residual (random variation)")

        return decomposition

    def create_visualizations(self, window: int = 7):
        """Create and save required visualizations."""
        print("\n🖼️ CREATING VISUALIZATIONS...")

        os.makedirs("results", exist_ok=True)

        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        fig.patch.set_facecolor("white")

        # Plot 1: Historical + MA
        axes[0].plot(self.df.index, self.df["volume"], label="Actual", alpha=0.6)
        ma_col = f"MA_{window}"
        if ma_col in self.df.columns:
            axes[0].plot(self.df.index, self.df[ma_col], label=f"{window}-Day MA", linewidth=2)
        axes[0].set_title("Historical Volume Data")
        axes[0].set_ylabel("Volume")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Plot 2: Average volume by day of week
        weekly_avg = self.df.groupby(self.df.index.dayofweek)["volume"].mean()
        day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        axes[1].bar(day_labels, weekly_avg.values)
        axes[1].set_title("Average Volume by Day of Week")
        axes[1].set_ylabel("Average Volume")
        axes[1].grid(True, alpha=0.3, axis="y")

        for i, v in enumerate(weekly_avg.values):
            axes[1].text(i, v, f"{v:.0f}", ha="center", va="bottom")

        # Plot 3: Average volume by month
        monthly_avg = self.df.groupby(self.df.index.month)["volume"].mean()
        month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        axes[2].plot(month_labels, monthly_avg.values, marker="o", linewidth=2)
        axes[2].set_title("Average Volume by Month")
        axes[2].set_ylabel("Average Volume")
        axes[2].set_xlabel("Month")
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        out_path = os.path.join("results", "volume_analysis.png")
        plt.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)

        print(f"✅ Saved visualization to {out_path}")
        return out_path


def main():
    # Ensure folders exist
    os.makedirs("data", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    data_path = os.path.join("data", "historical_volumes.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            "Missing data/historical_volumes.csv. Run: python data_generator.py first."
        )

    forecaster = VolumeForecaster(data_path)

    patterns = forecaster.analyze_patterns()
    ma_forecast, ma_error = forecaster.moving_average_forecast(window=7)
    exp_forecast, exp_error = forecaster.exponential_smoothing(alpha=0.3)
    forecaster.decompose_time_series()
    plot_path = forecaster.create_visualizations(window=7)

    # Save a short progress log for “proof”
    progress_path = os.path.join("results", "analysis_summary.txt")
    with open(progress_path, "w", encoding="utf-8") as f:
        f.write("WEEK 7 ANALYSIS SUMMARY\n")
        f.write("=" * 60 + "\n")
        f.write(f"Moving Average (7-day) forecast: {ma_forecast:.0f}\n")
        f.write(f"Moving Average MAE: {ma_error:.2f}\n")
        f.write(f"Exp Smoothing forecast (alpha=0.3): {exp_forecast:.0f}\n")
        f.write(f"Exp Smoothing MAE: {exp_error:.2f}\n")
        f.write(f"Visualization: {plot_path}\n")
        f.write("\nMoments:\n")
        for k, v in patterns["moments"].items():
            f.write(f"  {k}: {v}\n")

    print("\n" + "=" * 60)
    print("✅ ANALYSIS COMPLETE!")
    print("=" * 60)
    print(f"Moving Average (7-day): {ma_forecast:.0f} units | MAE={ma_error:.2f}")
    print(f"Exponential Smoothing: {exp_forecast:.0f} units | MAE={exp_error:.2f}")
    print(f"Saved: {plot_path}")
    print(f"Saved: {progress_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())