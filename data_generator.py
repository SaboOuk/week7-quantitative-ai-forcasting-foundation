# data_generator.py
# Quantitative AI: Generate synthetic numerical data with mathematical patterns

import os
from datetime import datetime

import numpy as np
import pandas as pd


def generate_volume_data(seed: int = 42) -> pd.DataFrame:
    """
    Generate 3 years of daily numerical volume data using mathematical functions.
    Includes deterministic + stochastic components:
      1) Linear trend
      2) Yearly sinusoidal seasonality
      3) Weekly pattern (weekday boost)
      4) Monthly spike (end-of-month-ish)
      5) Gaussian noise
    """
    start_date = datetime(2021, 1, 1)
    end_date = datetime(2023, 12, 31)
    dates = pd.date_range(start=start_date, end=end_date, freq="D")

    n_days = len(dates)

    # 1) Linear trend
    trend = np.linspace(1000, 1500, n_days)  # growth over 3 years

    # 2) Yearly sinusoidal seasonality
    seasonal_year = 200 * np.sin(2 * np.pi * np.arange(n_days) / 365.0)

    # 3) Weekly pattern: weekdays higher than weekends
    weekly = np.array([150 if d.weekday() < 5 else 50 for d in dates])

    # 4) Monthly spike: after day 25
    monthly = np.array([100 if d.day > 25 else 0 for d in dates])

    # 5) Noise
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 50, n_days)

    volume = trend + seasonal_year + weekly + monthly + noise
    volume = np.maximum(volume, 0)  # non-negative
    volume = np.round(volume).astype(int)

    df = pd.DataFrame(
        {
            "date": dates,
            "volume": volume,
            "day_of_week": [d.day_name() for d in dates],
            "month": [d.month for d in dates],
            "year": [d.year for d in dates],
        }
    )

    print("=" * 60)
    print("📈 QUANTITATIVE AI DATA GENERATOR")
    print("=" * 60)
    print(f"Generated {len(df)} numerical data points")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"Mean (μ): {df['volume'].mean():.2f}")
    print(f"Std Dev (σ): {df['volume'].std():.2f}")
    print(f"Coefficient of Variation: {(df['volume'].std()/df['volume'].mean()):.3f}")
    print("=" * 60)

    return df


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    data = generate_volume_data()
    out_path = os.path.join("data", "historical_volumes.csv")
    data.to_csv(out_path, index=False)
    print(f"✅ Data saved to {out_path}")