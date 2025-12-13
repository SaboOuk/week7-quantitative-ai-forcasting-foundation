# ml_forecast.py
# Quantitative AI - Machine Learning Forecasting Module (Week 8)

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# XGBoost is optional
try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    HAS_XGB = False


class MLForecaster:
    """
    Quantitative ML forecaster for time-series volume forecasting.

    - Engineers numerical features (lags, rolling stats, calendar encodings)
    - Trains multiple models (Linear Regression, Random Forest, optional XGBoost)
    - Produces daily forecasts and an ensemble forecast
    """

    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()

        if "date" in self.data.columns:
            self.data["date"] = pd.to_datetime(self.data["date"])
            self.data = self.data.set_index("date")

        if not isinstance(self.data.index, pd.DatetimeIndex):
            raise ValueError("Data index must be a DatetimeIndex.")

        if "volume" not in self.data.columns:
            raise ValueError("Data must contain a 'volume' column.")

        self.models = {}
        self.metrics = {}
        self.feature_cols = None

        print("\033[91m" + "=" * 70 + "\033[0m")
        print("\033[97m" + "🧠 QUANTITATIVE ML FORECASTER (Week 8)" + "\033[0m")
        print("\033[94m" + "=" * 70 + "\033[0m")

    def engineer_quantitative_features(self, window: int = 30) -> pd.DataFrame:
        print(f"\n\033[94m🛠 ENGINEERING FEATURES (window={window})...\033[0m")
        df = self.data.copy()

        # Lag features
        for i in range(1, window + 1):
            df[f"lag_{i}"] = df["volume"].shift(i)

        # Rolling features
        df["rolling_mean_7"] = df["volume"].rolling(7).mean()
        df["rolling_mean_30"] = df["volume"].rolling(30).mean()
        df["rolling_std_7"] = df["volume"].rolling(7).std()
        df["rolling_std_30"] = df["volume"].rolling(30).std()
        df["rolling_min_7"] = df["volume"].rolling(7).min()
        df["rolling_max_7"] = df["volume"].rolling(7).max()

        # Transformations
        df["log_volume"] = np.log1p(df["volume"])
        df["sqrt_volume"] = np.sqrt(df["volume"].clip(lower=0))
        df["volume_squared"] = df["volume"] ** 2

        # Calendar encoding
        df["day_of_week"] = df.index.dayofweek
        df["day_of_month"] = df.index.day
        df["month"] = df.index.month
        df["quarter"] = df.index.quarter
        df["year"] = df.index.year
        df["day_sin"] = np.sin(2 * np.pi * df.index.dayofyear / 365.0)
        df["day_cos"] = np.cos(2 * np.pi * df.index.dayofyear / 365.0)

        # Binary indicators
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
        df["is_month_end"] = (df["day_of_month"] >= 25).astype(int)

        # Trend index
        df["time_index"] = np.arange(len(df))

        df = df.dropna()

        self.feature_cols = [c for c in df.columns if c != "volume"]

        print(f"✓ Engineered {len(self.feature_cols)} features")
        print(f"✓ Sample size: {len(df)} rows")
        return df

    def train_models(self, test_size: int = 90, feature_window: int = 30):
        print("\n\033[94m🏋 TRAINING ML MODELS...\033[0m")
        df = self.engineer_quantitative_features(window=feature_window)

        X = df[self.feature_cols]
        y = df["volume"]

        if len(df) <= test_size + 10:
            raise ValueError("Not enough data for train/test split. Reduce test_size or add more data.")

        train_size = len(df) - test_size
        X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
        y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

        # 1) Linear Regression
        print("\n📈 Training Linear Regression...")
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        pred_lr = lr.predict(X_test)
        self.models["linear"] = lr
        self.metrics["linear"] = self._calc_metrics(y_test, pred_lr)

        # 2) Random Forest
        print("\n🌲 Training Random Forest...")
        rf = RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)
        pred_rf = rf.predict(X_test)
        self.models["rf"] = rf
        self.metrics["rf"] = self._calc_metrics(y_test, pred_rf)

        # 3) XGBoost (optional)
        if HAS_XGB:
            print("\n🚀 Training XGBoost...")
            xg = xgb.XGBRegressor(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.08,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=42
            )
            xg.fit(X_train, y_train)
            pred_xg = xg.predict(X_test)
            self.models["xgb"] = xg
            self.metrics["xgb"] = self._calc_metrics(y_test, pred_xg)
        else:
            print("\n⚠ XGBoost not installed. Skipping XGBoost model.")
            print("   (Optional install: pip install xgboost)")

        # Top features from RF
        print("\n\033[92m🔍 TOP 5 IMPORTANT FEATURES (Random Forest)\033[0m")
        importances = pd.Series(self.models["rf"].feature_importances_, index=self.feature_cols).sort_values(ascending=False)
        for feat, val in importances.head(5).items():
            bar = "█" * int(val * 50)
            print(f"{feat:22s} {bar} {val:.4f}")

        return self.models

    def forecast_future(self, periods: int = 13 * 30, feature_window: int = 30) -> pd.DataFrame:
        if not self.models:
            raise RuntimeError("No trained models. Run train_models() first.")

        print(f"\n\033[95m📅 GENERATING {periods}-DAY FORECAST (~13 months)...\033[0m")

        # Ensure features computed so feature_cols exist
        _ = self.engineer_quantitative_features(window=feature_window)
        last_date = self.data.index.max()

        # Rolling history for lag features
        history = self.data["volume"].copy()

        model_names = list(self.models.keys())
        forecasts = {m: [] for m in model_names}
        dates = []

        # time_index should continue beyond observed length
        time_index_base = len(self.data)

        for i in range(1, periods + 1):
            future_date = last_date + pd.Timedelta(days=i)
            dates.append(future_date)

            feat = {}

            # Lags
            for lag in range(1, feature_window + 1):
                feat[f"lag_{lag}"] = float(history.iloc[-lag])

            # Rolling stats
            last_7 = history.iloc[-7:]
            last_30 = history.iloc[-30:]
            feat["rolling_mean_7"] = float(last_7.mean())
            feat["rolling_mean_30"] = float(last_30.mean())
            feat["rolling_std_7"] = float(last_7.std(ddof=0))
            feat["rolling_std_30"] = float(last_30.std(ddof=0))
            feat["rolling_min_7"] = float(last_7.min())
            feat["rolling_max_7"] = float(last_7.max())

            # Transformations based on last known volume
            last_vol = float(history.iloc[-1])
            feat["log_volume"] = float(np.log1p(max(last_vol, 0.0)))
            feat["sqrt_volume"] = float(np.sqrt(max(last_vol, 0.0)))
            feat["volume_squared"] = float(last_vol ** 2)

            # Calendar
            feat["day_of_week"] = future_date.dayofweek
            feat["day_of_month"] = future_date.day
            feat["month"] = future_date.month
            feat["quarter"] = ((future_date.month - 1) // 3) + 1
            feat["year"] = future_date.year
            doy = future_date.timetuple().tm_yday
            feat["day_sin"] = float(np.sin(2 * np.pi * doy / 365.0))
            feat["day_cos"] = float(np.cos(2 * np.pi * doy / 365.0))

            feat["is_weekend"] = 1 if feat["day_of_week"] >= 5 else 0
            feat["is_month_end"] = 1 if feat["day_of_month"] >= 25 else 0

            feat["time_index"] = time_index_base + i

            X_future = pd.DataFrame([feat])[self.feature_cols]

            preds = []
            for name, model in self.models.items():
                p = float(model.predict(X_future)[0])
                p = max(p, 0.0)
                forecasts[name].append(p)
                preds.append(p)

            # feedback uses mean prediction
            ensemble_pred = float(np.mean(preds))
            history = pd.concat([history, pd.Series([ensemble_pred], index=[future_date])])

        out = pd.DataFrame({"date": dates})
        for name, vals in forecasts.items():
            out[name] = vals

        out["ensemble"] = out[[c for c in forecasts.keys()]].mean(axis=1)
        return out

    @staticmethod
    def _calc_metrics(y_true, y_pred):
        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        r2 = float(r2_score(y_true, y_pred))
        print(f"   MAE:  {mae:.2f}")
        print(f"   RMSE: {rmse:.2f}")
        print(f"   R²:   {r2:.4f}")
        return {"mae": mae, "rmse": rmse, "r2": r2}