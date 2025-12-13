# ml_forecast.py
# Quantitative AI - Machine Learning Forecasting Module

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Try XGBoost, fallback if not installed
try:
    import xgboost as xgb
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False
    from sklearn.ensemble import GradientBoostingRegressor


class MLForecaster:
    """
    Quantitative Machine Learning implementation for numerical forecasting
    Applies mathematical learning algorithms + numerical feature engineering
    """

    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()
        self.models = {}
        self.test_predictions = {}
        self.metrics = {}
        self.forecast_df = None

        print("\033[91m" + "=" * 60)  # Red
        print("\033[97m" + " 🤖 QUANTITATIVE ML FORECASTER 🤖 " + "\033[0m")  # White
        print("\033[94m" + "=" * 60 + "\033[0m")  # Blue

    def engineer_quantitative_features(self, window: int = 30) -> pd.DataFrame:
        """Create numerical features using mathematical transformations"""
        print("\n\033[95m🔧 ENGINEERING QUANTITATIVE FEATURES...\033[0m")

        df = self.data.copy()

        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                df = df.set_index("date")
            else:
                raise ValueError("Data must have DatetimeIndex or a 'date' column.")

        # Lag features (autoregressive components)
        for i in range(1, window + 1):
            df[f"lag_{i}"] = df["volume"].shift(i)

        # Rolling statistics
        df["rolling_mean_7"] = df["volume"].rolling(7).mean()
        df["rolling_mean_30"] = df["volume"].rolling(30).mean()
        df["rolling_std_7"] = df["volume"].rolling(7).std()
        df["rolling_std_30"] = df["volume"].rolling(30).std()
        df["rolling_min_7"] = df["volume"].rolling(7).min()
        df["rolling_max_7"] = df["volume"].rolling(7).max()

        # Mathematical transformations
        df["log_volume"] = np.log1p(df["volume"])
        df["sqrt_volume"] = np.sqrt(df["volume"])
        df["volume_squared"] = df["volume"] ** 2

        # Numerical time encoding
        df["day_of_week"] = df.index.dayofweek
        df["day_of_month"] = df.index.day
        df["month"] = df.index.month
        df["quarter"] = df.index.quarter
        df["year"] = df.index.year
        df["day_sin"] = np.sin(2 * np.pi * df.index.dayofyear / 365.0)
        df["day_cos"] = np.cos(2 * np.pi * df.index.dayofyear / 365.0)

        # Binary indicators (step functions)
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
        df["is_month_end"] = (df["day_of_month"] > 25).astype(int)

        # Linear time index for trend
        df["time_index"] = np.arange(len(df))

        # Drop NaNs caused by lagging/rolling
        df = df.dropna()

        print(f"✓ Engineered {len(df.columns) - 1} quantitative features")
        print(f"✓ Feature dimensionality: {df.shape[1] - 1}")
        print(f"✓ Sample size: {len(df)} observations")

        return df

    def train_models(self, test_size: int = 90, window: int = 30):
        """Train multiple ML models with time-series split (no shuffle)"""
        print("\n\033[93m🧠 TRAINING ML MODELS...\033[0m")

        df = self.engineer_quantitative_features(window=window)

        feature_cols = [c for c in df.columns if c != "volume"]
        X = df[feature_cols]
        y = df["volume"]

        train_size = len(X) - test_size
        if train_size <= 0:
            raise ValueError("Not enough data for requested test_size.")

        X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
        y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

        # 1) Linear Regression
        print("\n📈 Training Linear Regression...")
        self.models["linear"] = LinearRegression()
        self.models["linear"].fit(X_train, y_train)
        linear_pred = self.models["linear"].predict(X_test)
        self._store_metrics("linear", y_test, linear_pred)

        # 2) Random Forest
        print("\n🌲 Training Random Forest...")
        self.models["rf"] = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.models["rf"].fit(X_train, y_train)
        rf_pred = self.models["rf"].predict(X_test)
        self._store_metrics("rf", y_test, rf_pred)

        # 3) XGBoost (or fallback)
        if _HAS_XGB:
            print("\n⚡ Training XGBoost...")
            self.models["xgb"] = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=42
            )
        else:
            print("\n⚠️ xgboost not installed; using GradientBoostingRegressor fallback...")
            self.models["xgb"] = GradientBoostingRegressor(random_state=42)

        self.models["xgb"].fit(X_train, y_train)
        xgb_pred = self.models["xgb"].predict(X_test)
        self._store_metrics("xgb", y_test, xgb_pred)

        self.test_predictions = {
            "actual": y_test,
            "linear": linear_pred,
            "rf": rf_pred,
            "xgb": xgb_pred
        }

        # Feature importance (Random Forest)
        print("\n\033[92m📌 TOP 5 IMPORTANT FEATURES:\033[0m")
        importances = pd.DataFrame({
            "feature": feature_cols,
            "importance": self.models["rf"].feature_importances_
        }).sort_values("importance", ascending=False).head(5)

        for _, row in importances.iterrows():
            bar = "█" * int(row["importance"] * 100)
            print(f"{row['feature']:<20} {bar} {row['importance']:.3f}")

        return self.models

    def _store_metrics(self, name: str, y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        self.metrics[name] = {"MAE": mae, "RMSE": rmse, "R2": r2}
        print(f"   MAE: {mae:.2f} | RMSE: {rmse:.2f} | R2: {r2:.3f}")

    def forecast_future(self, periods: int = 13 * 30, window: int = 30):
        """
        Generate future forecasts (daily) for ~13 months (13*30 days)
        Creates daily forecast dataframe with model predictions + ensemble + CI
        """
        print("\n\033[95m📅 GENERATING 13-MONTH FORECAST...\033[0m")

        if not self.models:
            raise RuntimeError("Train models first (train_models).")

        df_feat = self.engineer_quantitative_features(window=window)
        feature_cols = [c for c in df_feat.columns if c != "volume"]

        last_features = df_feat[feature_cols].iloc[-1:].copy()
        last_date = df_feat.index[-1]

        forecasts = {"linear": [], "rf": [], "xgb": []}
        future_dates = []

        # Iterative forecasting
        for i in range(1, periods + 1):
            future_date = last_date + pd.Timedelta(days=i)
            future_dates.append(future_date)

            # Update time-based features
            last_features.loc[:, "day_of_week"] = future_date.dayofweek
            last_features.loc[:, "day_of_month"] = future_date.day
            last_features.loc[:, "month"] = future_date.month
            last_features.loc[:, "quarter"] = ((future_date.month - 1) // 3) + 1
            last_features.loc[:, "year"] = future_date.year
            doy = future_date.timetuple().tm_yday
            last_features.loc[:, "day_sin"] = np.sin(2 * np.pi * doy / 365.0)
            last_features.loc[:, "day_cos"] = np.cos(2 * np.pi * doy / 365.0)
            last_features.loc[:, "is_weekend"] = int(future_date.dayofweek >= 5)
            last_features.loc[:, "is_month_end"] = int(future_date.day > 25)
            last_features.loc[:, "time_index"] = last_features["time_index"].values[0] + 1

            # Predict
            for model_name, model in self.models.items():
                pred = float(model.predict(last_features)[0])
                pred = max(pred, 0.0)
                forecasts[model_name].append(pred)

            # Ensemble prediction to update lag_1 (and shift lags)
            ensemble_pred = float(np.mean([
                forecasts["linear"][-1],
                forecasts["rf"][-1],
                forecasts["xgb"][-1]
            ]))

            # Shift lag features down
            for lag in range(window, 1, -1):
                col_a = f"lag_{lag}"
                col_b = f"lag_{lag-1}"
                if col_a in last_features.columns and col_b in last_features.columns:
                    last_features.loc[:, col_a] = last_features[col_b].values

            # Set newest lag_1 to ensemble prediction
            if "lag_1" in last_features.columns:
                last_features.loc[:, "lag_1"] = ensemble_pred

            # Keep rolling features roughly aligned (simple approach)
            if "rolling_mean_7" in last_features.columns:
                last_features.loc[:, "rolling_mean_7"] = ensemble_pred
            if "rolling_mean_30" in last_features.columns:
                last_features.loc[:, "rolling_mean_30"] = ensemble_pred
            if "rolling_std_7" in last_features.columns:
                last_features.loc[:, "rolling_std_7"] = 0.0
            if "rolling_std_30" in last_features.columns:
                last_features.loc[:, "rolling_std_30"] = 0.0
            if "rolling_min_7" in last_features.columns:
                last_features.loc[:, "rolling_min_7"] = ensemble_pred
            if "rolling_max_7" in last_features.columns:
                last_features.loc[:, "rolling_max_7"] = ensemble_pred

            # Transform features tied to volume
            if "log_volume" in last_features.columns:
                last_features.loc[:, "log_volume"] = np.log1p(ensemble_pred)
            if "sqrt_volume" in last_features.columns:
                last_features.loc[:, "sqrt_volume"] = np.sqrt(max(ensemble_pred, 0.0))
            if "volume_squared" in last_features.columns:
                last_features.loc[:, "volume_squared"] = ensemble_pred ** 2

        out = pd.DataFrame({
            "date": future_dates,
            "linear": forecasts["linear"],
            "rf": forecasts["rf"],
            "xgb": forecasts["xgb"],
        })

        # Ensemble + CI (std across models)
        out["weighted_ensemble"] = out[["linear", "rf", "xgb"]].mean(axis=1)
        std_models = out[["linear", "rf", "xgb"]].std(axis=1)
        out["lower_bound"] = out