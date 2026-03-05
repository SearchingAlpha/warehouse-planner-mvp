"""LightGBM-based time series forecaster with quantile regression.

Uses lag features, rolling statistics, and calendar features to produce
probabilistic forecasts (P10, P50, P90) via quantile regression.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

_LAG_DAYS = [1, 2, 3, 7, 14, 21, 28]
_ROLLING_WINDOWS = [7, 14, 28]


def _build_features(series: pd.Series, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Build ML features from a time series.

    Features:
        - Lags: value at t-1, t-2, t-3, t-7, t-14, t-21, t-28
        - Rolling mean/std over 7, 14, 28 day windows
        - Calendar: day_of_week, day_of_month, week_of_year, month, is_weekend
    """
    df = pd.DataFrame({"date": dates, "value": series.values})

    # Lag features
    for lag in _LAG_DAYS:
        df[f"lag_{lag}"] = df["value"].shift(lag)

    # Rolling statistics
    for window in _ROLLING_WINDOWS:
        df[f"rolling_mean_{window}"] = df["value"].shift(1).rolling(window).mean()
        df[f"rolling_std_{window}"] = df["value"].shift(1).rolling(window).std()

    # Calendar features
    dt = pd.to_datetime(df["date"])
    df["day_of_week"] = dt.dt.dayofweek
    df["day_of_month"] = dt.dt.day
    df["week_of_year"] = dt.dt.isocalendar().week.astype(int)
    df["month"] = dt.dt.month
    df["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)

    return df


def _get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return the list of feature column names (everything except date and value)."""
    return [c for c in df.columns if c not in ("date", "value")]


# ---------------------------------------------------------------------------
# Forecaster class
# ---------------------------------------------------------------------------


class LightGBMForecaster:
    """Probabilistic time series forecaster using LightGBM quantile regression.

    Trains three models (P10, P50, P90) on lag/rolling/calendar features
    derived from the historical series. For multi-step forecasting, uses
    recursive prediction (each step feeds back as a new lag).
    """

    def __init__(
        self,
        n_estimators: int = 500,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        num_leaves: int = 31,
        min_child_samples: int = 20,
        verbose: int = -1,
    ):
        self._base_params = {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "num_leaves": num_leaves,
            "min_child_samples": min_child_samples,
            "verbose": verbose,
        }
        self._models: dict[str, lgb.LGBMRegressor] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forecast(
        self,
        history: pd.Series | np.ndarray,
        dates: pd.DatetimeIndex,
        horizon: int = 28,
    ) -> dict[str, np.ndarray]:
        """Train on history and produce a probabilistic forecast.

        Args:
            history: Historical daily volume values.
            dates: DatetimeIndex aligned with history.
            horizon: Number of days to forecast.

        Returns:
            dict with keys ``'p10'``, ``'p50'``, ``'p90'`` — arrays of
            length *horizon*.
        """
        history = pd.Series(np.asarray(history, dtype=float))
        dates = pd.DatetimeIndex(dates)

        # Build feature matrix from history
        feat_df = _build_features(history, dates)
        feature_cols = _get_feature_cols(feat_df)

        # Drop rows with NaN from the lag/rolling window warm-up
        train_df = feat_df.dropna(subset=feature_cols).copy()
        X_train = train_df[feature_cols]
        y_train = train_df["value"]

        # Train one model per quantile
        for quantile, label in [(0.10, "p10"), (0.50, "p50"), (0.90, "p90")]:
            model = lgb.LGBMRegressor(
                objective="quantile",
                alpha=quantile,
                **self._base_params,
            )
            model.fit(X_train, y_train)
            self._models[label] = model

        # Recursive multi-step forecast
        return self._predict_recursive(feat_df, feature_cols, dates, horizon)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _predict_recursive(
        self,
        feat_df: pd.DataFrame,
        feature_cols: list[str],
        dates: pd.DatetimeIndex,
        horizon: int,
    ) -> dict[str, np.ndarray]:
        """Generate forecasts by feeding each prediction back as a lag."""

        # We'll extend the feature dataframe one row at a time
        extended = feat_df.copy()
        last_date = dates[-1]

        predictions: dict[str, list[float]] = {"p10": [], "p50": [], "p90": []}

        for step in range(horizon):
            next_date = last_date + pd.Timedelta(days=step + 1)

            # Build a single-row feature vector for the next date
            row = self._build_next_row(extended, next_date)
            X = pd.DataFrame([row], columns=feature_cols)

            for label in ("p10", "p50", "p90"):
                pred = float(self._models[label].predict(X)[0])
                predictions[label].append(max(0.0, pred))

            # Append the P50 prediction as the "value" for future lags
            new_row = {c: row.get(c, np.nan) for c in extended.columns}
            new_row["date"] = next_date
            new_row["value"] = predictions["p50"][-1]
            extended = pd.concat(
                [extended, pd.DataFrame([new_row])], ignore_index=True,
            )

        return {k: np.array(v) for k, v in predictions.items()}

    @staticmethod
    def _build_next_row(df: pd.DataFrame, next_date: pd.Timestamp) -> dict:
        """Construct feature values for a single future date."""
        values = df["value"].values
        n = len(values)
        row: dict = {}

        # Lags
        for lag in _LAG_DAYS:
            idx = n - lag
            row[f"lag_{lag}"] = float(values[idx]) if idx >= 0 else np.nan

        # Rolling stats (shifted by 1 — we use values up to t-1)
        for window in _ROLLING_WINDOWS:
            if n >= window:
                window_vals = values[-window:]
                row[f"rolling_mean_{window}"] = float(np.mean(window_vals))
                row[f"rolling_std_{window}"] = float(np.std(window_vals, ddof=1))
            else:
                row[f"rolling_mean_{window}"] = float(np.mean(values)) if n > 0 else 0.0
                row[f"rolling_std_{window}"] = float(np.std(values, ddof=1)) if n > 1 else 0.0

        # Calendar
        dt = pd.Timestamp(next_date)
        row["day_of_week"] = dt.dayofweek
        row["day_of_month"] = dt.day
        row["week_of_year"] = dt.isocalendar().week
        row["month"] = dt.month
        row["is_weekend"] = int(dt.dayofweek >= 5)

        return row
