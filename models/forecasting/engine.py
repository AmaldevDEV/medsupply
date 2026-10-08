"""
Demand Forecasting Engine with Time-Based Split Validation,
Baseline Models (MA, EMA), ML Models (RandomForest / GradientBoosting),
Multi-Horizon Projections, and Evaluation Metrics (MAE, RMSE, MAPE).
"""

import sys
import pickle
import math
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import BASE_DIR, SAVED_MODELS_DIR, DATABASE_PATH
from database.database import get_connection, query_df

def safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculates Mean Absolute Percentage Error safely avoiding division by zero."""
    mask = y_true > 0
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)

class TimeSeriesFeatureExtractor:
    """Extracts lag, rolling statistical, calendar, and patient load features."""
    
    @staticmethod
    def extract_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        df = df.copy().sort_values("date").reset_index(drop=True)
        df["date_dt"] = pd.to_datetime(df["date"])
        
        # Calendar features
        df["day_of_week"] = df["date_dt"].dt.dayofweek
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
        df["month"] = df["date_dt"].dt.month
        df["day_of_month"] = df["date_dt"].dt.day
        df["trend"] = np.arange(len(df))
        
        # Lag features
        df["lag_1"] = df["daily_consumption"].shift(1)
        df["lag_2"] = df["daily_consumption"].shift(2)
        df["lag_7"] = df["daily_consumption"].shift(7)
        df["lag_14"] = df["daily_consumption"].shift(14)
        
        # Rolling features
        df["rolling_mean_7"] = df["daily_consumption"].shift(1).rolling(7).mean()
        df["rolling_mean_14"] = df["daily_consumption"].shift(1).rolling(14).mean()
        df["rolling_mean_28"] = df["daily_consumption"].shift(1).rolling(28).mean()
        df["rolling_std_7"] = df["daily_consumption"].shift(1).rolling(7).std().fillna(1.0)
        
        # Patient load features
        df["patient_load_lag1"] = df["patient_load"].shift(1)
        df["emergency_load_lag1"] = df["emergency_load"].shift(1)
        df["outbreak_flag"] = df["is_outbreak"]
        
        feature_cols = [
            "day_of_week", "is_weekend", "month", "trend",
            "lag_1", "lag_2", "lag_7", "lag_14",
            "rolling_mean_7", "rolling_mean_14", "rolling_mean_28", "rolling_std_7",
            "patient_load_lag1", "emergency_load_lag1", "outbreak_flag"
        ]
        
        # Drop rows where lag features are NaN (first 28 days)
        df_clean = df.dropna(subset=feature_cols).reset_index(drop=True)
        return df_clean, feature_cols

class DemandForecastingPipeline:
    def __init__(self, model_type: str = "random_forest"):
        self.model_type = model_type
        self.model = None
        self.feature_cols = []
        self.last_train_date = None
        self.metrics = {}

    def train_and_evaluate(
        self,
        series_df: pd.DataFrame,
        hospital_id: str,
        medicine_id: str,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15
    ) -> Dict[str, Any]:
        """
        Executes strict time-based split:
        First 70% Train, Next 15% Validation, Final 15% Test.
        Compares Baseline (Moving Average & EMA) vs ML Model.
        """
        df_feat, feature_cols = TimeSeriesFeatureExtractor.extract_features(series_df)
        self.feature_cols = feature_cols
        
        n = len(df_feat)
        if n < 50:
            raise ValueError(f"Insufficient history ({n} rows) for training")
            
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        train_df = df_feat.iloc[:train_end]
        val_df = df_feat.iloc[train_end:val_end]
        test_df = df_feat.iloc[val_end:]
        
        X_train, y_train = train_df[feature_cols], train_df["daily_consumption"].values
        X_val, y_val = val_df[feature_cols], val_df["daily_consumption"].values
        X_test, y_test = test_df[feature_cols], test_df["daily_consumption"].values
        
        # 1. Baseline Model: 7-day Moving Average on test set
        baseline_preds = test_df["rolling_mean_7"].values
        base_mae = float(mean_absolute_error(y_test, baseline_preds))
        base_rmse = float(np.sqrt(mean_squared_error(y_test, baseline_preds)))
        base_mape = safe_mape(y_test, baseline_preds)
        
        # 2. Fit ML Model (Random Forest)
        if self.model_type == "random_forest":
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        else:
            model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.08,
                random_state=42
            )
            
        # Fit on combined train + validation for final test evaluation
        X_train_val = pd.concat([X_train, X_val])
        y_train_val = np.concatenate([y_train, y_val])
        model.fit(X_train_val, y_train_val)
        
        ml_preds = model.predict(X_test)
        ml_mae = float(mean_absolute_error(y_test, ml_preds))
        ml_rmse = float(np.sqrt(mean_squared_error(y_test, ml_preds)))
        ml_mape = safe_mape(y_test, ml_preds)
        
        improvement_pct = round(((base_mae - ml_mae) / max(0.001, base_mae)) * 100.0, 1)
        
        # Refit on 100% of data for future forecasting
        model.fit(df_feat[feature_cols], df_feat["daily_consumption"].values)
        self.model = model
        self.last_train_date = pd.to_datetime(series_df["date"].max())
        
        # Residual std for uncertainty interval
        residuals = y_test - ml_preds
        self.residual_std = max(1.0, float(np.std(residuals)))
        
        self.metrics = {
            "hospital_id": hospital_id,
            "medicine_id": medicine_id,
            "baseline_mae": round(base_mae, 2),
            "baseline_rmse": round(base_rmse, 2),
            "baseline_mape": round(base_mape, 2),
            "ml_mae": round(ml_mae, 2),
            "ml_rmse": round(ml_rmse, 2),
            "ml_mape": round(ml_mape, 2),
            "improvement_pct": improvement_pct,
            "test_samples": len(y_test),
            "evaluated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return self.metrics

    def generate_multistep_forecast(
        self,
        series_df: pd.DataFrame,
        horizon_days: int = 30
    ) -> pd.DataFrame:
        """
        Generates day-by-day forward predictions for next 1, 7, 14, 30 days.
        Uses recursive autoregressive feature updates.
        """
        if self.model is None:
            raise RuntimeError("Model must be trained before generating forecast")
            
        df_history = series_df.copy().sort_values("date").reset_index(drop=True)
        recent_consumption = list(df_history["daily_consumption"].values[-35:])
        recent_patient_load = float(df_history["patient_load"].iloc[-7:].mean())
        recent_emergency_load = float(df_history["emergency_load"].iloc[-7:].mean())
        
        last_date = pd.to_datetime(df_history["date"].iloc[-1])
        base_trend = len(df_history)
        
        forecast_records = []
        for h in range(1, horizon_days + 1):
            target_date = last_date + timedelta(days=h)
            dow = target_date.dayofweek
            is_wknd = 1 if dow in [5, 6] else 0
            month = target_date.month
            
            # Autoregressive lag lookups
            lag_1 = recent_consumption[-1]
            lag_2 = recent_consumption[-2]
            lag_7 = recent_consumption[-7] if len(recent_consumption) >= 7 else lag_1
            lag_14 = recent_consumption[-14] if len(recent_consumption) >= 14 else lag_7
            
            r7 = float(np.mean(recent_consumption[-7:]))
            r14 = float(np.mean(recent_consumption[-14:]))
            r28 = float(np.mean(recent_consumption[-28:]))
            r_std7 = max(1.0, float(np.std(recent_consumption[-7:])))
            
            feat_row = pd.DataFrame([{
                "day_of_week": dow,
                "is_weekend": is_wknd,
                "month": month,
                "trend": base_trend + h,
                "lag_1": lag_1,
                "lag_2": lag_2,
                "lag_7": lag_7,
                "lag_14": lag_14,
                "rolling_mean_7": r7,
                "rolling_mean_14": r14,
                "rolling_mean_28": r28,
                "rolling_std_7": r_std7,
                "patient_load_lag1": recent_patient_load,
                "emergency_load_lag1": recent_emergency_load,
                "outbreak_flag": 0
            }])[self.feature_cols]
            
            pred = float(self.model.predict(feat_row)[0])
            pred = max(1.0, pred)
            
            # Expanding uncertainty bounds with horizon sqrt(h)
            uncert = 1.28 * self.residual_std * np.sqrt(1.0 + h * 0.05) # 80% CI
            lower_b = max(0.0, round(pred - uncert, 1))
            upper_b = round(pred + uncert, 1)
            
            forecast_records.append({
                "target_date": target_date.strftime("%Y-%m-%d"),
                "day_ahead": h,
                "predicted_demand": round(pred, 1),
                "lower_bound": lower_b,
                "upper_bound": upper_b,
                "model_type": self.model_type
            })
            
            # Append prediction to history for next autoregressive step
            recent_consumption.append(pred)
            
        return pd.DataFrame(forecast_records)

def train_and_save_all_forecasts():
    """
    Trains ML and Baseline models for all 10 hospitals x 20 medicines (200 pairs),
    evaluates holdout test metrics, saves metrics and 30-day forecasts into SQLite.
    """
    print("=" * 60)
    print("TRAINING DEMAND FORECASTING MODELS & COMPUTING METRICS")
    print("=" * 60)
    
    demand_df = query_df("SELECT * FROM demand_history ORDER BY hospital_id, medicine_id, date")
    hospitals = demand_df["hospital_id"].unique()
    medicines = demand_df["medicine_id"].unique()
    
    all_metrics = []
    all_forecasts = []
    
    total_pairs = len(hospitals) * len(medicines)
    count = 0
    
    for h_id in hospitals:
        for m_id in medicines:
            count += 1
            subset = demand_df[(demand_df["hospital_id"] == h_id) & (demand_df["medicine_id"] == m_id)].reset_index(drop=True)
            
            pipeline = DemandForecastingPipeline(model_type="random_forest")
            metrics = pipeline.train_and_evaluate(subset, h_id, m_id)
            all_metrics.append(metrics)
            
            forecast_df = pipeline.generate_multistep_forecast(subset, horizon_days=30)
            forecast_df["hospital_id"] = h_id
            forecast_df["medicine_id"] = m_id
            forecast_df["forecast_date"] = subset["date"].max()
            all_forecasts.append(forecast_df)
            
            if count % 25 == 0 or count == total_pairs:
                print(f"  Processed [{count}/{total_pairs}] series... Latest MAE: ML={metrics['ml_mae']} vs Base={metrics['baseline_mae']} (Improvement: {metrics['improvement_pct']}%)")
                
    # Save metrics to DB
    metrics_df = pd.DataFrame(all_metrics)
    forecasts_df = pd.concat(all_forecasts, ignore_index=True)
    
    conn = get_connection(DATABASE_PATH)
    metrics_df.to_sql("model_metrics", conn, if_exists="replace", index=False)
    forecasts_df.to_sql("forecasts", conn, if_exists="replace", index=False)
    conn.commit()
    conn.close()
    
    # Save summarized models
    with open(SAVED_MODELS_DIR / "latest_metrics.pkl", "wb") as f:
        pickle.dump(metrics_df, f)
        
    avg_base_mae = metrics_df["baseline_mae"].mean()
    avg_ml_mae = metrics_df["ml_mae"].mean()
    avg_imp = ((avg_base_mae - avg_ml_mae) / avg_base_mae) * 100.0
    
    print("\nFORECASTING PIPELINE EXECUTION COMPLETE!")
    print(f"  - Total Evaluated Series: {len(metrics_df)}")
    print(f"  - Average Baseline MAE: {avg_base_mae:.2f} units")
    print(f"  - Average ML Model MAE: {avg_ml_mae:.2f} units")
    print(f"  - Overall Accuracy Improvement: {avg_imp:.1f}%")
    print("=" * 60)
    
    return metrics_df, forecasts_df

if __name__ == "__main__":
    train_and_save_all_forecasts()
