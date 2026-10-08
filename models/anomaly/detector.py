"""
Anomaly & Outbreak Detection Engine
Detects coordinated regional demand surges and sudden patient influxes
using rolling statistical Z-scores and multi-facility spatial correlation.
Follows clinical decision support guidelines: uses explainable phrasing
like "Possible regional demand anomaly detected."
"""

from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from database.database import query_df

class RegionalAnomalyDetector:
    def __init__(self, z_threshold: float = 2.0, lookback_days: int = 30):
        self.z_threshold = z_threshold
        self.lookback_days = lookback_days

    def detect_anomalies(self, recent_window_days: int = 7) -> List[Dict[str, Any]]:
        """
        Scans all hospitals and medicines for demand surges.
        Correlates across facilities to identify possible regional anomalies.
        """
        demand_df = query_df("SELECT * FROM demand_history ORDER BY date ASC")
        if demand_df.empty:
            return []

        hospitals_df = query_df("SELECT hospital_id, hospital_name FROM hospitals")
        hosp_names = dict(zip(hospitals_df["hospital_id"], hospitals_df["hospital_name"]))
        
        meds_df = query_df("SELECT medicine_id, medicine_name, category FROM medicines")
        med_names = dict(zip(meds_df["medicine_id"], meds_df["medicine_name"]))
        med_cats = dict(zip(meds_df["medicine_id"], meds_df["category"]))

        max_date = pd.to_datetime(demand_df["date"].max())
        cutoff_date = (max_date - timedelta(days=self.lookback_days)).strftime("%Y-%m-%d")
        recent_date = (max_date - timedelta(days=recent_window_days)).strftime("%Y-%m-%d")

        df_window = demand_df[demand_df["date"] >= cutoff_date].copy()
        
        detected_points = []
        for (h_id, m_id), grp in df_window.groupby(["hospital_id", "medicine_id"]):
            grp = grp.sort_values("date")
            baseline_vals = grp[grp["date"] < recent_date]["daily_consumption"].values
            recent_vals = grp[grp["date"] >= recent_date]["daily_consumption"].values
            
            if len(baseline_vals) < 7 or len(recent_vals) < 3:
                continue
                
            base_mean = float(np.mean(baseline_vals))
            base_std = max(1.0, float(np.std(baseline_vals)))
            recent_mean = float(np.mean(recent_vals))
            
            z_score = (recent_mean - base_mean) / base_std
            percent_increase = ((recent_mean - base_mean) / max(1.0, base_mean)) * 100.0
            
            if z_score >= self.z_threshold:
                detected_points.append({
                    "hospital_id": h_id,
                    "hospital_name": hosp_names.get(h_id, h_id),
                    "medicine_id": m_id,
                    "medicine_name": med_names.get(m_id, m_id),
                    "category": med_cats.get(m_id, "Unknown"),
                    "z_score": round(z_score, 2),
                    "base_mean_daily": round(base_mean, 1),
                    "recent_mean_daily": round(recent_mean, 1),
                    "percent_increase": round(percent_increase, 1)
                })

        # Group by medicine to detect regional coordination
        anomalies_summary = []
        if detected_points:
            pt_df = pd.DataFrame(detected_points)
            for m_id, grp in pt_df.groupby("medicine_id"):
                aff_hosps = grp["hospital_name"].tolist()
                med_name = grp["medicine_name"].iloc[0]
                avg_z = float(grp["z_score"].mean())
                avg_inc = float(grp["percent_increase"].mean())
                
                is_regional = len(aff_hosps) >= 2
                severity = "HIGH" if avg_z >= 3.0 or len(aff_hosps) >= 3 else ("MODERATE" if avg_z >= 2.0 else "LOW")
                
                if is_regional:
                    explanation = (
                        f"Possible regional demand anomaly detected: Coordinated demand surge for "
                        f"{med_name} across {', '.join(aff_hosps)}. Average consumption is "
                        f"{avg_inc:.1f}% above 30-day baseline (mean Z-score = +{avg_z:.2f})."
                    )
                else:
                    explanation = (
                        f"Possible localized demand anomaly detected: {aff_hosps[0]} exhibits "
                        f"elevated demand for {med_name} (+{avg_inc:.1f}% vs baseline, Z-score = +{avg_z:.2f})."
                    )
                    
                anomalies_summary.append({
                    "medicine_id": m_id,
                    "medicine_name": med_name,
                    "affected_hospitals": aff_hosps,
                    "num_hospitals": len(aff_hosps),
                    "is_regional": is_regional,
                    "avg_z_score": round(avg_z, 2),
                    "avg_percent_increase": round(avg_inc, 1),
                    "severity": severity,
                    "explanation": explanation
                })

        return anomalies_summary
