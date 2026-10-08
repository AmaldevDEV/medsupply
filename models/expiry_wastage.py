"""
Batch-Level Expiry and Wastage Risk Detection Engine
Calculates days until expiry, forecast demand before expiry, expected usage,
and potential unused stock at batch level to prevent clinical waste.
"""

from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATABASE_PATH, EXPIRY_WARNING_DAYS
from database.database import query_df

class ExpiryWastageDetector:
    def __init__(self, warning_horizon_days: int = 60):
        self.warning_horizon_days = warning_horizon_days

    def evaluate_batch_expiry_risk(self, batch_id: str) -> Optional[Dict[str, Any]]:
        """Evaluates a single inventory batch for expiry and wastage risk."""
        batch_q = """
            SELECT i.inventory_id, i.hospital_id, i.medicine_id, i.batch_id,
                   i.quantity, i.received_date, i.expiry_date, i.days_until_expiry,
                   h.hospital_name, h.safety_stock_policy_days,
                   m.medicine_name, m.category, m.criticality,
                   COALESCE(m.unit_cost_inr, i.unit_cost_inr, 500.0) as unit_cost_inr
            FROM inventory i
            JOIN hospitals h ON i.hospital_id = h.hospital_id
            JOIN medicines m ON i.medicine_id = m.medicine_id
            WHERE i.batch_id = ?
        """
        df = query_df(batch_q, (batch_id,))
        if df.empty:
            return None
        row = df.iloc[0]
        return self._evaluate_row(row)

    def scan_all_batches(self) -> pd.DataFrame:
        """
        Scans all batches in the network, identifies imminent expiries,
        calculates expected usage before expiry, and flags potential waste.
        """
        q = """
            SELECT i.inventory_id, i.hospital_id, i.medicine_id, i.batch_id,
                   i.quantity, i.received_date, i.expiry_date, i.days_until_expiry,
                   h.hospital_name, h.safety_stock_policy_days,
                   m.medicine_name, m.category, m.criticality,
                   COALESCE(m.unit_cost_inr, i.unit_cost_inr, 500.0) as unit_cost_inr
            FROM inventory i
            JOIN hospitals h ON i.hospital_id = h.hospital_id
            JOIN medicines m ON i.medicine_id = m.medicine_id
            ORDER BY i.days_until_expiry ASC
        """
        batches_df = query_df(q)
        if batches_df.empty:
            return pd.DataFrame()

        results = []
        for _, row in batches_df.iterrows():
            eval_res = self._evaluate_row(row)
            if eval_res:
                results.append(eval_res)

        res_df = pd.DataFrame(results)
        if not res_df.empty:
            # Sort by at_risk_waste_units and financial risk
            res_df = res_df.sort_values(["at_risk_waste_units", "potential_financial_loss_inr"], ascending=[False, False]).reset_index(drop=True)
        return res_df

    def _evaluate_row(self, row: pd.Series) -> Dict[str, Any]:
        h_id = row["hospital_id"]
        h_name = row["hospital_name"]
        m_id = row["medicine_id"]
        m_name = row["medicine_name"]
        b_id = row["batch_id"]
        qty = int(row["quantity"])
        days_to_exp = int(row["days_until_expiry"])
        unit_cost_inr = float(row["unit_cost_inr"].iloc[0]) if isinstance(row["unit_cost_inr"], pd.Series) else float(row["unit_cost_inr"])
        safety_days = int(row["safety_stock_policy_days"])

        # Fetch daily forecast demand for this hospital-medicine
        fc_q = """
            SELECT day_ahead, predicted_demand
            FROM forecasts
            WHERE hospital_id = ? AND medicine_id = ?
            ORDER BY day_ahead ASC
        """
        fc_df = query_df(fc_q, (h_id, m_id))

        if not fc_df.empty:
            demands = fc_df["predicted_demand"].values
            avg_daily = float(np.mean(demands))
            # Demand before expiry (sum up to days_to_exp)
            if days_to_exp <= len(demands):
                forecast_usage = float(np.sum(demands[:max(1, days_to_exp)]))
            else:
                forecast_usage = float(np.sum(demands)) + (days_to_exp - len(demands)) * avg_daily
        else:
            # Fallback to historical daily consumption average
            hist_q = "SELECT AVG(daily_consumption) as avg_d FROM demand_history WHERE hospital_id = ? AND medicine_id = ?"
            h_df = query_df(hist_q, (h_id, m_id))
            avg_daily = float(h_df["avg_d"].iloc[0]) if not h_df.empty and pd.notnull(h_df["avg_d"].iloc[0]) else 20.0
            forecast_usage = avg_daily * max(1, days_to_exp)

        expected_usage_before_expiry = int(round(forecast_usage))
        safety_stock_protect = int(round(safety_days * avg_daily))

        # Potential Surplus / Waste Risk:
        # Potential Surplus = Current Batch Stock - Expected Usage Before Expiry - Required Safety Stock
        potential_surplus = qty - expected_usage_before_expiry - safety_stock_protect
        at_risk_waste_units = max(0, potential_surplus)
        financial_loss_inr = round(at_risk_waste_units * unit_cost_inr, 2)
        financial_loss_usd = round(financial_loss_inr / 83.0, 2)

        # Risk tier classification
        if days_to_exp <= 15 and at_risk_waste_units > 0:
            risk_tier = "CRITICAL WASTE RISK"
        elif days_to_exp <= 30 and at_risk_waste_units > 0:
            risk_tier = "HIGH WASTE RISK"
        elif days_to_exp <= 60 and at_risk_waste_units > 0:
            risk_tier = "MODERATE SURPLUS RISK"
        elif days_to_exp <= 30:
            risk_tier = "EXPIRING SOON (LOCAL USAGE SUFFICIENT)"
        else:
            risk_tier = "STABLE"

        # Generate explainable narrative
        if at_risk_waste_units > 0:
            explanation = (
                f"{h_name} has {qty:,} units of {m_name} (Batch {b_id}) expiring in {days_to_exp} days. "
                f"Forecast usage before expiry is {expected_usage_before_expiry:,} units. "
                f"After protecting safety stock ({safety_stock_protect:,} units), approximately "
                f"{at_risk_waste_units:,} units are at risk of remaining unused (₹{financial_loss_inr:,.2f} financial loss)."
            )
        else:
            explanation = (
                f"{h_name} batch {b_id} ({qty:,} units) expires in {days_to_exp} days. "
                f"Expected consumption ({expected_usage_before_expiry:,} units) is sufficient to utilize this stock before expiry."
            )

        return {
            "inventory_id": row["inventory_id"],
            "batch_id": b_id,
            "hospital_id": h_id,
            "hospital_name": h_name,
            "medicine_id": m_id,
            "medicine_name": m_name,
            "category": row["category"],
            "criticality": int(row["criticality"]),
            "quantity": qty,
            "expiry_date": row["expiry_date"],
            "days_until_expiry": days_to_exp,
            "expected_usage_before_expiry": expected_usage_before_expiry,
            "safety_stock_protect": safety_stock_protect,
            "at_risk_waste_units": at_risk_waste_units,
            "unit_cost_inr": unit_cost_inr,
            "potential_financial_loss_inr": financial_loss_inr,
            "potential_financial_loss_usd": financial_loss_usd,
            "risk_tier": risk_tier,
            "explanation": explanation
        }
