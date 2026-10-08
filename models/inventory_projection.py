"""
Inventory Projection & Stockout Prediction Engine
Simulates future inventory day-by-day using ML demand forecasts,
determines exact days until stockout, evaluates supplier lead times,
and flags procurement lead-time risks.
"""

from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATABASE_PATH, DEFAULT_SAFETY_STOCK_DAYS
from database.database import query_df

class InventoryProjectionEngine:
    def __init__(self, simulation_horizon_days: int = 30):
        self.simulation_horizon_days = simulation_horizon_days

    def project_inventory(
        self,
        hospital_id: str,
        medicine_id: str,
        incoming_orders: Optional[Dict[int, int]] = None
    ) -> Dict[str, Any]:
        """
        Simulates future inventory trajectory day-by-day:
        Projected Stock(t) = Projected Stock(t-1) - Forecast Demand(t) + Incoming Supply(t)
        Detects stockout when projected stock drops below the hospital's required safety stock.
        """
        incoming_orders = incoming_orders or {}
        
        # 1. Fetch current on-hand inventory
        inv_query = """
            SELECT SUM(quantity) as total_stock
            FROM inventory
            WHERE hospital_id = ? AND medicine_id = ?
        """
        inv_df = query_df(inv_query, (hospital_id, medicine_id))
        current_stock = int(inv_df["total_stock"].iloc[0]) if not inv_df.empty and pd.notnull(inv_df["total_stock"].iloc[0]) else 0

        # 2. Fetch hospital parameters (safety stock policy, beds, name)
        hosp_query = "SELECT hospital_name, safety_stock_policy_days, beds FROM hospitals WHERE hospital_id = ?"
        hosp_df = query_df(hosp_query, (hospital_id,))
        if hosp_df.empty:
            raise ValueError(f"Hospital {hospital_id} not found")
        hosp_name = hosp_df["hospital_name"].iloc[0]
        safety_days = int(hosp_df["safety_stock_policy_days"].iloc[0])

        # 3. Fetch medicine info
        med_query = "SELECT medicine_name, category, criticality, unit_cost_inr, has_alternative, alternative_medicine_id FROM medicines WHERE medicine_id = ?"
        med_df = query_df(med_query, (medicine_id,))
        if med_df.empty:
            raise ValueError(f"Medicine {medicine_id} not found")
        med_name = med_df["medicine_name"].iloc[0]
        criticality = int(med_df["criticality"].iloc[0])
        has_alt = bool(med_df["has_alternative"].iloc[0])
        alt_id = med_df["alternative_medicine_id"].iloc[0]

        # 4. Fetch supplier lead time
        sup_query = """
            SELECT MIN(lead_time_days) as min_lead_time
            FROM suppliers
            WHERE medicine_id = ?
        """
        sup_df = query_df(sup_query, (medicine_id,))
        supplier_lead_time = int(sup_df["min_lead_time"].iloc[0]) if not sup_df.empty and pd.notnull(sup_df["min_lead_time"].iloc[0]) else 7

        # 5. Fetch ML Demand Forecasts
        fc_query = """
            SELECT target_date, day_ahead, predicted_demand
            FROM forecasts
            WHERE hospital_id = ? AND medicine_id = ?
            ORDER BY day_ahead ASC
        """
        fc_df = query_df(fc_query, (hospital_id, medicine_id))
        
        # Fallback if forecast table is still generating: use recent demand
        if fc_df.empty:
            recent_demand_query = """
                SELECT AVG(daily_consumption) as avg_c
                FROM demand_history
                WHERE hospital_id = ? AND medicine_id = ?
            """
            rec_df = query_df(recent_demand_query, (hospital_id, medicine_id))
            avg_daily = float(rec_df["avg_c"].iloc[0]) if not rec_df.empty and pd.notnull(rec_df["avg_c"].iloc[0]) else 20.0
            forecast_demands = [avg_daily] * self.simulation_horizon_days
            target_dates = [(datetime.now() + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(1, self.simulation_horizon_days + 1)]
        else:
            forecast_demands = fc_df["predicted_demand"].values.tolist()[:self.simulation_horizon_days]
            target_dates = fc_df["target_date"].values.tolist()[:self.simulation_horizon_days]
            
        # Ensure we cover the full horizon
        avg_demand_rate = float(np.mean(forecast_demands)) if forecast_demands else 20.0
        while len(forecast_demands) < self.simulation_horizon_days:
            forecast_demands.append(avg_demand_rate)
            next_d = (datetime.now() + timedelta(days=len(forecast_demands))).strftime("%Y-%m-%d")
            target_dates.append(next_d)

        # 6. Calculate required safety stock
        # Safety Stock = safety_stock_policy_days * average daily forecast
        safety_stock_threshold = int(round(safety_days * avg_demand_rate))

        # 7. Day-by-Day Forward Inventory Simulation
        trajectory = []
        stock = float(current_stock)
        days_until_stockout = None
        days_until_zero = None
        
        for t, (t_date, dem) in enumerate(zip(target_dates, forecast_demands), start=1):
            incoming = incoming_orders.get(t, 0)
            stock = stock - dem + incoming
            stock_rounded = max(0, int(round(stock)))
            
            # Check safety stock threshold breach
            if stock_rounded <= safety_stock_threshold and days_until_stockout is None:
                days_until_stockout = t
                
            # Check absolute zero stock
            if stock_rounded <= 0 and days_until_zero is None:
                days_until_zero = t
                
            trajectory.append({
                "day_ahead": t,
                "date": t_date,
                "forecast_demand": round(dem, 1),
                "incoming_supply": incoming,
                "projected_stock": stock_rounded,
                "safety_stock_threshold": safety_stock_threshold,
                "is_below_safety": stock_rounded <= safety_stock_threshold,
                "is_zero_stock": stock_rounded == 0
            })

        if days_until_stockout is None:
            days_until_stockout = self.simulation_horizon_days + 1 # Safe beyond horizon
        if days_until_zero is None:
            days_until_zero = self.simulation_horizon_days + 1

        # 8. Evaluate Procurement Lead-Time Risk
        # Compare days_until_stockout vs supplier_lead_time
        if days_until_stockout <= 0 or current_stock <= safety_stock_threshold:
            lead_time_risk = "CRITICAL"
            risk_explanation = (
                f"Procurement Lead-Time Risk: CRITICAL. Immediate Shortage Alert: Current stock ({current_stock} units) is already at or below "
                f"safety threshold ({safety_stock_threshold} units). Supplier lead time is {supplier_lead_time} days."
            )
        elif days_until_stockout < supplier_lead_time:
            lead_time_risk = "CRITICAL"
            risk_explanation = (
                f"Procurement Lead-Time Risk: CRITICAL. Projected stockout in {days_until_stockout} days, "
                f"but supplier lead time is {supplier_lead_time} days. An emergency or redistribution transfer is necessary."
            )
        elif days_until_stockout <= supplier_lead_time + 2:
            lead_time_risk = "HIGH"
            risk_explanation = (
                f"Procurement Lead-Time Risk: HIGH. Projected stockout in {days_until_stockout} days vs "
                f"supplier lead time of {supplier_lead_time} days (buffer is only {days_until_stockout - supplier_lead_time} days)."
            )
        elif days_until_stockout <= 7:
            lead_time_risk = "MODERATE"
            risk_explanation = f"Stockout approaching within 7 days ({days_until_stockout} days remaining)."
        else:
            lead_time_risk = "LOW"
            risk_explanation = f"Inventory posture stable. Sufficient stock for {days_until_stockout} days."

        # 9. Calculate Transferable Surplus (14-day horizon)
        # Transferable Surplus = max(0, Current Stock - Forecast 14-Day Demand - Safety Stock)
        forecast_14d_demand = int(round(sum(forecast_demands[:14])))
        transferable_surplus = max(0, current_stock - forecast_14d_demand - safety_stock_threshold)

        return {
            "hospital_id": hospital_id,
            "hospital_name": hosp_name,
            "medicine_id": medicine_id,
            "medicine_name": med_name,
            "category": med_df["category"].iloc[0],
            "criticality": criticality,
            "current_stock": current_stock,
            "safety_stock_policy_days": safety_days,
            "safety_stock_threshold": safety_stock_threshold,
            "avg_daily_forecast": round(avg_demand_rate, 1),
            "forecast_14d_demand": forecast_14d_demand,
            "transferable_surplus": transferable_surplus,
            "days_until_stockout": days_until_stockout,
            "days_until_zero": days_until_zero,
            "supplier_lead_time_days": supplier_lead_time,
            "lead_time_risk": lead_time_risk,
            "risk_explanation": risk_explanation,
            "has_alternative": has_alt,
            "alternative_medicine_id": alt_id,
            "trajectory": trajectory
        }

    def scan_all_shortage_risks(self) -> pd.DataFrame:
        """Scans all hospital-medicine combinations and ranks those facing shortage risk."""
        hosp_df = query_df("SELECT hospital_id FROM hospitals")
        med_df = query_df("SELECT medicine_id FROM medicines")
        
        results = []
        for h_id in hosp_df["hospital_id"]:
            for m_id in med_df["medicine_id"]:
                proj = self.project_inventory(h_id, m_id)
                results.append({
                    "hospital_id": proj["hospital_id"],
                    "hospital_name": proj["hospital_name"],
                    "medicine_id": proj["medicine_id"],
                    "medicine_name": proj["medicine_name"],
                    "category": proj["category"],
                    "criticality": proj["criticality"],
                    "current_stock": proj["current_stock"],
                    "safety_threshold": proj["safety_stock_threshold"],
                    "avg_daily_demand": proj["avg_daily_forecast"],
                    "days_until_stockout": proj["days_until_stockout"],
                    "supplier_lead_time": proj["supplier_lead_time_days"],
                    "lead_time_risk": proj["lead_time_risk"],
                    "transferable_surplus": proj["transferable_surplus"],
                    "risk_explanation": proj["risk_explanation"],
                    "has_alternative": proj["has_alternative"],
                    "alternative_medicine_id": proj["alternative_medicine_id"]
                })
        return pd.DataFrame(results).sort_values(["days_until_stockout", "criticality"], ascending=[True, False]).reset_index(drop=True)
