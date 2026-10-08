"""
Supply Priority Engine
Computes transparent, multi-criteria priority scores (0-100) when multiple
hospitals compete for scarce medical inventory. Supports configurable weights
and provides granular feature attribution / explainability.
"""

from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DEFAULT_PRIORITY_WEIGHTS, DATABASE_PATH
from database.database import query_df

class SupplyPriorityEngine:
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """
        :param weights: Dictionary containing keys:
            - shortage_urgency (default 0.30)
            - emergency_demand (default 0.25)
            - patient_load (default 0.20)
            - alternative_scarcity (default 0.15)
            - lead_time_risk (default 0.10)
        """
        self.weights = weights or DEFAULT_PRIORITY_WEIGHTS
        # Normalize weights to sum to 1.0
        total_w = sum(self.weights.values())
        self.weights = {k: v / total_w for k, v in self.weights.items()}

    def calculate_priority(
        self,
        hospital_id: str,
        medicine_id: str,
        days_until_stockout: int,
        supplier_lead_time: int,
        current_stock: int,
        emergency_patient_load: int,
        current_occupancy: float,
        criticality: int,
        has_alternative: bool
    ) -> Dict[str, Any]:
        """
        Calculates normalized components (0-100) and composite priority score:
        Priority Score =
            w1 * Shortage Urgency
            + w2 * Emergency Demand
            + w3 * Patient Load
            + w4 * Alternative Scarcity
            + w5 * Lead-Time Risk
        """
        # 1. Shortage Urgency (0-100)
        # 0 days or less = 100, 1-3 days = 90-70, 7 days = 40, >= 14 days = 0
        if days_until_stockout <= 1:
            shortage_urgency = 100.0
        elif days_until_stockout <= 3:
            shortage_urgency = 85.0 + (3 - days_until_stockout) * 5.0
        elif days_until_stockout <= 7:
            shortage_urgency = 50.0 + (7 - days_until_stockout) * 8.0
        elif days_until_stockout <= 14:
            shortage_urgency = max(0.0, 50.0 - (days_until_stockout - 7) * 7.0)
        else:
            shortage_urgency = 0.0

        # Scale by clinical criticality of the medicine (tier 1-5)
        # Criticality 5 gets 1.0x, criticality 1 gets 0.6x
        crit_factor = 0.5 + 0.1 * criticality
        shortage_urgency = min(100.0, shortage_urgency * crit_factor)

        # 2. Emergency Demand (0-100)
        # Normal emergency load ranges from 15 to 120 patients
        # Baseline 100 pts = 100 score
        emergency_demand = min(100.0, (emergency_patient_load / 110.0) * 100.0)

        # 3. Patient Load / Capacity Acuity (0-100)
        # Occupancy 0.5 to 1.0 -> maps to 20 to 100
        patient_load = min(100.0, max(0.0, current_occupancy * 100.0))

        # 4. Alternative Scarcity (0-100)
        # If no alternative exists (has_alternative = False) -> 100
        # If alternative exists -> 25
        alternative_scarcity = 100.0 if not has_alternative else 25.0

        # 5. Lead-Time Risk (0-100)
        # Compares stockout days vs supplier lead time
        # If stockout arrives BEFORE supplier delivery: max danger
        lead_time_gap = supplier_lead_time - days_until_stockout
        if lead_time_gap >= 3:
            lead_time_risk = 100.0
        elif lead_time_gap >= 0:
            lead_time_risk = 75.0 + lead_time_gap * 8.0
        elif days_until_stockout <= supplier_lead_time + 3:
            lead_time_risk = 40.0
        else:
            lead_time_risk = 10.0

        # Composite Score Calculation
        w = self.weights
        composite_score = (
            w["shortage_urgency"] * shortage_urgency
            + w["emergency_demand"] * emergency_demand
            + w["patient_load"] * patient_load
            + w["alternative_scarcity"] * alternative_scarcity
            + w["lead_time_risk"] * lead_time_risk
        )
        composite_score = round(min(100.0, max(0.0, composite_score)), 1)

        # Qualitative Rationale Generator
        primary_drivers = []
        if shortage_urgency >= 75:
            primary_drivers.append(f"Imminent stockout ({days_until_stockout}d remaining)")
        if lead_time_risk >= 75:
            primary_drivers.append(f"Supplier delivery lag ({supplier_lead_time}d lead time)")
        if emergency_demand >= 75:
            primary_drivers.append(f"Severe emergency patient burden ({emergency_patient_load} pts)")
        if alternative_scarcity >= 80:
            primary_drivers.append("Zero clinical drug substitutes available")
        if patient_load >= 85:
            primary_drivers.append(f"Near-maximum hospital occupancy ({int(current_occupancy*100)}%)")

        if not primary_drivers:
            rationale = "Low operational urgency; adequate reserve buffer."
        else:
            rationale = "High priority driven by: " + "; ".join(primary_drivers) + "."

        return {
            "hospital_id": hospital_id,
            "medicine_id": medicine_id,
            "composite_priority_score": composite_score,
            "shortage_urgency_score": round(shortage_urgency, 1),
            "emergency_demand_score": round(emergency_demand, 1),
            "patient_load_score": round(patient_load, 1),
            "alternative_scarcity_score": round(alternative_scarcity, 1),
            "lead_time_risk_score": round(lead_time_risk, 1),
            "weights_used": self.weights,
            "priority_tier": "CRITICAL" if composite_score >= 80 else ("HIGH" if composite_score >= 65 else ("MODERATE" if composite_score >= 45 else "LOW")),
            "rationale": rationale
        }

    def rank_all_competing_facilities(self, medicine_id: str) -> pd.DataFrame:
        """
        Ranks all 10 hospitals competing for a specific scarce medicine.
        Provides a transparent audit comparison table.
        """
        q = """
            SELECT h.hospital_id, h.hospital_name, h.emergency_patient_load,
                   h.current_occupancy, h.safety_stock_policy_days,
                   m.medicine_name, m.criticality, m.has_alternative,
                   MIN(s.lead_time_days) as supplier_lead_time,
                   COALESCE(SUM(i.quantity), 0) as current_stock
            FROM hospitals h
            CROSS JOIN medicines m
            LEFT JOIN suppliers s ON m.medicine_id = s.medicine_id
            LEFT JOIN inventory i ON h.hospital_id = i.hospital_id AND m.medicine_id = i.medicine_id
            WHERE m.medicine_id = ?
            GROUP BY h.hospital_id
        """
        df = query_df(q, (medicine_id,))
        if df.empty:
            return pd.DataFrame()

        rows = []
        for _, row in df.iterrows():
            h_id = row["hospital_id"]
            cur_stock = int(row["current_stock"])
            lead_time = int(row["supplier_lead_time"]) if pd.notnull(row["supplier_lead_time"]) else 7

            # Fetch forecast daily consumption
            fc_q = "SELECT AVG(predicted_demand) as avg_d FROM forecasts WHERE hospital_id = ? AND medicine_id = ?"
            fc_df = query_df(fc_q, (h_id, medicine_id))
            avg_d = float(fc_df["avg_d"].iloc[0]) if not fc_df.empty and pd.notnull(fc_df["avg_d"].iloc[0]) else 20.0
            
            # Simple day-by-day simulation or stockout estimation
            safety_target = int(round(int(row["safety_stock_policy_days"]) * avg_d))
            if cur_stock <= safety_target:
                days_to_stockout = 1
            else:
                days_to_stockout = max(1, int(round((cur_stock - safety_target) / max(1.0, avg_d))))

            p_res = self.calculate_priority(
                hospital_id=h_id,
                medicine_id=medicine_id,
                days_until_stockout=days_to_stockout,
                supplier_lead_time=lead_time,
                current_stock=cur_stock,
                emergency_patient_load=int(row["emergency_patient_load"]),
                current_occupancy=float(row["current_occupancy"]),
                criticality=int(row["criticality"]),
                has_alternative=bool(row["has_alternative"])
            )

            rows.append({
                "rank": 0,
                "hospital_id": h_id,
                "hospital_name": row["hospital_name"],
                "medicine_id": medicine_id,
                "medicine_name": row["medicine_name"],
                "current_stock": cur_stock,
                "days_until_stockout": days_to_stockout,
                "supplier_lead_time": lead_time,
                "priority_tier": p_res["priority_tier"],
                "priority_score": p_res["composite_priority_score"],
                "urgency_component": p_res["shortage_urgency_score"],
                "emergency_component": p_res["emergency_demand_score"],
                "acuity_component": p_res["patient_load_score"],
                "alternative_scarcity": p_res["alternative_scarcity_score"],
                "lead_time_risk": p_res["lead_time_risk_score"],
                "rationale": p_res["rationale"]
            })

        ranked_df = pd.DataFrame(rows).sort_values("priority_score", ascending=False).reset_index(drop=True)
        ranked_df["rank"] = np.arange(1, len(ranked_df) + 1)
        return ranked_df
