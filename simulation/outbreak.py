"""
Regional Outbreak Simulator
Simulates targeted infectious surges across designated hospital clusters
and medicine categories (antivirals, respiratory drugs, IV fluids, antibiotics).
Calculates acute inventory strain, bed occupancy escalation, and inter-hospital relief flows.
"""

from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATABASE_PATH
from database.database import query_df
from models.inventory_projection import InventoryProjectionEngine
from optimization.redistribution import RedistributionOptimizer

class OutbreakSimulator:
    def __init__(self):
        self.projection_engine = InventoryProjectionEngine()
        self.redist_optimizer = RedistributionOptimizer()

    def simulate_regional_outbreak(
        self,
        cluster_hospital_ids: List[str] = ["HOSP_D", "HOSP_E", "HOSP_F"],
        target_medicine_ids: List[str] = ["MED007", "MED016", "MED005"],
        severity_multiplier: float = 2.2,
        duration_days: int = 30
    ) -> Dict[str, Any]:
        """
        Runs dedicated outbreak stress test:
        1. Multiplies demand for affected medicines in selected hospital clusters.
        2. Escalates emergency patient loads.
        3. Identifies emerging stockout cliffs.
        4. Calculates required surge redistribution from unaffected network hospitals.
        """
        hosp_query = "SELECT hospital_id, hospital_name, beds, emergency_patient_load FROM hospitals"
        hosp_df = query_df(hosp_query)
        hosp_names = dict(zip(hosp_df["hospital_id"], hosp_df["hospital_name"]))
        
        med_query = "SELECT medicine_id, medicine_name, category, criticality FROM medicines"
        med_df = query_df(med_query)
        med_names = dict(zip(med_df["medicine_id"], med_df["medicine_name"]))

        impacted_hospitals = [hosp_names.get(h, h) for h in cluster_hospital_ids]
        impacted_medicines = [med_names.get(m, m) for m in target_medicine_ids]

        facility_impacts = []
        new_critical_shortages = 0
        total_surge_units_needed = 0

        for h_id in cluster_hospital_ids:
            h_name = hosp_names.get(h_id, h_id)
            for m_id in target_medicine_ids:
                proj = self.projection_engine.project_inventory(h_id, m_id)
                normal_demand = proj["avg_daily_forecast"]
                surge_demand = normal_demand * severity_multiplier
                cur_stock = proj["current_stock"]
                lead_time = proj["supplier_lead_time_days"]
                safety_threshold = int(round(proj["safety_stock_policy_days"] * surge_demand))

                # New stockout under surge
                if cur_stock <= safety_threshold:
                    surge_days_to_stockout = 1
                else:
                    surge_days_to_stockout = max(1, int(round((cur_stock - safety_threshold) / max(1.0, surge_demand))))

                # Calculate deficit
                target_30d_stock = int(round(surge_demand * 14))
                deficit = max(0, target_30d_stock - cur_stock)
                total_surge_units_needed += deficit

                is_critical = (surge_days_to_stockout < lead_time)
                if is_critical:
                    new_critical_shortages += 1

                facility_impacts.append({
                    "hospital_id": h_id,
                    "hospital_name": h_name,
                    "medicine_id": m_id,
                    "medicine_name": med_names.get(m_id, m_id),
                    "normal_daily_demand": round(normal_demand, 1),
                    "surge_daily_demand": round(surge_demand, 1),
                    "current_stock": cur_stock,
                    "baseline_days_to_stockout": proj["days_until_stockout"],
                    "surge_days_to_stockout": surge_days_to_stockout,
                    "supplier_lead_time": lead_time,
                    "lead_time_breached": is_critical,
                    "additional_units_needed": deficit
                })

        # Relief options from unaffected hospitals
        unaffected_hosps = [h for h in hosp_df["hospital_id"] if h not in cluster_hospital_ids]
        potential_relief = []
        for uh_id in unaffected_hosps:
            for m_id in target_medicine_ids:
                proj = self.projection_engine.project_inventory(uh_id, m_id)
                if proj["transferable_surplus"] >= 100:
                    potential_relief.append({
                        "donor_hospital_id": uh_id,
                        "donor_hospital_name": hosp_names.get(uh_id, uh_id),
                        "medicine_id": m_id,
                        "medicine_name": med_names.get(m_id, m_id),
                        "safe_transferable_units": proj["transferable_surplus"]
                    })

        total_relief_available = sum(p["safe_transferable_units"] for p in potential_relief)
        relief_coverage_pct = min(100.0, round((total_relief_available / max(1, total_surge_units_needed)) * 100.0, 1))

        return {
            "outbreak_cluster": impacted_hospitals,
            "target_medicines": impacted_medicines,
            "severity_multiplier": severity_multiplier,
            "duration_days": duration_days,
            "new_critical_shortages": new_critical_shortages,
            "total_surge_units_needed": total_surge_units_needed,
            "total_relief_available": total_relief_available,
            "relief_coverage_pct": relief_coverage_pct,
            "facility_impacts": facility_impacts,
            "potential_relief_sources": potential_relief,
            "summary_narrative": (
                f"Simulated {severity_multiplier:.1f}x outbreak across {', '.join(impacted_hospitals)} "
                f"creates {new_critical_shortages} immediate critical supply breaches. "
                f"Total network surge deficit is {total_surge_units_needed:,} units. "
                f"Unaffected facilities can supply {total_relief_available:,} safe transferable units "
                f"({relief_coverage_pct}% relief coverage)."
            )
        }
