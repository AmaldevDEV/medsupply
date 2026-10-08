"""
What-If Scenario Simulation Engine
Allows interactive modeling of operational disruptions:
- Demand surges (+20% to +100%)
- Outbreak escalation in specific regions
- Supplier lead-time delays (+2 to +10 days)
- Donor capacity restrictions (e.g. Hospital B stock drops or caps donation at 500 units)
- Transport transit delays
Reruns forecast projections, stockout risks, priority rankings, redistribution,
and emergency sourcing to produce comprehensive Before vs After impact analysis.
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
from optimization.priority import SupplyPriorityEngine
from optimization.redistribution import RedistributionOptimizer
from optimization.emergency_matching import EmergencySourcingEngine

class WhatIfSimulator:
    def __init__(self):
        self.projection_engine = InventoryProjectionEngine()
        self.priority_engine = SupplyPriorityEngine()
        self.redist_optimizer = RedistributionOptimizer()
        self.emergency_engine = EmergencySourcingEngine()

    def run_simulation(
        self,
        demand_surge_pct: float = 0.0,
        outbreak_hospitals: Optional[List[str]] = None,
        outbreak_severity_mult: float = 1.0,
        supplier_lead_time_added_days: int = 0,
        donor_stock_reduction_pct: float = 0.0,
        hospital_b_max_donation: Optional[int] = None,
        transport_slowdown_factor: float = 1.0,
        emergency_request_id: str = "EMERG_REQ_001"
    ) -> Dict[str, Any]:
        """
        Executes end-to-end Before vs After simulation.
        """
        outbreak_hospitals = outbreak_hospitals or []

        # 1. BASELINE CALCULATIONS
        baseline_shortages = self.projection_engine.scan_all_shortage_risks()
        base_crit_count = int((baseline_shortages["lead_time_risk"] == "CRITICAL").sum())
        base_high_count = int((baseline_shortages["lead_time_risk"] == "HIGH").sum())
        base_avg_days_to_stockout = float(baseline_shortages["days_until_stockout"].mean())
        base_total_surplus = int(baseline_shortages["transferable_surplus"].sum())

        # Baseline Emergency Sourcing
        base_emerg = self.emergency_engine.evaluate_emergency_request(emergency_request_id)
        base_emerg_source = base_emerg["recommended_source"]["candidate_hospital_name"] if base_emerg["recommended_source"] else "None"
        base_emerg_feasible = base_emerg["has_feasible_source"]

        # Baseline Redistribution
        base_redist = self.redist_optimizer.generate_redistribution_plan()
        base_redist_count = len(base_redist)
        base_transferred_units = sum(r["quantity"] for r in base_redist)

        # 2. SIMULATED CALCULATIONS (Applying Scenario Parameters)
        hosp_df = query_df("SELECT * FROM hospitals")
        med_df = query_df("SELECT * FROM medicines")
        
        sim_results = []
        for h_id in hosp_df["hospital_id"]:
            for m_id in med_df["medicine_id"]:
                proj = self.projection_engine.project_inventory(h_id, m_id)
                
                # Apply demand modifiers
                base_demand = proj["avg_daily_forecast"]
                sim_demand = base_demand * (1.0 + demand_surge_pct / 100.0)
                
                # Regional outbreak surge
                if h_id in outbreak_hospitals:
                    sim_demand = sim_demand * outbreak_severity_mult
                    
                # Apply donor stock reduction (e.g. Hospital B)
                sim_stock = proj["current_stock"]
                if h_id == "HOSP_B" and donor_stock_reduction_pct > 0:
                    sim_stock = int(sim_stock * (1.0 - donor_stock_reduction_pct / 100.0))
                    
                # Apply supplier lead time extension
                sim_lead_time = proj["supplier_lead_time_days"] + supplier_lead_time_added_days
                
                # Recalculate days until stockout under simulated demand
                sim_safety_threshold = int(round(proj["safety_stock_policy_days"] * sim_demand))
                if sim_demand > 0:
                    if sim_stock <= sim_safety_threshold:
                        sim_days_to_stockout = 1
                    else:
                        sim_days_to_stockout = max(1, int(round((sim_stock - sim_safety_threshold) / sim_demand)))
                else:
                    sim_days_to_stockout = 30
                    
                # Recalculate lead time risk
                if sim_days_to_stockout <= 1 or sim_stock <= sim_safety_threshold:
                    sim_risk = "CRITICAL"
                elif sim_days_to_stockout < sim_lead_time:
                    sim_risk = "CRITICAL"
                elif sim_days_to_stockout <= sim_lead_time + 2:
                    sim_risk = "HIGH"
                elif sim_days_to_stockout <= 7:
                    sim_risk = "MODERATE"
                else:
                    sim_risk = "LOW"
                    
                # Recalculate transferable surplus
                sim_14d_demand = int(round(sim_demand * 14))
                sim_surplus = max(0, sim_stock - sim_14d_demand - sim_safety_threshold)
                if h_id == "HOSP_B" and hospital_b_max_donation is not None:
                    sim_surplus = min(sim_surplus, hospital_b_max_donation)

                sim_results.append({
                    "hospital_id": h_id,
                    "hospital_name": proj["hospital_name"],
                    "medicine_id": m_id,
                    "medicine_name": proj["medicine_name"],
                    "sim_demand": round(sim_demand, 1),
                    "sim_stock": sim_stock,
                    "sim_days_until_stockout": sim_days_to_stockout,
                    "sim_lead_time": sim_lead_time,
                    "sim_lead_time_risk": sim_risk,
                    "sim_transferable_surplus": sim_surplus
                })

        sim_df = pd.DataFrame(sim_results)
        sim_crit_count = int((sim_df["sim_lead_time_risk"] == "CRITICAL").sum())
        sim_high_count = int((sim_df["sim_lead_time_risk"] == "HIGH").sum())
        sim_avg_days_to_stockout = float(sim_df["sim_days_until_stockout"].mean())
        sim_total_surplus = int(sim_df["sim_transferable_surplus"].sum())

        # Simulated Emergency Sourcing (incorporating transport slowdown & stock cap)
        # Fetch base emergency candidate list and adjust
        sim_emerg_feasible = False
        sim_emerg_source = "None"
        sim_emerg_reason = ""
        
        for cand in base_emerg["all_evaluated_candidates"]:
            c_id = cand["candidate_hospital"]
            adjusted_transit = cand["transit_time_hours"] * transport_slowdown_factor
            adjusted_arrival_dt = cand["estimated_arrival_dt"] + timedelta(hours=(adjusted_transit - cand["transit_time_hours"]))
            
            # Check stock
            avail_stock = cand["current_stock"]
            if c_id == "HOSP_B":
                if donor_stock_reduction_pct > 0:
                    avail_stock = int(avail_stock * (1.0 - donor_stock_reduction_pct / 100.0))
                if hospital_b_max_donation is not None and hospital_b_max_donation < base_emerg["quantity_required"]:
                    # Cannot donate required amount!
                    continue

            # Check deadline with adjusted transit
            req_by_dt = pd.to_datetime(base_emerg["required_by"])
            arrives_on_time = (adjusted_arrival_dt <= req_by_dt)
            donor_safe = (avail_stock - base_emerg["quantity_required"] >= cand["safety_stock_threshold"])

            if arrives_on_time and donor_safe and avail_stock >= base_emerg["quantity_required"]:
                sim_emerg_feasible = True
                sim_emerg_source = cand["candidate_hospital_name"]
                sim_emerg_reason = (
                    f"Selected under simulation: {sim_emerg_source} delivers before surgical deadline "
                    f"even with {transport_slowdown_factor:.1f}x transport slowdown (transit = {adjusted_transit:.1f}h)."
                )
                break

        if not sim_emerg_feasible:
            sim_emerg_reason = (
                "Under simulated stress conditions (slowdowns/stock reductions), NO hospital can deliver "
                "the required quantity before the surgical deadline while preserving safety stock."
            )

        # 3. Before vs After Comparison Summary
        comparison = {
            "critical_shortages": {
                "before": base_crit_count,
                "after": sim_crit_count,
                "delta": sim_crit_count - base_crit_count
            },
            "high_risk_count": {
                "before": base_high_count,
                "after": sim_high_count,
                "delta": sim_high_count - base_high_count
            },
            "average_days_until_stockout": {
                "before": round(base_avg_days_to_stockout, 1),
                "after": round(sim_avg_days_to_stockout, 1),
                "delta": round(sim_avg_days_to_stockout - base_avg_days_to_stockout, 1)
            },
            "network_transferable_surplus_units": {
                "before": base_total_surplus,
                "after": sim_total_surplus,
                "delta": sim_total_surplus - base_total_surplus
            },
            "emergency_sourcing": {
                "before_feasible": base_emerg_feasible,
                "before_source": base_emerg_source,
                "after_feasible": sim_emerg_feasible,
                "after_source": sim_emerg_source,
                "after_reason": sim_emerg_reason
            }
        }

        # Narrative executive summary
        summary_lines = []
        if sim_crit_count > base_crit_count:
            summary_lines.append(f"Critical shortages surged from {base_crit_count} to {sim_crit_count} (+{sim_crit_count - base_crit_count}).")
        if sim_total_surplus < base_total_surplus:
            summary_lines.append(f"Network transferable buffer contracted by {base_total_surplus - sim_total_surplus:,} units.")
        if base_emerg_feasible and not sim_emerg_feasible:
            summary_lines.append("EMERGENCY FAILURE: Inter-hospital emergency sourcing collapsed due to compounding delays and capacity loss.")
        elif sim_emerg_source != base_emerg_source:
            summary_lines.append(f"Emergency source shifted from {base_emerg_source} to {sim_emerg_source}.")

        return {
            "scenario_parameters": {
                "demand_surge_pct": demand_surge_pct,
                "outbreak_hospitals": outbreak_hospitals,
                "outbreak_severity_mult": outbreak_severity_mult,
                "supplier_lead_time_added_days": supplier_lead_time_added_days,
                "donor_stock_reduction_pct": donor_stock_reduction_pct,
                "hospital_b_max_donation": hospital_b_max_donation,
                "transport_slowdown_factor": transport_slowdown_factor
            },
            "comparison": comparison,
            "executive_summary": " ".join(summary_lines) if summary_lines else "Network resilience held stable under simulated parameters.",
            "simulated_details": sim_df.to_dict("records")
        }
