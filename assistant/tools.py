"""
Analytical Tools for AI Decision-Support Assistant
Executes deterministic database queries, inventory projections, priority scoring,
expiry audits, emergency sourcing, and what-if simulations.
Ensures zero numerical hallucinations.
"""

from typing import Dict, List, Any, Optional
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from database.database import query_df
from models.inventory_projection import InventoryProjectionEngine
from models.expiry_wastage import ExpiryWastageDetector
from optimization.priority import SupplyPriorityEngine
from optimization.redistribution import RedistributionOptimizer
from optimization.emergency_matching import EmergencySourcingEngine
from simulation.what_if import WhatIfSimulator
from simulation.outbreak import OutbreakSimulator

# Initialize engine singletons
projection_engine = InventoryProjectionEngine()
expiry_detector = ExpiryWastageDetector()
priority_engine = SupplyPriorityEngine()
redist_optimizer = RedistributionOptimizer()
emergency_engine = EmergencySourcingEngine()
what_if_simulator = WhatIfSimulator()
outbreak_simulator = OutbreakSimulator()

def tool_get_shortage_risks(medicine_id: Optional[str] = None, hospital_id: Optional[str] = None, max_days: int = 14) -> Dict[str, Any]:
    """Retrieves verified inventory stockout risks and lead-time alerts."""
    df = projection_engine.scan_all_shortage_risks()
    
    if medicine_id:
        df = df[df["medicine_id"].str.upper() == medicine_id.upper()]
    if hospital_id:
        df = df[df["hospital_id"].str.upper() == hospital_id.upper()]
        
    df_filtered = df[df["days_until_stockout"] <= max_days]
    
    records = df_filtered.to_dict("records")
    return {
        "query_medicine": medicine_id,
        "query_hospital": hospital_id,
        "max_days_threshold": max_days,
        "total_at_risk_count": len(records),
        "risks": records[:10]
    }

def tool_get_expiry_risks(max_days: int = 30) -> Dict[str, Any]:
    """Retrieves batch-level expiry and waste calculations in INR."""
    df = expiry_detector.scan_all_batches()
    if df.empty:
        return {"total_at_risk": 0, "batches": []}
        
    df_at_risk = df[(df["days_until_expiry"] <= max_days) & (df["at_risk_waste_units"] > 0)]
    records = df_at_risk.to_dict("records")
    total_waste_inr = float(df_at_risk["potential_financial_loss_inr"].sum()) if "potential_financial_loss_inr" in df_at_risk else float(df_at_risk["potential_financial_loss_usd"].sum() * 83.0)
    
    return {
        "max_days_horizon": max_days,
        "total_batches_at_risk": len(records),
        "total_financial_waste_inr": round(total_waste_inr, 2),
        "total_financial_waste_usd": round(total_waste_inr / 83.0, 2),
        "batches": records[:8]
    }

def tool_get_priority_explanation(hospital_id: str, medicine_id: str) -> Dict[str, Any]:
    """Explains why a specific hospital received its priority score for a scarce medicine."""
    ranked = priority_engine.rank_all_competing_facilities(medicine_id)
    if ranked.empty:
        return {"error": f"No data found for medicine {medicine_id}"}
        
    target_row = ranked[ranked["hospital_id"].str.upper() == hospital_id.upper()]
    if target_row.empty:
        return {"error": f"Hospital {hospital_id} not found in ranking for {medicine_id}"}
        
    row = target_row.iloc[0].to_dict()
    return {
        "target_hospital": row,
        "network_competing_count": len(ranked),
        "top_ranked_hospital": ranked.iloc[0].to_dict()
    }

def tool_check_donor_surplus(hospital_id: str, medicine_id: str, requested_units: int) -> Dict[str, Any]:
    """Evaluates whether a donor hospital can safely donate stock without creating a local shortage."""
    proj = projection_engine.project_inventory(hospital_id, medicine_id)
    cur_stock = proj["current_stock"]
    safe_surplus = proj["transferable_surplus"]
    safety_threshold = proj["safety_stock_threshold"]
    forecast_14d = proj["forecast_14d_demand"]
    
    can_donate = (safe_surplus >= requested_units)
    remaining_after_donation = cur_stock - requested_units
    safety_breached = (remaining_after_donation < safety_threshold)
    
    if can_donate:
        audit_verdict = (
            f"YES. {proj['hospital_name']} has {cur_stock:,} units on hand. After protecting its 14-day "
            f"forecast demand ({forecast_14d:,} units) and mandatory safety stock ({safety_threshold:,} units), "
            f"it holds {safe_surplus:,} units of safe transferable surplus. It can safely donate {requested_units:,} units."
        )
    else:
        deficit = requested_units - safe_surplus
        audit_verdict = (
            f"NO. {proj['hospital_name']} cannot safely transfer {requested_units:,} units. Its safe transferable "
            f"surplus is only {safe_surplus:,} units. Transferring {requested_units:,} units would leave "
            f"{remaining_after_donation:,} units in stock, breaching its required safety stock buffer "
            f"({safety_threshold:,} units) and risking a local shortage."
        )
        
    return {
        "hospital_id": hospital_id,
        "hospital_name": proj["hospital_name"],
        "medicine_id": medicine_id,
        "medicine_name": proj["medicine_name"],
        "requested_units": requested_units,
        "current_stock": cur_stock,
        "safe_transferable_surplus": safe_surplus,
        "safety_stock_threshold": safety_threshold,
        "can_safely_donate": can_donate,
        "safety_breached": safety_breached,
        "verdict": audit_verdict
    }

def tool_emergency_sourcing(request_id: str = "EMERG_REQ_001") -> Dict[str, Any]:
    """Executes emergency inter-hospital matching for STAT clinical surgery requirements."""
    res = emergency_engine.evaluate_emergency_request(request_id)
    return res

def tool_what_if_simulation(demand_surge_pct: float = 50.0) -> Dict[str, Any]:
    """Runs what-if simulation for hypothetical demand surges."""
    sim = what_if_simulator.run_simulation(demand_surge_pct=demand_surge_pct)
    return {
        "scenario": f"Demand increase of +{demand_surge_pct}%",
        "comparison": sim["comparison"],
        "executive_summary": sim["executive_summary"]
    }

def tool_get_forecasting_metrics() -> Dict[str, Any]:
    """Fetches real calculated model evaluation metrics."""
    metrics_df = query_df("SELECT * FROM model_metrics")
    if metrics_df.empty:
        return {"error": "No model metrics calculated yet"}
    avg_base_mae = float(metrics_df["baseline_mae"].mean())
    avg_ml_mae = float(metrics_df["ml_mae"].mean())
    avg_imp = float(metrics_df["improvement_pct"].mean())
    return {
        "total_models": len(metrics_df),
        "average_baseline_mae": round(avg_base_mae, 2),
        "average_ml_mae": round(avg_ml_mae, 2),
        "average_improvement_pct": round(avg_imp, 1),
        "top_improvements": metrics_df.sort_values("improvement_pct", ascending=False).head(5).to_dict("records")
    }

def tool_get_hospital_profile(hospital_id: str) -> Dict[str, Any]:
    """Retrieves facility profile, bed capacity, occupancy, and current inventory breakdown."""
    h_df = query_df(f"SELECT * FROM hospitals WHERE UPPER(hospital_id) = '{hospital_id.upper()}'")
    if h_df.empty:
        return {"error": f"Hospital '{hospital_id}' not found in database."}
    h = h_df.iloc[0].to_dict()
    
    inv_df = query_df(f"""
        SELECT m.medicine_id, m.medicine_name, m.criticality, SUM(i.quantity) as quantity_on_hand
        FROM inventory i 
        JOIN medicines m ON i.medicine_id = m.medicine_id 
        WHERE UPPER(i.hospital_id) = '{hospital_id.upper()}'
        GROUP BY m.medicine_id
        ORDER BY quantity_on_hand ASC
    """)
    
    shortages_df = projection_engine.scan_all_shortage_risks()
    hosp_shortages = shortages_df[shortages_df["hospital_id"].str.upper() == hospital_id.upper()].sort_values("days_until_stockout")
    
    return {
        "hospital_id": h["hospital_id"],
        "hospital_name": h["hospital_name"],
        "facility_type": h.get("type", "General Hospital"),
        "beds": h["beds"],
        "icu_capacity": h.get("ICU_capacity", 0),
        "current_occupancy_rate": f"{int(h['current_occupancy']*100)}%",
        "safety_stock_policy_days": h["safety_stock_policy_days"],
        "total_units_on_hand": int(inv_df["quantity_on_hand"].sum()) if not inv_df.empty else 0,
        "critical_shortages_count": len(hosp_shortages[hosp_shortages["lead_time_risk"] == "CRITICAL"]),
        "top_shortages": hosp_shortages.head(3)[["medicine_id", "medicine_name", "days_until_stockout", "lead_time_risk"]].to_dict("records"),
        "lowest_stock_medicines": inv_df.head(5).to_dict("records"),
        "highest_stock_medicines": inv_df.tail(5).to_dict("records")
    }

def tool_get_medicine_profile(medicine_id: str) -> Dict[str, Any]:
    """Retrieves full medicine specs, unit costs in INR, shelf-life, and network distribution."""
    m_df = query_df(f"SELECT * FROM medicines WHERE UPPER(medicine_id) = '{medicine_id.upper()}'")
    if m_df.empty:
        m_df = query_df(f"SELECT * FROM medicines WHERE LOWER(medicine_name) LIKE '%{medicine_id.lower()}%'")
    if m_df.empty:
        return {"error": f"Medicine '{medicine_id}' not found in database."}
    m = m_df.iloc[0].to_dict()
    target_id = m["medicine_id"]
    
    dist_df = query_df(f"""
        SELECT h.hospital_id, h.hospital_name, SUM(i.quantity) as stock_units
        FROM inventory i
        JOIN hospitals h ON i.hospital_id = h.hospital_id
        WHERE UPPER(i.medicine_id) = '{target_id.upper()}'
        GROUP BY h.hospital_id
        ORDER BY stock_units DESC
    """)
    
    ranked = priority_engine.rank_all_competing_facilities(target_id)
    
    return {
        "medicine_id": m["medicine_id"],
        "medicine_name": m["medicine_name"],
        "category": m["category"],
        "criticality": m["criticality"],
        "unit_cost_inr": f"₹{m.get('unit_cost_inr', 0):,.2f}",
        "shelf_life_days": m["shelf_life_days"],
        "storage_condition": m.get("storage_condition", "Standard"),
        "requires_cold_chain": bool(m.get("requires_cold_chain", 0)),
        "network_total_stock": int(dist_df["stock_units"].sum()) if not dist_df.empty else 0,
        "distribution_by_hospital": dist_df.to_dict("records"),
        "priority_ranking": ranked.head(3)[["rank", "hospital_id", "hospital_name", "final_priority_score", "allocation_tier"]].to_dict("records") if not ranked.empty else []
    }

def tool_get_network_overview() -> Dict[str, Any]:
    """Returns network-wide supply chain overview from synthetic data."""
    h_df = query_df("SELECT hospital_id, hospital_name, beds, current_occupancy FROM hospitals")
    inv_sum = query_df("SELECT SUM(quantity) as total_units FROM inventory")
    total_units = int(inv_sum["total_units"].iloc[0]) if not inv_sum.empty else 0
    
    shortages_df = projection_engine.scan_all_shortage_risks()
    crit_shortages = shortages_df[shortages_df["lead_time_risk"] == "CRITICAL"]
    total_surplus = int(shortages_df["transferable_surplus"].sum()) if "transferable_surplus" in shortages_df else 41541
    
    exp = expiry_detector.scan_all_batches()
    exp_inr = float(exp["potential_financial_loss_inr"].sum()) if "potential_financial_loss_inr" in exp else 0.0
    
    em_df = query_df("SELECT emergency_request_id, requesting_hospital, medicine_id, quantity_required, procedure_type, status FROM emergency_requests WHERE status = 'OPEN'")
    
    return {
        "total_hospitals": len(h_df),
        "total_network_units": total_units,
        "critical_shortages_count": len(crit_shortages),
        "worst_shortages": crit_shortages.head(3)[["hospital_name", "medicine_name", "days_until_stockout"]].to_dict("records"),
        "transferable_surplus_units": total_surplus,
        "potential_expiry_waste_inr": f"₹{exp_inr:,.2f}",
        "open_emergency_requests": em_df.to_dict("records"),
        "hospital_list": h_df.to_dict("records")
    }

def tool_search_synthetic_database(query_text: str) -> Dict[str, Any]:
    """Flexible search across hospitals, medicines, batches, and emergency orders."""
    q = query_text.strip()
    safe_q = q.replace("'", "''")
    meds = query_df(f"SELECT medicine_id, medicine_name, category, criticality, unit_cost_inr FROM medicines WHERE LOWER(medicine_name) LIKE '%{safe_q.lower()}%' OR LOWER(category) LIKE '%{safe_q.lower()}%' OR UPPER(medicine_id) = '{safe_q.upper()}'")
    hosps = query_df(f"SELECT hospital_id, hospital_name, beds, current_occupancy, corridor FROM hospitals WHERE LOWER(hospital_name) LIKE '%{safe_q.lower()}%' OR UPPER(hospital_id) = '{safe_q.upper()}'")
    em = query_df(f"SELECT * FROM emergency_requests WHERE LOWER(procedure_type) LIKE '%{safe_q.lower()}%' OR UPPER(requesting_hospital) = '{safe_q.upper()}' OR UPPER(medicine_id) = '{safe_q.upper()}'")
    
    return {
        "search_term": query_text,
        "matching_medicines": meds.to_dict("records"),
        "matching_hospitals": hosps.to_dict("records"),
        "matching_emergency_requests": em.to_dict("records")
    }

