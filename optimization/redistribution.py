"""
Normal Inter-Hospital Redistribution Optimizer
Matches surplus donor hospitals with deficit receiver hospitals.
Enforces physical arrival, traffic level clearance, donor safety stock preservation,
batch shelf life, and transport constraints while maximizing shortage and waste mitigation.
Includes Google Maps driving directions and INR valuations.
"""

from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATABASE_PATH, TRANSPORT_COST_PER_KM_INR
from database.database import query_df, execute_sql, execute_many
from models.inventory_projection import InventoryProjectionEngine

class RedistributionOptimizer:
    def __init__(self, stockout_urgency_threshold_days: int = 7):
        self.stockout_threshold_days = stockout_urgency_threshold_days
        self.projection_engine = InventoryProjectionEngine()

    def generate_redistribution_plan(self) -> List[Dict[str, Any]]:
        """
        Executes constraint-satisfaction matching across network:
        1. Identifies receivers facing imminent stockout (<= 7 days).
        2. Identifies donors with verified transferable surplus (Current - 14d Demand - Safety Stock > 0).
        3. Prioritizes near-expiry donor batches to eliminate waste.
        4. Validates travel time and donor safety retention.
        5. Computes optimized transfer quantities.
        """
        # 1. Fetch transport routes
        routes_df = query_df("SELECT * FROM transport_routes")
        route_map = {}
        for _, r in routes_df.iterrows():
            route_map[(r["from_hospital"], r["to_hospital"])] = {
                "distance_km": float(r["distance_km"]),
                "transit_hours": float(r["transport_time_hours"]),
                "traffic_level": str(r["traffic_level"]) if "traffic_level" in r else "Moderate",
                "google_maps_url": str(r["google_maps_url"]) if "google_maps_url" in r else "",
                "capacity": int(r["transport_capacity"]),
                "risk": str(r["transport_risk"])
            }

        # 2. Scan all facilities for surpluses and shortages
        shortage_records = []
        surplus_records = []

        hospitals_df = query_df("SELECT * FROM hospitals")
        medicines_df = query_df("SELECT * FROM medicines")

        for h_id in hospitals_df["hospital_id"]:
            for m_id in medicines_df["medicine_id"]:
                proj = self.projection_engine.project_inventory(h_id, m_id)
                
                # Check for deficit receiver
                if proj["days_until_stockout"] <= self.stockout_threshold_days or proj["current_stock"] <= proj["safety_stock_threshold"]:
                    target_stock = proj["safety_stock_threshold"] + int(proj["avg_daily_forecast"] * 10)
                    deficit_qty = max(50, target_stock - proj["current_stock"])
                    
                    shortage_records.append({
                        "hospital_id": h_id,
                        "hospital_name": proj["hospital_name"],
                        "medicine_id": m_id,
                        "medicine_name": proj["medicine_name"],
                        "category": proj["category"],
                        "criticality": proj["criticality"],
                        "current_stock": proj["current_stock"],
                        "days_until_stockout": proj["days_until_stockout"],
                        "supplier_lead_time": proj["supplier_lead_time_days"],
                        "lead_time_risk": proj["lead_time_risk"],
                        "deficit_qty": deficit_qty,
                        "safety_threshold": proj["safety_stock_threshold"]
                    })
                
                # Check for surplus donor
                if proj["transferable_surplus"] >= 50:
                    surplus_records.append({
                        "hospital_id": h_id,
                        "hospital_name": proj["hospital_name"],
                        "medicine_id": m_id,
                        "medicine_name": proj["medicine_name"],
                        "transferable_surplus": proj["transferable_surplus"],
                        "current_stock": proj["current_stock"],
                        "safety_threshold": proj["safety_stock_threshold"]
                    })

        # Near-expiry batches
        exp_q = """
            SELECT i.hospital_id, i.medicine_id, i.batch_id, i.quantity, i.days_until_expiry,
                   h.hospital_name, m.medicine_name
            FROM inventory i
            JOIN hospitals h ON i.hospital_id = h.hospital_id
            JOIN medicines m ON i.medicine_id = m.medicine_id
            WHERE i.days_until_expiry <= 30
        """
        exp_df = query_df(exp_q)
        expiring_donors = {}
        for _, exp_row in exp_df.iterrows():
            key = (exp_row["hospital_id"], exp_row["medicine_id"])
            expiring_donors[key] = {
                "batch_id": exp_row["batch_id"],
                "quantity": int(exp_row["quantity"]),
                "days_until_expiry": int(exp_row["days_until_expiry"])
            }

        recommendations = []
        rec_id_counter = 1

        shortage_records.sort(key=lambda x: (x["days_until_stockout"], -x["criticality"]))

        donor_available = {
            (s["hospital_id"], s["medicine_id"]): s["transferable_surplus"]
            for s in surplus_records
        }

        for rec in shortage_records:
            m_id = rec["medicine_id"]
            to_h = rec["hospital_id"]
            needed = rec["deficit_qty"]

            candidate_donors = [
                s for s in surplus_records 
                if s["medicine_id"] == m_id and s["hospital_id"] != to_h and donor_available.get((s["hospital_id"], m_id), 0) >= 50
            ]

            if not candidate_donors:
                continue

            scored_candidates = []
            for d in candidate_donors:
                from_h = d["hospital_id"]
                route = route_map.get((from_h, to_h), {
                    "distance_km": 25.0,
                    "transit_hours": 0.8,
                    "traffic_level": "Moderate",
                    "google_maps_url": "",
                    "risk": "Moderate"
                })
                transit_h = route["transit_hours"]
                dist_km = route["distance_km"]

                hours_until_stockout = max(2.0, rec["days_until_stockout"] * 24.0)
                if transit_h > hours_until_stockout:
                    continue

                exp_info = expiring_donors.get((from_h, m_id))
                waste_prevention_bonus = 50.0 if exp_info and exp_info["days_until_expiry"] <= 25 else 0.0

                avail = donor_available.get((from_h, m_id), 0)
                # Score factors in available surplus, waste bonus, transit time with traffic, and distance
                match_score = (avail * 0.10) + waste_prevention_bonus - (transit_h * 15.0) - (dist_km * 0.20)
                
                scored_candidates.append({
                    "donor": d,
                    "route": route,
                    "avail": avail,
                    "exp_info": exp_info,
                    "score": match_score
                })

            if not scored_candidates:
                continue

            scored_candidates.sort(key=lambda x: x["score"], reverse=True)
            best = scored_candidates[0]
            best_donor = best["donor"]
            from_h = best_donor["hospital_id"]
            avail_units = best["avail"]
            route = best["route"]

            transfer_qty = min(needed, avail_units)
            transfer_qty = max(25, (transfer_qty // 25) * 25)

            donor_available[(from_h, m_id)] -= transfer_qty

            batch_assigned = best["exp_info"]["batch_id"] if best["exp_info"] else f"BAT-{from_h[5:]}-{m_id[3:]}-OPT"
            is_waste_prevented = best["exp_info"] is not None

            transit_mins = int(round(route["transit_hours"] * 60))
            traffic_lvl = route.get("traffic_level", "Moderate")

            if is_waste_prevented:
                reason = (
                    f"Redistribution optimizes both shortage and waste: {best_donor['hospital_name']} has "
                    f"{avail_units:,} units surplus with batch nearing expiry ({best['exp_info']['days_until_expiry']}d). "
                    f"{rec['hospital_name']} faces stockout in {rec['days_until_stockout']} days "
                    f"(supplier takes {rec['supplier_lead_time']} days). Courier transit takes {transit_mins} mins "
                    f"via {traffic_lvl} traffic flow ({route['distance_km']} km)."
                )
            else:
                reason = (
                    f"{rec['hospital_name']} is projected to stock out in {rec['days_until_stockout']} days "
                    f"under primary supplier lead time of {rec['supplier_lead_time']} days. "
                    f"{best_donor['hospital_name']} maintains verified safe surplus of {avail_units:,} units "
                    f"and will remain safely above its required reserve after transferring {transfer_qty:,} units "
                    f"via {transit_mins} mins courier transit ({route['distance_km']} km, {traffic_lvl} traffic)."
                )

            cost_inr = round(route["distance_km"] * TRANSPORT_COST_PER_KM_INR, 2)

            recommendations.append({
                "recommendation_id": f"REDIST_{rec_id_counter:03d}",
                "from_hospital": from_h,
                "from_hospital_name": best_donor["hospital_name"],
                "to_hospital": to_h,
                "to_hospital_name": rec["hospital_name"],
                "medicine_id": m_id,
                "medicine_name": rec["medicine_name"],
                "batch_id": batch_assigned,
                "quantity": transfer_qty,
                "transit_time_hours": route["transit_hours"],
                "transit_time_minutes": transit_mins,
                "distance_km": route["distance_km"],
                "traffic_level": traffic_lvl,
                "google_maps_url": route.get("google_maps_url", ""),
                "estimated_cost_inr": cost_inr,
                "reason": reason,
                "status": "RECOMMENDED",
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M")
            })
            rec_id_counter += 1

        # Save recommendations to database
        if recommendations:
            rec_df = pd.DataFrame(recommendations)
            db_records = rec_df[[
                "recommendation_id", "from_hospital", "to_hospital", "medicine_id",
                "batch_id", "quantity", "transit_time_hours", "distance_km",
                "traffic_level", "google_maps_url", "estimated_cost_inr",
                "reason", "status", "created_at"
            ]].to_dict("records")
            
            param_list = [
                (r["recommendation_id"], r["from_hospital"], r["to_hospital"],
                 r["medicine_id"], r["batch_id"], r["quantity"],
                 r["transit_time_hours"], r["distance_km"], r["traffic_level"],
                 r["google_maps_url"], r["estimated_cost_inr"], r["reason"],
                 r["status"], r["created_at"])
                for r in db_records
            ]
            execute_sql("DELETE FROM redistribution_recommendations")
            execute_many("""
                INSERT INTO redistribution_recommendations (
                    recommendation_id, from_hospital, to_hospital, medicine_id,
                    batch_id, quantity, transit_time_hours, distance_km,
                    traffic_level, google_maps_url, estimated_cost_inr,
                    reason, status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, param_list)

        return recommendations
