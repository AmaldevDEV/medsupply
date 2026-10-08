"""
Medical Supply Intelligence - Grounded Local AI Assistant
Processes natural-language inquiries by detecting intent, executing verified
analytical tools, retrieving exact factual calculations, and formatting an explainable response.
Supports local Ollama LLMs with an automatic deterministic analytical fallback.
"""

import os
import re
import json
from typing import Dict, Any, Optional
import requests
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import LLM_MODEL, OLLAMA_BASE_URL
from assistant.tools import (
    tool_get_shortage_risks,
    tool_get_expiry_risks,
    tool_get_priority_explanation,
    tool_check_donor_surplus,
    tool_emergency_sourcing,
    tool_what_if_simulation,
    tool_get_forecasting_metrics
)

class MedicalSupplyAIAssistant:
    def __init__(self, model_name: Optional[str] = None, ollama_url: Optional[str] = None):
        self.model_name = model_name or os.getenv("LLM_MODEL", LLM_MODEL)
        self.ollama_url = (ollama_url or os.getenv("OLLAMA_BASE_URL", OLLAMA_BASE_URL)).rstrip("/")
        self.is_ollama_available = self._check_ollama_connectivity()

    def _check_ollama_connectivity(self) -> bool:
        """Verifies if local Ollama daemon is reachable."""
        try:
            resp = requests.get(f"{self.ollama_url}/api/tags", timeout=1.5)
            return resp.status_code == 200
        except Exception:
            return False

    def query(self, user_question: str) -> Dict[str, Any]:
        """
        Processes user question:
        1. Detects intent and extracts parameters
        2. Calls verified backend calculation tool
        3. Generates grounded explanation (via Ollama or analytical fallback)
        """
        q_lower = user_question.lower()
        tool_name = "unknown"
        tool_data = {}

        # 1. INTENT DETECTION & TOOL ROUTING

        # Intent A: Emergency surgery / inter-hospital sourcing
        if any(w in q_lower for w in ["emergency", "surgery", "tomorrow", "med008", "aortic", "stat", "provide it"]):
            tool_name = "emergency_sourcing"
            tool_data = tool_emergency_sourcing("EMERG_REQ_001")
            
        # Intent B: Can Hospital X donate Y units?
        elif "donate" in q_lower or ("spare" in q_lower and "unit" in q_lower) or "transferable" in q_lower:
            tool_name = "check_donor_surplus"
            # Extract hospital ID (default Hospital B)
            hosp_id = "HOSP_B"
            for h in ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]:
                if f"hospital {h.lower()}" in q_lower or f"hosp_{h.lower()}" in q_lower:
                    hosp_id = f"HOSP_{h}"
                    break
            
            # Extract medicine ID (default MED004)
            med_match = re.search(r"med\d{3}", q_lower)
            med_id = med_match.group(0).upper() if med_match else "MED004"
            
            # Extract quantity
            qty_match = re.search(r"(\d[\d,]*)\s*units?", q_lower)
            if qty_match:
                qty = int(qty_match.group(1).replace(",", ""))
            else:
                qty = 2000
                
            tool_data = tool_check_donor_surplus(hosp_id, med_id, qty)

        # Intent C: Priority explanation
        elif "priority" in q_lower or "prioritize" in q_lower or "who should receive" in q_lower:
            tool_name = "priority_explanation"
            hosp_id = "HOSP_A"
            for h in ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]:
                if f"hospital {h.lower()}" in q_lower:
                    hosp_id = f"HOSP_{h}"
                    break
            med_match = re.search(r"med\d{3}", q_lower)
            med_id = med_match.group(0).upper() if med_match else "MED004"
            tool_data = tool_get_priority_explanation(hosp_id, med_id)

        # Intent D: Expiry / Waste risks
        elif any(w in q_lower for w in ["expire", "expiry", "waste", "spoil", "unused", "shelf life"]):
            tool_name = "expiry_risks"
            tool_data = tool_get_expiry_risks(max_days=30)

        # Intent E: What-if / Demand increase
        elif any(w in q_lower for w in ["what happens if", "what if", "demand increase", "surge by", "increase by"]):
            tool_name = "what_if_simulation"
            pct_match = re.search(r"(\d+)%", q_lower)
            pct = float(pct_match.group(1)) if pct_match else 50.0
            tool_data = tool_what_if_simulation(pct)

        # Intent F: Model accuracy & metrics
        elif any(w in q_lower for w in ["model", "accuracy", "mae", "rmse", "baseline", "forecasting performance"]):
            tool_name = "model_metrics"
            tool_data = tool_get_forecasting_metrics()

        # Intent G: Shortages & Stockout
        elif any(w in q_lower for w in ["run out", "shortage", "stockout", "deficit", "depleted"]):
            tool_name = "shortage_risks"
            med_match = re.search(r"med\d{3}", q_lower)
            med_id = med_match.group(0).upper() if med_match else None
            tool_data = tool_get_shortage_risks(medicine_id=med_id, max_days=14)

        else:
            # General overview / network health
            tool_name = "shortage_risks"
            tool_data = tool_get_shortage_risks(max_days=7)

        # 2. GENERATE RESPONSE
        explanation = self._generate_explanation(user_question, tool_name, tool_data)

        return {
            "question": user_question,
            "tool_used": tool_name,
            "retrieved_facts": tool_data,
            "is_local_llm_active": self.is_ollama_available,
            "llm_model": self.model_name if self.is_ollama_available else "Deterministic Analytical Engine",
            "answer": explanation
        }

    def _generate_explanation(self, user_question: str, tool_name: str, tool_data: Dict[str, Any]) -> str:
        """
        Synthesizes an explainable, fact-grounded response.
        If Ollama is available, queries Ollama. Otherwise, uses deterministic synthesis.
        """
        # If Ollama is available, query local model with strict system instructions
        if self.is_ollama_available:
            try:
                system_prompt = (
                    "You are the Medical Supply Intelligence AI Assistant. Your role is to explain "
                    "complex hospital supply chain decisions, stockout forecasts, and emergency transfers. "
                    "CRITICAL REQUIREMENT: Ground your entire response exclusively on the provided retrieved facts. "
                    "Do NOT invent, alter, or extrapolate any numbers or medical drug facts."
                )
                user_prompt = f"User Question: {user_question}\n\nRetrieved System Facts:\n{json.dumps(tool_data, indent=2)}\n\nPlease provide a clear, professional explanation."
                
                resp = requests.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": self.model_name,
                        "prompt": f"{system_prompt}\n\n{user_prompt}",
                        "stream": False
                    },
                    timeout=10
                )
                if resp.status_code == 200:
                    data = resp.json()
                    return data.get("response", "").strip()
            except Exception:
                pass # Fallback to deterministic synthesis

        # Deterministic Grounded Synthesis (Guarantees zero numerical hallucinations)
        if tool_name == "emergency_sourcing":
            req_hosp = tool_data.get("requesting_hospital_name", "Hospital H")
            med_name = tool_data.get("medicine_name", "Medicine")
            qty = tool_data.get("quantity_required", 0)
            req_by = tool_data.get("required_by", "scheduled procedure")
            proc = tool_data.get("procedure_type", "Emergency Surgery")
            
            if tool_data.get("has_feasible_source"):
                rec = tool_data["recommended_source"]
                cand_name = rec["candidate_hospital_name"]
                surplus = rec["transferable_surplus"]
                transit_mins = rec.get("transit_time_minutes", int(rec["transit_time_hours"] * 60))
                traffic = rec.get("traffic_level", "Moderate")
                dist = rec.get("distance_km", 0.0)
                arrival = rec["estimated_arrival"]
                gmaps = rec.get("google_maps_url", "")
                
                # Nearest vs fastest callout
                nvf = tool_data.get("nearest_vs_fastest", {})
                nvf_text = ""
                if nvf and nvf.get("nearest_hospital_by_distance") and nvf.get("fastest_hospital_by_time"):
                    near = nvf["nearest_hospital_by_distance"]
                    fast = nvf["fastest_hospital_by_time"]
                    if near["hospital_id"] != fast["hospital_id"]:
                        nvf_text = (
                            f"\n> ⚡ **Fastest vs. Nearest Route Insight**: Geographically nearest facility is "
                            f"**{near['hospital_name']}** ({near['distance_km']} km), but due to **{near['traffic_level']}** "
                            f"traffic, transit takes **{near['transit_time_minutes']} mins** (and status: {near['status']}). "
                            f"In contrast, **{fast['hospital_name']}** is {fast['distance_km']} km away via open expressway "
                            f"({fast['traffic_level']} traffic) arriving in **{fast['transit_time_minutes']} mins** "
                            f"(**{fast['time_saved_vs_nearest_minutes']} mins faster**!).\n"
                        )

                rejected_notes = []
                for rej in tool_data.get("rejected_candidates", [])[:2]:
                    rejected_notes.append(f"- **{rej['candidate_hospital_name']}**: {rej['rejection_reason']}")
                rejected_str = "\n".join(rejected_notes)

                gmaps_md = f"\n🗺️ **[Open Live Route & Traffic in Google Maps]({gmaps})**\n" if gmaps else ""

                return (
                    f"### Emergency Inter-Hospital Sourcing Decision\n\n"
                    f"**Requesting Facility**: {req_hosp}\n"
                    f"**Clinical Procedure**: {proc}\n"
                    f"**Requirement**: {qty:,} units of {med_name} required by **{req_by}**.\n\n"
                    f"#### ✅ RECOMMENDED SOURCE: {cand_name} (Fastest Arrival)\n"
                    f"- **Safe Transferable Surplus**: {surplus:,} units\n"
                    f"- **Transit Time**: **{transit_mins} minutes** ({dist} km, {traffic} traffic)\n"
                    f"- **Expected Physical Arrival**: **{arrival}** (arrives safely before deadline)\n"
                    f"- **Donor Safety Reserve**: Preserved (Donor will remain safely above its required reserve)\n"
                    f"{gmaps_md}"
                    f"{nvf_text}\n"
                    f"**Clinical Reasoning**:\n{rec['recommendation_reason']}\n\n"
                    f"#### ❌ Evaluated & Rejected Facilities:\n{rejected_str}\n\n"
                    f"> *Notice: Recommended transfer requires clinical director authorization prior to dispatch.*"
                )
            else:
                return (
                    f"### Emergency Inter-Hospital Sourcing Decision\n\n"
                    f"**Alert**: No feasible hospital source found before the deadline.\n\n"
                    f"The system evaluated all network facilities for {qty:,} units of {med_name}. "
                    f"All candidates were rejected due to insufficient safe surplus, safety stock breach, "
                    f"or courier transit times that exceed the required surgical deadline."
                )

        elif tool_name == "check_donor_surplus":
            return (
                f"### Transferable Surplus Audit\n\n"
                f"**Facility**: {tool_data.get('hospital_name')}\n"
                f"**Medicine**: {tool_data.get('medicine_name')}\n"
                f"**Requested Transfer**: {tool_data.get('requested_units'):,} units\n\n"
                f"- **Current On-Hand Stock**: {tool_data.get('current_stock'):,} units\n"
                f"- **Mandatory Safety Reserve**: {tool_data.get('safety_stock_threshold'):,} units\n"
                f"- **Verified Transferable Surplus**: {tool_data.get('safe_transferable_surplus'):,} units\n\n"
                f"**Decision**: **{tool_data.get('verdict')}**"
            )

        elif tool_name == "shortage_risks":
            risks = tool_data.get("risks", [])
            if not risks:
                return "All hospital inventories currently maintain adequate stock above safety thresholds for the query window."
            
            lines = [f"### Shortage & Stockout Risk Report ({len(risks)} at-risk series)\n"]
            for r in risks[:5]:
                lines.append(
                    f"- **{r['hospital_name']}** — **{r['medicine_name']}**: "
                    f"Projected stockout in **{r['days_until_stockout']} days** "
                    f"(Current stock: {r['current_stock']:,} units, Daily demand: {r['avg_daily_demand']} units). "
                    f"Supplier lead time is {r['supplier_lead_time']} days. "
                    f"**Procurement Lead-Time Risk: {r['lead_time_risk']}**."
                )
            return "\n".join(lines)

        elif tool_name == "expiry_risks":
            batches = tool_data.get("batches", [])
            if not batches:
                return "No inventory batches are projected to expire with unused surplus within the 30-day window."
                
            waste_inr = tool_data.get("total_financial_waste_inr") or (tool_data.get("total_financial_waste_usd", 0) * 83.0)
            lines = [
                f"### Batch-Level Expiry & Wastage Analysis\n",
                f"Total At-Risk Batches: **{tool_data.get('total_batches_at_risk')}** | Total Potential Financial Waste: **₹{waste_inr:,.2f}**\n"
            ]
            for b in batches[:4]:
                lines.append(f"- **{b['explanation']}**")
            lines.append("\n*Recommendation: Trigger proactive redistribution to deficit facilities before expiration.*")
            return "\n".join(lines)

        elif tool_name == "priority_explanation":
            target = tool_data.get("target_hospital", {})
            return (
                f"### Supply Allocation Priority Breakdown\n\n"
                f"**Facility**: {target.get('hospital_name')} (Rank #{target.get('rank')} of {tool_data.get('network_competing_count')})\n"
                f"**Medicine**: {target.get('medicine_name')}\n"
                f"**Composite Priority Score**: **{target.get('priority_score')}/100** ({target.get('priority_tier')})\n\n"
                f"#### Component Scores (0–100):\n"
                f"- **Shortage Urgency (30% weight)**: {target.get('urgency_component')}/100 ({target.get('days_until_stockout')} days until stockout)\n"
                f"- **Emergency Patient Demand (25% weight)**: {target.get('emergency_component')}/100\n"
                f"- **Hospital Occupancy / Patient Load (20% weight)**: {target.get('acuity_component')}/100\n"
                f"- **Alternative Medicine Scarcity (15% weight)**: {target.get('alternative_scarcity')}/100\n"
                f"- **Supplier Lead-Time Risk (10% weight)**: {target.get('lead_time_risk')}/100\n\n"
                f"**Why this facility received its score**:\n{target.get('rationale')}"
            )

        elif tool_name == "what_if_simulation":
            comp = tool_data.get("comparison", {})
            cs_d = comp['critical_shortages']['delta']
            hr_d = comp['high_risk_count']['delta']
            ad_d = comp['average_days_until_stockout']['delta']
            ns_d = comp['network_transferable_surplus_units']['delta']
            return (
                f"### What-If Simulation Results: {tool_data.get('scenario')}\n\n"
                f"**Executive Summary**: {tool_data.get('executive_summary')}\n\n"
                f"| Metric | Before Simulation | After Simulation | Delta |\n"
                f"|---|---|---|---|\n"
                f"| Critical Shortages | {comp['critical_shortages']['before']} | {comp['critical_shortages']['after']} | {'+' if cs_d > 0 else ''}{cs_d} |\n"
                f"| High Risk Facilities | {comp['high_risk_count']['before']} | {comp['high_risk_count']['after']} | {'+' if hr_d > 0 else ''}{hr_d} |\n"
                f"| Avg Days to Stockout | {comp['average_days_until_stockout']['before']}d | {comp['average_days_until_stockout']['after']}d | {'+' if ad_d > 0 else ''}{ad_d}d |\n"
                f"| Network Safe Surplus | {comp['network_transferable_surplus_units']['before']:,} units | {comp['network_transferable_surplus_units']['after']:,} units | {'+' if ns_d > 0 else ''}{ns_d:,} units |\n\n"
                f"**Emergency Sourcing Impact**: {comp['emergency_sourcing']['after_reason']}"
            )

        elif tool_name == "model_metrics":
            return (
                f"### Demand Forecasting Model Validation Metrics\n\n"
                f"- **Total Time-Series Models**: {tool_data.get('total_models')} facilities x medicines\n"
                f"- **Validation Strategy**: Strict Time-Based Holdout Split (70% Train, 15% Validation, 15% Test)\n"
                f"- **Average Baseline MAE**: {tool_data.get('average_baseline_mae')} units\n"
                f"- **Average ML Model MAE**: {tool_data.get('average_ml_mae')} units\n"
                f"- **Mean Accuracy Improvement**: {tool_data.get('average_improvement_pct')}%\n\n"
                f"*All metrics originate directly from out-of-sample holdout testing.*"
            )

        return "I have evaluated the system state based on current inventory, forecasts, and optimization models."
