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
    tool_get_forecasting_metrics,
    tool_get_hospital_profile,
    tool_get_medicine_profile,
    tool_get_network_overview,
    tool_search_synthetic_database
)

SYNTHETIC_DATA_BASELINE = """
=== MEDINATIVE REGIONAL SYNTHETIC HEALTHCARE NETWORK DATABASE ===
HOSPITAL FACILITIES (10 Regional Network Nodes):
- Hospital A (HOSP_A): Level 1 Trauma, 750 beds, 88% occupancy, 18,833 units total stock, CRITICAL shortage risk on MED004 (stockout in 4.2 days vs 7-day supplier lead time), safety stock: 5 days. Needs 2,500 units redistributed from Hospital B.
- Hospital B (HOSP_B): Academic Medical Center, 520 beds, 72% occupancy, 18,214 units stock, SAFE status, primary donor hub with 12,400+ units of safe transferable surplus. Recommended donor for Hospital A.
- Hospital C (HOSP_C): Specialty Surgical Institute, 310 beds, 68% occupancy, 11,150 units stock, SAFE status, holds 2,800+ transferable units.
- Hospital D (HOSP_D): Community Acute Hospital, 240 beds, 93% occupancy, 5,784 units stock, CRITICAL status. Active emergency request EMERG_REQ_002: 800 units MED004 needed for Mass Casualty Multi-Trauma Resuscitation.
- Hospital E (HOSP_E): Trauma & Emergency Hospital, 380 beds, 89% occupancy, 8,639 units stock, HIGH risk.
- Hospital F (HOSP_F): Rural Specialty Care, 120 beds, 70% occupancy, 2,762 units stock, MODERATE risk.
- Hospital G (HOSP_G): Pediatric & General Facility, 200 beds, 75% occupancy, 5,591 units stock. Active emergency request EMERG_REQ_003: 80 units MED018 (Critical-Biologic-A, ₹18,500/unit) for Severe Immunologic Crisis.
- Hospital H (HOSP_H): Cardiac & Critical Center, 340 beds, 82% occupancy, 7,604 units stock, CRITICAL. Active emergency request EMERG_REQ_001: 500 units MED008 needed by tomorrow 10:00 AM for Emergency Aortic Repair / Reversal.
- Hospital I (HOSP_I): Suburban Satellite Hospital, 160 beds, 69% occupancy, 3,635 units stock.
- Hospital J (HOSP_J): Metropolitan General Hospital, 420 beds, 74% occupancy, 10,691 units stock, SAFE status.

MEDICINES CATALOG (20 Monitored Formulary Medicines):
- MED001: Antibiotic-A (Broad-Spectrum Antibiotic, Crit: 4, Cost: ₹850, Shelf life: 270d)
- MED002: Analgesic-A (Critical Analgesic & Sedation, Crit: 4, Cost: ₹280, Shelf life: 365d)
- MED003: Insulin-A (Metabolic & Endocrine, Crit: 5, Cost: ₹420, Shelf life: 180d, Cold Chain: Yes)
- MED004: Emergency-Medicine-A (Emergency Resuscitation & Vasopressor, Crit: 5, Cost: ₹350, Shelf life: 365d)
- MED005: IV-Fluid-A (High-Volume Hydration Fluid, Crit: 3, Cost: ₹55, Shelf life: 730d)
- MED006: Surgical-Supply-A (Surgical Hemostatic Matrix, Crit: 4, Cost: ₹3,200, Shelf life: 365d)
- MED007: Antiviral-A (Targeted Antiviral Infusion, Crit: 4, Cost: ₹4,800, Shelf life: 180d)
- MED008: Emergency-Injection-A (Urgent Surgery Reversal & Antidote, Crit: 5, Cost: ₹2,400, Shelf life: 240d)
- MED009: Sedative-A (ICU Sedation & Anesthesia, Crit: 4, Cost: ₹380, Shelf life: 365d)
- MED010: Anticoagulant-A (Critical Cardiovascular Anticoagulant, Crit: 4, Cost: ₹290, Shelf life: 540d)
- MED011: Antibiotic-B (Second-Line Antibiotic, Crit: 3, Cost: ₹620, Shelf life: 300d)
- MED012: IV-Fluid-B (Balanced Electrolyte Fluid, Crit: 3, Cost: ₹65, Shelf life: 730d)
- MED013: Sedative-B (Short-Acting Sedative Anxiolytic, Crit: 3, Cost: ₹190, Shelf life: 365d)
- MED014: Antiviral-B (Oral Antiviral Capsule, Crit: 3, Cost: ₹1,450, Shelf life: 365d)
- MED015: Cardiovascular-A (Antiarrhythmic Injection, Crit: 4, Cost: ₹340, Shelf life: 365d)
- MED016: Respiratory-A (Emergency Bronchodilator Nebulizer, Crit: 4, Cost: ₹280, Shelf life: 450d)
- MED017: Dialysis-Supply-A (Renal Replacement Solution, Crit: 4, Cost: ₹1,200, Shelf life: 540d)
- MED018: Critical-Biologic-A (Specialized Monoclonal Immunoglobulin, Crit: 5, Cost: ₹18,500, Shelf life: 90d, Cold Chain: Yes)
- MED019: Surgical-Suture-A (Vascular Synthetic Hemostatic Suture, Crit: 3, Cost: ₹1,800, Shelf life: 730d)
- MED020: Burn-Care-A (Advanced Antimicrobial Wound Dressing, Crit: 3, Cost: ₹750, Shelf life: 540d)

NETWORK TOTALS & LIVE METRICS:
- Total Network Inventory: 92,903 units across all 10 hospitals.
- Critical Shortages: 20 medicine/facility alerts (worst: Hospital A on MED004 with 4.2 days remaining).
- Transferable Surplus: 41,541 units available across safe facilities (Hospital B has largest surplus).
- Expiry / Wastage Risk: ₹16,544,250 worth of batches at risk within 30 days.
- Active Emergency Orders: 3 open orders (Hospital H, Hospital D, Hospital G).
- Primary Recommended Transfer: Hospital B transfers 2,500 units of MED004 to Hospital A (transit: 48 mins).
=============================================================
"""

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

        # Intent H: Specific Hospital Inquiry
        elif any(f"hospital {h.lower()}" in q_lower or f"hosp_{h.lower()}" in q_lower for h in ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]):
            tool_name = "hospital_profile"
            hosp_id = "HOSP_A"
            for h in ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]:
                if f"hospital {h.lower()}" in q_lower or f"hosp_{h.lower()}" in q_lower:
                    hosp_id = f"HOSP_{h}"
                    break
            tool_data = tool_get_hospital_profile(hosp_id)

        # Intent I: Specific Medicine Inquiry
        elif re.search(r"med\d{3}", q_lower) or any(w in q_lower for w in ["antibiotic", "analgesic", "insulin", "iv-fluid", "surgical", "antiviral", "sedative", "anticoagulant", "biologic", "suture"]):
            tool_name = "medicine_profile"
            med_match = re.search(r"med\d{3}", q_lower)
            if med_match:
                med_id = med_match.group(0).upper()
            else:
                for w in ["antibiotic", "analgesic", "insulin", "iv-fluid", "surgical", "antiviral", "sedative", "anticoagulant", "biologic", "suture"]:
                    if w in q_lower:
                        med_id = w
                        break
                else:
                    med_id = "MED004"
            tool_data = tool_get_medicine_profile(med_id)

        # Intent J: Network Overview / Total Synthetic Data
        elif any(w in q_lower for w in ["network", "total", "overview", "summary", "synthetic", "data", "facilities", "all hospitals", "inventory"]):
            tool_name = "network_overview"
            tool_data = tool_get_network_overview()

        else:
            # Fallback search across synthetic database
            tool_name = "synthetic_search"
            tool_data = tool_search_synthetic_database(user_question)
            if not tool_data.get("matching_medicines") and not tool_data.get("matching_hospitals"):
                tool_name = "network_overview"
                tool_data = tool_get_network_overview()

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
                    "You are the Medinative Clinical Supply Intelligence AI Assistant for this regional hospital network.\n"
                    "CRITICAL INSTRUCTION: You have FULL AND DIRECT access to the regional hospital network's live synthetic operations database.\n"
                    "NEVER state 'I do not have real-time data access' or 'I am an AI without access to hospital inventory'. You have the complete database provided below.\n"
                    "Answer the user's question directly, clinically, and factually using the synthetic network baseline and retrieved system facts.\n"
                    "Always mention specific facility names (e.g. Hospital A, Hospital B), medicine names/IDs, unit quantities, and costs in Indian Rupees (₹).\n\n"
                    f"{SYNTHETIC_DATA_BASELINE}"
                )
                user_prompt = f"User Question: {user_question}\n\nRetrieved Analytical System Facts:\n{json.dumps(tool_data, indent=2)}\n\nPlease provide a clear, professional, clinical explanation."
                
                resp = requests.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": self.model_name,
                        "prompt": f"{system_prompt}\n\n{user_prompt}",
                        "stream": False
                    },
                    timeout=45
                )
                if resp.status_code == 200:
                    data = resp.json()
                    res_text = data.get("response", "").strip()
                    if res_text:
                        return res_text
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

        elif tool_name == "hospital_profile":
            h_name = tool_data.get("hospital_name", "Hospital")
            beds = tool_data.get("beds", 0)
            occ = tool_data.get("current_occupancy_rate", "N/A")
            units = tool_data.get("total_units_on_hand", 0)
            crit_cnt = tool_data.get("critical_shortages_count", 0)
            top_s = tool_data.get("top_shortages", [])
            s_lines = [f"- **{s['medicine_name']} ({s['medicine_id']})**: Stockout in **{s['days_until_stockout']} days** (Risk: {s['lead_time_risk']})" for s in top_s]
            s_text = "\n".join(s_lines) if s_lines else "None - all stock levels healthy."
            return (
                f"### Facility Profile: {h_name}\n\n"
                f"- **Capacity**: {beds} beds | ICU: {tool_data.get('icu_capacity')} beds\n"
                f"- **Current Occupancy**: {occ}\n"
                f"- **Safety Stock Policy**: {tool_data.get('safety_stock_policy_days')} days mandatory reserve\n"
                f"- **Total On-Hand Inventory**: **{units:,} units** across 20 monitored medicines\n"
                f"- **Critical Stockout Alerts**: **{crit_cnt} alerts**\n\n"
                f"#### Shortage Projections:\n{s_text}"
            )

        elif tool_name == "medicine_profile":
            m_name = tool_data.get("medicine_name", "Medicine")
            m_id = tool_data.get("medicine_id", "")
            cat = tool_data.get("category", "")
            crit = tool_data.get("criticality", 3)
            cost = tool_data.get("unit_cost_inr", "₹0.00")
            stock = tool_data.get("network_total_stock", 0)
            cold = "Yes (2–8°C)" if tool_data.get("requires_cold_chain") else "Standard Ambient"
            dist = tool_data.get("distribution_by_hospital", [])
            top_h = ", ".join([f"{d['hospital_name']} ({d['stock_units']:,}u)" for d in dist[:4]])
            return (
                f"### Medicine Profile: {m_name} (`{m_id}`)\n\n"
                f"- **Therapeutic Category**: {cat}\n"
                f"- **Clinical Criticality Tier**: **Level {crit}/5**\n"
                f"- **Formulary Unit Cost**: **{cost}**\n"
                f"- **Shelf Life**: {tool_data.get('shelf_life_days')} days\n"
                f"- **Cold Chain Requirement**: {cold}\n"
                f"- **Network-Wide Total Inventory**: **{stock:,} units**\n\n"
                f"**Top Facility Stock**: {top_h}"
            )

        elif tool_name == "network_overview":
            tot_h = tool_data.get("total_hospitals", 10)
            tot_u = tool_data.get("total_network_units", 0)
            crit_c = tool_data.get("critical_shortages_count", 0)
            surplus = tool_data.get("transferable_surplus_units", 0)
            waste = tool_data.get("potential_expiry_waste_inr", "₹0.00")
            open_em = tool_data.get("open_emergency_requests", [])
            em_lines = [f"- **{e['emergency_request_id']}**: {e['requesting_hospital']} requested {e['quantity_required']} units of {e['medicine_id']} for *{e['procedure_type']}*" for e in open_em]
            em_text = "\n".join(em_lines) if em_lines else "No open emergency orders."
            return (
                f"### Medinative Regional Healthcare Network Overview\n\n"
                f"- **Monitored Facilities**: {tot_h} regional hospitals (Hospital A through Hospital J)\n"
                f"- **Total Network Inventory**: **{tot_u:,} units** across 20 monitored medicines\n"
                f"- **Active Critical Shortages**: **{crit_c} alerts** requiring intervention\n"
                f"- **Safe Transferable Surplus**: **{surplus:,} units** (Hospital B primary donor)\n"
                f"- **Potential 30-Day Expiry Risk**: **{waste}**\n\n"
                f"#### Active Emergency Requests ({len(open_em)}):\n{em_text}"
            )

        elif tool_name == "synthetic_search":
            meds = tool_data.get("matching_medicines", [])
            hosps = tool_data.get("matching_hospitals", [])
            lines = [f"### Database Search Results for '{tool_data.get('search_term')}'\n"]
            if hosps:
                lines.append("**Hospitals:**")
                for h in hosps:
                    lines.append(f"- **{h['hospital_name']} ({h['hospital_id']})**: {h['beds']} beds, {int(h['current_occupancy']*100)}% occupancy, {h['corridor']}")
            if meds:
                lines.append("\n**Medicines:**")
                for m in meds:
                    lines.append(f"- **{m['medicine_name']} ({m['medicine_id']})**: {m['category']}, Criticality {m['criticality']}/5, ₹{m['unit_cost_inr']:,.2f}")
            return "\n".join(lines)

        return "I have evaluated the system state based on current inventory, forecasts, and optimization models."
