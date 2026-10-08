"""
Medical Supply Intelligence - FastAPI Backend
Provides RESTful APIs for inventory projections, shortage risks, expiry audits,
priority rankings, normal redistribution, emergency sourcing, simulation, and AI assistant.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import sys

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from database.database import query_df, execute_sql
from models.inventory_projection import InventoryProjectionEngine
from models.expiry_wastage import ExpiryWastageDetector
from optimization.priority import SupplyPriorityEngine
from optimization.redistribution import RedistributionOptimizer
from optimization.emergency_matching import EmergencySourcingEngine
from simulation.what_if import WhatIfSimulator
from simulation.outbreak import OutbreakSimulator
from assistant.agent import MedicalSupplyAIAssistant

app = FastAPI(
    title="Medical Supply Intelligence API",
    description="Explainable local decision-support system for hospital supply networks",
    version="1.0.0"
)

# Enable CORS for local dashboards and frontend clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Service Singletons
projection_engine = InventoryProjectionEngine()
expiry_detector = ExpiryWastageDetector()
priority_engine = SupplyPriorityEngine()
redist_optimizer = RedistributionOptimizer()
emergency_engine = EmergencySourcingEngine()
what_if_simulator = WhatIfSimulator()
outbreak_simulator = OutbreakSimulator()
ai_assistant = MedicalSupplyAIAssistant()

# --- Request / Response Models ---

class EmergencyRequestCreate(BaseModel):
    requesting_hospital: str = Field(..., example="HOSP_H")
    medicine_id: str = Field(..., example="MED008")
    quantity_required: int = Field(..., example=500)
    required_by: str = Field(..., example="2026-10-09 10:00")
    procedure_type: str = Field(..., example="Emergency Vascular Surgery")
    urgency_level: str = Field(..., example="STAT Critical")

class WhatIfRequest(BaseModel):
    demand_surge_pct: float = Field(default=0.0, example=50.0)
    outbreak_hospitals: List[str] = Field(default=[], example=["HOSP_D", "HOSP_E"])
    outbreak_severity_mult: float = Field(default=1.0, example=2.0)
    supplier_lead_time_added_days: int = Field(default=0, example=3)
    donor_stock_reduction_pct: float = Field(default=0.0, example=25.0)
    hospital_b_max_donation: Optional[int] = Field(default=None, example=500)
    transport_slowdown_factor: float = Field(default=1.0, example=1.5)

class OutbreakSimRequest(BaseModel):
    cluster_hospital_ids: List[str] = Field(default=["HOSP_D", "HOSP_E", "HOSP_F"])
    target_medicine_ids: List[str] = Field(default=["MED007", "MED016", "MED005"])
    severity_multiplier: float = Field(default=2.2, example=2.2)
    duration_days: int = Field(default=30, example=30)

class AssistantQueryRequest(BaseModel):
    question: str = Field(..., example="Which hospitals are likely to run out of Medicine MED004 next week?")

# --- Endpoints ---

@app.get("/")
def get_system_root():
    return {
        "system": "Medical Supply Intelligence Decision Support Engine",
        "status": "OPERATIONAL",
        "timestamp": datetime.now().isoformat(),
        "endpoints": [
            "/hospitals", "/medicines", "/inventory", "/forecasts",
            "/shortage-risks", "/expiry-risks", "/surplus", "/redistribution",
            "/emergency-request", "/priority", "/simulate-outbreak",
            "/what-if", "/assistant"
        ]
    }

@app.get("/hospitals")
def get_hospitals():
    """Returns all 10 hospitals and clinical capacity profiles."""
    df = query_df("SELECT * FROM hospitals")
    return df.to_dict("records")

@app.get("/medicines")
def get_medicines():
    """Returns all 20 medicines and criticalities."""
    df = query_df("SELECT * FROM medicines")
    return df.to_dict("records")

@app.get("/inventory")
def get_inventory(hospital_id: Optional[str] = None, medicine_id: Optional[str] = None):
    """Returns batch-level inventory records."""
    q = "SELECT * FROM inventory WHERE 1=1"
    params = []
    if hospital_id:
        q += " AND hospital_id = ?"
        params.append(hospital_id)
    if medicine_id:
        q += " AND medicine_id = ?"
        params.append(medicine_id)
    df = query_df(q, tuple(params))
    return df.to_dict("records")

@app.get("/forecasts")
def get_forecasts(hospital_id: str, medicine_id: str):
    """Returns 30-day multi-horizon ML demand forecast."""
    q = "SELECT * FROM forecasts WHERE hospital_id = ? AND medicine_id = ? ORDER BY day_ahead ASC"
    df = query_df(q, (hospital_id, medicine_id))
    if df.empty:
        raise HTTPException(status_code=404, detail="Forecast not found for specified series")
    return df.to_dict("records")

@app.get("/shortage-risks")
def get_shortage_risks(max_days: int = 14):
    """Returns ranked list of hospital-medicine pairs facing stockout risk."""
    df = projection_engine.scan_all_shortage_risks()
    if max_days:
        df = df[df["days_until_stockout"] <= max_days]
    return df.to_dict("records")

@app.get("/expiry-risks")
def get_expiry_risks(max_days: int = 30):
    """Returns batch-level expiry and financial waste risks."""
    df = expiry_detector.scan_all_batches()
    if df.empty:
        return []
    filtered = df[(df["days_until_expiry"] <= max_days) & (df["at_risk_waste_units"] > 0)]
    return filtered.to_dict("records")

@app.get("/surplus")
def get_surplus():
    """Returns all facilities holding safe transferable surplus."""
    df = projection_engine.scan_all_shortage_risks()
    surplus_df = df[df["transferable_surplus"] >= 50].sort_values("transferable_surplus", ascending=False)
    return surplus_df.to_dict("records")

@app.get("/priority")
def get_priority(medicine_id: str):
    """Ranks competing facilities for a scarce medicine with normalized component attribution."""
    ranked = priority_engine.rank_all_competing_facilities(medicine_id)
    return ranked.to_dict("records")

@app.get("/redistribution")
def get_redistribution():
    """Generates and returns optimized inter-hospital redistribution plan."""
    plan = redist_optimizer.generate_redistribution_plan()
    return plan

@app.post("/emergency-request")
def create_emergency_request(req: EmergencyRequestCreate):
    """Creates a new emergency medicine request and executes inter-hospital matching."""
    req_id = f"EMERG_{int(datetime.now().timestamp())}"
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    # Query current requesting hospital stock
    cur_q = "SELECT COALESCE(SUM(quantity), 0) as s FROM inventory WHERE hospital_id = ? AND medicine_id = ?"
    cur_df = query_df(cur_q, (req.requesting_hospital, req.medicine_id))
    cur_stock = int(cur_df["s"].iloc[0]) if not cur_df.empty else 0
    
    execute_sql("""
        INSERT INTO emergency_requests (
            emergency_request_id, requesting_hospital, medicine_id,
            quantity_required, required_by, procedure_type, urgency_level,
            current_stock, supplier_available, supplier_delivery_time, status, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        req_id, req.requesting_hospital, req.medicine_id,
        req.quantity_required, req.required_by, req.procedure_type, req.urgency_level,
        cur_stock, 0, "48 hours (Contract Lead Time)", "OPEN", now_str
    ))
    
    evaluation = emergency_engine.evaluate_emergency_request(req_id)
    return evaluation

@app.get("/emergency-request/{request_id}")
def get_emergency_request(
    request_id: str,
    simulated_traffic: Optional[str] = Query(None, description="Optional traffic condition: Low, Moderate, Heavy, Severe")
):
    """Fetches evaluation and feasibility ranking for an emergency request with optional traffic simulation."""
    evaluation = emergency_engine.evaluate_emergency_request(request_id, simulated_traffic_condition=simulated_traffic)
    return evaluation

@app.post("/what-if")
def simulate_what_if(req: WhatIfRequest):
    """Executes multi-parameter what-if stress simulation."""
    res = what_if_simulator.run_simulation(
        demand_surge_pct=req.demand_surge_pct,
        outbreak_hospitals=req.outbreak_hospitals,
        outbreak_severity_mult=req.outbreak_severity_mult,
        supplier_lead_time_added_days=req.supplier_lead_time_added_days,
        donor_stock_reduction_pct=req.donor_stock_reduction_pct,
        hospital_b_max_donation=req.hospital_b_max_donation,
        transport_slowdown_factor=req.transport_slowdown_factor
    )
    return res

@app.post("/simulate-outbreak")
def simulate_outbreak(req: OutbreakSimRequest):
    """Simulates targeted regional outbreak surge across designated hospital cluster."""
    res = outbreak_simulator.simulate_regional_outbreak(
        cluster_hospital_ids=req.cluster_hospital_ids,
        target_medicine_ids=req.target_medicine_ids,
        severity_multiplier=req.severity_multiplier,
        duration_days=req.duration_days
    )
    return res

@app.post("/assistant")
def query_assistant(req: AssistantQueryRequest):
    """Routes natural-language query to grounded AI assistant."""
    res = ai_assistant.query(req.question)
    return res
