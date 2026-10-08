"""
Medical Supply Intelligence - System Configuration & Master Definitions
Track 2: Medical Supply Intelligence - Hackathon Prototype
Configured for Regional Healthcare Network with Realistic INR (₹) Valuations & Traffic Awareness
"""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
SYNTHETIC_DATA_DIR = DATA_DIR / "synthetic"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RAW_DATA_DIR = DATA_DIR / "raw"
MODELS_DIR = BASE_DIR / "models"
SAVED_MODELS_DIR = MODELS_DIR / "saved"
DATABASE_PATH = BASE_DIR / "database" / "medical_supply.db"

# Ensure directories exist
for p in [DATA_DIR, SYNTHETIC_DATA_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR, MODELS_DIR, SAVED_MODELS_DIR, BASE_DIR / "database"]:
    p.mkdir(parents=True, exist_ok=True)

# Currency and Valuation Constants (Indian Rupee - INR)
CURRENCY_SYMBOL = "₹"
CURRENCY_CODE = "INR"
TRANSPORT_COST_PER_KM_INR = 35.0  # Realistic ₹35/km for dedicated medical courier/ambulance transit

# LLM Assistant Settings
LLM_MODEL = os.getenv("LLM_MODEL", "mistral")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Traffic Level Dynamics & Multipliers
TRAFFIC_LEVELS = {
    "Low": {
        "label": "Clear / Expressway Flow",
        "delay_factor": 1.0,
        "avg_speed_kmh": 55.0,
        "badge_color": "green"
    },
    "Moderate": {
        "label": "Moderate City Traffic",
        "delay_factor": 1.25,
        "avg_speed_kmh": 38.0,
        "badge_color": "blue"
    },
    "Heavy": {
        "label": "Heavy Traffic / Peak Rush",
        "delay_factor": 1.70,
        "avg_speed_kmh": 22.0,
        "badge_color": "orange"
    },
    "Severe": {
        "label": "Severe Bottleneck / Gridlock",
        "delay_factor": 2.40,
        "avg_speed_kmh": 12.0,
        "badge_color": "red"
    }
}

# 10 Fictional Facilities (Hospital A to Hospital J)
# Positioned realistically across a major metropolitan healthcare zone
# (Coordinates allow real Google Maps navigation and driving routing)
HOSPITALS_CONFIG = [
    {
        "hospital_id": "HOSP_A",
        "hospital_name": "Hospital A",
        "type": "Level 1 Trauma & Tertiary Referral Center",
        "latitude": 28.5672,
        "longitude": 77.2100,
        "beds": 750,
        "daily_patient_capacity": 650,
        "average_patient_load": 580,
        "emergency_patient_load": 120,
        "ICU_capacity": 90,
        "current_occupancy": 0.88,
        "safety_stock_policy_days": 5,
        "corridor": "Ring Road Central Corridor",
        "description": "High acute demand epicenter. Facing upcoming shortage of critical emergency resuscitation medicines."
    },
    {
        "hospital_id": "HOSP_B",
        "hospital_name": "Hospital B",
        "type": "Metropolitan General Hospital",
        "latitude": 28.5284,
        "longitude": 77.2185,
        "beds": 520,
        "daily_patient_capacity": 450,
        "average_patient_load": 360,
        "emergency_patient_load": 70,
        "ICU_capacity": 55,
        "current_occupancy": 0.72,
        "safety_stock_policy_days": 5,
        "corridor": "Elevated Expressway Corridor",
        "description": "Stable regional center with high excess stock, safe transferable surplus, and rapid expressway access."
    },
    {
        "hospital_id": "HOSP_C",
        "hospital_name": "Hospital C",
        "type": "Community Acute Care Hospital",
        "latitude": 28.6219,
        "longitude": 77.0878,
        "beds": 310,
        "daily_patient_capacity": 270,
        "average_patient_load": 210,
        "emergency_patient_load": 40,
        "ICU_capacity": 24,
        "current_occupancy": 0.68,
        "safety_stock_policy_days": 4,
        "corridor": "West District Radial Arterial",
        "description": "Community facility with short-expiry inventory batch at imminent risk of wastage."
    },
    {
        "hospital_id": "HOSP_D",
        "hospital_name": "Hospital D",
        "type": "Urban District Hospital",
        "latitude": 28.6517,
        "longitude": 77.1906,
        "beds": 240,
        "daily_patient_capacity": 210,
        "average_patient_load": 195,
        "emergency_patient_load": 85,
        "ICU_capacity": 18,
        "current_occupancy": 0.93,
        "safety_stock_policy_days": 4,
        "corridor": "Central Commercial Dense Grid",
        "description": "Outbreak epicenter hospital experiencing sudden demand surge and high emergency influx."
    },
    {
        "hospital_id": "HOSP_E",
        "hospital_name": "Hospital E",
        "type": "Regional Specialty & Emergency Care",
        "latitude": 28.5494,
        "longitude": 77.2736,
        "beds": 380,
        "daily_patient_capacity": 330,
        "average_patient_load": 300,
        "emergency_patient_load": 95,
        "ICU_capacity": 38,
        "current_occupancy": 0.89,
        "safety_stock_policy_days": 5,
        "corridor": "South-East Industrial Bypass",
        "description": "Specialty center with sustained high emergency patient load and regional outbreak exposure."
    },
    {
        "hospital_id": "HOSP_F",
        "hospital_name": "Hospital F",
        "type": "Peripheral District Clinic & Hospital",
        "latitude": 28.4595,
        "longitude": 77.0266,
        "beds": 120,
        "daily_patient_capacity": 100,
        "average_patient_load": 80,
        "emergency_patient_load": 25,
        "ICU_capacity": 8,
        "current_occupancy": 0.70,
        "safety_stock_policy_days": 6,
        "corridor": "National Highway Corridor",
        "description": "Peripheral facility burdened by unusually long primary supplier lead times (12-14 days)."
    },
    {
        "hospital_id": "HOSP_G",
        "hospital_name": "Hospital G",
        "type": "Suburban Surgical Center",
        "latitude": 28.5700,
        "longitude": 77.3200,
        "beds": 200,
        "daily_patient_capacity": 180,
        "average_patient_load": 140,
        "emergency_patient_load": 30,
        "ICU_capacity": 16,
        "current_occupancy": 0.75,
        "safety_stock_policy_days": 4,
        "corridor": "Trans-River Tollway Arterial",
        "description": "Facing primary critical antibiotic shortage, but holds ample viable clinical alternative stock."
    },
    {
        "hospital_id": "HOSP_H",
        "hospital_name": "Hospital H",
        "type": "Children & Adult Surgical Specialty Hospital",
        "latitude": 28.6304,
        "longitude": 77.2177,
        "beds": 340,
        "daily_patient_capacity": 300,
        "average_patient_load": 250,
        "emergency_patient_load": 50,
        "ICU_capacity": 32,
        "current_occupancy": 0.82,
        "safety_stock_policy_days": 5,
        "corridor": "Central Civic Hub",
        "description": "Scheduled urgent emergency vascular surgery requiring specialized injection unavailable locally."
    },
    {
        "hospital_id": "HOSP_I",
        "hospital_name": "Hospital I",
        "type": "Eastside Community Care",
        "latitude": 28.6280,
        "longitude": 77.2789,
        "beds": 160,
        "daily_patient_capacity": 140,
        "average_patient_load": 110,
        "emergency_patient_load": 20,
        "ICU_capacity": 10,
        "current_occupancy": 0.69,
        "safety_stock_policy_days": 4,
        "corridor": "Bridge Choke Point Corridor (High Traffic Congestion)",
        "description": "Geographically nearest facility to Hospital H (6.2 km), but suffers severe bottleneck bridge traffic and lacks surplus."
    },
    {
        "hospital_id": "HOSP_J",
        "hospital_name": "Hospital J",
        "type": "North County Health Hub",
        "latitude": 28.7041,
        "longitude": 77.1025,
        "beds": 420,
        "daily_patient_capacity": 370,
        "average_patient_load": 290,
        "emergency_patient_load": 45,
        "ICU_capacity": 40,
        "current_occupancy": 0.74,
        "safety_stock_policy_days": 5,
        "corridor": "Outer Ring Road Signal-Free Flow",
        "description": "Has sufficient safe surplus for emergency requirements; connected via Outer Ring Road flow."
    }
]

# 20 Fictional / Generic Medicines & Supplies with Realistic INR (₹) Valuations
MEDICINES_CONFIG = [
    {
        "medicine_id": "MED001",
        "medicine_name": "Antibiotic-A",
        "category": "Broad-Spectrum Antibiotic",
        "criticality": 4,
        "unit": "vials (1g)",
        "shelf_life_days": 270,
        "has_alternative": 1,
        "alternative_medicine_id": "MED011",
        "unit_cost_inr": 850.0,
        "storage_condition": "Ambient (15-25°C)",
        "requires_cold_chain": False
    },
    {
        "medicine_id": "MED002",
        "medicine_name": "Analgesic-A",
        "category": "Critical Analgesic & Sedation",
        "criticality": 4,
        "unit": "ampoules (50mcg/mL)",
        "shelf_life_days": 365,
        "has_alternative": 1,
        "alternative_medicine_id": "MED013",
        "unit_cost_inr": 280.0,
        "storage_condition": "Controlled Ambient",
        "requires_cold_chain": False
    },
    {
        "medicine_id": "MED003",
        "medicine_name": "Insulin-A",
        "category": "Metabolic & Endocrine",
        "criticality": 5,
        "unit": "vials (100U/mL)",
        "shelf_life_days": 180,
        "has_alternative": 0,
        "alternative_medicine_id": None,
        "unit_cost_inr": 420.0,
        "storage_condition": "Cold Chain (2-8°C)",
        "requires_cold_chain": True
    },
    {
        "medicine_id": "MED004",
        "medicine_name": "Emergency-Medicine-A",
        "category": "Emergency Resuscitation & Vasopressor",
        "criticality": 5,
        "unit": "ampoules (1mg/mL)",
        "shelf_life_days": 365,
        "has_alternative": 0,
        "alternative_medicine_id": None,
        "unit_cost_inr": 350.0,
        "storage_condition": "Ambient (Protected from light)",
        "requires_cold_chain": False
    },
    {
        "medicine_id": "MED005",
        "medicine_name": "IV-Fluid-A",
        "category": "High-Volume Hydration Fluid",
        "criticality": 3,
        "unit": "bags (1000mL)",
        "shelf_life_days": 730,
        "has_alternative": 1,
        "alternative_medicine_id": "MED012",
        "unit_cost_inr": 55.0,
        "storage_condition": "Ambient (15-30°C)",
        "requires_cold_chain": False
    },
    {
        "medicine_id": "MED006",
        "medicine_name": "Surgical-Supply-A",
        "category": "Surgical Hemostatic Matrix",
        "criticality": 4,
        "unit": "kits",
        "shelf_life_days": 365,
        "has_alternative": 0,
        "alternative_medicine_id": None,
        "unit_cost_inr": 3200.0,
        "storage_condition": "Ambient (15-25°C)",
        "requires_cold_chain": False
    },
    {
        "medicine_id": "MED007",
        "medicine_name": "Antiviral-A",
        "category": "Targeted Antiviral Infusion",
        "criticality": 4,
        "unit": "vials (100mg)",
        "shelf_life_days": 180,
        "has_alternative": 1,
        "alternative_medicine_id": "MED014",
        "unit_cost_inr": 4800.0,
        "storage_condition": "Cold Chain (2-8°C)",
        "requires_cold_chain": True
    },
    {
        "medicine_id": "MED008",
        "medicine_name": "Emergency-Injection-A",
        "category": "Urgent Surgery Reversal & Antidote",
        "criticality": 5,
        "unit": "ampoules (10mg/mL)",
        "shelf_life_days": 240,
        "has_alternative": 0,
        "alternative_medicine_id": None,
        "unit_cost_inr": 2400.0,
        "storage_condition": "Cold Chain (2-8°C)",
        "requires_cold_chain": True
    },
    {
        "medicine_id": "MED009",
        "medicine_name": "Sedative-A",
        "category": "ICU Sedation & Anesthesia",
        "criticality": 4,
        "unit": "vials (20mL)",
        "shelf_life_days": 365,
        "has_alternative": 1,
        "alternative_medicine_id": "MED013",
        "unit_cost_inr": 380.0,
        "storage_condition": "Ambient (20-25°C)",
        "requires_cold_chain": False
    },
    {
        "medicine_id": "MED010",
        "medicine_name": "Anticoagulant-A",
        "category": "Critical Cardiovascular Anticoagulant",
        "criticality": 4,
        "unit": "vials (5000U/mL)",
        "shelf_life_days": 540,
        "has_alternative": 0,
        "alternative_medicine_id": None,
        "unit_cost_inr": 290.0,
        "storage_condition": "Ambient",
        "requires_cold_chain": False
    },
    {
        "medicine_id": "MED011",
        "medicine_name": "Antibiotic-B",
        "category": "Second-Line Antibiotic",
        "criticality": 3,
        "unit": "vials (1g)",
        "shelf_life_days": 300,
        "has_alternative": 1,
        "alternative_medicine_id": "MED001",
        "unit_cost_inr": 620.0,
        "storage_condition": "Ambient",
        "requires_cold_chain": False
    },
    {
        "medicine_id": "MED012",
        "medicine_name": "IV-Fluid-B",
        "category": "Balanced Electrolyte Fluid",
        "criticality": 3,
        "unit": "bags (1000mL)",
        "shelf_life_days": 730,
        "has_alternative": 1,
        "alternative_medicine_id": "MED005",
        "unit_cost_inr": 65.0,
        "storage_condition": "Ambient",
        "requires_cold_chain": False
    },
    {
        "medicine_id": "MED013",
        "medicine_name": "Sedative-B",
        "category": "Short-Acting Sedative Anxiolytic",
        "criticality": 3,
        "unit": "ampoules (5mg/mL)",
        "shelf_life_days": 365,
        "has_alternative": 1,
        "alternative_medicine_id": "MED009",
        "unit_cost_inr": 190.0,
        "storage_condition": "Ambient",
        "requires_cold_chain": False
    },
    {
        "medicine_id": "MED014",
        "medicine_name": "Antiviral-B",
        "category": "Oral Antiviral Capsule",
        "criticality": 3,
        "unit": "packs (30 caps)",
        "shelf_life_days": 365,
        "has_alternative": 1,
        "alternative_medicine_id": "MED007",
        "unit_cost_inr": 1450.0,
        "storage_condition": "Ambient",
        "requires_cold_chain": False
    },
    {
        "medicine_id": "MED015",
        "medicine_name": "Cardiovascular-A",
        "category": "Antiarrhythmic Injection",
        "criticality": 4,
        "unit": "ampoules (150mg/3mL)",
        "shelf_life_days": 365,
        "has_alternative": 0,
        "alternative_medicine_id": None,
        "unit_cost_inr": 340.0,
        "storage_condition": "Ambient",
        "requires_cold_chain": False
    },
    {
        "medicine_id": "MED016",
        "medicine_name": "Respiratory-A",
        "category": "Emergency Bronchodilator Nebulizer",
        "criticality": 4,
        "unit": "boxes (30 nebules)",
        "shelf_life_days": 450,
        "has_alternative": 0,
        "alternative_medicine_id": None,
        "unit_cost_inr": 280.0,
        "storage_condition": "Ambient",
        "requires_cold_chain": False
    },
    {
        "medicine_id": "MED017",
        "medicine_name": "Dialysis-Supply-A",
        "category": "Renal Replacement Solution",
        "criticality": 4,
        "unit": "canisters (5L)",
        "shelf_life_days": 540,
        "has_alternative": 0,
        "alternative_medicine_id": None,
        "unit_cost_inr": 1200.0,
        "storage_condition": "Ambient",
        "requires_cold_chain": False
    },
    {
        "medicine_id": "MED018",
        "medicine_name": "Critical-Biologic-A",
        "category": "Specialized Monoclonal Immunoglobulin",
        "criticality": 5,
        "unit": "vials (500mg)",
        "shelf_life_days": 90,  # Highly perishable biologic
        "has_alternative": 0,
        "alternative_medicine_id": None,
        "unit_cost_inr": 18500.0,
        "storage_condition": "Cold Chain (2-8°C)",
        "requires_cold_chain": True
    },
    {
        "medicine_id": "MED019",
        "medicine_name": "Surgical-Suture-A",
        "category": "Vascular Synthetic Hemostatic Suture",
        "criticality": 3,
        "unit": "boxes (36 sutures)",
        "shelf_life_days": 730,
        "has_alternative": 0,
        "alternative_medicine_id": None,
        "unit_cost_inr": 1800.0,
        "storage_condition": "Ambient",
        "requires_cold_chain": False
    },
    {
        "medicine_id": "MED020",
        "medicine_name": "Burn-Care-A",
        "category": "Advanced Antimicrobial Wound Dressing",
        "criticality": 3,
        "unit": "packs (10 dressings)",
        "shelf_life_days": 540,
        "has_alternative": 0,
        "alternative_medicine_id": None,
        "unit_cost_inr": 750.0,
        "storage_condition": "Ambient",
        "requires_cold_chain": False
    }
]

# Supplier Directory & Lead Times
SUPPLIERS_CONFIG = [
    {
        "supplier_id": "SUP001",
        "supplier_name": "Apex Regional Pharma Distribution",
        "lead_time_days": 5,
        "reliability_score": 0.96,
        "supplies": ["MED001", "MED002", "MED004", "MED005", "MED009", "MED010", "MED015"]
    },
    {
        "supplier_id": "SUP002",
        "supplier_name": "Beacon Biologics & Cold Chain Logistics",
        "lead_time_days": 7,
        "reliability_score": 0.94,
        "supplies": ["MED003", "MED007", "MED008", "MED018"]
    },
    {
        "supplier_id": "SUP003",
        "supplier_name": "Crestline Surgical & Specialty Supplies",
        "lead_time_days": 4,
        "reliability_score": 0.98,
        "supplies": ["MED006", "MED019", "MED020"]
    },
    {
        "supplier_id": "SUP004",
        "supplier_name": "Delta National Wholesale",
        "lead_time_days": 6,
        "reliability_score": 0.91,
        "supplies": ["MED001", "MED005", "MED011", "MED012", "MED014", "MED016", "MED017"]
    },
    {
        "supplier_id": "SUP005",
        "supplier_name": "Eastern Rural & District Medical Supply",
        "lead_time_days": 13,  # Long lead time affecting Hospital F!
        "reliability_score": 0.82,
        "supplies": ["MED001", "MED002", "MED004", "MED005", "MED009", "MED011"]
    },
    {
        "supplier_id": "SUP006",
        "supplier_name": "Global Anesthesia & Critical Care Depot",
        "lead_time_days": 5,
        "reliability_score": 0.95,
        "supplies": ["MED002", "MED009", "MED010", "MED013", "MED015"]
    },
    {
        "supplier_id": "SUP007",
        "supplier_name": "Highland Emergency Infusion Direct",
        "lead_time_days": 3,
        "reliability_score": 0.97,
        "supplies": ["MED004", "MED005", "MED008", "MED012"]
    },
    {
        "supplier_id": "SUP008",
        "supplier_name": "Metro Standard Health Distributor",
        "lead_time_days": 8,
        "reliability_score": 0.89,
        "supplies": ["MED001", "MED003", "MED007", "MED011", "MED016"]
    }
]

# Priority Engine Configurable Weights
DEFAULT_PRIORITY_WEIGHTS = {
    "shortage_urgency": 0.30,
    "emergency_demand": 0.25,
    "patient_load": 0.20,
    "alternative_scarcity": 0.15,
    "lead_time_risk": 0.10
}

# Operational Constants
DEFAULT_SAFETY_STOCK_DAYS = 5
STOCKOUT_RED_ALERT_DAYS = 5
EXPIRY_WARNING_DAYS = 30
COURIER_AVG_SPEED_KMH = 45.0
DISPATCH_PREPARATION_HOURS = 0.25  # 15 minutes preparation
URBAN_CIRCUITY_FACTOR = 1.30
