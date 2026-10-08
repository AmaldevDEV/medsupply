"""
Synthetic Data Generation Engine for Medical Supply Intelligence
Produces realistic time-series demand, batch-level inventories with shelf lives in INR (₹),
supplier lead times, transport distance & traffic level matrices, Google Maps direction URLs,
outbreaks, and emergency clinical requests.
"""

import math
import random
from datetime import datetime, timedelta
from pathlib import Path
import sys
import numpy as np
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    BASE_DIR, DATA_DIR, SYNTHETIC_DATA_DIR, DATABASE_PATH,
    HOSPITALS_CONFIG, MEDICINES_CONFIG, SUPPLIERS_CONFIG,
    TRAFFIC_LEVELS, COURIER_AVG_SPEED_KMH, DISPATCH_PREPARATION_HOURS,
    URBAN_CIRCUITY_FACTOR, TRANSPORT_COST_PER_KM_INR
)
from database.database import init_db, get_connection

def haversine_distance_km(lat1, lon1, lat2, lon2):
    """Calculates great-circle distance between two coordinates in kilometers."""
    R = 6371.0 # Earth radius in km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def generate_hospitals():
    """Generates hospitals dataframe."""
    df = pd.DataFrame(HOSPITALS_CONFIG)
    return df

def generate_medicines():
    """Generates medicines dataframe with realistic INR valuations."""
    rows = []
    for m in MEDICINES_CONFIG:
        rec = dict(m)
        rec["unit_cost_usd"] = round(m["unit_cost_inr"] / 83.0, 2)
        rows.append(rec)
    return pd.DataFrame(rows)

def generate_suppliers():
    """Generates supplier catalogue dataframe."""
    rows = []
    for s in SUPPLIERS_CONFIG:
        for med_id in s["supplies"]:
            rows.append({
                "supplier_id": s["supplier_id"],
                "supplier_name": s["supplier_name"],
                "medicine_id": med_id,
                "lead_time_days": s["lead_time_days"],
                "minimum_order_quantity": 50 if "MED018" in med_id else 100,
                "maximum_supply": 25000,
                "reliability_score": s["reliability_score"]
            })
    return pd.DataFrame(rows)

def generate_transport_routes():
    """
    Generates transport matrix between all pairs of 10 hospitals.
    Incorporates traffic congestion levels (Low, Moderate, Heavy, Severe),
    effective road speeds, and direct Google Maps direction URLs.
    """
    routes = []
    
    # Corridor traffic mapping for realistic regional dynamics
    # E.g. East Delhi bridge crossings (Hospital I) experience Severe bottleneck traffic
    # South Elevated corridors (Hospital B) experience Low / Expressway flow
    
    for h1 in HOSPITALS_CONFIG:
        id1 = h1["hospital_id"]
        lat1, lon1 = h1["latitude"], h1["longitude"]
        
        for h2 in HOSPITALS_CONFIG:
            id2 = h2["hospital_id"]
            lat2, lon2 = h2["latitude"], h2["longitude"]
            
            if id1 == id2:
                dist_km = 0.0
                traffic = "Low"
                delay_factor = 1.0
                base_time = 0.0
                transit_hours = 0.0
                risk = "None"
                gmaps_url = f"https://www.google.com/maps/search/?api=1&query={lat1},{lon1}"
            else:
                raw_km = haversine_distance_km(lat1, lon1, lat2, lon2)
                dist_km = max(3.5, round(raw_km * URBAN_CIRCUITY_FACTOR, 1))
                
                # Determine traffic conditions based on urban corridors:
                # 1. Hospital I <-> Hospital H (East Yamuna Bridge Choke Point)
                if (id1 == "HOSP_I" and id2 == "HOSP_H") or (id1 == "HOSP_H" and id2 == "HOSP_I"):
                    dist_km = 6.2  # Geographically nearest!
                    traffic = "Severe"
                    delay_factor = 2.40  # 2.4x delay due to bridge gridlock
                    base_speed = 35.0
                    base_time = dist_km / base_speed
                    transit_hours = round((base_time * delay_factor) + DISPATCH_PREPARATION_HOURS, 2)
                    # ~0.68 hours = 41 mins despite being only 6.2 km away!

                # 2. Hospital B <-> Hospital H (Signal-Free Elevated Corridor)
                elif (id1 == "HOSP_B" and id2 == "HOSP_H") or (id1 == "HOSP_H" and id2 == "HOSP_B"):
                    dist_km = 17.5  # Further away by distance!
                    traffic = "Low"
                    delay_factor = 1.00  # Clear elevated expressway
                    base_speed = 55.0
                    base_time = dist_km / base_speed
                    transit_hours = round((base_time * delay_factor) + DISPATCH_PREPARATION_HOURS, 2)
                    # ~0.47 hours = 28 mins! Reaches 13 minutes FASTER than Hospital I!

                # 3. Hospital J <-> Hospital H (Outer Ring Road Flow)
                elif (id1 == "HOSP_J" and id2 == "HOSP_H") or (id1 == "HOSP_H" and id2 == "HOSP_J"):
                    dist_km = 21.8
                    traffic = "Moderate"
                    delay_factor = 1.25
                    base_speed = 50.0
                    base_time = dist_km / base_speed
                    transit_hours = round((base_time * delay_factor) + DISPATCH_PREPARATION_HOURS, 2)
                    # ~0.70 hours = 42 mins

                # 4. Hospital B <-> Hospital A (Direct South Corridor)
                elif (id1 == "HOSP_B" and id2 == "HOSP_A") or (id1 == "HOSP_A" and id2 == "HOSP_B"):
                    dist_km = 8.5
                    traffic = "Moderate"
                    delay_factor = 1.20
                    base_speed = 42.0
                    base_time = dist_km / base_speed
                    transit_hours = round((base_time * delay_factor) + DISPATCH_PREPARATION_HOURS, 2)

                # Standard calculation for other pairs
                else:
                    # Random deterministic traffic based on hospital hash
                    h_hash = (hash(id1) + hash(id2)) % 4
                    traffic_keys = ["Low", "Moderate", "Heavy", "Severe"]
                    traffic = traffic_keys[h_hash]
                    delay_factor = TRAFFIC_LEVELS[traffic]["delay_factor"]
                    base_speed = TRAFFIC_LEVELS[traffic]["avg_speed_kmh"]
                    base_time = dist_km / base_speed
                    transit_hours = round((base_time * delay_factor) + DISPATCH_PREPARATION_HOURS, 2)

                if transit_hours <= 0.6:
                    risk = "Low"
                elif transit_hours <= 1.2:
                    risk = "Moderate"
                else:
                    risk = "High"

                # Generate real Google Maps driving route URL
                gmaps_url = f"https://www.google.com/maps/dir/?api=1&origin={lat1},{lon1}&destination={lat2},{lon2}&travelmode=driving"

            routes.append({
                "from_hospital": id1,
                "to_hospital": id2,
                "distance_km": dist_km,
                "traffic_level": traffic,
                "traffic_delay_factor": delay_factor,
                "base_travel_time_hours": round(base_time, 2) if id1 != id2 else 0.0,
                "transport_time_hours": transit_hours,
                "transport_capacity": 5000,
                "transport_risk": risk,
                "google_maps_url": gmaps_url
            })
            
    return pd.DataFrame(routes)

def generate_outbreaks(end_date):
    """Generates synthetic outbreak scenario record."""
    start_dt = end_date - timedelta(days=120)
    end_dt = start_dt + timedelta(days=28)
    outbreak = [{
        "outbreak_id": "OUTBREAK_2026_01",
        "start_date": start_dt.strftime("%Y-%m-%d"),
        "end_date": end_dt.strftime("%Y-%m-%d"),
        "affected_hospitals": "HOSP_D,HOSP_E,HOSP_F",
        "affected_medicines": "MED005,MED007,MED016",
        "severity": 2.4, # 2.4x surge
        "description": "Regional respiratory viral surge with high pulmonary morbidity and acute care admissions."
    }]
    return pd.DataFrame(outbreak)

def generate_demand_history(end_date, num_days=450):
    """
    Generates 450 days of daily consumption for 10 hospitals x 20 medicines.
    Demand = baseline + patient_load_effect + seasonal_effect + weekday_effect + outbreak_effect + random_noise
    """
    np.random.seed(42)
    random.seed(42)

    start_date = end_date - timedelta(days=num_days - 1)
    dates = [start_date + timedelta(days=i) for i in range(num_days)]
    
    outbreak_start_idx = 250
    outbreak_end_idx = 280
    
    demand_records = []
    
    med_consumption_rate = {
        "MED001": 22, "MED002": 35, "MED003": 18, "MED004": 15, "MED005": 140,
        "MED006": 8,  "MED007": 12, "MED008": 6,  "MED009": 20, "MED010": 25,
        "MED011": 14, "MED012": 80, "MED013": 12, "MED014": 10, "MED015": 9,
        "MED016": 28, "MED017": 15, "MED018": 4,  "MED019": 11, "MED020": 16
    }
    
    for hosp in HOSPITALS_CONFIG:
        h_id = hosp["hospital_id"]
        bed_factor = hosp["beds"] / 350.0
        avg_pat = hosp["average_patient_load"]
        avg_emerg = hosp["emergency_patient_load"]
        
        for day_idx, dt in enumerate(dates):
            dow = dt.weekday()
            doy = dt.timetuple().tm_yday
            
            pat_noise = np.random.normal(0, avg_pat * 0.05)
            pat_weekend_mult = 0.88 if dow in [5, 6] else 1.03
            daily_patient_load = int(max(20, round((avg_pat + pat_noise) * pat_weekend_mult)))
            
            emerg_weekend_mult = 1.15 if dow in [5, 6] else 0.95
            emerg_noise = np.random.normal(0, avg_emerg * 0.08)
            daily_emergency_load = int(max(5, round((avg_emerg + emerg_noise) * emerg_weekend_mult)))
            
            is_outbreak_day = 0
            if outbreak_start_idx <= day_idx <= outbreak_end_idx and h_id in ["HOSP_D", "HOSP_E", "HOSP_F"]:
                is_outbreak_day = 1
                daily_patient_load = int(round(daily_patient_load * 1.35))
                daily_emergency_load = int(round(daily_emergency_load * 1.60))

            annual_season = math.sin(2 * math.pi * (doy - 20) / 365.25)
            
            for med in MEDICINES_CONFIG:
                m_id = med["medicine_id"]
                base_rate = med_consumption_rate[m_id]
                baseline = base_rate * bed_factor
                
                pat_effect = (daily_patient_load / max(1, avg_pat) - 1.0) * baseline * 0.45
                emerg_weight = 0.55 if m_id in ["MED004", "MED008", "MED002", "MED015"] else 0.15
                emerg_effect = (daily_emergency_load / max(1, avg_emerg) - 1.0) * baseline * emerg_weight
                
                if m_id in ["MED006", "MED019", "MED009", "MED001"]:
                    weekday_effect = baseline * (-0.25 if dow in [5, 6] else 0.10)
                else:
                    weekday_effect = baseline * (-0.05 if dow in [5, 6] else 0.02)
                
                if m_id in ["MED007", "MED016", "MED014"]:
                    seasonal_effect = baseline * 0.35 * annual_season
                else:
                    seasonal_effect = baseline * 0.10 * annual_season
                
                outbreak_effect = 0.0
                if is_outbreak_day:
                    if m_id in ["MED007", "MED016"]:
                        outbreak_effect = baseline * 1.85
                    elif m_id in ["MED005", "MED001", "MED009"]:
                        outbreak_effect = baseline * 0.75
                
                noise = np.random.normal(0, max(1.5, baseline * 0.08))
                
                if h_id == "HOSP_A" and m_id == "MED004" and day_idx >= (num_days - 14):
                    noise += 15.0
                if h_id == "HOSP_D" and day_idx >= (num_days - 7):
                    noise += baseline * 0.30

                total_demand = baseline + pat_effect + emerg_effect + weekday_effect + seasonal_effect + outbreak_effect + noise
                final_consumption = max(1, int(round(total_demand)))
                
                demand_records.append({
                    "date": dt.strftime("%Y-%m-%d"),
                    "hospital_id": h_id,
                    "medicine_id": m_id,
                    "daily_consumption": final_consumption,
                    "patient_load": daily_patient_load,
                    "emergency_load": daily_emergency_load,
                    "is_outbreak": is_outbreak_day
                })
                
    return pd.DataFrame(demand_records)

def generate_inventory(simulation_date):
    """
    Generates batch-level inventory explicitly honoring all evaluation scenarios
    with realistic INR (₹) unit costs.
    """
    np.random.seed(42)
    batches = []
    inv_counter = 1000
    
    med_cost_inr = {m["medicine_id"]: m["unit_cost_inr"] for m in MEDICINES_CONFIG}

    # 1. Scenario-Specific Key Batches
    
    # --- HOSPITAL A: Upcoming Shortage of MED004 (Emergency-Medicine-A) ---
    batches.append({
        "inventory_id": f"INV_{inv_counter}",
        "hospital_id": "HOSP_A",
        "medicine_id": "MED004",
        "batch_id": "BAT-A-004-1",
        "quantity": 180,
        "received_date": (simulation_date - timedelta(days=60)).strftime("%Y-%m-%d"),
        "expiry_date": (simulation_date + timedelta(days=180)).strftime("%Y-%m-%d"),
        "days_until_expiry": 180,
        "supplier_id": "SUP001",
        "unit_cost_inr": med_cost_inr["MED004"] # ₹350
    })
    inv_counter += 1

    # --- HOSPITAL B: Excess Stock & Safe Transferable Surplus of MED004 and MED008 ---
    batches.append({
        "inventory_id": f"INV_{inv_counter}",
        "hospital_id": "HOSP_B",
        "medicine_id": "MED004",
        "batch_id": "BAT-B-004-1",
        "quantity": 2500,
        "received_date": (simulation_date - timedelta(days=40)).strftime("%Y-%m-%d"),
        "expiry_date": (simulation_date + timedelta(days=240)).strftime("%Y-%m-%d"),
        "days_until_expiry": 240,
        "supplier_id": "SUP001",
        "unit_cost_inr": med_cost_inr["MED004"]
    })
    inv_counter += 1
    batches.append({
        "inventory_id": f"INV_{inv_counter}",
        "hospital_id": "HOSP_B",
        "medicine_id": "MED004",
        "batch_id": "BAT-B-004-2",
        "quantity": 2000,
        "received_date": (simulation_date - timedelta(days=15)).strftime("%Y-%m-%d"),
        "expiry_date": (simulation_date + timedelta(days=320)).strftime("%Y-%m-%d"),
        "days_until_expiry": 320,
        "supplier_id": "SUP001",
        "unit_cost_inr": med_cost_inr["MED004"]
    })
    inv_counter += 1

    # MED008: 2,500 units on hand at Hospital B
    batches.append({
        "inventory_id": f"INV_{inv_counter}",
        "hospital_id": "HOSP_B",
        "medicine_id": "MED008",
        "batch_id": "BAT-B-008-1",
        "quantity": 2500,
        "received_date": (simulation_date - timedelta(days=30)).strftime("%Y-%m-%d"),
        "expiry_date": (simulation_date + timedelta(days=210)).strftime("%Y-%m-%d"),
        "days_until_expiry": 210,
        "supplier_id": "SUP002",
        "unit_cost_inr": med_cost_inr["MED008"] # ₹2,400
    })
    inv_counter += 1

    # --- HOSPITAL C: Short Expiry Batch & Imminent Wastage ---
    # 4,000 units of MED001 expiring in 12 days. Potential waste = ~2,300 units (₹19,55,000 loss)
    batches.append({
        "inventory_id": f"INV_{inv_counter}",
        "hospital_id": "HOSP_C",
        "medicine_id": "MED001",
        "batch_id": "BAT-C-001-WASTE",
        "quantity": 4000,
        "received_date": (simulation_date - timedelta(days=258)).strftime("%Y-%m-%d"),
        "expiry_date": (simulation_date + timedelta(days=12)).strftime("%Y-%m-%d"),
        "days_until_expiry": 12,
        "supplier_id": "SUP001",
        "unit_cost_inr": med_cost_inr["MED001"] # ₹850
    })
    inv_counter += 1

    # --- HOSPITAL G: Critical Shortage of MED001, but Alternative MED011 Available ---
    batches.append({
        "inventory_id": f"INV_{inv_counter}",
        "hospital_id": "HOSP_G",
        "medicine_id": "MED001",
        "batch_id": "BAT-G-001-LOW",
        "quantity": 40,
        "received_date": (simulation_date - timedelta(days=80)).strftime("%Y-%m-%d"),
        "expiry_date": (simulation_date + timedelta(days=120)).strftime("%Y-%m-%d"),
        "days_until_expiry": 120,
        "supplier_id": "SUP001",
        "unit_cost_inr": med_cost_inr["MED001"]
    })
    inv_counter += 1
    batches.append({
        "inventory_id": f"INV_{inv_counter}",
        "hospital_id": "HOSP_G",
        "medicine_id": "MED011",
        "batch_id": "BAT-G-011-ALT",
        "quantity": 1250,
        "received_date": (simulation_date - timedelta(days=20)).strftime("%Y-%m-%d"),
        "expiry_date": (simulation_date + timedelta(days=260)).strftime("%Y-%m-%d"),
        "days_until_expiry": 260,
        "supplier_id": "SUP004",
        "unit_cost_inr": med_cost_inr["MED011"] # ₹620
    })
    inv_counter += 1

    # --- HOSPITAL H: Stock of MED008 is ZERO ---

    # --- HOSPITAL I: Geographically closest to H (6.2 km) but Insufficient Surplus ---
    batches.append({
        "inventory_id": f"INV_{inv_counter}",
        "hospital_id": "HOSP_I",
        "medicine_id": "MED008",
        "batch_id": "BAT-I-008-LOW",
        "quantity": 35,
        "received_date": (simulation_date - timedelta(days=50)).strftime("%Y-%m-%d"),
        "expiry_date": (simulation_date + timedelta(days=140)).strftime("%Y-%m-%d"),
        "days_until_expiry": 140,
        "supplier_id": "SUP002",
        "unit_cost_inr": med_cost_inr["MED008"]
    })
    inv_counter += 1

    # --- HOSPITAL J: Farther from H (21.8 km) but SUFFICIENT Safe Surplus ---
    batches.append({
        "inventory_id": f"INV_{inv_counter}",
        "hospital_id": "HOSP_J",
        "medicine_id": "MED008",
        "batch_id": "BAT-J-008-SURP",
        "quantity": 1200,
        "received_date": (simulation_date - timedelta(days=25)).strftime("%Y-%m-%d"),
        "expiry_date": (simulation_date + timedelta(days=190)).strftime("%Y-%m-%d"),
        "days_until_expiry": 190,
        "supplier_id": "SUP002",
        "unit_cost_inr": med_cost_inr["MED008"]
    })
    inv_counter += 1

    # Standard Batches for All Other Pairs
    handled_pairs = {
        ("HOSP_A", "MED004"), ("HOSP_B", "MED004"), ("HOSP_B", "MED008"),
        ("HOSP_C", "MED001"), ("HOSP_G", "MED001"), ("HOSP_G", "MED011"),
        ("HOSP_H", "MED008"), ("HOSP_I", "MED008"), ("HOSP_J", "MED008")
    }

    for hosp in HOSPITALS_CONFIG:
        h_id = hosp["hospital_id"]
        bed_factor = hosp["beds"] / 350.0

        for med in MEDICINES_CONFIG:
            m_id = med["medicine_id"]
            if (h_id, m_id) in handled_pairs:
                continue
            
            daily_est = 20 * bed_factor
            target_stock = int(daily_est * np.random.uniform(14, 28))
            
            num_batches = np.random.choice([1, 2, 3], p=[0.25, 0.50, 0.25])
            fractions = np.random.dirichlet(np.ones(num_batches))
            
            for i, frac in enumerate(fractions):
                batch_qty = max(15, int(round(target_stock * frac)))
                exp_days = int(np.random.uniform(45, med["shelf_life_days"]))
                rec_days_ago = int(np.random.uniform(10, 60))
                
                supplier = "SUP001"
                for s in SUPPLIERS_CONFIG:
                    if m_id in s["supplies"]:
                        supplier = s["supplier_id"]
                        break

                batches.append({
                    "inventory_id": f"INV_{inv_counter}",
                    "hospital_id": h_id,
                    "medicine_id": m_id,
                    "batch_id": f"BAT-{h_id[5:]}-{m_id[3:]}-{i+1}",
                    "quantity": batch_qty,
                    "received_date": (simulation_date - timedelta(days=rec_days_ago)).strftime("%Y-%m-%d"),
                    "expiry_date": (simulation_date + timedelta(days=exp_days)).strftime("%Y-%m-%d"),
                    "days_until_expiry": exp_days,
                    "supplier_id": supplier,
                    "unit_cost_inr": med_cost_inr[m_id]
                })
                inv_counter += 1

    return pd.DataFrame(batches)

def generate_emergency_requests(simulation_date):
    """
    Generates synthetic emergency requests containing:
    1. Benchmark Hospital H request: Emergency vascular surgery tomorrow at 10:00 AM, needing 500 units MED008 (Solvable).
    2. Mass Casualty Trauma request at Hospital D: Needing 800 units MED004 in 15 mins (Impossible deadline).
    3. Oncology/Biologic urgent request at Hospital G: Needing 80 units MED018 in 24 hours (Solvable).
    """
    requests = [
        {
            "emergency_request_id": "EMERG_REQ_001",
            "requesting_hospital": "HOSP_H",
            "medicine_id": "MED008",
            "quantity_required": 500,
            "required_by": (simulation_date + timedelta(hours=18.5)).strftime("%Y-%m-%d %H:%M"), # Tomorrow 10:00 AM
            "procedure_type": "Emergency Aortic Repair / Reversal",
            "urgency_level": "STAT Critical",
            "current_stock": 0,
            "supplier_available": 0,
            "supplier_delivery_time": "48 hours (Contract Lead Time)",
            "status": "OPEN",
            "created_at": simulation_date.strftime("%Y-%m-%d %H:%M")
        },
        {
            "emergency_request_id": "EMERG_REQ_002",
            "requesting_hospital": "HOSP_D",
            "medicine_id": "MED004",
            "quantity_required": 800,
            "required_by": (simulation_date + timedelta(minutes=15)).strftime("%Y-%m-%d %H:%M"), # In 15 minutes! Impossible deadline!
            "procedure_type": "Mass Casualty Multi-Trauma Resuscitation",
            "urgency_level": "Immediate Code Red",
            "current_stock": 15,
            "supplier_available": 0,
            "supplier_delivery_time": "24 hours",
            "status": "OPEN",
            "created_at": simulation_date.strftime("%Y-%m-%d %H:%M")
        },
        {
            "emergency_request_id": "EMERG_REQ_003",
            "requesting_hospital": "HOSP_G",
            "medicine_id": "MED018",
            "quantity_required": 80,
            "required_by": (simulation_date + timedelta(hours=24.0)).strftime("%Y-%m-%d %H:%M"),
            "procedure_type": "Severe Refractory Immunologic Crisis",
            "urgency_level": "Urgent Priority",
            "current_stock": 5,
            "supplier_available": 1,
            "supplier_delivery_time": "7 days",
            "status": "OPEN",
            "created_at": simulation_date.strftime("%Y-%m-%d %H:%M")
        }
    ]
    return pd.DataFrame(requests)

def generate_all_data():
    """Generates all synthetic tables, writes CSVs, and inserts into SQLite database."""
    print("=" * 60)
    print("MEDICAL SUPPLY INTELLIGENCE - SYNTHETIC DATA GENERATOR (INR & TRAFFIC)")
    print("=" * 60)
    
    simulation_date = datetime(2026, 10, 8, 15, 30)
    
    print("[1/7] Generating 10 Hospitals with Real Coordinates & Corridors...")
    hospitals_df = generate_hospitals()
    hospitals_df.to_csv(SYNTHETIC_DATA_DIR / "hospitals.csv", index=False)
    
    print("[2/7] Generating 20 Medical Supplies with Realistic INR Valuations...")
    medicines_df = generate_medicines()
    medicines_df.to_csv(SYNTHETIC_DATA_DIR / "medicines.csv", index=False)
    
    print("[3/7] Generating Regional Supplier Network & Lead Times...")
    suppliers_df = generate_suppliers()
    suppliers_df.to_csv(SYNTHETIC_DATA_DIR / "suppliers.csv", index=False)
    
    print("[4/7] Generating Transport Matrix with Traffic Levels & Google Maps Links...")
    routes_df = generate_transport_routes()
    routes_df.to_csv(SYNTHETIC_DATA_DIR / "transport_routes.csv", index=False)
    
    print("[5/7] Generating Regional Outbreak Scenario...")
    outbreaks_df = generate_outbreaks(simulation_date)
    outbreaks_df.to_csv(SYNTHETIC_DATA_DIR / "outbreaks.csv", index=False)
    
    print("[6/7] Generating 450 days of daily historical consumption (200 series)...")
    demand_df = generate_demand_history(simulation_date, num_days=450)
    demand_df.to_csv(SYNTHETIC_DATA_DIR / "demand_history.csv", index=False)
    
    print("[7/7] Generating Batch-Level Inventories in INR & Emergency Requests...")
    inventory_df = generate_inventory(simulation_date)
    inventory_df.to_csv(SYNTHETIC_DATA_DIR / "inventory_batches.csv", index=False)
    
    emergency_df = generate_emergency_requests(simulation_date)
    emergency_df.to_csv(SYNTHETIC_DATA_DIR / "emergency_requests.csv", index=False)
    
    print("\n[DB] Initializing SQLite database and loading tables...")
    init_db(DATABASE_PATH)
    conn = get_connection(DATABASE_PATH)
    
    hospitals_df.to_sql("hospitals", conn, if_exists="replace", index=False)
    medicines_df.to_sql("medicines", conn, if_exists="replace", index=False)
    suppliers_df.to_sql("suppliers", conn, if_exists="replace", index=False)
    routes_df.to_sql("transport_routes", conn, if_exists="replace", index=False)
    outbreaks_df.to_sql("outbreaks", conn, if_exists="replace", index=False)
    demand_df.to_sql("demand_history", conn, if_exists="replace", index=False)
    inventory_df.to_sql("inventory", conn, if_exists="replace", index=False)
    emergency_df.to_sql("emergency_requests", conn, if_exists="replace", index=False)
    
    conn.commit()
    conn.close()
    
    print("\nSYNTHETIC DATA GENERATION COMPLETE!")
    print(f"  - Hospitals: {len(hospitals_df)}")
    print(f"  - Medicines: {len(medicines_df)} (Valued in INR)")
    print(f"  - Transport Routes: {len(routes_df)} (With Traffic & Google Maps URLs)")
    print(f"  - Historical Demand Records: {len(demand_df)}")
    print(f"  - Inventory Batches: {len(inventory_df)}")
    print(f"  - Emergency Requests: {len(emergency_df)}")
    print(f"  - Database: {DATABASE_PATH}")
    print("=" * 60)

if __name__ == "__main__":
    generate_all_data()
