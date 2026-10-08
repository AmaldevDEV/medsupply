"""
SQLite Schema Definitions for Medical Supply Intelligence
Updated with Indian Rupee (INR) valuations, traffic congestion parameters, and Google Maps links.
"""

CREATE_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS hospitals (
    hospital_id TEXT PRIMARY KEY,
    hospital_name TEXT NOT NULL,
    type TEXT,
    latitude REAL NOT NULL,
    longitude REAL NOT NULL,
    beds INTEGER NOT NULL,
    daily_patient_capacity INTEGER NOT NULL,
    average_patient_load INTEGER NOT NULL,
    emergency_patient_load INTEGER NOT NULL,
    ICU_capacity INTEGER NOT NULL,
    current_occupancy REAL NOT NULL,
    safety_stock_policy_days INTEGER NOT NULL,
    corridor TEXT,
    description TEXT
);

CREATE TABLE IF NOT EXISTS medicines (
    medicine_id TEXT PRIMARY KEY,
    medicine_name TEXT NOT NULL,
    category TEXT NOT NULL,
    criticality INTEGER NOT NULL,
    unit TEXT NOT NULL,
    shelf_life_days INTEGER NOT NULL,
    has_alternative INTEGER NOT NULL,
    alternative_medicine_id TEXT,
    unit_cost_inr REAL NOT NULL,
    unit_cost_usd REAL,
    storage_condition TEXT NOT NULL,
    requires_cold_chain INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS suppliers (
    supplier_id TEXT NOT NULL,
    supplier_name TEXT NOT NULL,
    medicine_id TEXT NOT NULL,
    lead_time_days INTEGER NOT NULL,
    minimum_order_quantity INTEGER NOT NULL,
    maximum_supply INTEGER NOT NULL,
    reliability_score REAL NOT NULL,
    PRIMARY KEY (supplier_id, medicine_id)
);

CREATE TABLE IF NOT EXISTS transport_routes (
    from_hospital TEXT NOT NULL,
    to_hospital TEXT NOT NULL,
    distance_km REAL NOT NULL,
    traffic_level TEXT NOT NULL,
    traffic_delay_factor REAL NOT NULL,
    base_travel_time_hours REAL NOT NULL,
    transport_time_hours REAL NOT NULL,
    transport_capacity INTEGER NOT NULL,
    transport_risk TEXT NOT NULL,
    google_maps_url TEXT NOT NULL,
    PRIMARY KEY (from_hospital, to_hospital)
);

CREATE TABLE IF NOT EXISTS outbreaks (
    outbreak_id TEXT PRIMARY KEY,
    start_date TEXT NOT NULL,
    end_date TEXT NOT NULL,
    affected_hospitals TEXT NOT NULL,
    affected_medicines TEXT NOT NULL,
    severity REAL NOT NULL,
    description TEXT
);

CREATE TABLE IF NOT EXISTS inventory (
    inventory_id TEXT PRIMARY KEY,
    hospital_id TEXT NOT NULL,
    medicine_id TEXT NOT NULL,
    batch_id TEXT NOT NULL,
    quantity INTEGER NOT NULL,
    received_date TEXT NOT NULL,
    expiry_date TEXT NOT NULL,
    days_until_expiry INTEGER NOT NULL,
    supplier_id TEXT NOT NULL,
    unit_cost_inr REAL NOT NULL,
    FOREIGN KEY(hospital_id) REFERENCES hospitals(hospital_id),
    FOREIGN KEY(medicine_id) REFERENCES medicines(medicine_id)
);

CREATE TABLE IF NOT EXISTS demand_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,
    hospital_id TEXT NOT NULL,
    medicine_id TEXT NOT NULL,
    daily_consumption INTEGER NOT NULL,
    patient_load INTEGER NOT NULL,
    emergency_load INTEGER NOT NULL,
    is_outbreak INTEGER DEFAULT 0,
    FOREIGN KEY(hospital_id) REFERENCES hospitals(hospital_id),
    FOREIGN KEY(medicine_id) REFERENCES medicines(medicine_id)
);

CREATE INDEX IF NOT EXISTS idx_demand_hosp_med_date ON demand_history(hospital_id, medicine_id, date);

CREATE TABLE IF NOT EXISTS forecasts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    hospital_id TEXT NOT NULL,
    medicine_id TEXT NOT NULL,
    forecast_date TEXT NOT NULL,
    target_date TEXT NOT NULL,
    day_ahead INTEGER NOT NULL,
    predicted_demand REAL NOT NULL,
    lower_bound REAL NOT NULL,
    upper_bound REAL NOT NULL,
    model_type TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_forecasts ON forecasts(hospital_id, medicine_id, target_date);

CREATE TABLE IF NOT EXISTS risk_scores (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    hospital_id TEXT NOT NULL,
    medicine_id TEXT NOT NULL,
    current_stock INTEGER NOT NULL,
    days_until_stockout INTEGER NOT NULL,
    supplier_lead_time INTEGER NOT NULL,
    lead_time_risk TEXT NOT NULL,
    transferable_surplus INTEGER NOT NULL,
    shortage_urgency_score REAL NOT NULL,
    emergency_demand_score REAL NOT NULL,
    patient_load_score REAL NOT NULL,
    alternative_scarcity_score REAL NOT NULL,
    lead_time_risk_score REAL NOT NULL,
    composite_priority_score REAL NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS redistribution_recommendations (
    recommendation_id TEXT PRIMARY KEY,
    from_hospital TEXT NOT NULL,
    to_hospital TEXT NOT NULL,
    medicine_id TEXT NOT NULL,
    batch_id TEXT,
    quantity INTEGER NOT NULL,
    transit_time_hours REAL NOT NULL,
    distance_km REAL NOT NULL,
    traffic_level TEXT,
    google_maps_url TEXT,
    estimated_cost_inr REAL,
    reason TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS emergency_requests (
    emergency_request_id TEXT PRIMARY KEY,
    requesting_hospital TEXT NOT NULL,
    medicine_id TEXT NOT NULL,
    quantity_required INTEGER NOT NULL,
    required_by TEXT NOT NULL,
    procedure_type TEXT NOT NULL,
    urgency_level TEXT NOT NULL,
    current_stock INTEGER NOT NULL,
    supplier_available INTEGER NOT NULL,
    supplier_delivery_time TEXT,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS emergency_sources (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    emergency_request_id TEXT NOT NULL,
    candidate_hospital TEXT NOT NULL,
    available_stock INTEGER NOT NULL,
    transferable_surplus INTEGER NOT NULL,
    distance_km REAL NOT NULL,
    transit_time_hours REAL NOT NULL,
    traffic_level TEXT,
    traffic_delay_factor REAL,
    google_maps_url TEXT,
    is_fastest_route INTEGER DEFAULT 0,
    is_nearest_route INTEGER DEFAULT 0,
    estimated_arrival TEXT NOT NULL,
    donor_remains_safe INTEGER NOT NULL,
    is_feasible INTEGER NOT NULL,
    rejection_reason TEXT,
    rank_order INTEGER,
    recommendation_reason TEXT,
    FOREIGN KEY(emergency_request_id) REFERENCES emergency_requests(emergency_request_id)
);

CREATE TABLE IF NOT EXISTS model_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    hospital_id TEXT NOT NULL,
    medicine_id TEXT NOT NULL,
    model_name TEXT NOT NULL,
    mae REAL NOT NULL,
    rmse REAL NOT NULL,
    mape REAL NOT NULL,
    test_samples INTEGER NOT NULL,
    evaluated_at TEXT NOT NULL
);
"""
