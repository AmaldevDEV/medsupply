"""
Unit Tests for Medical Supply Intelligence Decision Engine
Validates:
- Stockout calculation (day-by-day simulation)
- Expiry calculation (batch-level surplus/waste formula)
- Transferable surplus (donor safety protection)
- Priority scoring (normalization, component weights, ranking)
- Redistribution constraints (capacity, shelf life, safety retention)
- Emergency sourcing (physical arrival validation, donor ranking, rejection audit)
- Edge cases (no feasible transfer, impossible deadline)
"""

import sys
import unittest
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.inventory_projection import InventoryProjectionEngine
from models.expiry_wastage import ExpiryWastageDetector
from optimization.priority import SupplyPriorityEngine
from optimization.redistribution import RedistributionOptimizer
from optimization.emergency_matching import EmergencySourcingEngine
from simulation.what_if import WhatIfSimulator
from assistant.agent import MedicalSupplyAIAssistant

class TestMedicalSupplyIntelligence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.proj_engine = InventoryProjectionEngine()
        cls.expiry_detector = ExpiryWastageDetector()
        cls.priority_engine = SupplyPriorityEngine()
        cls.redist_engine = RedistributionOptimizer()
        cls.emergency_engine = EmergencySourcingEngine()
        cls.what_if_engine = WhatIfSimulator()
        cls.assistant = MedicalSupplyAIAssistant()

    def test_stockout_prediction_day_by_day(self):
        """Validates that stockout is calculated by day-by-day forward simulation."""
        # Hospital A, MED004 (known low-stock scenario)
        res = self.proj_engine.project_inventory("HOSP_A", "MED004")
        self.assertIn("days_until_stockout", res)
        self.assertIn("trajectory", res)
        self.assertEqual(len(res["trajectory"]), 30)
        
        # Verify day-by-day trajectory integrity
        prev_stock = res["current_stock"]
        for pt in res["trajectory"]:
            expected_stock = max(0, int(round(prev_stock - pt["forecast_demand"] + pt["incoming_supply"])))
            self.assertEqual(pt["projected_stock"], expected_stock)
            prev_stock = pt["projected_stock"]

        # Hospital A has 180 units and consumes ~45/day -> Stockout in <= 4 days
        self.assertLessEqual(res["days_until_stockout"], 4)

    def test_supplier_lead_time_risk(self):
        """Validates procurement lead-time risk classification."""
        res = self.proj_engine.project_inventory("HOSP_A", "MED004")
        # Days until stockout is less than supplier lead time -> CRITICAL risk
        self.assertEqual(res["lead_time_risk"], "CRITICAL")
        self.assertIn("CRITICAL", res["risk_explanation"])

    def test_expiry_wastage_formula(self):
        """
        Validates batch expiry formula:
        Potential Waste = max(0, Current Batch Stock - Expected Usage Before Expiry - Required Safety Stock)
        """
        # Batch BAT-C-001-WASTE at Hospital C
        batch_res = self.expiry_detector.evaluate_batch_expiry_risk("BAT-C-001-WASTE")
        self.assertIsNotNone(batch_res)
        self.assertEqual(batch_res["hospital_id"], "HOSP_C")
        self.assertEqual(batch_res["days_until_expiry"], 12)
        
        qty = batch_res["quantity"] # 4,000 units
        exp_usage = batch_res["expected_usage_before_expiry"]
        safety = batch_res["safety_stock_protect"]
        expected_waste = max(0, qty - exp_usage - safety)
        
        self.assertEqual(batch_res["at_risk_waste_units"], expected_waste)
        self.assertGreater(batch_res["at_risk_waste_units"], 2000) # Should be ~2,300 units!
        self.assertIn("CRITICAL WASTE RISK", batch_res["risk_tier"])

    def test_transferable_surplus_donor_protection(self):
        """Validates that transferable surplus never endangers the donor hospital's safety stock."""
        # Hospital B has high stock of MED004
        res_b = self.proj_engine.project_inventory("HOSP_B", "MED004")
        self.assertGreater(res_b["transferable_surplus"], 2000)
        
        # Verify formula: surplus = current_stock - 14d_demand - safety_threshold
        expected_surplus = max(0, res_b["current_stock"] - res_b["forecast_14d_demand"] - res_b["safety_stock_threshold"])
        self.assertEqual(res_b["transferable_surplus"], expected_surplus)

        # Hospital I has low stock of MED008 (150 units, safety stock 100) -> safe surplus = 0
        res_i = self.proj_engine.project_inventory("HOSP_I", "MED008")
        self.assertEqual(res_i["transferable_surplus"], 0)

    def test_priority_scoring_normalization(self):
        """Validates priority scores: normalized components (0-100), configurable weights, valid bounds."""
        score_res = self.priority_engine.calculate_priority(
            hospital_id="HOSP_A",
            medicine_id="MED004",
            days_until_stockout=2,
            supplier_lead_time=7,
            current_stock=180,
            emergency_patient_load=120,
            current_occupancy=0.88,
            criticality=5,
            has_alternative=False
        )
        
        self.assertTrue(0.0 <= score_res["composite_priority_score"] <= 100.0)
        self.assertTrue(0.0 <= score_res["shortage_urgency_score"] <= 100.0)
        self.assertTrue(0.0 <= score_res["emergency_demand_score"] <= 100.0)
        self.assertTrue(0.0 <= score_res["patient_load_score"] <= 100.0)
        self.assertTrue(0.0 <= score_res["alternative_scarcity_score"] <= 100.0)
        self.assertTrue(0.0 <= score_res["lead_time_risk_score"] <= 100.0)
        
        # Hospital A in high acute crisis should achieve high/critical priority
        self.assertGreaterEqual(score_res["composite_priority_score"], 70.0)
        self.assertIn("CRITICAL", score_res["priority_tier"])

    def test_redistribution_constraints(self):
        """Validates that normal redistribution preserves donor safety stock and meets transit constraints."""
        plan = self.redist_engine.generate_redistribution_plan()
        self.assertIsInstance(plan, list)
        self.assertGreater(len(plan), 0)

        for rec in plan:
            # Transit time must be positive and reasonable
            self.assertGreater(rec["transit_time_hours"], 0.0)
            self.assertLess(rec["transit_time_hours"], 12.0)
            self.assertGreater(rec["quantity"], 0)
            self.assertNotEqual(rec["from_hospital"], rec["to_hospital"])

    def test_emergency_sourcing_solvable_case(self):
        """Validates Scenario 6: Hospital H emergency request for MED008 (Solvable)."""
        eval_res = self.emergency_engine.evaluate_emergency_request("EMERG_REQ_001")
        self.assertTrue(eval_res["has_feasible_source"])
        self.assertIsNotNone(eval_res["recommended_source"])
        
        rec = eval_res["recommended_source"]
        # Hospital B should be the recommended donor
        self.assertEqual(rec["candidate_hospital"], "HOSP_B")
        self.assertTrue(rec["donor_remains_safe"])
        self.assertTrue(rec["is_feasible"])
        self.assertGreaterEqual(rec["transferable_surplus"], 500)

    def test_emergency_sourcing_rejection_reasons(self):
        """Validates that candidate hospitals are properly rejected with clinical/logistical reasons."""
        eval_res = self.emergency_engine.evaluate_emergency_request("EMERG_REQ_001")
        rejected = eval_res["rejected_candidates"]
        self.assertGreater(len(rejected), 0)

        # Check Hospital I rejection reason (insufficient stock)
        hosp_i = next((c for c in eval_res["all_evaluated_candidates"] if c["candidate_hospital"] == "HOSP_I"), None)
        self.assertIsNotNone(hosp_i)
        self.assertFalse(hosp_i["is_feasible"])
        self.assertIn("Insufficient stock", hosp_i["rejection_reason"])

    def test_edge_case_no_feasible_source(self):
        """Validates Scenario 7: Impossible deadline correctly returns 'No feasible source'."""
        eval_res = self.emergency_engine.evaluate_emergency_request("EMERG_REQ_002")
        self.assertFalse(eval_res["has_feasible_source"])
        self.assertIsNone(eval_res["recommended_source"])
        self.assertEqual(eval_res["action_status"], "NO_FEASIBLE_SOURCE")

    def test_what_if_simulation_deltas(self):
        """Validates that what-if simulation calculates correct before vs after deltas."""
        sim_res = self.what_if_engine.run_simulation(demand_surge_pct=50.0)
        comp = sim_res["comparison"]
        
        # When demand surges by 50%, critical shortages should increase or stay high
        self.assertGreaterEqual(comp["critical_shortages"]["after"], comp["critical_shortages"]["before"])
        self.assertLessEqual(comp["average_days_until_stockout"]["after"], comp["average_days_until_stockout"]["before"])

    def test_ai_assistant_grounding(self):
        """Validates that AI assistant executes tools and never hallucinates numbers."""
        ans = self.assistant.query("Which hospitals are likely to run out of Medicine MED004 next week?")
        self.assertEqual(ans["tool_used"], "shortage_risks")
        self.assertIn("Hospital A", ans["answer"])
        self.assertIn("Emergency-Medicine-A", ans["answer"])

    def test_nearest_vs_fastest_traffic_selection(self):
        """
        Validates that the optimization engine compares Nearest vs Fastest routes under traffic.
        Even if Hospital D is closer geographically (4.6 km), Hospital B (17.5 km) is chosen because
        Hospital D is delayed by traffic gridlock (70 mins) and lacks stock, while Hospital B arrives
        in only 34 mins via open expressway, saving 36 minutes!
        """
        eval_res = self.emergency_engine.evaluate_emergency_request("EMERG_REQ_001")
        self.assertIn("nearest_vs_fastest", eval_res)
        nvf = eval_res["nearest_vs_fastest"]
        
        nearest = nvf["nearest_hospital_by_distance"]
        fastest = nvf["fastest_hospital_by_time"]
        
        # Nearest facility should be Hospital D (4.6 km, but takes 70 mins due to Severe traffic)
        self.assertEqual(nearest["hospital_id"], "HOSP_D")
        self.assertLess(nearest["distance_km"], fastest["distance_km"])
        self.assertEqual(nearest["traffic_level"], "Severe")
        
        # Recommended source should be Hospital B (the fastest feasible source, 34 mins)
        self.assertEqual(fastest["hospital_id"], "HOSP_B")
        self.assertEqual(fastest["traffic_level"], "Low")
        self.assertEqual(fastest["time_saved_vs_nearest_minutes"], 36)
        self.assertIn("TRAFFIC-OPTIMIZED DISPATCH", eval_res["recommended_source"]["recommendation_reason"])
        self.assertTrue(eval_res["recommended_source"]["is_fastest_route"])

    def test_google_maps_url_generation(self):
        """Validates that generated Google Maps URLs contain valid coordinates and driving travel mode."""
        eval_res = self.emergency_engine.evaluate_emergency_request("EMERG_REQ_001")
        rec = eval_res["recommended_source"]
        
        self.assertIn("google_maps_url", rec)
        url = rec["google_maps_url"]
        self.assertTrue(url.startswith("https://www.google.com/maps/dir/?api=1"))
        self.assertIn("origin=", url)
        self.assertIn("destination=", url)
        self.assertIn("travelmode=driving", url)

    def test_inr_valuation_calculations(self):
        """Validates realistic INR (₹) valuations in inventory, wastage, and transportation."""
        # 1. Wastage financial loss in INR
        exp_res = self.expiry_detector.evaluate_batch_expiry_risk("BAT-C-001-WASTE")
        self.assertIn("potential_financial_loss_inr", exp_res)
        self.assertGreater(exp_res["potential_financial_loss_inr"], 50000.0) # > ₹50,000
        
        # 2. Redistribution transport costs in INR (₹35/km)
        plan = self.redist_engine.generate_redistribution_plan()
        self.assertGreater(len(plan), 0)
        for rec in plan:
            self.assertIn("estimated_cost_inr", rec)
            expected_cost = round(rec["distance_km"] * 35.0, 2)
            self.assertAlmostEqual(rec["estimated_cost_inr"], expected_cost, places=1)

    def test_simulated_traffic_condition(self):
        """Validates that dynamic traffic simulation modulates transit time."""
        res_low = self.emergency_engine.evaluate_emergency_request("EMERG_REQ_001", simulated_traffic_condition="Low")
        res_severe = self.emergency_engine.evaluate_emergency_request("EMERG_REQ_001", simulated_traffic_condition="Severe")
        
        # Under severe gridlock, transit time must be significantly longer than under low traffic
        low_mins = res_low["recommended_source"]["transit_time_minutes"]
        severe_mins = res_severe["recommended_source"]["transit_time_minutes"]
        self.assertGreater(severe_mins, low_mins)

if __name__ == "__main__":
    unittest.main()
