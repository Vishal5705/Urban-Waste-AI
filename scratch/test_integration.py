import sys
import unittest
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from utils import (
    load_forecasting_pipeline,
    load_bin_data,
    load_waste_data,
    build_forecast_input_df,
    generate_forecast_priorities,
    solve_ortools_tsp,
    solve_csv_order_route,
    solve_nearest_neighbor_route,
    calculate_haversine_km,
    build_distance_matrix_metres,
    detect_isolated_bins
)


class TestUrbanWasteAIRouteAndForecasting(unittest.TestCase):

    def test_01_model_loading_and_prediction(self):
        pipeline_dict = load_forecasting_pipeline()
        self.assertIsNotNone(pipeline_dict, "Saved model pipeline dictionary should not be None")
        self.assertIn("model", pipeline_dict)
        self.assertIn("preprocessor", pipeline_dict)
        self.assertIn("features", pipeline_dict)
        self.assertIn("evaluation", pipeline_dict)

        eval_metrics = pipeline_dict["evaluation"]
        self.assertGreater(eval_metrics["R2"], 0.8, "R2 score should be > 0.8")
        self.assertLess(eval_metrics["MAE"], 200.0, "MAE should be < 200.0 tons (Actual RF MAE ~139 tons)")

    def test_02_bin_dataset_integrity(self):
        df_bins = load_bin_data()
        self.assertFalse(df_bins.empty, "Chennai_Bins.csv should not be empty")
        self.assertEqual(len(df_bins), 11058, "Should contain 11,058 bin records")
        self.assertTrue(df_bins["LATITUDE"].between(-90, 90).all(), "All latitudes must be valid (-90 to 90)")
        self.assertTrue(df_bins["LONGTITUDE"].between(-180, 180).all(), "All longitudes must be valid (-180 to 180)")
        self.assertTrue(np.isfinite(df_bins["LATITUDE"]).all(), "Latitudes must be finite numbers")
        self.assertTrue(np.isfinite(df_bins["LONGTITUDE"]).all(), "Longitudes must be finite numbers")
        self.assertEqual(df_bins.isnull().sum().sum(), 0, "No missing values allowed in bin dataset")

    def test_03_priority_engine(self):
        report = generate_forecast_priorities(2026, 10)
        self.assertIsNotNone(report, "Priority report should be generated")
        self.assertFalse(report.empty)
        self.assertIn("Collection_Priority", report.columns)
        self.assertIn("Priority_Percentile", report.columns)
        priorities = set(report["Collection_Priority"].unique())
        self.assertTrue(priorities.issubset({"High", "Medium", "Low"}))

    def test_04_ortools_tsp_solver_division1(self):
        df_bins = load_bin_data()
        div1_bins = df_bins[(df_bins["ZONE_CODE"] == "N01") & (df_bins["DIV_CODE"] == "N001")].reset_index(drop=True)
        self.assertGreaterEqual(len(div1_bins), 2)

        lats = div1_bins["LATITUDE"].tolist()
        lons = div1_bins["LONGTITUDE"].tolist()

        res = solve_ortools_tsp(lats, lons, depot=0, time_limit_sec=2)
        self.assertIsNotNone(res)
        self.assertEqual(res["optimized_route"][0], 0, "Route must start at depot 0")
        self.assertEqual(res["optimized_route"][-1], 0, "Route must return to depot 0")
        self.assertEqual(len(res["optimized_route"]), len(div1_bins) + 1, "Must visit all nodes plus return")
        self.assertEqual(len(set(res["optimized_route"][:-1])), len(div1_bins), "Must visit every node exactly once")
        self.assertLess(res["optimized_distance_km"], res["csv_distance_km"], "Optimized route must be shorter than CSV baseline")

    def test_05_ortools_tsp_solver_tnagar_all_and_core(self):
        df_bins = load_bin_data()
        tnagar_bins = df_bins[df_bins["DIV_CODE"] == "N136"].reset_index(drop=True)
        self.assertEqual(len(tnagar_bins), 157, "Division N136 (T. Nagar test area) must have 157 bin locations")

        # Test A: Including ALL 157 bins
        lats_all = tnagar_bins["LATITUDE"].tolist()
        lons_all = tnagar_bins["LONGTITUDE"].tolist()

        res_all = solve_ortools_tsp(lats_all, lons_all, depot=0, time_limit_sec=5)
        self.assertIsNotNone(res_all)
        self.assertEqual(res_all["locations_visited"], 157)
        self.assertEqual(res_all["optimized_route"][0], 0)
        self.assertEqual(res_all["optimized_route"][-1], 0)
        self.assertEqual(len(set(res_all["optimized_route"][:-1])), 157)

        self.assertAlmostEqual(res_all["csv_distance_km"], 86.633, delta=0.5)
        self.assertAlmostEqual(res_all["nn_distance_km"], 27.142, delta=1.0)
        self.assertLessEqual(res_all["optimized_distance_km"], 25.0)
        self.assertGreater(res_all["csv_reduction_pct"], 70.0, "CSV reduction should be ~73%")

        # Test B: Core 152 bins (Excluding 5 isolated >2km outliers)
        isolated = detect_isolated_bins(tnagar_bins, threshold_km=2.0)
        self.assertEqual(len(isolated), 5, "Must detect exactly 5 isolated >2km outliers")

        tnagar_core = tnagar_bins[~tnagar_bins["OBJECTID"].isin(isolated["OBJECTID"])].reset_index(drop=True)
        self.assertEqual(len(tnagar_core), 152, "Core cluster must have 152 bins")

        lats_core = tnagar_core["LATITUDE"].tolist()
        lons_core = tnagar_core["LONGTITUDE"].tolist()

        res_core = solve_ortools_tsp(lats_core, lons_core, depot=0, time_limit_sec=5)
        self.assertIsNotNone(res_core)
        self.assertEqual(res_core["locations_visited"], 152)
        self.assertEqual(res_core["optimized_route"][0], 0)
        self.assertEqual(res_core["optimized_route"][-1], 0)

        self.assertAlmostEqual(res_core["csv_distance_km"], 46.564, delta=0.5)
        self.assertAlmostEqual(res_core["nn_distance_km"], 17.366, delta=1.0)
        self.assertLessEqual(res_core["optimized_distance_km"], 15.0)

    def test_06_haversine_calculation(self):
        # Distance between Chennai Central and T. Nagar (~6.5 km)
        d = calculate_haversine_km(13.0827, 80.2707, 13.0418, 80.2341)
        self.assertGreater(d, 5.0)
        self.assertLess(d, 8.0)

    def test_07_forecast_input_builder_and_prediction_path(self):
        pipeline_dict = load_forecasting_pipeline()
        df_waste = load_waste_data()
        self.assertIsNotNone(pipeline_dict)

        model = pipeline_dict["model"]
        preprocessor = pipeline_dict["preprocessor"]
        feature_names = pipeline_dict["features"]

        base_all, X_all = build_forecast_input_df(
            pipeline_dict=pipeline_dict,
            df_waste=df_waste,
            forecast_year=2026,
            forecast_month=10,
            waste_type_select="All Waste Types",
            user_overrides={"Population": 7500000}
        )

        self.assertGreaterEqual(len(X_all), 1, "Must contain at least 1 row")
        self.assertEqual(list(X_all.columns), feature_names)

        X_proc_all = preprocessor.transform(X_all)
        preds_all = model.predict(X_proc_all)
        self.assertEqual(len(preds_all), len(X_all))
        self.assertTrue(np.all(preds_all >= 0))

    def test_08_styler_map_compatibility(self):
        report = generate_forecast_priorities(2026, 10)
        self.assertIsNotNone(report)

        def highlight_priority(val):
            if val == "High":
                return "background-color: #FEE2E2; color: #991B1B; font-weight: bold;"
            return "background-color: #D1FAE5; color: #065F46; font-weight: bold;"

        styler = report.style.map(highlight_priority, subset=["Collection_Priority"])
        html = styler.to_html()
        self.assertIn("<table", html)

    def test_09_solver_validation_and_error_handling(self):
        # Test 1: Single location (less than 2)
        with self.assertRaises(ValueError):
            solve_ortools_tsp([13.0827], [80.2707], depot=0)

        # Test 2: Out-of-bounds depot index
        with self.assertRaises(ValueError):
            solve_ortools_tsp([13.0827, 13.0418], [80.2707, 80.2341], depot=5)

        # Test 3: NaN or Infinite coordinates
        with self.assertRaises(ValueError):
            solve_ortools_tsp([13.0827, np.nan], [80.2707, 80.2341], depot=0)

        # Test 4: Out-of-bounds latitude
        with self.assertRaises(ValueError):
            solve_ortools_tsp([13.0827, 105.0], [80.2707, 80.2341], depot=0)

    def test_10_distance_matrix_properties(self):
        lats = [13.0827, 13.0418, 13.0067]
        lons = [80.2707, 80.2341, 80.2570]

        matrix = build_distance_matrix_metres(lats, lons)
        n = len(lats)

        self.assertEqual(len(matrix), n)
        self.assertEqual(len(matrix[0]), n)

        for i in range(n):
            self.assertEqual(matrix[i][i], 0, "Diagonal entries must be 0")
            for j in range(n):
                self.assertGreaterEqual(matrix[i][j], 0, "Distances must be non-negative")
                self.assertEqual(matrix[i][j], matrix[j][i], "Distance matrix must be symmetric")


if __name__ == "__main__":
    unittest.main()
