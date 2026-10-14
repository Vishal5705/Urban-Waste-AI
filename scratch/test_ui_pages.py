import unittest
from pathlib import Path
from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")

class TestDashboardUI(unittest.TestCase):
    def test_city_overview_renders(self):
        at = AppTest.from_file(APP_PATH, default_timeout=30)
        at.run()
        self.assertFalse(at.exception, f"App threw exception on default page: {at.exception}")
        all_md = " ".join([m.value for m in at.markdown])
        self.assertIn("Urban Waste Intelligence", all_md)

    def test_waste_forecasting_page_renders(self):
        at = AppTest.from_file(APP_PATH, default_timeout=30)
        at.run()
        # Select 'Waste Forecasting'
        radio = at.sidebar.radio[0]
        radio.set_value("Waste Forecasting").run()
        self.assertFalse(at.exception, f"App threw exception on Waste Forecasting page: {at.exception}")

    def test_route_optimization_page_renders(self):
        at = AppTest.from_file(APP_PATH, default_timeout=30)
        at.run()
        radio = at.sidebar.radio[0]
        radio.set_value("Collection Route Optimization").run()
        self.assertFalse(at.exception, f"App threw exception on Route Optimization page: {at.exception}")

    def test_bin_locations_page_renders(self):
        at = AppTest.from_file(APP_PATH, default_timeout=30)
        at.run()
        radio = at.sidebar.radio[0]
        radio.set_value("Bin Locations").run()
        self.assertFalse(at.exception, f"App threw exception on Bin Locations page: {at.exception}")

    def test_model_performance_page_renders(self):
        at = AppTest.from_file(APP_PATH, default_timeout=30)
        at.run()
        radio = at.sidebar.radio[0]
        radio.set_value("Model Performance").run()
        self.assertFalse(at.exception, f"App threw exception on Model Performance page: {at.exception}")

    def test_reports_and_downloads_page_renders(self):
        at = AppTest.from_file(APP_PATH, default_timeout=30)
        at.run()
        radio = at.sidebar.radio[0]
        radio.set_value("Reports and Downloads").run()
        self.assertFalse(at.exception, f"App threw exception on Reports and Downloads page: {at.exception}")

if __name__ == "__main__":
    unittest.main()
