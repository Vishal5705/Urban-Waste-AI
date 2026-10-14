# AI-Driven Urban Waste Generation Forecasting and Dynamic Collection Route Optimization

**Final-Year B.Tech CSE Major Project**

---

## 📌 Project Overview
This project presents an integrated decision-support system designed for sustainable municipal solid waste management in metropolitan cities, specifically evaluated on **Greater Chennai Corporation**. It combines machine learning for urban waste generation forecasting, percentile-based collection prioritization, interactive GIS bin mapping, and Google OR-Tools constraint optimization for vehicle routing.

---

## 🚀 Key Features

1. **Waste Generation Forecasting**:
   - Uses a trained **Random Forest Regression** model on historical demographic, environmental, and temporal features.
   - Evaluated on holdout test data (2024–2025): **MAE = 139.13 tons/day**, **RMSE = 203.18 tons/day**, **R² = 0.9654** (explaining 96.5% of variance vs baseline MAE 983.14 tons/day).
   - Allows interactive scenario simulation by year, month, population, AQI, rainfall, and urbanization rate.

2. **Collection Priority Analysis**:
   - Classifies waste categories into **High (🔴 ≥ 75th percentile)**, **Medium (🟡 25th–75th percentile)**, and **Low (🟢 < 25th percentile)** urgency tiers.
   - Generates priority ranking reports to optimize vehicle fleet allocation.

3. **Chennai Bin Mapping**:
   - Maps **11,058 physical garbage bins** across all 200 divisions of Greater Chennai Corporation.
   - Interactive filtering by Area Name and Division Code with popups showing bin IDs, street names, and exact coordinates.
   - Flags geographically isolated bins located >2 km from median cluster centers.

4. **Dynamic Route Optimization**:
   - Implements **Google OR-Tools Guided Local Search** to solve the Traveling Salesperson Problem (TSP).
   - Demonstrates dramatic distance reductions on real bin clusters (e.g., **T. Nagar 157 bins**: **23.38 km** OR-Tools vs **86.63 km** CSV-order baseline — a **73.01% distance reduction**).
   - Solves routes interactively with depot selection and configurable search timeouts.

5. **Baseline Method Comparison**:
   - Quantitative evaluation across CSV-Order Baseline, Nearest-Neighbor Heuristic, and OR-Tools Metaheuristic.

6. **Exportable Reports & CSV Downloads**:
   - One-click export for prediction summaries, priority reports, stop-by-stop visit sequences, and baseline comparison tables.

---

## 📁 Project Directory Structure

```text
Urban_waste_AI/
├── app.py                            # Streamlit Web Dashboard Entry Point
├── utils.py                          # Utility Module (Loaders, Solvers, Priority Engine)
├── train_model.py                    # Model Training Pipeline Script
├── collection_priority.py            # CLI Collection Priority Engine
├── route_optimizer.py                # CLI Route Optimizer (Division 1 Test)
├── route_optimizer_tnagar.py         # CLI Route Optimizer (T. Nagar Test)
├── requirements.txt                  # Python Project Dependencies
├── README.md                         # Documentation & Setup Guide
├── PROJECT_REPORT_OUTLINE.md         # Comprehensive Report & Presentation Outline
│
├── dataset/                          # CSV Datasets
│   ├── Chennai_Bins.csv              # 11,058 Chennai Bin Locations
│   ├── Garbage_collection_clean.csv  # Cleaned Division Garbage Collection Data
│   ├── Waste_Management_India_20K.csv# 20,000 Waste Generation Records
│   ├── Route_Test_Division_1.csv     # Division 1 Routing Test Subset (14 bins)
│   └── Route_Test_TNagar.csv         # T. Nagar Routing Test Subset (157 bins)
│
├── models/                           # Trained ML Model Artifacts
│   └── waste_forecasting_pipeline.pkl# Saved Random Forest Model & Preprocessor
│
├── outputs/                          # Generated Results & Exports
│   ├── collection_priorities.csv
│   ├── tnagar_optimized_route.csv
│   ├── tnagar_route_comparison.csv
│   └── tnagar_isolated_locations_review.csv
│
└── scratch/                          # Internal Test Scripts
    └── test_integration.py           # Automated Integration Test Suite
```

---

## 🛠️ Installation & Setup (Windows)

### Prerequisites
- Python **3.9 to 3.12** installed on Windows.
- Git & PowerShell.

### Step 1: Open PowerShell in Project Folder
```powershell
cd c:\Users\visha\OneDrive\Documents\Urban_waste_AI
```

### Step 2: Install Required Dependencies
```powershell
pip install -r requirements.txt
```

---

## 🖥️ Launching the Application

Run the following command in PowerShell to start the Streamlit web dashboard:

```powershell
streamlit run app.py
```

The application will automatically open in your default browser at:
`http://localhost:8501`

---

## 🧪 Running Automated Integration Tests

To verify that all components, saved models, distance matrices, and solvers pass integration testing:

```powershell
python scratch/test_integration.py
```

---

## 🎓 College Evaluator Demonstration Guide (Viva)

When presenting this project to external evaluators or professors, structure your 10-minute demo as follows:

1. **📌 Overview (1 min)**:
   - Highlight the problem of municipal waste in Chennai (11,058 bins across 200 divisions).
   - Explain the 2-phase approach: ML Forecasting + OR-Tools TSP Routing.

2. **🔮 Forecasting & Priority Demo (3 mins)**:
   - Show the **Waste Generation Forecasting** tab. Explain the Random Forest model ($R^2 = 0.9654$).
   - Switch to **Collection Priority Analysis**. Demonstrate how percentile ranks assign High/Medium/Low priority.

3. **🗺️ Bin GIS Map Demo (2 mins)**:
   - Open **Chennai Bin Map**. Filter by Area (e.g., T. Nagar). Show popup metadata and geographically isolated bin warnings.

4. **🚛 Route Optimization & Benchmark Demo (4 mins)**:
   - Open **Route Optimization (OR-Tools)**. Select **T. Nagar** (157 bins). Click **Run OR-Tools Route Optimizer**.
   - Point out the **73.01% distance reduction** (23.38 km vs 86.63 km CSV order).
   - Download the route CSV file to demonstrate operational deliverability.

---

## ⚠️ Known Limitations & Scope
- **Distance Metric Standard**: Route distances are computed using straight-line (Haversine) spherical distance matrices in integer metres. They serve as Euclidean baseline benchmarks and do not model real road network topography or traffic lights.
- **Priority Scope**: Priority levels reflect relative waste-type volume percentiles rather than individual IoT bin fill-level sensors.
