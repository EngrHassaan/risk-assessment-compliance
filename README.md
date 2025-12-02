# 🌿 Pro_DESG Materiality & Sustainability Tool Kit

**Pro_DESG Materiality & Sustainability Tool Kit** is an interactive **data analysis and decision support system** built in Python using **Streamlit**.  
It enables organizations and sustainability teams to perform **Risk Assessment**, **ESGFP Scoring**, **Scenario Analysis (MCDA)**, and **Validation (DEA + Monte Carlo)** — all within a single, easy-to-use interface.

---

## 🚀 Features

### 🔹 1. Risk Assessment
- Add and visualize organizational or project risks.  
- Compute risk ratings (Probability × Severity).  
- Interactive risk bubble charts (Plotly).  
- Export risk tables and charts as PDF or CSV.

### 🔹 2. ESGFP Scoring
- Define ESGFP pillars and key issues.  
- Add technologies and score them manually or via CSV upload.  
- Automatically compute pillar averages and visualize results via:
  - Heatmaps  
  - Radar charts  
  - Parallel coordinate plots  
- Export all visuals and results to PDF.

### 🔹 3. Scenario Analysis (MCDA)
- Compare technologies using multiple decision-making methods:
  - **Weighted Average**, **WPM**, **TOPSIS**, **VIKOR**, **EDAS**, **MAUT**, **PCA**
- Assign pillar weights and normalize results for comparison.
- Visualize with interactive bar charts.
- Export scenario results to CSV or PDF.

### 🔹 4. Validation (DEA + Monte Carlo)
- Validate performance consistency via **approximate DEA (Data Envelopment Analysis)**.  
- Conduct **Monte Carlo sensitivity analysis** with random weight perturbations.  
- Visualize results (Frontier Probability, Peer Heatmaps, Rankograms).  
- Export summary tables and diagnostics to PDF.

### 🔹 5. Report Generation
- Export **individual tab reports** (Risk, ESGFP, Scenarios, Validation) as PDFs.
- Export **full combined report** with all charts, tables, and a title cover page.

---

## 🧰 Tech Stack

| Layer | Technology |
|-------|-------------|
| Frontend | [Streamlit](https://streamlit.io/) |
| Visualization | Plotly, Matplotlib |
| PDF Generation | ReportLab |
| Data Handling | Pandas, NumPy |
| Image Conversion | Pillow, Kaleido |
| Core Model Logic | `risk_assessment.py` (custom ESGFP + Risk model) |

---

## 🧩 Folder Structure

📂 pro_desg_toolkit/
├── risk_assessment.py # Main Streamlit application
├── risk_assessment_model.py # Core computation and model logic
├── requirements.txt # Dependencies
├── README.md # Project documentation
└── /data # Optional: Example CSVs, reports, etc.

⚡ Quick Demo

You can load and execute a demo dataset without external files:

In the sidebar, click “Run Example End-to-End”.
This will:

Load sample ESGFP data

Run risk, ESGFP, scenario, and validation workflows

Generate ready-to-export charts and PDF reports

🧾 PDF Reports

Each export includes:

Cover Page — with project title and generation timestamp.

Data Tables — risk, ESGFP, pillar averages, MCDA, DEA, etc.

Visuals — all charts embedded automatically (Matplotlib + Plotly).

You can export:

Individual tab reports, or

A Full Combined Report containing all results.

📊 Example Workflows
Step	Description1️⃣ Risk Input	Add multiple risks with probability and severity.
2️⃣ ESGFP Scores	Input or upload technology ESGFP scores (1–9).
3️⃣ Scenario Run	Choose decision methods and assign weights.
4️⃣ Validation	Run DEA + Monte Carlo to verify model stability.
5️⃣ Export	Download PDF report with full results.
🧱 Core Model (risk_assessment.py)

The model implements:

TOPSIS, VIKOR, WPM, EDAS, MAUT, PCA for multi-criteria decision analysis.

DEA diagnostics for frontier validation.

Monte Carlo simulations for sensitivity analysis.

Matplotlib-based visualization utilities.

This file can also be used standalone (e.g., integrated with other analytical pipelines).

🧠 License & Usage

This project is internal to Pro_DESG for sustainability and ESG data management.
All rights reserved © 2025 Pro_DESG Analytics Team.

💡 Future Enhancements

 User authentication (login-based access).

 Cloud data sync for team collaboration.

 Custom branding (logo on PDF reports).

 API endpoints for automated report generation.

