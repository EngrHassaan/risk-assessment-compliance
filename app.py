"""
Pro_DESG Materiality & Sustainability Tool Kit
Integrated Streamlit app with three pages:
1. Home - Overview and instructions
2. Risk Analysis - Existing risk-assessment.py functionality
3. Compliance TEA - New compliance.py integration
"""

import importlib.util
import os
import io
import tempfile
import sys
from datetime import datetime
from typing import List, Dict

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import json

# PDF tools
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image as RLImage,
    Table as RLTable,
    TableStyle,
    PageBreak,
)

# ---------------- CONFIG ----------------
MODEL_PATH = "model/risk_assessment_model.py"
COMPLIANCE_PATH = "model/compliance.py"
APP_TITLE = "Pro_DESG Materiality & Sustainability Tool Kit"

# ----------------- Utilities: dynamic import -----------------
def load_model_module(path: str):
    if not os.path.exists(path):
        st.error(f"Model file not found at: {path}")
        st.stop()
    spec = importlib.util.spec_from_file_location("esgfp_model", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# Load risk assessment module
try:
    mod = load_model_module(MODEL_PATH)
    DEFAULT_ESGFP = mod.DEFAULT_ESGFP if hasattr(mod, 'DEFAULT_ESGFP') else {}
except Exception as e:
    st.error(f"Failed to load risk assessment model: {e}")
    mod = None
    DEFAULT_ESGFP = {}

# ----------------- Session State Initialization -----------------
st.set_page_config(page_title=APP_TITLE, layout="wide", initial_sidebar_state="expanded")

# Initialize session state for Risk Analysis
if "risks" not in st.session_state:
    st.session_state.risks = []
if "scores_by_tech" not in st.session_state:
    st.session_state.scores_by_tech = {}
if "pillars" not in st.session_state:
    st.session_state.pillars = dict(DEFAULT_ESGFP)
if "pillar_avgs_df" not in st.session_state:
    st.session_state.pillar_avgs_df = pd.DataFrame()
if "last_weights" not in st.session_state:
    st.session_state.last_weights = None
if "last_methods" not in st.session_state:
    st.session_state.last_methods = None
if "scenario_results" not in st.session_state:
    st.session_state.scenario_results = {}

# Initialize session state for Compliance TEA
if "tea_params" not in st.session_state:
    st.session_state.tea_params = {}
if "tea_results" not in st.session_state:
    st.session_state.tea_results = None

# ----------------- TEA Calculation Functions -----------------
def calculate_capex(params: Dict) -> float:
    """Calculate total capital expenditure."""
    C_PE = params.get("C_PE", 1e8)
    f_ins = params.get("f_ins", 0.30)
    f_pipe = params.get("f_pipe", 0.45)
    f_elec = params.get("f_elec", 0.10)
    f_bldg = params.get("f_bldg", 0.05)
    f_util = params.get("f_util", 0.06)
    f_stor = params.get("f_stor", 0.02)
    f_safe = params.get("f_safe", 0.01)
    f_waste = params.get("f_waste", 0.01)
    
    # Direct costs
    direct_cost = C_PE * (1 + f_ins + f_pipe + f_elec + f_bldg + f_util + f_stor + f_safe + f_waste)
    
    # Indirect costs
    f_eng = params.get("f_eng", 0.12)
    f_cons = params.get("f_cons", 0.10)
    f_licn = params.get("f_licn", 0.00)
    f_cont = params.get("f_cont", 0.02)
    
    indirect_cost = direct_cost * (f_eng + f_cons + f_licn + f_cont)
    
    # Contingency and other costs
    f_contg = params.get("f_contg", 0.0)
    f_insur = params.get("f_insur", 0.01)
    f_own = params.get("f_own", 0.02)
    f_start = params.get("f_start", 0.01)
    
    other_cost = direct_cost * (f_contg + f_insur + f_own + f_start)
    
    total_capex = direct_cost + indirect_cost + other_cost
    return total_capex

def calculate_opex(params: Dict, capex: float) -> float:
    """Calculate annual operating expenses."""
    C_RM = params.get("C_RM", 4e7)
    C_UT = params.get("C_UT", 1.2e7)
    COL = params.get("COL", 1e7)
    C_CAT = params.get("C_CAT", 2e6)
    f_esg = params.get("f_esg", 0.07)
    f_pack = params.get("f_pack", 0.02)
    f_risk_op = params.get("f_risk_op", 0.05)
    
    # Raw materials, utilities, labor, catalysts
    direct_opex = C_RM + C_UT + COL + C_CAT
    
    # ESG compliance cost
    esg_cost = direct_opex * f_esg
    
    # Packaging cost
    packaging_cost = direct_opex * f_pack
    
    # Risk operational cost
    risk_cost = direct_opex * f_risk_op
    
    # Maintenance as percentage of CAPEX (typically 2-5%)
    maintenance_cost = capex * 0.03
    
    total_opex = direct_opex + esg_cost + packaging_cost + risk_cost + maintenance_cost
    return total_opex

def calculate_carbon_cost(params: Dict, production: float) -> float:
    """Calculate carbon emission costs."""
    tau_CO2 = params.get("tau_CO2", 50.0)
    E_CO2 = params.get("E_CO2", 200000.0)
    
    # Carbon emission cost
    carbon_cost = E_CO2 * tau_CO2
    return carbon_cost

def calculate_depreciation(capex: float, salvage_value: float, years: int, method: str = "SL") -> List[float]:
    """Calculate depreciation schedule."""
    depreciable_value = capex - salvage_value
    
    if method == "SL":  # Straight Line
        annual_dep = depreciable_value / years
        return [annual_dep] * years
    elif method == "SYD":  # Sum of Years' Digits
        schedule = []
        sum_years = years * (years + 1) / 2
        for i in range(years, 0, -1):
            dep = depreciable_value * (i / sum_years)
            schedule.append(dep)
        return schedule
    else:  # DDB - Double Declining Balance
        schedule = []
        book_value = capex
        rate = 2 / years
        for _ in range(years):
            dep = book_value * rate
            # Ensure we don't depreciate below salvage value
            if book_value - dep < salvage_value:
                dep = book_value - salvage_value
            schedule.append(dep)
            book_value -= dep
        return schedule

def calculate_crf(discount_rate: float, years: int) -> float:
    """Calculate Capital Recovery Factor."""
    if discount_rate == 0:
        return 1 / years
    return discount_rate * (1 + discount_rate)**years / ((1 + discount_rate)**years - 1)

def run_tea_analysis(params: Dict) -> Dict:
    """Run complete TEA analysis."""
    try:
        # Extract parameters
        Q_prod = params.get("Q_prod", 5e5)  # tons/year
        P_prod = params.get("P_prod", 550.0)  # $/ton
        N_project = params.get("N_project", 20)  # years
        L_asset = params.get("L_asset", 20)  # years
        salv_frac = params.get("salv_frac", 0.10)
        i_base = params.get("i_base", 0.08)
        delta_risk = params.get("delta_risk", 0.03)
        dep_method = params.get("dep_method", "SL")
        
        # Calculate metrics
        total_capex = calculate_capex(params)
        salvage_value = total_capex * salv_frac
        annual_opex = calculate_opex(params, total_capex)
        carbon_cost = calculate_carbon_cost(params, Q_prod)
        
        # Revenue calculation
        annual_revenue = Q_prod * P_prod
        
        # Depreciation schedule
        depreciation = calculate_depreciation(total_capex, salvage_value, L_asset, dep_method)
        
        # Initialize cash flow arrays
        years = list(range(N_project + 1))  # Year 0 to N_project
        cash_flow = [0] * (N_project + 1)
        discounted_cash_flow = [0] * (N_project + 1)
        cumulative_cf = [0] * (N_project + 1)
        
        # Year 0: Initial investment (negative cash flow)
        cash_flow[0] = -total_capex
        discounted_cash_flow[0] = cash_flow[0]  # No discount for year 0
        cumulative_cf[0] = cash_flow[0]
        
        # Years 1 to N_project
        total_discounted_benefits = 0
        total_discounted_costs = 0
        
        for year in range(1, N_project + 1):
            # Depreciation for this year (if asset still in use)
            dep = depreciation[min(year - 1, len(depreciation) - 1)] if year <= L_asset else 0
            
            # Operating profit
            operating_profit = annual_revenue - annual_opex - carbon_cost - dep
            
            # Tax calculation (assuming 25% tax rate)
            tax_rate = 0.25
            tax = max(operating_profit, 0) * tax_rate
            
            # Net cash flow
            net_cash = operating_profit - tax + dep
            
            # Add salvage value in final year if asset fully depreciated
            if year == N_project and year >= L_asset:
                net_cash += salvage_value
            
            cash_flow[year] = net_cash
            
            # Discount cash flow
            discount_factor = 1 / ((1 + i_base + delta_risk) ** year)
            discounted_cash_flow[year] = net_cash * discount_factor
            
            # Cumulative calculations
            cumulative_cf[year] = cumulative_cf[year - 1] + cash_flow[year]
            
            # For BCR calculation
            if net_cash > 0:
                total_discounted_benefits += discounted_cash_flow[year]
            else:
                total_discounted_costs += abs(discounted_cash_flow[year])
        
        # Calculate NPV
        npv = sum(discounted_cash_flow)
        
        # Calculate IRR (simplified)
        irr = 0.0
        if npv > 0:
            # Simple IRR approximation
            try:
                import numpy as np
                irr = np.irr(cash_flow)
                if np.isnan(irr):
                    irr = 0.15  # Default reasonable IRR
            except:
                irr = 0.15  # Default reasonable IRR
        
        # Calculate LCOx (Levelized Cost of Production)
        total_discounted_cost = sum(discounted_cash_flow[1:])  # Exclude initial investment
        total_discounted_production = sum([Q_prod / ((1 + i_base + delta_risk) ** year) for year in range(1, N_project + 1)])
        lcox = -total_discounted_cost / total_discounted_production if total_discounted_production > 0 else 0
        
        # Calculate Capital Recovery Factor
        crf = calculate_crf(i_base + delta_risk, N_project)
        annualized_capex = total_capex * crf
        
        # Calculate BCR (Benefit-Cost Ratio)
        bcr = total_discounted_benefits / abs(total_discounted_costs) if total_discounted_costs > 0 else float('inf')
        
        # Payback period
        payback_period = None
        for year in range(N_project + 1):
            if cumulative_cf[year] >= 0:
                payback_period = year
                break
        
        # Return results
        results = {
            "CAPEX": total_capex,
            "Annual_OPEX": annual_opex,
            "Annual_Revenue": annual_revenue,
            "NPV": npv,
            "IRR": irr,
            "LCOx": lcox,
            "Payback_Period": payback_period,
            "BCR": bcr,
            "CRF": crf,
            "Annualized_CAPEX": annualized_capex,
            "CF": cash_flow,
            "Discounted_CF": discounted_cash_flow,
            "Cumulative_CF": cumulative_cf,
            "Depreciation_Schedule": depreciation,
            "Salvage_Value": salvage_value,
            "Carbon_Cost": carbon_cost,
            "Production_Rate": Q_prod,
            "Product_Price": P_prod,
            "Project_Life": N_project,
            "Discount_Rate": i_base + delta_risk,
            "Parameters_Used": params
        }
        
        return results
        
    except Exception as e:
        st.error(f"Error in TEA calculation: {str(e)}")
        return None

# ----------------- Sidebar Navigation -----------------
st.sidebar.title("🔧 Navigation")
page = st.sidebar.radio("Go to:", ["🏠 Home", "📊 Risk Analysis", "💰 Compliance TEA"])

# Sidebar quick actions (for Risk Analysis)
st.sidebar.markdown("---")
st.sidebar.header("Quick Actions")

if page == "📊 Risk Analysis" and mod:
    if st.sidebar.button("Load example dataset"):
        pillars = dict(DEFAULT_ESGFP)
        techs = {}
        # Create example data for 2 technologies
        for tech_idx, tech_name in enumerate(["Process Design A", "Process Design B"]):
            tech_scores = {}
            for pillar_idx, (pillar, subs) in enumerate(pillars.items()):
                for sub_idx, sub in enumerate(subs):
                    # Create varying scores
                    base_score = 5.0 + tech_idx * 1.5  # Tech A: 5.0, Tech B: 6.5
                    exposure = 0.25 * ((pillar_idx + sub_idx) % 3)  # Varying exposure
                    final_score = round(base_score * (1.0 + exposure), 3)
                    tech_scores[f"{pillar}:{sub}"] = final_score
            techs[tech_name] = tech_scores
        
        st.session_state.pillars = pillars
        st.session_state.scores_by_tech = techs
        st.sidebar.success("Example dataset loaded with 2 process designs.")
    
    def make_example_csv_bytes(pillars: Dict[str, List[str]]) -> bytes:
        cols = []
        for p, subs in pillars.items():
            for sub in subs:
                cols.append(f"{p}:{sub}")
        df = pd.DataFrame(columns=["Tech"] + cols)
        df.loc[0] = ["Tech A"] + [5.0 for _ in cols]
        df.loc[1] = ["Tech B"] + [6.0 for _ in cols]
        b = io.BytesIO()
        df.to_csv(b, index=False)
        return b.getvalue()

st.sidebar.markdown("---")
st.sidebar.write("**Dependencies:**")
st.sidebar.write("streamlit, pandas, numpy, matplotlib, plotly, kaleido, pillow, reportlab")

# ----------------- Home Page -----------------
if page == "🏠 Home":
    st.title("🏠 Welcome to Pro_DESG Materiality & Sustainability Tool Kit")
    
    st.markdown("""
    ## 📋 Overview
    
    This integrated toolkit provides comprehensive analysis for:
    
    **1. 📊 Risk Analysis & ESGFP Assessment**
    - Materiality assessment and risk identification
    - ESGFP (Environmental, Social, Governance, Financial, Process) scoring
    - Multi-criteria decision analysis (MCDA) with 8+ methods
    - DEA validation and Monte Carlo sensitivity analysis
    
    **2. 💰 Compliance & ESG TEA (Techno-Economic Analysis)**
    - Economic feasibility assessment
    - ESG compliance cost integration
    - Cash flow analysis and financial metrics
    - Scenario analysis and sensitivity testing
    
    ## 🚀 Getting Started
    
    1. **Risk Analysis Tab**: Start with risk assessment and ESGFP scoring
    2. **Compliance TEA Tab**: Run techno-economic analysis with ESG integration
    3. **Export Results**: Generate comprehensive PDF reports
    """)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("""
        **Risk Assessment**
        
        Identify and quantify project risks with probability-severity analysis
        """)
    with col2:
        st.success("""
        **ESGFP Scoring**
        
        Evaluate sustainability across 5 pillars with customizable key issues
        """)
    with col3:
        st.warning("""
        **TEA Analysis**
        
        Calculate financial viability with ESG compliance costs
        """)
    
    st.markdown("---")
    st.markdown("### 📈 Sample Workflows")
    
    workflow = st.selectbox("Choose a sample workflow:", 
                          ["Quick Assessment", "Full Analysis", "Compliance Check"])
    
    if workflow == "Quick Assessment":
        st.markdown("""
        1. Go to **Risk Analysis** tab
        2. Load example dataset
        3. Run scenario analysis with default weights
        4. View results and export
        """)
    elif workflow == "Full Analysis":
        st.markdown("""
        1. Upload your ESGFP data in CSV format
        2. Configure risk parameters
        3. Run MCDA with multiple methods
        4. Validate with DEA and Monte Carlo
        5. Export full PDF report
        """)
    else:
        st.markdown("""
        1. Go to **Compliance TEA** tab
        2. Configure TEA parameters
        3. Run analysis with ESG cost factors
        4. Review financial metrics and cash flows
        """)

# ----------------- Risk Analysis Page -----------------
elif page == "📊 Risk Analysis":
    if mod is None:
        st.error("Risk assessment model failed to load. Please check model/risk_assessment_model.py")
        st.stop()
    
    st.title("📊 Risk Analysis & ESGFP Assessment")
    
    # Create tabs for different sections
    risk_tabs = st.tabs(["Risk Assessment", "ESGFP Scoring", "Scenario Analysis", "Validation", "Export"])
    
    # ----- Risk Assessment Tab -----
    with risk_tabs[0]:
        st.header("🔍 Risk Assessment")
        col_l, col_r = st.columns([2, 1])
        
        with col_l:
            st.subheader("Add / Edit Risks")
            with st.form("risk_form", clear_on_submit=True):
                rname = st.text_input("Risk name", "New Risk")
                prob = st.number_input("Probability (0 < p ≤ 1)", min_value=0.0001, max_value=1.0, value=0.1, format="%.4f")
                sev = st.number_input("Severity (1–10)", min_value=1.0, max_value=10.0, value=5.0, format="%.2f")
                add = st.form_submit_button("➕ Add risk")
                if add:
                    st.session_state.risks.append({"name": rname, "prob": float(prob), "sev": float(sev)})
                    st.success(f"Added risk {rname}.")
            
            if st.session_state.risks:
                df_r = pd.DataFrame([
                    {"Risk": r["name"], "Probability": r["prob"], "Severity": r["sev"], "Rating": r["prob"] * r["sev"]}
                    for r in st.session_state.risks
                ])
                st.dataframe(df_r.round(4))
                if st.button("🗑️ Clear all risks"):
                    st.session_state.risks = []
                    st.rerun()
            else:
                st.info("No risks added yet. Add your first risk above.")
        
        with col_r:
            st.subheader("Risk Visuals")
            if st.session_state.risks:
                risks_objs = [mod.Risk(name=r["name"], probability=r["prob"], severity=r["sev"]) for r in st.session_state.risks]
                df_for_plot = mod.risk_dataframe(risks_objs)
                
                # Generate and display risk plots
                try:
                    # Create a figure for heatmap
                    fig1, ax1 = plt.subplots(figsize=(8, 6))
                    attrs = df_for_plot.set_index("Risk")[["Probability", "Severity", "Rating"]]
                    im = ax1.imshow(attrs.values)
                    ax1.set_xticks(range(attrs.shape[1]))
                    ax1.set_xticklabels(attrs.columns)
                    ax1.set_yticks(range(attrs.shape[0]))
                    ax1.set_yticklabels(attrs.index)
                    for i in range(attrs.shape[0]):
                        for j in range(attrs.shape[1]):
                            ax1.text(j, i, f"{attrs.values[i, j]:.2f}", ha="center", va="center")
                    ax1.set_title("Risk Attributes Heatmap")
                    fig1.colorbar(im, ax=ax1)
                    fig1.tight_layout()
                    st.pyplot(fig1)
                    
                    # Create bar plot
                    fig2, ax2 = plt.subplots(figsize=(10, 6))
                    df_for_plot.plot(x="Risk", y="Rating", kind="bar", rot=45, ax=ax2, legend=False)
                    ax2.set_ylabel("Risk Rating")
                    ax2.set_title("Risk Rating Bar Plot")
                    ax2.grid(True)
                    fig2.tight_layout()
                    st.pyplot(fig2)
                    
                    # Create bubble plot (materiality assessment)
                    fig3, ax3 = plt.subplots(figsize=(16, 9))
                    x_min, x_max = 0.0, 1.0
                    y_min, y_max = 1.0, 10.0
                    nx, ny = 500, 500
                    gx = np.linspace(x_min, x_max, nx)
                    gy = np.linspace(y_min, y_max, ny)
                    gy01 = (gy - y_min) / (y_max - y_min)
                    Z = gy01[:, None] + gx[None, :]
                    Z = (Z - Z.min()) / (Z.max() - Z.min())
                    ax3.imshow(
                        Z, extent=[x_min, x_max, y_min, y_max], origin="lower",
                        cmap="RdYlGn_r", alpha=0.85, aspect="auto", interpolation="bilinear",
                    )
                    x = df_for_plot["Probability"].to_numpy()
                    y = df_for_plot["Severity"].to_numpy()
                    r = df_for_plot["Rating"].to_numpy()
                    labels = df_for_plot["Risk"].astype(str).to_list()
                    sizes = np.clip(r, 0.05, None) * 600.0
                    ax3.scatter(x, y, s=sizes, edgecolors="black", linewidths=0.8, alpha=0.95, zorder=3)
                    for xi, yi, lab in zip(x, y, labels):
                        offset_x = 18 if xi < 0.7 else -18
                        ha = "left" if xi < 0.7 else "right"
                        ax3.annotate(
                            lab, xy=(xi, yi), xytext=(offset_x, 10), textcoords="offset points",
                            ha=ha, va="bottom", fontsize=13, color="black",
                            arrowprops=dict(arrowstyle="-", color="gray", lw=0.9, alpha=0.9),
                            zorder=4,
                        )
                    ax3.set_xlim(x_min, x_max)
                    ax3.set_ylim(y_min, y_max)
                    ax3.set_xlabel("Probability (Likelihood)")
                    ax3.set_ylabel("Severity (Impact)")
                    ax3.set_title("Materiality Assessment Diagram – Likelihood × Impact")
                    ax3.grid(False)
                    fig3.tight_layout()
                    st.pyplot(fig3)
                    
                except Exception as e:
                    st.error(f"Error generating plots: {e}")
                
                # Plotly bubble chart
                fig = px.scatter(
                    df_for_plot,
                    x="Probability", y="Severity", size="Rating", hover_name="Risk",
                    title="Risk Bubble – Likelihood × Impact",
                    labels={"Probability":"Probability", "Severity":"Severity"}
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Download
                csv_b = io.BytesIO()
                df_for_plot.to_csv(csv_b, index=False)
                csv_b.seek(0)
                st.download_button("📥 Download risk table (CSV)", data=csv_b, file_name="risk_table.csv", mime="text/csv")
            else:
                st.info("Add risks to see visuals.")
    
    # ----- ESGFP Scoring Tab -----
    with risk_tabs[1]:
        st.header("📦 ESGFP Scoring")
        left, right = st.columns([2, 1])
        
        with left:
            st.subheader("Pillars & Key Issues")
            for p, subs in st.session_state.pillars.items():
                with st.expander(f"📌 {p}"):
                    st.write(f"**Key Issues:** {', '.join(subs)}")
            
            st.markdown("---")
            st.subheader("Add Process Design")
            with st.form("add_tech_form", clear_on_submit=True):
                tech_name = st.text_input("Process Design name", f"Process Design {len(st.session_state.scores_by_tech)+1}")
                inputs = {}
                for p, subs in st.session_state.pillars.items():
                    st.markdown(f"**{p}**")
                    for sub in subs:
                        key = f"{p}:{sub}"
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            base = st.number_input(f"{sub} base (1–9)", min_value=1.0, max_value=9.0, value=5.0, 
                                                 key=f"{tech_name}_{key}_base")
                        with col2:
                            exposure = st.slider("Exp", min_value=0.0, max_value=1.0, value=0.0, 
                                               key=f"{tech_name}_{key}_exp")
                        inputs[key] = round(base * (1.0 + exposure), 4)
                added = st.form_submit_button("➕ Add Process Design")
                if added:
                    label = tech_name
                    k = 2
                    while label in st.session_state.scores_by_tech:
                        label = f"{tech_name} ({k})"; k += 1
                    st.session_state.scores_by_tech[label] = inputs
                    st.success(f"Added {label}.")
            
            st.markdown("---")
            st.subheader("Upload ESGFP CSV")
            uploaded = st.file_uploader("Choose CSV file", type=["csv"], 
                                      help="CSV should have 'Tech' column and 'Pillar:KeyIssue' columns")
            if uploaded:
                try:
                    df_up = pd.read_csv(uploaded)
                    if "Tech" not in df_up.columns and df_up.shape[1] >= 1:
                        df_up.columns.values[0] = "Tech"
                    df_up = df_up.set_index("Tech")
                    
                    # Update pillars with any new columns
                    for col in df_up.columns:
                        if ":" in col:
                            p, sub = col.split(":", 1)
                            if p not in st.session_state.pillars:
                                st.session_state.pillars[p] = []
                            if sub not in st.session_state.pillars[p]:
                                st.session_state.pillars[p].append(sub)
                    
                    # Update scores
                    for tech, row in df_up.iterrows():
                        st.session_state.scores_by_tech[str(tech)] = row.dropna().to_dict()
                    
                    st.success(f"Uploaded {len(df_up)} process designs.")
                except Exception as e:
                    st.error(f"Error reading CSV: {e}")
            
            st.markdown("---")
            st.subheader("Current Process Designs")
            if st.session_state.scores_by_tech:
                st.dataframe(pd.DataFrame(st.session_state.scores_by_tech).round(4))
                if st.button("🗑️ Clear all designs"):
                    st.session_state.scores_by_tech = {}
                    st.rerun()
            else:
                st.info("No process designs added yet. Add manually, upload CSV, or load example dataset.")
        
        with right:
            st.subheader("Visualization")
            if not st.session_state.scores_by_tech:
                st.info("Add process designs first.")
            else:
                pillar_avgs = mod.pillar_averages_multi(st.session_state.scores_by_tech, st.session_state.pillars)
                st.session_state.pillar_avgs_df = pillar_avgs.copy()
                
                st.write("**Pillar Averages:**")
                st.dataframe(pillar_avgs.round(4))
                
                # Generate and display pillar heatmaps
                try:
                    # Heatmap 1: Raw values
                    fig1, ax1 = plt.subplots(figsize=(10, 6))
                    data = pillar_avgs.copy()
                    im = ax1.imshow(data.values, aspect="auto")
                    ax1.set_xticks(range(data.shape[1]))
                    ax1.set_xticklabels(data.columns, rotation=45, ha="right")
                    ax1.set_yticks(range(data.shape[0]))
                    ax1.set_yticklabels(data.index)
                    for i in range(data.shape[0]):
                        for j in range(data.shape[1]):
                            ax1.text(j, i, f"{data.values[i, j]:.2f}", ha="center", va="center", fontsize=9)
                    ax1.set_title("Pillar Averages – Raw")
                    fig1.colorbar(im, ax=ax1, shrink=0.8, label="Score")
                    fig1.tight_layout()
                    st.pyplot(fig1)
                    
                    # Heatmap 2: Z-score
                    fig2, ax2 = plt.subplots(figsize=(10, 6))
                    mean = data.mean(axis=1)
                    std = data.std(axis=1).replace(0, 1.0)
                    z = data.sub(mean, axis=0).div(std, axis=0)
                    im = ax2.imshow(z.values, aspect="auto", cmap="coolwarm")
                    ax2.set_xticks(range(z.shape[1]))
                    ax2.set_xticklabels(z.columns, rotation=45, ha="right")
                    ax2.set_yticks(range(z.shape[0]))
                    ax2.set_yticklabels(z.index)
                    for i in range(z.shape[0]):
                        for j in range(z.shape[1]):
                            ax2.text(j, i, f"{z.values[i, j]:.2f}", ha="center", va="center", fontsize=9)
                    ax2.set_title("Pillar Averages – Z-score by Pillar")
                    fig2.colorbar(im, ax=ax2, shrink=0.8, label="z")
                    fig2.tight_layout()
                    st.pyplot(fig2)
                    
                except Exception as e:
                    st.error(f"Error generating heatmaps: {e}")
                
                # Radar chart
                try:
                    fig3, ax3 = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
                    pillars = pillar_avgs.index.tolist()
                    techs = pillar_avgs.columns.tolist()
                    norm = (pillar_avgs / mod.WEIGHTED_THEORETICAL_MAX) * mod.OUTPUT_SCALE
                    
                    angles = np.linspace(0, 2 * np.pi, len(pillars), endpoint=False).tolist()
                    angles += angles[:1]
                    
                    for tech in techs:
                        vals = norm[tech].tolist()
                        vals += vals[:1]
                        ax3.plot(angles, vals, linewidth=2, alpha=0.9, label=tech)
                        ax3.fill(angles, vals, alpha=0.1)
                    
                    ax3.set_xticks(angles[:-1])
                    ax3.set_xticklabels(pillars)
                    ax3.set_yticklabels([])
                    ax3.set_ylim(0, 10)
                    ax3.set_title("Pillar Profiles (Radar, 0–10)")
                    ax3.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), frameon=False)
                    fig3.tight_layout()
                    st.pyplot(fig3)
                    
                except Exception as e:
                    st.error(f"Error generating radar chart: {e}")
                
                # Parallel coordinates
                try:
                    fig4, ax4 = plt.subplots(figsize=(12, 6))
                    data = pillar_avgs.copy()
                    lo = data.min(axis=1)
                    hi = data.max(axis=1)
                    denom = (hi - lo).replace(0, 1.0)
                    norm = data.sub(lo, axis=0).div(denom, axis=0)
                    x = np.arange(len(data.index))
                    
                    for tech in norm.columns:
                        ax4.plot(x, norm[tech].values, marker="o", linewidth=2, alpha=0.9, label=tech)
                    
                    ax4.set_xticks(x)
                    ax4.set_xticklabels(data.index)
                    ax4.set_ylim(0, 1)
                    ax4.set_ylabel("Normalized (0–1)")
                    ax4.set_title("Parallel Coordinates – Pillar Profiles")
                    ax4.legend(loc="upper right", ncol=2, frameon=False)
                    fig4.tight_layout()
                    st.pyplot(fig4)
                    
                except Exception as e:
                    st.error(f"Error generating parallel coordinates: {e}")
                
                # Download
                csv_b = io.BytesIO()
                pillar_avgs.to_csv(csv_b)
                csv_b.seek(0)
                st.download_button("📥 Download averages", data=csv_b, 
                                 file_name="pillar_averages.csv", mime="text/csv")
    
    # ----- Scenario Analysis Tab -----
    with risk_tabs[2]:
        st.header("🎯 Scenario Analysis (MCDA)")
        
        if st.session_state.pillar_avgs_df.empty:
            st.info("Create pillar averages in ESGFP Scoring tab first.")
        else:
            pillar_avgs = st.session_state.pillar_avgs_df.copy()
            
            st.subheader("Configure Weights")
            weights = {}
            cols = st.columns(len(pillar_avgs.index))
            for i, p in enumerate(pillar_avgs.index):
                with cols[i]:
                    weights[p] = st.slider(f"{p} %", 0, 100, 
                                         int(100/len(pillar_avgs.index)), 
                                         key=f"weight_{p}")
            
            total = sum(weights.values())
            if total != 100:
                st.warning(f"Weights sum to {total}%. Adjust to sum to 100%.")
            
            st.subheader("Select Methods")
            methods = st.multiselect(
                "Choose MCDA methods:",
                ["WEIGHTED", "WPM", "RANK", "TOPSIS", "VIKOR", "EDAS", "MAUT", "PCA"],
                default=["WEIGHTED", "TOPSIS"]
            )
            
            norm_flag = st.checkbox("Normalize to 0-10 scale", value=True)
            
            if st.button("🚀 Run Scenario Analysis", type="primary"):
                if total != 100:
                    st.error("Weights must sum to 100%")
                else:
                    with st.spinner("Running analysis..."):
                        per_method = {}
                        if "WEIGHTED" in methods:
                            per_method["WEIGHTED"] = mod.method_weighted(pillar_avgs, weights)
                        if "WPM" in methods:
                            per_method["WPM"] = mod.method_wpm(pillar_avgs, weights)
                        if "RANK" in methods:
                            per_method["RANK"] = mod.method_rank(pillar_avgs, weights)
                        if "TOPSIS" in methods:
                            per_method["TOPSIS"] = mod.method_topsis(pillar_avgs, weights)
                        if "VIKOR" in methods:
                            per_method["VIKOR"] = mod.method_vikor(pillar_avgs, weights)
                        if "EDAS" in methods:
                            per_method["EDAS"] = mod.method_edas(pillar_avgs, weights)
                        if "MAUT" in methods:
                            per_method["MAUT"] = mod.method_maut(pillar_avgs, weights)
                        if "PCA" in methods:
                            per_method["PCA"] = mod.method_pca(pillar_avgs)
                        
                        scenario_df = pd.DataFrame(per_method)
                        scenario_df.index.name = "Alternative"
                        
                        if norm_flag:
                            scaled_cols = {
                                col: mod._scale_series_by_method(scenario_df[col], col)
                                for col in scenario_df.columns
                            }
                            scenario_df_scaled = pd.DataFrame(scaled_cols, index=scenario_df.index)
                        else:
                            scenario_df_scaled = scenario_df.copy()
                        
                        scenario_name = f"Scenario {len(st.session_state.scenario_results) + 1}"
                        st.session_state.scenario_results[scenario_name] = {
                            "weights": weights,
                            "methods": methods,
                            "results": scenario_df_scaled,
                        }
                        st.session_state.last_weights = weights
                        st.session_state.last_methods = methods
                        
                        st.success("Analysis complete!")
            
            # Display results
            if st.session_state.scenario_results:
                st.subheader("📊 Results")
                for name, sdata in st.session_state.scenario_results.items():
                    with st.expander(f"{name}"):
                        st.write(f"**Methods:** {', '.join(sdata['methods'])}")
                        st.dataframe(sdata["results"].round(4))
                        
                        # Plot
                        fig = px.bar(
                            sdata["results"].reset_index().melt(
                                id_vars="Alternative",
                                var_name="Method",
                                value_name="Score"
                            ),
                            x="Alternative",
                            y="Score",
                            color="Method",
                            barmode="group",
                            title=f"{name} Results"
                        )
                        st.plotly_chart(fig, use_container_width=True)
    
    # ----- Validation Tab -----
    with risk_tabs[3]:
        st.header("🔎 Validation")
        
        if st.session_state.pillar_avgs_df.empty:
            st.info("Run scenario analysis first.")
        else:
            pillar_avgs = st.session_state.pillar_avgs_df.copy()
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("DEA Settings")
                dea_samples = st.number_input("Samples", 100, 50000, 5000, 100)
                peer_cut = st.number_input("Peer cutoff", 0.0, 1.0, 0.05, 0.01)
            
            with col2:
                st.subheader("Monte Carlo")
                mc_sims = st.number_input("Simulations", 100, 50000, 2000, 100)
                alpha = st.number_input("Alpha", 0.1, 10.0, 1.0, 0.1)
                sigma = st.number_input("Sigma", 0.0, 0.5, 0.03, 0.01)
            
            if st.button("🔬 Run Validation", type="primary"):
                with st.spinner("Running DEA diagnostics..."):
                    dea_summary, peer_matrix, bottleneck_matrix, targets = mod.approx_dea_diagnostics(
                        pillar_avgs, samples=int(dea_samples), min_peer_lambda=float(peer_cut)
                    )
                
                st.subheader("DEA Results")
                st.dataframe(dea_summary.round(4))
                
                # Dominance matrix
                dom = mod.compute_dominance_matrix(pillar_avgs)
                with st.expander("Dominance Matrix"):
                    st.dataframe(dom)
                
                # Monte Carlo
                with st.spinner("Running Monte Carlo..."):
                    Pbest, MeanRank, StdRank, RankDist = mod.run_monte_carlo_sensitivity(
                        pillar_avgs,
                        st.session_state.last_weights or {p: 100.0/len(pillar_avgs.index) for p in pillar_avgs.index},
                        st.session_state.last_methods or ["WEIGHTED"],
                        sims=int(mc_sims),
                        weight_alpha=float(alpha),
                        score_noise_sigma=float(sigma),
                    )
                
                st.subheader("Monte Carlo Results")
                tab1, tab2, tab3 = st.tabs(["P(Best)", "Mean Rank", "Std Dev"])
                
                with tab1:
                    st.dataframe(Pbest.round(4))
                    if "WEIGHTED" in Pbest.columns:
                        fig = px.bar(Pbest.reset_index(), x="Tech", y="WEIGHTED", 
                                    title="Probability of Being Best (Weighted)")
                        st.plotly_chart(fig, use_container_width=True)
                
                with tab2:
                    st.dataframe(MeanRank.round(4))
                
                with tab3:
                    st.dataframe(StdRank.round(4))
    
    # ----- Export Tab -----
    with risk_tabs[4]:
        st.header("📤 Export Results")
        
        def build_pdf_report_simple():
            """Build PDF report for risk analysis."""
            buf = io.BytesIO()
            doc = SimpleDocTemplate(buf, pagesize=A4)
            styles = getSampleStyleSheet()
            story = []
            
            # Title
            story.append(Paragraph("Risk Analysis Report", styles["Title"]))
            story.append(Spacer(1, 12))
            story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles["Normal"]))
            story.append(PageBreak())
            
            # Add sections based on available data
            if st.session_state.risks:
                story.append(Paragraph("Risk Assessment", styles["Heading1"]))
                df_r = pd.DataFrame([
                    {"Risk": r["name"], "Probability": r["prob"], "Severity": r["sev"], "Rating": r["prob"] * r["sev"]}
                    for r in st.session_state.risks
                ])
                story.append(RLTable([df_r.columns.tolist()] + df_r.values.tolist()))
                story.append(Spacer(1, 12))
            
            if not st.session_state.pillar_avgs_df.empty:
                story.append(Paragraph("ESGFP Results", styles["Heading1"]))
                story.append(RLTable(
                    [["Pillar"] + list(st.session_state.pillar_avgs_df.columns)] +
                    [[idx] + list(row) for idx, row in st.session_state.pillar_avgs_df.iterrows()]
                ))
            
            doc.build(story)
            buf.seek(0)
            return buf.getvalue()
        
        export_format = st.radio("Export format:", ["PDF Report", "CSV Data", "JSON"])
        
        if export_format == "PDF Report":
            if st.button("📄 Generate PDF Report"):
                pdf_bytes = build_pdf_report_simple()
                st.download_button(
                    "⬇️ Download PDF",
                    data=pdf_bytes,
                    file_name="risk_analysis_report.pdf",
                    mime="application/pdf"
                )
        
        elif export_format == "CSV Data":
            col1, col2 = st.columns(2)
            with col1:
                if st.session_state.risks:
                    df_r = pd.DataFrame([
                        {"Risk": r["name"], "Probability": r["prob"], "Severity": r["sev"], "Rating": r["prob"] * r["sev"]}
                        for r in st.session_state.risks
                    ])
                    csv_b = df_r.to_csv(index=False).encode()
                    st.download_button(
                        "📥 Risks CSV",
                        data=csv_b,
                        file_name="risks.csv",
                        mime="text/csv"
                    )
            
            with col2:
                if not st.session_state.pillar_avgs_df.empty:
                    csv_b = st.session_state.pillar_avgs_df.to_csv().encode()
                    st.download_button(
                        "📥 Pillar Averages CSV",
                        data=csv_b,
                        file_name="pillar_averages.csv",
                        mime="text/csv"
                    )
        
        else:  # JSON
            data = {
                "risks": st.session_state.risks,
                "pillar_averages": st.session_state.pillar_avgs_df.to_dict() if not st.session_state.pillar_avgs_df.empty else {},
                "scenarios": len(st.session_state.scenario_results)
            }
            json_str = json.dumps(data, indent=2)
            st.download_button(
                "📥 Export JSON",
                data=json_str,
                file_name="risk_analysis.json",
                mime="application/json"
            )

# ----------------- Compliance TEA Page -----------------
elif page == "💰 Compliance TEA":
    st.title("💰 Compliance & ESG Techno-Economic Analysis")
    
    # Initialize default parameters if not set
    if not st.session_state.tea_params:
        st.session_state.tea_params = {
            "C_PE": 1e8,
            "COL": 1e7,
            "C_RM": 4e7,
            "C_UT": 1.2e7,
            "C_CAT": 2e6,
            "Q_prod": 5e5,
            "P_prod": 550.0,
            "f_ins": 0.30,
            "f_pipe": 0.45,
            "f_elec": 0.10,
            "f_bldg": 0.05,
            "f_util": 0.06,
            "f_stor": 0.02,
            "f_safe": 0.01,
            "f_waste": 0.01,
            "f_eng": 0.12,
            "f_cons": 0.10,
            "f_licn": 0.00,
            "f_cont": 0.02,
            "f_contg": 0.0,
            "f_insur": 0.01,
            "f_own": 0.02,
            "f_start": 0.01,
            "N_project": 20,
            "L_asset": 20,
            "salv_frac": 0.10,
            "f_risk_op": 0.05,
            "tau_CO2": 50.0,
            "E_CO2": 200000.0,
            "f_pack": 0.02,
            "f_esg": 0.07,
            "i_base": 0.08,
            "delta_risk": 0.03,
            "dep_method": "SL"
        }
    
    # Tabs for different TEA sections
    tea_tabs = st.tabs(["⚙️ Parameters", "📊 Analysis", "📈 Visualizations", "📤 Export"])
    
    # ----- Parameters Tab -----
    with tea_tabs[0]:
        st.header("⚙️ TEA Parameters Configuration")
        
        st.markdown("""
        ### Capital Costs
        Configure the major capital expenditure components for your project.
        """)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Major Equipment")
            st.session_state.tea_params["C_PE"] = st.number_input(
                "Process Equipment (C_PE) USD", 
                value=float(st.session_state.tea_params["C_PE"]),
                format="%.0f",
                help="Cost of major process equipment"
            )
            
            st.session_state.tea_params["f_ins"] = st.slider(
                "Instrumentation (f_ins)", 
                min_value=0.0, max_value=1.0, 
                value=float(st.session_state.tea_params["f_ins"]),
                format="%.2f",
                help="Fraction of C_PE for instrumentation"
            )
            
            st.session_state.tea_params["f_pipe"] = st.slider(
                "Piping (f_pipe)", 
                min_value=0.0, max_value=1.0, 
                value=float(st.session_state.tea_params["f_pipe"]),
                format="%.2f"
            )
        
        with col2:
            st.subheader("Infrastructure")
            st.session_state.tea_params["f_elec"] = st.slider(
                "Electrical (f_elec)", 
                min_value=0.0, max_value=1.0, 
                value=float(st.session_state.tea_params["f_elec"]),
                format="%.2f"
            )
            
            st.session_state.tea_params["f_bldg"] = st.slider(
                "Buildings (f_bldg)", 
                min_value=0.0, max_value=1.0, 
                value=float(st.session_state.tea_params["f_bldg"]),
                format="%.2f"
            )
            
            st.session_state.tea_params["f_util"] = st.slider(
                "Utilities (f_util)", 
                min_value=0.0, max_value=1.0, 
                value=float(st.session_state.tea_params["f_util"]),
                format="%.2f"
            )
        
        with col3:
            st.subheader("Indirect Costs")
            st.session_state.tea_params["f_eng"] = st.slider(
                "Engineering (f_eng)", 
                min_value=0.0, max_value=1.0, 
                value=float(st.session_state.tea_params["f_eng"]),
                format="%.2f"
            )
            
            st.session_state.tea_params["f_cons"] = st.slider(
                "Construction (f_cons)", 
                min_value=0.0, max_value=1.0, 
                value=float(st.session_state.tea_params["f_cons"]),
                format="%.2f"
            )
            
            st.session_state.tea_params["f_contg"] = st.slider(
                "Contingency (f_contg)", 
                min_value=0.0, max_value=0.5, 
                value=float(st.session_state.tea_params["f_contg"]),
                format="%.2f"
            )
        
        st.markdown("---")
        st.markdown("### Operating Parameters")
        
        col4, col5, col6 = st.columns(3)
        
        with col4:
            st.session_state.tea_params["Q_prod"] = st.number_input(
                "Production Rate (Q_prod) tons/year", 
                value=float(st.session_state.tea_params["Q_prod"]),
                format="%.0f"
            )
            
            st.session_state.tea_params["P_prod"] = st.number_input(
                "Product Price (P_prod) USD/ton", 
                value=float(st.session_state.tea_params["P_prod"]),
                format="%.2f"
            )
            
            st.session_state.tea_params["C_RM"] = st.number_input(
                "Raw Materials (C_RM) USD/year", 
                value=float(st.session_state.tea_params["C_RM"]),
                format="%.0f"
            )
        
        with col5:
            st.session_state.tea_params["COL"] = st.number_input(
                "Operating Labor (COL) USD/year", 
                value=float(st.session_state.tea_params["COL"]),
                format="%.0f"
            )
            
            st.session_state.tea_params["C_UT"] = st.number_input(
                "Utilities (C_UT) USD/year", 
                value=float(st.session_state.tea_params["C_UT"]),
                format="%.0f"
            )
            
            st.session_state.tea_params["C_CAT"] = st.number_input(
                "Catalysts (C_CAT) USD/year", 
                value=float(st.session_state.tea_params["C_CAT"]),
                format="%.0f"
            )
        
        with col6:
            st.session_state.tea_params["f_esg"] = st.slider(
                "ESG Compliance Factor (f_esg)", 
                min_value=0.0, max_value=0.5, 
                value=float(st.session_state.tea_params["f_esg"]),
                format="%.3f",
                help="Additional cost factor for ESG compliance"
            )
            
            st.session_state.tea_params["tau_CO2"] = st.number_input(
                "CO2 Tax (tau_CO2) USD/ton", 
                value=float(st.session_state.tea_params["tau_CO2"]),
                format="%.2f"
            )
            
            st.session_state.tea_params["E_CO2"] = st.number_input(
                "CO2 Emissions (E_CO2) tons/year", 
                value=float(st.session_state.tea_params["E_CO2"]),
                format="%.0f"
            )
        
        st.markdown("---")
        st.markdown("### Financial Parameters")
        
        col7, col8, col9 = st.columns(3)
        
        with col7:
            st.session_state.tea_params["N_project"] = st.number_input(
                "Project Life (N_project) years", 
                value=int(st.session_state.tea_params["N_project"]),
                min_value=1, max_value=50
            )
            
            st.session_state.tea_params["L_asset"] = st.number_input(
                "Asset Life (L_asset) years", 
                value=int(st.session_state.tea_params["L_asset"]),
                min_value=1, max_value=50
            )
        
        with col8:
            st.session_state.tea_params["i_base"] = st.slider(
                "Discount Rate (i_base)", 
                min_value=0.0, max_value=0.5, 
                value=float(st.session_state.tea_params["i_base"]),
                format="%.3f"
            )
            
            st.session_state.tea_params["delta_risk"] = st.slider(
                "Risk Premium (delta_risk)", 
                min_value=0.0, max_value=0.2, 
                value=float(st.session_state.tea_params["delta_risk"]),
                format="%.3f"
            )
        
        with col9:
            st.session_state.tea_params["salv_frac"] = st.slider(
                "Salvage Value Fraction", 
                min_value=0.0, max_value=1.0, 
                value=float(st.session_state.tea_params["salv_frac"]),
                format="%.2f"
            )
            
            st.session_state.tea_params["dep_method"] = st.selectbox(
                "Depreciation Method", 
                options=["SL", "SYD", "DDB"],
                index=["SL", "SYD", "DDB"].index(st.session_state.tea_params["dep_method"])
            )
        
        # JSON Editor for advanced users
        with st.expander("🛠️ Advanced JSON Editor"):
            params_json = st.text_area(
                "Edit all parameters as JSON:",
                value=json.dumps(st.session_state.tea_params, indent=2),
                height=300
            )
            if st.button("Apply JSON Configuration"):
                try:
                    new_params = json.loads(params_json)
                    st.session_state.tea_params.update(new_params)
                    st.success("Parameters updated successfully!")
                    st.rerun()
                except json.JSONDecodeError:
                    st.error("Invalid JSON format")
    
    # ----- Analysis Tab -----
    with tea_tabs[1]:
        st.header("📊 TEA Analysis")
        
        if st.button("🚀 Run Complete TEA Analysis", type="primary"):
            with st.spinner("Calculating TEA metrics..."):
                results = run_tea_analysis(st.session_state.tea_params)
                
                if results:
                    st.session_state.tea_results = results
                    st.success("✅ Analysis complete!")
                else:
                    st.error("❌ Analysis failed. Check parameters.")
        
        if "tea_results" in st.session_state and st.session_state.tea_results:
            results = st.session_state.tea_results
            
            st.markdown("### 📈 Key Financial Metrics")
            
            # Display metrics in cards
            metric_cols = st.columns(4)
            with metric_cols[0]:
                st.metric(
                    "Net Present Value (NPV)",
                    f"${results['NPV']:,.0f}",
                    delta="Positive" if results['NPV'] > 0 else "Negative",
                    delta_color="normal" if results['NPV'] > 0 else "inverse"
                )
            
            with metric_cols[1]:
                st.metric(
                    "Internal Rate of Return (IRR)",
                    f"{results['IRR']*100:.1f}%",
                    delta=f"vs {results['Discount_Rate']*100:.1f}% hurdle",
                    delta_color="normal" if results['IRR'] > results['Discount_Rate'] else "inverse"
                )
            
            with metric_cols[2]:
                st.metric(
                    "Levelized Cost (LCOx)",
                    f"${results['LCOx']:,.2f}/ton",
                    delta=f"vs ${results['Product_Price']:,.2f}/ton price",
                    delta_color="normal" if results['LCOx'] < results['Product_Price'] else "inverse"
                )
            
            with metric_cols[3]:
                if results['Payback_Period']:
                    st.metric(
                        "Payback Period",
                        f"{results['Payback_Period']} years",
                        delta=f"of {results['Project_Life']} year project",
                        delta_color="normal" if results['Payback_Period'] < results['Project_Life']/2 else "inverse"
                    )
                else:
                    st.metric("Payback Period", "> Project Life")
            
            st.markdown("---")
            st.markdown("### 💰 Cost Breakdown")
            
            cost_cols = st.columns(3)
            with cost_cols[0]:
                st.metric("Total CAPEX", f"${results['CAPEX']:,.0f}")
                st.metric("Annual OPEX", f"${results['Annual_OPEX']:,.0f}/year")
            
            with cost_cols[1]:
                st.metric("Annual Revenue", f"${results['Annual_Revenue']:,.0f}/year")
                st.metric("Carbon Cost", f"${results['Carbon_Cost']:,.0f}/year")
            
            with cost_cols[2]:
                st.metric("Salvage Value", f"${results['Salvage_Value']:,.0f}")
                st.metric("Production Rate", f"{results['Production_Rate']:,.0f} tons/year")
    
    # ----- Visualizations Tab -----
    with tea_tabs[2]:
        st.header("📈 TEA Visualizations")
        
        if "tea_results" not in st.session_state or not st.session_state.tea_results:
            st.info("Run TEA analysis first to generate visualizations.")
        else:
            results = st.session_state.tea_results
            
            # Cash Flow Chart
            st.subheader("💵 Cash Flow Analysis")
            
            fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # Annual Cash Flow
            years = list(range(len(results["CF"])))
            ax1.bar(years, results["CF"], color='skyblue', edgecolor='navy', alpha=0.8)
            ax1.axhline(y=0, color='black', linewidth=0.5)
            ax1.set_xlabel("Year", fontsize=12)
            ax1.set_ylabel("Cash Flow (USD)", fontsize=12)
            ax1.set_title("Annual Cash Flow Over Project Life", fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            
            # Cumulative Cash Flow
            ax2.plot(years, results["Cumulative_CF"], marker='o', linewidth=2, color='darkgreen', markersize=6)
            ax2.axhline(y=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
            ax2.fill_between(years, 0, results["Cumulative_CF"], where=np.array(results["Cumulative_CF"]) >= 0, 
                           alpha=0.3, color='green', label='Profit')
            ax2.fill_between(years, 0, results["Cumulative_CF"], where=np.array(results["Cumulative_CF"]) < 0, 
                           alpha=0.3, color='red', label='Loss')
            ax2.set_xlabel("Year", fontsize=12)
            ax2.set_ylabel("Cumulative Cash Flow (USD)", fontsize=12)
            ax2.set_title("Cumulative Cash Flow (Payback Analysis)", fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            plt.tight_layout()
            st.pyplot(fig1)
            
            # Payback Period Marker
            if results['Payback_Period']:
                st.info(f"**Payback Period:** {results['Payback_Period']} years")
            else:
                st.warning("**Payback Period:** Exceeds project life")
            
            st.markdown("---")
            
            # Cost Breakdown Pie Chart
            st.subheader("📊 Cost Structure Analysis")
            
            cost_data = {
                "CAPEX": results["CAPEX"],
                "Annual OPEX": results["Annual_OPEX"],
                "Carbon Cost": results["Carbon_Cost"],
                "ESG Compliance": results["Annual_OPEX"] * st.session_state.tea_params.get("f_esg", 0.07)
            }
            
            fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(14, 6))
            
            # Pie chart for cost structure
            labels = list(cost_data.keys())
            sizes = list(cost_data.values())
            colors_pie = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99']
            ax3.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%', startangle=90)
            ax3.set_title("Cost Structure Breakdown", fontsize=14, fontweight='bold')
            
            # Bar chart for comparison
            x_pos = np.arange(len(cost_data))
            ax4.bar(x_pos, sizes, color=colors_pie, edgecolor='black', alpha=0.8)
            ax4.set_xlabel("Cost Category", fontsize=12)
            ax4.set_ylabel("USD", fontsize=12)
            ax4.set_title("Cost Magnitude Comparison", fontsize=14, fontweight='bold')
            ax4.set_xticks(x_pos)
            ax4.set_xticklabels(labels, rotation=45, ha='right')
            ax4.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            
            plt.tight_layout()
            st.pyplot(fig2)
    
    # ----- Export Tab -----
    with tea_tabs[3]:
        st.header("📤 Export TEA Results")
        
        if "tea_results" not in st.session_state or not st.session_state.tea_results:
            st.info("Run TEA analysis first to export results.")
        else:
            results = st.session_state.tea_results
            
            export_format = st.radio(
                "Select export format:",
                ["PDF Report", "JSON Data", "CSV Tables"]
            )
            
            if export_format == "PDF Report":
                st.subheader("📄 PDF Report")
                
                if st.button("Generate PDF Report"):
                    # Create PDF report
                    buf = io.BytesIO()
                    doc = SimpleDocTemplate(buf, pagesize=A4)
                    styles = getSampleStyleSheet()
                    story = []
                    
                    # Title
                    story.append(Paragraph("Techno-Economic Analysis Report", styles["Title"]))
                    story.append(Spacer(1, 12))
                    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles["Normal"]))
                    story.append(Paragraph(f"Project Life: {results['Project_Life']} years", styles["Normal"]))
                    story.append(PageBreak())
                    
                    # Key Metrics
                    story.append(Paragraph("Key Financial Metrics", styles["Heading1"]))
                    metrics_text = f"""
                    Net Present Value (NPV): ${results['NPV']:,.0f}<br/>
                    Internal Rate of Return (IRR): {results['IRR']*100:.1f}%<br/>
                    Levelized Cost of Production (LCOx): ${results['LCOx']:,.2f}/ton<br/>
                    Payback Period: {results['Payback_Period'] if results['Payback_Period'] else '> Project Life'} years<br/>
                    Benefit-Cost Ratio (BCR): {results['BCR']:.2f if results['BCR'] != float('inf') else '∞'}<br/>
                    """
                    story.append(Paragraph(metrics_text, styles["Normal"]))
                    story.append(Spacer(1, 12))
                    
                    # Cost Breakdown
                    story.append(Paragraph("Cost Breakdown", styles["Heading1"]))
                    cost_text = f"""
                    Total CAPEX: ${results['CAPEX']:,.0f}<br/>
                    Annual OPEX: ${results['Annual_OPEX']:,.0f}/year<br/>
                    Annual Revenue: ${results['Annual_Revenue']:,.0f}/year<br/>
                    Carbon Cost: ${results['Carbon_Cost']:,.0f}/year<br/>
                    Salvage Value: ${results['Salvage_Value']:,.0f}<br/>
                    """
                    story.append(Paragraph(cost_text, styles["Normal"]))
                    
                    doc.build(story)
                    buf.seek(0)
                    
                    st.download_button(
                        "⬇️ Download PDF Report",
                        data=buf.getvalue(),
                        file_name="tea_analysis_report.pdf",
                        mime="application/pdf"
                    )
            
            elif export_format == "JSON Data":
                st.subheader("📊 JSON Data Export")
                
                # Prepare exportable results
                exportable_results = {}
                for key, value in results.items():
                    if isinstance(value, np.ndarray):
                        exportable_results[key] = value.tolist()
                    elif isinstance(value, (int, float, str, list, dict)):
                        exportable_results[key] = value
                
                json_str = json.dumps(exportable_results, indent=2, default=str)
                
                st.download_button(
                    "⬇️ Download JSON Data",
                    data=json_str,
                    file_name="tea_results.json",
                    mime="application/json"
                )
            
            else:  # CSV Tables
                st.subheader("📈 CSV Data Export")
                
                col_csv1, col_csv2 = st.columns(2)
                
                with col_csv1:
                    # Cash Flow CSV
                    cf_df = pd.DataFrame({
                        "Year": list(range(len(results["CF"]))),
                        "Cash_Flow": results["CF"],
                        "Discounted_Cash_Flow": results["Discounted_CF"],
                        "Cumulative_Cash_Flow": results["Cumulative_CF"]
                    })
                    csv_cf = cf_df.to_csv(index=False).encode()
                    st.download_button(
                        "⬇️ Download Cash Flow CSV",
                        data=csv_cf,
                        file_name="cash_flow_analysis.csv",
                        mime="text/csv"
                    )
                
                with col_csv2:
                    # Summary metrics CSV
                    summary_data = {
                        "Metric": ["NPV", "IRR", "LCOx", "CAPEX", "Annual_OPEX", "Annual_Revenue"],
                        "Value": [
                            results["NPV"],
                            results["IRR"],
                            results["LCOx"],
                            results["CAPEX"],
                            results["Annual_OPEX"],
                            results["Annual_Revenue"]
                        ],
                        "Unit": ["USD", "%", "USD/ton", "USD", "USD/year", "USD/year"]
                    }
                    summary_df = pd.DataFrame(summary_data)
                    csv_summary = summary_df.to_csv(index=False).encode()
                    st.download_button(
                        "⬇️ Download Summary Metrics CSV",
                        data=csv_summary,
                        file_name="tea_summary_metrics.csv",
                        mime="text/csv"
                    )

# ----------------- Footer -----------------
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
    Pro_DESG Materiality & Sustainability Tool Kit © 2024
    </div>
    """,
    unsafe_allow_html=True
)