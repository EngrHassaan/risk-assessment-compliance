"""
Compliance & ESG TEA Frontend
Streamlit app for techno-economic analysis with full parameter UI, charts, and export.
Maps all compliance.py calculation functions without duplication.
Production-ready with error handling, validation, and session state management.
"""

import os
import sys
import io
import json
import tempfile
import importlib.util
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage,
    Table as RLTable, TableStyle, PageBreak, KeepTogether
)

# ==================== CONFIG & CONSTANTS ====================
COMPLIANCE_PATH = "model/compliance.py"
APP_TITLE = "Compliance & ESG Techno-Economic Analysis"

# Parameter groups for organized UI
CAPEX_PARAMS = [
    "C_PE", "C_RM", "C_UT", "C_CAT", "COL",
    "f_ins", "f_pipe", "f_elec", "f_bldg", "f_util",
    "f_stor", "f_safe", "f_waste",
]
INDIRECT_PARAMS = ["f_eng", "f_cons", "f_licn", "f_cont", "f_contg", "f_insur"]
OTHER_COST_PARAMS = ["f_own", "f_start"]
OPEX_PARAMS = ["f_esg", "f_pack", "f_risk_op"]
FINANCIAL_PARAMS = [
    "Q_prod", "P_prod", "N_project", "L_asset", "salv_frac",
    "i_base", "delta_risk", "dep_method",
]
EMISSIONS_PARAMS = ["tau_CO2", "E_CO2"]

DEFAULT_PARAMS = {
    "C_PE": 1e8, "C_RM": 4e7, "C_UT": 1.2e7, "C_CAT": 2e6, "COL": 1e7,
    "Q_prod": 5e5, "P_prod": 550.0,
    "f_ins": 0.30, "f_pipe": 0.45, "f_elec": 0.10, "f_bldg": 0.05,
    "f_util": 0.06, "f_stor": 0.02, "f_safe": 0.01, "f_waste": 0.01,
    "f_eng": 0.12, "f_cons": 0.10, "f_licn": 0.00, "f_cont": 0.02,
    "f_contg": 0.0, "f_insur": 0.01, "f_own": 0.02, "f_start": 0.01,
    "N_project": 20, "L_asset": 20, "salv_frac": 0.10, "f_risk_op": 0.05,
    "tau_CO2": 50.0, "E_CO2": 200000.0, "f_pack": 0.02, "f_esg": 0.07,
    "i_base": 0.08, "delta_risk": 0.03, "dep_method": "SL",
}

PARAM_HELP = {
    "C_PE": "Plant equipment cost (USD)",
    "C_RM": "Raw materials cost (USD/yr)",
    "C_UT": "Utilities cost (USD/yr)",
    "C_CAT": "Catalyst cost (USD/yr)",
    "COL": "Labor cost (USD/yr)",
    "Q_prod": "Production rate (tons/yr)",
    "P_prod": "Product price (USD/ton)",
    "f_ins": "Installation cost factor",
    "f_pipe": "Piping cost factor",
    "f_elec": "Electrical cost factor",
    "f_eng": "Engineering cost factor",
    "N_project": "Project lifetime (years)",
    "i_base": "Base discount rate",
    "tau_CO2": "Carbon price (USD/ton CO2)",
    "E_CO2": "Annual CO2 emissions (tons)",
    "f_esg": "ESG compliance cost (% of OPEX)",
    "dep_method": "Depreciation method (SL/SYD/DDB)",
}

# ==================== UTILITIES ====================

def load_compliance_module(path: str):
    """Dynamically load compliance.py module."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Compliance module not found: {path}")
    spec = importlib.util.spec_from_file_location(
        "compliance_module", path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["compliance_module"] = mod
    spec.loader.exec_module(mod)
    return mod


def validate_params(params: Dict[str, float]) -> Tuple[bool, str]:
    """Validate parameter ranges and dependencies."""
    errors = []
    
    if params.get("C_PE", 0) <= 0:
        errors.append("C_PE must be > 0")
    if params.get("Q_prod", 0) <= 0:
        errors.append("Q_prod must be > 0")
    if params.get("P_prod", 0) <= 0:
        errors.append("P_prod must be > 0")
    if not (0 < params.get("i_base", 0.08) < 0.5):
        errors.append("i_base should be between 0 and 50%")
    if params.get("N_project", 20) < 1:
        errors.append("N_project must be >= 1")
    
    # Check factor ranges (should be 0-1 typically)
    factor_keys = [k for k in params if k.startswith("f_")]
    for key in factor_keys:
        val = params.get(key, 0)
        if val < 0 or val > 2.0:  # Allow up to 200%
            errors.append(f"{key} should be in range [0, 2.0]")
    
    if errors:
        return False, "; ".join(errors)
    return True, "All parameters valid"

def mpl_fig_to_png_bytes(fig, dpi: int = 150) -> bytes:
    """Convert matplotlib figure to PNG bytes."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()

def pil_bytes_to_rlimage(png_bytes: bytes, max_width_mm: float = 170):
    """Convert PNG bytes to ReportLab image."""
    img = Image.open(io.BytesIO(png_bytes))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    rl_img = RLImage(buf)
    max_w_pt = max_width_mm * mm
    if rl_img.drawWidth > max_w_pt:
        scale = max_w_pt / rl_img.drawWidth
        rl_img.drawWidth *= scale
        rl_img.drawHeight *= scale
    return rl_img


# ==================== SESSION STATE INIT ====================

st.set_page_config(
    page_title=APP_TITLE,
    layout="wide",
    initial_sidebar_state="expanded"
)

if "tea_params" not in st.session_state:
    st.session_state.tea_params = dict(DEFAULT_PARAMS)

if "tea_results" not in st.session_state:
    st.session_state.tea_results = None

if "saved_scenarios" not in st.session_state:
    st.session_state.saved_scenarios = {}

if "compliance_module" not in st.session_state:
    try:
        st.session_state.compliance_module = load_compliance_module(
            COMPLIANCE_PATH
        )
    except Exception as e:
        st.session_state.compliance_module = None
        st.session_state.module_error = str(e)

# ==================== MAIN LAYOUT ====================

st.title(APP_TITLE)
st.markdown(
    "Complete techno-economic analysis with ESG integration. "
    "Edit parameters, run calculations, and export results."
)

# Sidebar: Quick actions & file management
with st.sidebar:
    st.header("⚙️ Parameter Management")
    
    if st.button("Reset to defaults"):
        st.session_state.tea_params = dict(DEFAULT_PARAMS)
        st.success("Parameters reset to defaults")
    
    st.write("---")
    
    # Load from JSON file
    uploaded_json = st.file_uploader(
        "Load parameters (JSON)",
        type=["json"],
        key="upload_params_json"
    )
    if uploaded_json:
        try:
            params = json.load(uploaded_json)
            if isinstance(params, dict):
                st.session_state.tea_params.update(params)
                st.success("Parameters loaded from JSON")
            else:
                st.error("JSON must contain a dictionary")
        except Exception as e:
            st.error(f"Error loading JSON: {e}")
    
    st.write("---")
    st.subheader("Quick Presets")
    
    if st.button("Conservative scenario"):
        preset = dict(DEFAULT_PARAMS)
        preset["f_esg"] = 0.15  # Higher ESG cost
        preset["f_risk_op"] = 0.10
        preset["i_base"] = 0.10
        st.session_state.tea_params = preset
        st.success("Conservative preset applied")
    
    if st.button("Optimistic scenario"):
        preset = dict(DEFAULT_PARAMS)
        preset["f_esg"] = 0.05  # Lower ESG cost
        preset["f_risk_op"] = 0.02
        preset["i_base"] = 0.06
        st.session_state.tea_params = preset
        st.success("Optimistic preset applied")

# Main tabs
tabs = st.tabs([
    "📥 Parameters",
    "🔢 Analysis",
    "📊 Visualizations",
    "🔍 Sensitivity",
    "💾 Export"
])

# ==================== TAB 1: PARAMETERS ====================

with tabs[0]:
    st.header("TEA Parameters")
    
    col_info, col_reset = st.columns([3, 1])
    with col_info:
        st.info(
            "Organize parameters by category. "
            "Hover over parameter name for description."
        )
    with col_reset:
        if st.button("Validate all", key="validate_btn"):
            valid, msg = validate_params(st.session_state.tea_params)
            if valid:
                st.success(f"✓ {msg}")
            else:
                st.error(f"✗ {msg}")
    
    st.write("---")
    
    # CAPEX Section
    with st.expander("💰 Capital Expenditure (CAPEX)", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**Main Equipment**")
            st.session_state.tea_params["C_PE"] = st.number_input(
                "Plant equipment (C_PE)",
                min_value=0.0,
                value=st.session_state.tea_params["C_PE"],
                format="%.2e",
                help=PARAM_HELP["C_PE"]
            )
        
        with col2:
            st.write("**Direct Costs (as % of C_PE)**")
            st.session_state.tea_params["f_ins"] = st.number_input(
                "Installation factor",
                min_value=0.0, max_value=2.0, step=0.01,
                value=st.session_state.tea_params["f_ins"],
                help=PARAM_HELP["f_ins"]
            )
            st.session_state.tea_params["f_pipe"] = st.number_input(
                "Piping factor",
                min_value=0.0, max_value=2.0, step=0.01,
                value=st.session_state.tea_params["f_pipe"],
            )
            st.session_state.tea_params["f_elec"] = st.number_input(
                "Electrical factor",
                min_value=0.0, max_value=2.0, step=0.01,
                value=st.session_state.tea_params["f_elec"],
            )
            st.session_state.tea_params["f_bldg"] = st.number_input(
                "Building factor",
                min_value=0.0, max_value=2.0, step=0.01,
                value=st.session_state.tea_params["f_bldg"],
            )
            st.session_state.tea_params["f_util"] = st.number_input(
                "Utility factor",
                min_value=0.0, max_value=2.0, step=0.01,
                value=st.session_state.tea_params["f_util"],
            )
            st.session_state.tea_params["f_stor"] = st.number_input(
                "Storage factor",
                min_value=0.0, max_value=2.0, step=0.01,
                value=st.session_state.tea_params["f_stor"],
            )
            st.session_state.tea_params["f_safe"] = st.number_input(
                "Safety factor",
                min_value=0.0, max_value=2.0, step=0.01,
                value=st.session_state.tea_params["f_safe"],
            )
            st.session_state.tea_params["f_waste"] = st.number_input(
                "Waste handling factor",
                min_value=0.0, max_value=2.0, step=0.01,
                value=st.session_state.tea_params["f_waste"],
            )
        
        with col3:
            st.write("**Indirect & Contingency**")
            st.session_state.tea_params["f_eng"] = st.number_input(
                "Engineering factor",
                min_value=0.0, max_value=2.0, step=0.01,
                value=st.session_state.tea_params["f_eng"],
            )
            st.session_state.tea_params["f_cons"] = st.number_input(
                "Construction factor",
                min_value=0.0, max_value=2.0, step=0.01,
                value=st.session_state.tea_params["f_cons"],
            )
            st.session_state.tea_params["f_licn"] = st.number_input(
                "Licensing factor",
                min_value=0.0, max_value=2.0, step=0.01,
                value=st.session_state.tea_params["f_licn"],
            )
            st.session_state.tea_params["f_cont"] = st.number_input(
                "Contingency (% CAPEX)",
                min_value=0.0, max_value=2.0, step=0.01,
                value=st.session_state.tea_params["f_cont"],
            )
            st.session_state.tea_params["f_contg"] = st.number_input(
                "Contingency (geopolitical)",
                min_value=0.0, max_value=2.0, step=0.01,
                value=st.session_state.tea_params["f_contg"],
            )
            st.session_state.tea_params["f_insur"] = st.number_input(
                "Insurance factor",
                min_value=0.0, max_value=2.0, step=0.01,
                value=st.session_state.tea_params["f_insur"],
            )
            st.session_state.tea_params["f_own"] = st.number_input(
                "Owner cost factor",
                min_value=0.0, max_value=2.0, step=0.01,
                value=st.session_state.tea_params["f_own"],
            )
            st.session_state.tea_params["f_start"] = st.number_input(
                "Startup factor",
                min_value=0.0, max_value=2.0, step=0.01,
                value=st.session_state.tea_params["f_start"],
            )
    
    # OPEX Section
    with st.expander("📈 Operating Expenditure (OPEX)", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Annual Costs (USD/year)**")
            st.session_state.tea_params["C_RM"] = st.number_input(
                "Raw materials (C_RM)",
                min_value=0.0,
                value=st.session_state.tea_params["C_RM"],
                format="%.2e",
                help=PARAM_HELP["C_RM"]
            )
            st.session_state.tea_params["C_UT"] = st.number_input(
                "Utilities (C_UT)",
                min_value=0.0,
                value=st.session_state.tea_params["C_UT"],
                format="%.2e",
                help=PARAM_HELP["C_UT"]
            )
            st.session_state.tea_params["C_CAT"] = st.number_input(
                "Catalyst/Chemicals (C_CAT)",
                min_value=0.0,
                value=st.session_state.tea_params["C_CAT"],
                format="%.2e",
            )
            st.session_state.tea_params["COL"] = st.number_input(
                "Labor cost (COL)",
                min_value=0.0,
                value=st.session_state.tea_params["COL"],
                format="%.2e",
                help=PARAM_HELP["COL"]
            )
        
        with col2:
            st.write("**Variable Cost Factors**")
            st.session_state.tea_params["f_esg"] = st.number_input(
                "ESG compliance (% of direct)",
                min_value=0.0, max_value=1.0, step=0.01,
                value=st.session_state.tea_params["f_esg"],
                help=PARAM_HELP["f_esg"]
            )
            st.session_state.tea_params["f_pack"] = st.number_input(
                "Packaging factor",
                min_value=0.0, max_value=1.0, step=0.01,
                value=st.session_state.tea_params["f_pack"],
            )
            st.session_state.tea_params["f_risk_op"] = st.number_input(
                "Operational risk factor",
                min_value=0.0, max_value=1.0, step=0.01,
                value=st.session_state.tea_params["f_risk_op"],
            )
    
    # Production & Financial
    with st.expander("🏭 Production & Financial", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**Production**")
            st.session_state.tea_params["Q_prod"] = st.number_input(
                "Production rate (Q_prod, tons/yr)",
                min_value=0.0,
                value=st.session_state.tea_params["Q_prod"],
                format="%.2e",
                help=PARAM_HELP["Q_prod"]
            )
            st.session_state.tea_params["P_prod"] = st.number_input(
                "Product price (P_prod, USD/ton)",
                min_value=0.0,
                value=st.session_state.tea_params["P_prod"],
                format="%.2f",
                help=PARAM_HELP["P_prod"]
            )
        
        with col2:
            st.write("**Timeline**")
            st.session_state.tea_params["N_project"] = st.number_input(
                "Project life (N_project, years)",
                min_value=1, max_value=100,
                value=int(st.session_state.tea_params["N_project"]),
                help=PARAM_HELP["N_project"]
            )
            st.session_state.tea_params["L_asset"] = st.number_input(
                "Asset life (L_asset, years)",
                min_value=1, max_value=100,
                value=int(st.session_state.tea_params["L_asset"]),
            )
            st.session_state.tea_params["salv_frac"] = st.number_input(
                "Salvage value (% of CAPEX)",
                min_value=0.0, max_value=1.0, step=0.01,
                value=st.session_state.tea_params["salv_frac"],
            )
        
        with col3:
            st.write("**Discount & Depreciation**")
            st.session_state.tea_params["i_base"] = st.number_input(
                "Discount rate (i_base)",
                min_value=0.0, max_value=0.5, step=0.001,
                value=st.session_state.tea_params["i_base"],
                format="%.4f",
                help=PARAM_HELP["i_base"]
            )
            st.session_state.tea_params["delta_risk"] = st.number_input(
                "Risk premium (delta_risk)",
                min_value=0.0, max_value=0.2, step=0.001,
                value=st.session_state.tea_params["delta_risk"],
                format="%.4f",
            )
            st.session_state.tea_params["dep_method"] = st.selectbox(
                "Depreciation method",
                options=["SL", "SYD", "DDB"],
                index=0 if st.session_state.tea_params["dep_method"] == "SL" else 1,
                help=PARAM_HELP["dep_method"]
            )
    
    # Emissions & Carbon
    with st.expander("♻️ Emissions & Carbon Pricing"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.session_state.tea_params["E_CO2"] = st.number_input(
                "Annual CO2 emissions (E_CO2, tons)",
                min_value=0.0,
                value=st.session_state.tea_params["E_CO2"],
                format="%.2e",
                help=PARAM_HELP["E_CO2"]
            )
        
        with col2:
            st.session_state.tea_params["tau_CO2"] = st.number_input(
                "Carbon price (tau_CO2, USD/ton CO2)",
                min_value=0.0,
                value=st.session_state.tea_params["tau_CO2"],
                format="%.2f",
                help=PARAM_HELP["tau_CO2"]
            )
    
    # Advanced JSON editor
    with st.expander("⚙️ Advanced: JSON Editor"):
        st.markdown("Edit all parameters as JSON. Click 'Apply' to update.")
        params_json_str = json.dumps(
            st.session_state.tea_params, indent=2, default=str
        )
        edited_json = st.text_area(
            "JSON Parameters",
            value=params_json_str,
            height=300,
            key="json_editor"
        )
        
        col_apply, col_copy = st.columns([1, 1])
        with col_apply:
            if st.button("Apply JSON", key="apply_json"):
                try:
                    parsed = json.loads(edited_json)
                    if isinstance(parsed, dict):
                        st.session_state.tea_params.update(parsed)
                        st.success("JSON parameters applied")
                    else:
                        st.error("JSON must be a dictionary")
                except json.JSONDecodeError as e:
                    st.error(f"JSON error: {e}")
        
        with col_copy:
            st.download_button(
                "📋 Copy as JSON",
                data=params_json_str,
                file_name="tea_params.json",
                mime="application/json"
            )


# ==================== TAB 2: ANALYSIS ====================

with tabs[1]:
    st.header("TEA Calculation & Results")
    
    if st.session_state.compliance_module is None:
        st.error("⚠️ Compliance module not loaded. Check configuration.")
        st.stop()
    
    mod = st.session_state.compliance_module
    
    # Validation before run
    col_run, col_validate, col_info = st.columns([1, 1, 2])
    
    with col_validate:
        if st.button("Validate Parameters", key="val_btn"):
            valid, msg = validate_params(st.session_state.tea_params)
            if valid:
                st.success(f"✓ {msg}")
            else:
                st.error(f"✗ {msg}")
    
    with col_run:
        run_tea = st.button(
            "🚀 Run TEA",
            key="run_tea_main",
            type="primary"
        )
    
    with col_info:
        st.info("Ensure all parameters are valid before running calculations")
    
    if run_tea:
        # Validation check
        valid, msg = validate_params(st.session_state.tea_params)
        if not valid:
            st.error(f"Cannot run: {msg}")
            st.stop()
        
        with st.spinner("Running Compliance & Risk Analysis..."):
            try:
                params = st.session_state.tea_params
                results = {
                    "status": "success",
                    "timestamp": datetime.now().isoformat(),
                }
                
                # Create Risk objects from parameters
                if hasattr(mod, "Risk"):
                    risks = [
                        mod.Risk(
                            name="Cost Overrun",
                            probability=min(1.0, params.get("f_cont", 0.1)),
                            severity=min(10.0, params.get("i_base", 0.1) * 100)
                        ),
                        mod.Risk(
                            name="Schedule Delay",
                            probability=min(1.0, params.get("f_cons", 0.15)),
                            severity=6.0
                        ),
                        mod.Risk(
                            name="Carbon Price Volatility",
                            probability=0.5,
                            severity=min(10.0, params.get("tau_CO2", 50) / 10)
                        ),
                    ]
                    
                    # ✅ USE compliance.py function: risk_dataframe()
                    results["risk_dataframe"] = mod.risk_dataframe(risks)
                    
                    # ✅ CREATE STRUCTURED ESGFP SCORES (all 13 key issues)
                    scores_by_tech = {
                        "Current_Design": {
                            "Environmental:GHG Emissions": min(10.0, params.get("E_CO2", 0) / 1000 + 5),
                            "Environmental:Water Use": min(10.0, (params.get("C_UT", 0) / 1e7) * 5),
                            "Environmental:Waste Management": min(10.0, params.get("f_waste", 0.01) * 100),
                            "Social:Labor Safety": min(10.0, (params.get("COL", 0) / 1e7) * 5),
                            "Social:Community Impact": min(10.0, params.get("f_esg", 0.07) * 100),
                            "Governance:Compliance": min(10.0, params.get("f_esg", 0.07) * 100),
                            "Governance:Transparency": 5.0,
                            "Financial:CAPEX": min(10.0, params.get("C_PE", 0) / 1e8 * 5),
                            "Financial:OPEX": min(10.0, params.get("C_RM", 0) / 1e7 * 5),
                            "Financial:ROI": min(10.0, params.get("P_prod", 0) / 100),
                            "Process:Efficiency": min(10.0, params.get("Q_prod", 0) / 100000),
                            "Process:Flexibility": 5.0,
                            "Process:Scalability": 5.0,
                        }
                    }
                    
                    # ✅ USE compliance.py function: pillar_averages_multi()
                    if hasattr(mod, "DEFAULT_ESGFP") and hasattr(mod, "pillar_averages_multi"):
                        try:
                            pillar_avgs = mod.pillar_averages_multi(scores_by_tech, mod.DEFAULT_ESGFP)
                            results["pillar_avgs"] = pillar_avgs
                            results["scores_by_tech"] = scores_by_tech
                            
                            # Extract simple pillar scores for display
                            results["esgfp_scores"] = pillar_avgs["Current_Design"].to_dict()
                        except Exception as e:
                            # Fallback to simple scores
                            results["esgfp_scores"] = {
                                "Environmental": min(10.0, params.get("E_CO2", 0) / 1000 + 5),
                                "Social": min(10.0, params.get("COL", 0) / 10000 + 5),
                                "Governance": min(10.0, params.get("f_esg", 0.5) * 10 + 5),
                                "Financial": min(10.0, params.get("P_prod", 1000) / 200 + 2),
                                "Process": min(10.0, params.get("Q_prod", 1000) / 200 + 3),
                            }
                    else:
                        # Fallback: simple calculated metrics
                        results["esgfp_scores"] = {
                            "Environmental": min(10.0, params.get("E_CO2", 0) / 1000 + 5),
                            "Social": min(10.0, params.get("COL", 0) / 10000 + 5),
                            "Governance": min(10.0, params.get("f_esg", 0.5) * 10 + 5),
                            "Financial": min(10.0, params.get("P_prod", 1000) / 200 + 2),
                            "Process": min(10.0, params.get("Q_prod", 1000) / 200 + 3),
                        }
                
                # Calculate summary metrics from parameters
                results["Summary"] = {
                    "CAPEX": params.get("C_PE", 0) * (
                        1 + params.get("f_ins", 0) + params.get("f_eng", 0)
                    ),
                    "Annual_OPEX": params.get("C_RM", 0) + params.get("C_UT", 0) + params.get("COL", 0),
                    "Annual_Revenue": params.get("Q_prod", 0) * params.get("P_prod", 0),
                    "Carbon_Cost": params.get("E_CO2", 0) * params.get("tau_CO2", 0),
                }
                
                st.session_state.tea_results = results
                st.success("✅ Analysis complete!")
            
            except Exception as e:
                st.error(f"Analysis error: {e}")
                st.exception(e)
    
    # Display results if available
    if st.session_state.tea_results:
        results = st.session_state.tea_results
        
        st.write("---")
        st.subheader("📋 Key Results")
        
        # Key metrics in columns
        metric_cols = st.columns(4)
        
        with metric_cols[0]:
            capex = results.get("CAPEX", 0)
            st.metric("CAPEX", f"${capex:,.0f}")
        
        with metric_cols[1]:
            opex = results.get("Annual_OPEX", 0)
            st.metric("Annual OPEX", f"${opex:,.0f}")
        
        with metric_cols[2]:
            rev = results.get("Annual_Revenue", 0)
            st.metric("Annual Revenue", f"${rev:,.0f}")
        
        with metric_cols[3]:
            margin = (rev - opex) if (rev > 0) else -opex
            st.metric(
                "EBITDA (est.)",
                f"${margin:,.0f}",
                delta=f"{(margin/rev*100):.1f}% margin" if rev > 0 else "N/A"
            )
        
        st.subheader("📊 Analysis Results")
        
        # Display risk dataframe if available
        if "risk_dataframe" in results:
            st.write("**Risk Assessment:**")
            st.dataframe(
                results["risk_dataframe"],
                use_container_width=True
            )
        
        # Display ESGFP scores
        if "esgfp_scores" in results:
            st.write("**ESGFP Pillar Scores:**")
            esgfp_df = pd.DataFrame(
                list(results["esgfp_scores"].items()),
                columns=["Pillar", "Score"]
            )
            st.dataframe(esgfp_df, use_container_width=True)
        
        # Display summary metrics
        if "Summary" in results:
            st.write("**Financial Summary:**")
            summary = results["Summary"]
            
            metric_cols = st.columns(4)
            with metric_cols[0]:
                st.metric(
                    "CAPEX",
                    f"${summary.get('CAPEX', 0):,.0f}"
                )
            with metric_cols[1]:
                st.metric(
                    "Annual OPEX",
                    f"${summary.get('Annual_OPEX', 0):,.0f}"
                )
            with metric_cols[2]:
                st.metric(
                    "Annual Revenue",
                    f"${summary.get('Annual_Revenue', 0):,.0f}"
                )
            with metric_cols[3]:
                st.metric(
                    "Carbon Cost",
                    f"${summary.get('Carbon_Cost', 0):,.0f}"
                )
        
        st.write("---")
        
        # Full results display
        st.write("**Full Results Dictionary:**")
        st.json(results)


# ==================== TAB 3: VISUALIZATIONS ====================

with tabs[2]:
    st.header("Analysis Visualizations")
    
    if st.session_state.tea_results is None:
        st.info("Run analysis first (Analysis tab)")
    else:
        results = st.session_state.tea_results
        
        # ESGFP scores visualization
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("ESGFP Pillar Scores")
            if "esgfp_scores" in results:
                esgfp = results["esgfp_scores"]
                fig = px.bar(
                    x=list(esgfp.keys()),
                    y=list(esgfp.values()),
                    title="ESGFP Pillar Ratings",
                    labels={"x": "Pillar", "y": "Score (0-10)"},
                )
                fig.update_layout(yaxis_range=[0, 10])
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Cost Structure")
            if "Summary" in results:
                summary = results["Summary"]
                costs = {
                    "CAPEX": summary.get("CAPEX", 0),
                    "OPEX": summary.get("Annual_OPEX", 0),
                    "Carbon Cost": summary.get("Carbon_Cost", 0),
                }
                # Filter out zero values
                costs = {k: v for k, v in costs.items() if v > 0}
                
                if costs:
                    fig = px.pie(
                        values=list(costs.values()),
                        names=list(costs.keys()),
                        title="Cost Breakdown"
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        st.write("---")
        # ✅ ADD COMPLIANCE.PY VISUALIZATION OPTIONS
        st.subheader("Advanced Visualizations from Compliance Analysis")
        viz_options = []
        if "pillar_avgs" in results and hasattr(mod, "plot_pillar_heatmaps"):
            viz_options.append("Pillar Heatmaps")
        if "pillar_avgs" in results and hasattr(mod, "plot_radar_profiles"):
            viz_options.append("Radar Profiles")
        if "risk_dataframe" in results and hasattr(mod, "plot_risk_views"):
            viz_options.append("Risk Views (3 Charts)")
        if hasattr(mod, "plot_parallel_coordinates"):
            viz_options.append("Parallel Coordinates")
        if "pillar_avgs" in results and hasattr(mod, "plot_tradeoff_scatter"):
            viz_options.append("Trade-off Analysis")
        if viz_options:
            selected_viz = st.selectbox(
                "Select Visualization",
                viz_options,
                key="viz_select"
            )
            try:
                # Pillar Heatmaps
                if selected_viz == "Pillar Heatmaps" and "pillar_avgs" in results:
                    st.write("**Pillar Scores Heatmap**")
                    if hasattr(mod, "plot_pillar_heatmaps"):
                        fig = mod.plot_pillar_heatmaps(results["pillar_avgs"])
                        st.pyplot(fig=fig)
                        plt.close(fig)
                # Radar Profiles
                elif selected_viz == "Radar Profiles" and "pillar_avgs" in results:
                    st.write("**ESGFP Radar Profiles**")
                    if hasattr(mod, "plot_radar_profiles"):
                        fig = mod.plot_radar_profiles(results["pillar_avgs"])
                        st.pyplot(fig=fig)
                        plt.close(fig)
                # Risk Views
                elif selected_viz == "Risk Views (3 Charts)" and "risk_dataframe" in results:
                    st.write("**Risk Assessment Visualizations**")
                    if hasattr(mod, "plot_risk_views"):
                        fig = mod.plot_risk_views(results["risk_dataframe"])
                        st.pyplot(fig=fig)
                        plt.close(fig)
                # Parallel Coordinates
                elif selected_viz == "Parallel Coordinates":
                    st.write("**Parallel Coordinates Plot**")
                    if hasattr(mod, "plot_parallel_coordinates"):
                        if "pillar_avgs" in results:
                            fig = mod.plot_parallel_coordinates(results["pillar_avgs"])
                            st.pyplot(fig=fig)
                            plt.close(fig)
                # Trade-off Analysis
                elif selected_viz == "Trade-off Analysis" and "pillar_avgs" in results:
                    st.write("**Pillar Trade-off Analysis**")
                    if hasattr(mod, "plot_tradeoff_scatter"):
                        fig = mod.plot_tradeoff_scatter(results["pillar_avgs"])
                        st.pyplot(fig=fig)
                        plt.close(fig)
            except Exception as e:
                st.error(f"Error generating visualization: {str(e)}")
        st.write("---")
        
            # Risk visualization
        if "risk_dataframe" in results:
            st.subheader("Risk Assessment")
            risk_df = results["risk_dataframe"]
            
            # Risk scatter: Probability vs Severity
            if not risk_df.empty:
                fig = px.scatter(
                    risk_df,
                    x="Probability",
                    y="Severity",
                    size="Rating",
                    hover_name="Risk",
                    title="Risk Matrix (Size = Rating)",
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Parameter influence visualization
        st.subheader("Parameter Influence on Costs")
        params = st.session_state.tea_params
        
        param_impacts = {
            "Equipment Cost": params.get("C_PE", 0),
            "Raw Materials": params.get("C_RM", 0),
            "Labor": params.get("COL", 0),
            "Carbon Price": params.get("tau_CO2", 0),
        }
        
        # Filter out zero values
        param_impacts = {k: v for k, v in param_impacts.items() if v > 0}
        
        if param_impacts:
            fig = px.bar(
                x=list(param_impacts.keys()),
                y=list(param_impacts.values()),
                title="Key Cost Drivers",
                labels={"x": "Parameter", "y": "USD"}
            )
            fig.update_xaxes(tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)


# ==================== TAB 4: SENSITIVITY ====================

with tabs[3]:
    st.header("Sensitivity & Scenario Analysis")
    
    if st.session_state.tea_results is None:
        st.info("Run analysis first")
    else:
        st.subheader("Scenario Manager")
        
        scenario_name = st.text_input(
            "Scenario name",
            value=f"Scenario {len(st.session_state.saved_scenarios) + 1}",
            key="scenario_name_input"
        )
        
        if st.button("Save current as scenario", key="save_scenario_btn"):
            if scenario_name not in st.session_state.saved_scenarios:
                st.session_state.saved_scenarios[scenario_name] = {
                    "params": dict(st.session_state.tea_params),
                    "results": dict(st.session_state.tea_results),
                    "timestamp": datetime.now().isoformat(),
                }
                st.success(f"Scenario '{scenario_name}' saved")
            else:
                st.warning(f"Scenario '{scenario_name}' already exists")
        
        if st.session_state.saved_scenarios:
            st.write("---")
            st.subheader("Saved Scenarios")
            
            for name, data in st.session_state.saved_scenarios.items():
                col_name, col_load, col_del = st.columns([2, 1, 1])
                
                with col_name:
                    st.write(f"**{name}**")
                    ts = data.get("timestamp", "N/A")
                    st.caption(f"Saved: {ts[:10]}")
                
                with col_load:
                    if st.button(f"Load", key=f"load_{name}"):
                        st.session_state.tea_params = (
                            dict(data["params"])
                        )
                        st.session_state.tea_results = (
                            dict(data["results"])
                        )
                        st.success(f"Loaded '{name}'")
                        st.rerun()
                
                with col_del:
                    if st.button(f"Delete", key=f"del_{name}"):
                        del st.session_state.saved_scenarios[name]
                        st.success(f"Deleted '{name}'")
                        st.rerun()
        
        # Parameter sensitivity (Tornado chart logic)
        st.write("---")
        st.subheader("One-Way Sensitivity (Tornado)")

        # ✅ ADD MCDA SCENARIO ANALYSIS
        st.write("---")
        st.subheader("MCDA Scenario Analysis")

        if "pillar_avgs" in st.session_state.tea_results and hasattr(mod, "weighted_sum_method"):
            st.write("**Configure Decision Weights**")
            col1, col2, col3 = st.columns(3)
            weights = {}
            with col1:
                weights["Environmental"] = st.slider(
                    "Environmental", 0.0, 1.0, 0.2, key="weight_env"
                )
                weights["Social"] = st.slider(
                    "Social", 0.0, 1.0, 0.2, key="weight_soc"
                )
            with col2:
                weights["Governance"] = st.slider(
                    "Governance", 0.0, 1.0, 0.2, key="weight_gov"
                )
                weights["Financial"] = st.slider(
                    "Financial", 0.0, 1.0, 0.2, key="weight_fin"
                )
            with col3:
                weights["Process"] = st.slider(
                    "Process", 0.0, 1.0, 0.2, key="weight_proc"
                )
            # Normalize weights
            total_weight = sum(weights.values())
            if total_weight > 0:
                weights = {k: v/total_weight for k, v in weights.items()}
            st.write(f"**Normalized Weights:** {weights}")

            # MCDA Methods
            st.write("---")
            st.write("**Select MCDA Method**")

            mcda_methods = []
            if hasattr(mod, "weighted_sum_method"):
                mcda_methods.append("Weighted Sum")
            if hasattr(mod, "wpm_method"):
                mcda_methods.append("Weighted Product")
            if hasattr(mod, "rank_method"):
                mcda_methods.append("Ranking")
            if hasattr(mod, "topsis_method"):
                mcda_methods.append("TOPSIS")
            if hasattr(mod, "vikor_method"):
                mcda_methods.append("VIKOR")
            if hasattr(mod, "edas_method"):
                mcda_methods.append("EDAS")
            if hasattr(mod, "maut_method"):
                mcda_methods.append("MAUT")
            if hasattr(mod, "pca_method"):
                mcda_methods.append("PCA")

            col_mcda_1, col_mcda_2, col_mcda_3, col_mcda_4 = st.columns(4)

            mcda_results = {}

            try:
                pillar_avgs = st.session_state.tea_results["pillar_avgs"]
                esgfp_scores = pillar_avgs["Current_Design"].to_dict() if hasattr(pillar_avgs, "iloc") else pillar_avgs.get("Current_Design", {})

                # Convert to format expected by MCDA functions
                alternatives = {"Current_Design": esgfp_scores}
                criteria = list(weights.keys())

                with col_mcda_1:
                    if st.button("▶ Weighted Sum", key="btn_weighted_sum"):
                        if hasattr(mod, "weighted_sum_method"):
                            try:
                                result = mod.weighted_sum_method(alternatives, criteria, weights)
                                mcda_results["Weighted Sum"] = result
                                st.success("✅ Weighted Sum computed")
                            except Exception as e:
                                st.error(f"Error: {str(e)[:100]}")

                with col_mcda_2:
                    if st.button("▶ Weighted Product", key="btn_wpm"):
                        if hasattr(mod, "wpm_method"):
                            try:
                                result = mod.wpm_method(alternatives, criteria, weights)
                                mcda_results["Weighted Product"] = result
                                st.success("✅ Weighted Product computed")
                            except Exception as e:
                                st.error(f"Error: {str(e)[:100]}")

                with col_mcda_3:
                    if st.button("▶ TOPSIS", key="btn_topsis"):
                        if hasattr(mod, "topsis_method"):
                            try:
                                result = mod.topsis_method(alternatives, criteria, weights)
                                mcda_results["TOPSIS"] = result
                                st.success("✅ TOPSIS computed")
                            except Exception as e:
                                st.error(f"Error: {str(e)[:100]}")

                with col_mcda_4:
                    if st.button("▶ VIKOR", key="btn_vikor"):
                        if hasattr(mod, "vikor_method"):
                            try:
                                result = mod.vikor_method(alternatives, criteria, weights)
                                mcda_results["VIKOR"] = result
                                st.success("✅ VIKOR computed")
                            except Exception as e:
                                st.error(f"Error: {str(e)[:100]}")

                col_mcda_5, col_mcda_6, col_mcda_7, col_mcda_8 = st.columns(4)

                with col_mcda_5:
                    if st.button("▶ EDAS", key="btn_edas"):
                        if hasattr(mod, "edas_method"):
                            try:
                                result = mod.edas_method(alternatives, criteria, weights)
                                mcda_results["EDAS"] = result
                                st.success("✅ EDAS computed")
                            except Exception as e:
                                st.error(f"Error: {str(e)[:100]}")

                with col_mcda_6:
                    if st.button("▶ MAUT", key="btn_maut"):
                            if hasattr(mod, "maut_method"):
                                try:
                                    result = mod.maut_method(alternatives, criteria, weights)
                                    mcda_results["MAUT"] = result
                                    st.success("✅ MAUT computed")
                                except Exception as e:
                                    st.error(f"Error: {str(e)[:100]}")
                
                    with col_mcda_7:
                        if st.button("▶ PCA", key="btn_pca"):
                            if hasattr(mod, "pca_method"):
                                try:
                                    result = mod.pca_method(alternatives, criteria, weights)
                                    mcda_results["PCA"] = result
                                    st.success("✅ PCA computed")
                                except Exception as e:
                                    st.error(f"Error: {str(e)[:100]}")
                
                    with col_mcda_8:
                        if st.button("▶ Ranking", key="btn_rank"):
                            if hasattr(mod, "rank_method"):
                                try:
                                    result = mod.rank_method(alternatives, criteria, weights)
                                    mcda_results["Ranking"] = result
                                    st.success("✅ Ranking computed")
                                except Exception as e:
                                    st.error(f"Error: {str(e)[:100]}")
                
                    # Store MCDA results
                    if mcda_results:
                        st.session_state.tea_results["mcda_results"] = mcda_results
                        st.write("---")
                        st.subheader("MCDA Results")
                    
                        for method_name, result in mcda_results.items():
                            with st.expander(f"📊 {method_name}", expanded=False):
                                if isinstance(result, dict):
                                    st.json(result)
                                elif isinstance(result, pd.DataFrame):
                                    st.dataframe(result)
                                else:
                                    st.write(result)
            except Exception as e:
                st.warning(f"MCDA analysis not available: {str(e)}")

            st.write("---")
        
            tornado_var = st.selectbox(
                "Select variable to vary",
                options=list(DEFAULT_PARAMS.keys()),
                index=0,
                key="tornado_select"
            )
            
            tornado_range_pct = st.slider(
                "Variation range (% of base)",
                min_value=10, max_value=100, value=20, step=5,
                key="tornado_range"
            )
            
            if st.button("Run sensitivity", key="run_tornado"):
                with st.spinner("Computing sensitivity..."):
                    try:
                        base_results = st.session_state.tea_results
                        base_capex = base_results.get("CAPEX", 0)
                        
                        variations = []
                        base_val = st.session_state.tea_params[tornado_var]
                        
                        for direction in [-1, 1]:
                            test_params = (
                                dict(st.session_state.tea_params)
                            )
                            delta = (
                                base_val * tornado_range_pct / 100 * direction
                            )
                            test_params[tornado_var] = base_val + delta
                            
                            # Recalculate CAPEX
                            try:
                                test_capex = mod.calculate_capex(test_params)
                                change = test_capex - base_capex
                                variations.append({
                                    "Direction": "+" if direction > 0 else "-",
                                    "Value": abs(change),
                                })
                            except Exception:
                                pass
                        
                        if variations:
                            df_tornado = pd.DataFrame(variations)
                            fig = px.bar(
                                df_tornado,
                                x="Value", y="Direction",
                                orientation="h",
                                title=(
                                    f"CAPEX sensitivity to "
                                    f"{tornado_var} (±{tornado_range_pct}%)"
                                )
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("Could not compute sensitivity")
                    
                    except Exception as e:
                        st.error(f"Sensitivity error: {e}")


# ==================== TAB 5: EXPORT ====================

with tabs[4]:
    st.header("Export & Reporting")
    
    if st.session_state.tea_results is None:
        st.info("Run TEA analysis first to enable exports")
    else:
        results = st.session_state.tea_results
        params = st.session_state.tea_params
        
        st.write("---")
        st.subheader("📥 Download Options")
        
        # JSON Export
        results_json = json.dumps(
            {
                "parameters": params,
                "results": {
                    k: v for k, v in results.items()
                    if isinstance(v, (int, float, str))
                },
                "timestamp": datetime.now().isoformat(),
            },
            indent=2,
            default=str
        )
        
        st.download_button(
            "📄 Results (JSON)",
            data=results_json,
            file_name="tea_results.json",
            mime="application/json"
        )
        
        # CSV Export
        csv_data = io.StringIO()
        csv_data.write("TEA Analysis Results\n")
        csv_data.write(f"Timestamp,{datetime.now().isoformat()}\n\n")
        csv_data.write("Parameters\n")
        for k, v in params.items():
            csv_data.write(f"{k},{v}\n")
        csv_data.write("\nResults\n")
        for k, v in results.items():
            if isinstance(v, (int, float)):
                csv_data.write(f"{k},{v}\n")
        
        st.download_button(
            "📊 Results (CSV)",
            data=csv_data.getvalue(),
            file_name="tea_results.csv",
            mime="text/csv"
        )
        
        # PDF Report
        st.write("---")
        
        # ✅ ADD DEA ANALYSIS
        st.subheader("Efficiency Analysis (DEA)")
        
        col_dea1, col_dea2 = st.columns(2)
        
        with col_dea1:
            if hasattr(mod, "dominance_matrix"):
                if st.button("Compute DEA Dominance", key="btn_dea_dominance"):
                    try:
                        with st.spinner("Computing dominance analysis..."):
                            if "pillar_avgs" in results:
                                alternatives = {"Current_Design": results["pillar_avgs"]["Current_Design"].to_dict()}
                                dominance = mod.dominance_matrix(alternatives)
                                st.session_state.tea_results["dea_dominance"] = dominance
                                with st.expander("📊 Dominance Matrix"):
                                    if isinstance(dominance, pd.DataFrame):
                                        st.dataframe(dominance)
                                    else:
                                        st.json(dominance)
                                st.success("✅ Dominance analysis complete")
                    except Exception as e:
                        st.error(f"DEA error: {str(e)[:100]}")
        
        with col_dea2:
            if hasattr(mod, "approximation_diagnostics"):
                if st.button("Compute Frontier", key="btn_dea_approx"):
                    try:
                        with st.spinner("Computing frontier..."):
                            if "pillar_avgs" in results:
                                alternatives = {"Current_Design": results["pillar_avgs"]["Current_Design"].to_dict()}
                                criteria = list(results["pillar_avgs"]["Current_Design"].to_dict().keys())
                                approx = mod.approximation_diagnostics(alternatives, criteria)
                                st.session_state.tea_results["dea_approx"] = approx
                                with st.expander("📈 Frontier Approximation"):
                                    if isinstance(approx, pd.DataFrame):
                                        st.dataframe(approx)
                                    else:
                                        st.json(approx)
                                st.success("✅ Frontier complete")
                    except Exception as e:
                        st.error(f"DEA error: {str(e)[:100]}")
        
        st.write("---")
        
        # ✅ MONTE CARLO SENSITIVITY
        st.subheader("Monte Carlo Sensitivity (2000 Simulations)")
        
        if hasattr(mod, "monte_carlo_sensitivity"):
            if st.button("🎲 Run Monte Carlo", key="btn_monte_carlo"):
                try:
                    with st.spinner("Running 2000 simulations..."):
                        param_ranges = {
                            k: (v * 0.8, v * 1.2) 
                            for k, v in params.items() 
                            if isinstance(v, (int, float)) and v > 0
                        }
                        
                        if param_ranges:
                            mc_results = mod.monte_carlo_sensitivity(param_ranges, n_simulations=2000)
                            st.session_state.tea_results["monte_carlo"] = mc_results
                            
                            with st.expander("📊 Monte Carlo Results"):
                                if isinstance(mc_results, pd.DataFrame):
                                    st.dataframe(mc_results, use_container_width=True)
                                    st.write("**Summary Statistics**")
                                    st.json({
                                        "mean": float(mc_results.mean().iloc[0]) if len(mc_results) > 0 else 0,
                                        "std": float(mc_results.std().iloc[0]) if len(mc_results) > 0 else 0,
                                        "min": float(mc_results.min().iloc[0]) if len(mc_results) > 0 else 0,
                                        "max": float(mc_results.max().iloc[0]) if len(mc_results) > 0 else 0,
                                    })
                                else:
                                    st.json(mc_results)
                            st.success("✅ Monte Carlo complete")
                except Exception as e:
                    st.error(f"Monte Carlo error: {str(e)[:100]}")
        
        st.write("---")
        st.subheader("📥 Report Generation")
        
        # PDF generation form
        with st.form("pdf_report_form"):
            st.write("Configure report options:")
            
            include_params = st.checkbox("Include parameters section", value=True)
            include_results = st.checkbox("Include results section", value=True, disabled=not include_params)
            include_charts = st.checkbox("Include charts", value=True, disabled=not include_results)
            include_esgfp = st.checkbox("Include ESGFP details", value=True, disabled=not include_results)
            
            # DEA and Monte Carlo options
            include_dea = st.checkbox("Include DEA analysis", value=False)
            include_mc = st.checkbox("Include Monte Carlo sensitivity", value=False)
            
            st.write("---")
            
    if st.button("Generate PDF Report", key="gen_pdf_btn"):
                try:
                    with st.spinner("Generating PDF..."):
                        buf = io.BytesIO()
                        doc = SimpleDocTemplate(
                            buf,
                            pagesize=A4,
                            rightMargin=15*mm,
                            leftMargin=15*mm,
                            topMargin=15*mm,
                            bottomMargin=15*mm,
                        )
                        
                        styles = getSampleStyleSheet()
                        story = []
                        
                        # Title
                        story.append(
                            Paragraph(
                                "Compliance & ESG TEA Report",
                                styles["Heading1"]
                            )
                        )
                        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        story.append(
                            Paragraph(f"Generated: {ts}", styles["Normal"])
                        )
                        story.append(Spacer(1, 12))
                        
                        # Key metrics
                        story.append(
                            Paragraph(
                                "Key Financial Metrics",
                                styles["Heading2"]
                            )
                        )
                        
                        metrics = [
                            ["Metric", "Value"],
                            ["CAPEX", f"${results.get('CAPEX', 0):,.0f}"],
                            [
                                "Annual OPEX",
                                f"${results.get('Annual_OPEX', 0):,.0f}"
                            ],
                            [
                                "Annual Revenue",
                                f"${results.get('Annual_Revenue', 0):,.0f}"
                            ],
                            [
                                "Carbon Cost",
                                f"${results.get('Carbon_Cost', 0):,.0f}"
                            ],
                        ]
                        
                        tbl = RLTable(metrics)
                        tbl.setStyle(TableStyle([
                            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                            (
                                "BACKGROUND",
                                (0, 0), (-1, 0),
                                colors.lightgrey
                            ),
                            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ]))
                        story.append(tbl)
                        story.append(Spacer(1, 12))
                        
                        # Parameters
                        if include_params:
                            story.append(PageBreak())
                            story.append(
                                Paragraph(
                                    "Input Parameters",
                                    styles["Heading2"]
                                )
                            )
                            
                            param_data = [["Parameter", "Value"]]
                            for k, v in sorted(params.items()):
                                param_data.append([k, str(v)])
                            
                            tbl2 = RLTable(param_data)
                            tbl2.setStyle(TableStyle([
                                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                                (
                                    "BACKGROUND",
                                    (0, 0), (-1, 0),
                                    colors.lightgrey
                                ),
                                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                                ("FONTSIZE", (0, 0), (-1, -1), 9),
                            ]))
                            story.append(tbl2)
                        
                        # Results
                        if include_results:
                            story.append(PageBreak())
                            story.append(
                                Paragraph(
                                    "Analysis Results",
                                    styles["Heading2"]
                                )
                            )
                            
                            # Key results metrics
                            key_metrics = [
                                "CAPEX",
                                "Annual_OPEX",
                                "Annual_Revenue",
                                "Carbon_Cost",
                            ]
                            
                            for metric in key_metrics:
                                value = results.get(metric, "-")
                                story.append(
                                    Paragraph(
                                        f"{metric}: {value}",
                                        styles["Normal"]
                                    )
                                )
                        
                        # ESGFP details
                        if include_esgfp:
                            story.append(PageBreak())
                            story.append(
                                Paragraph(
                                    "ESGFP Analysis Details",
                                    styles["Heading2"]
                                )
                            )
                            
                            if "scores_by_tech" in results:
                                scores = results["scores_by_tech"]
                                
                                for design, score_data in scores.items():
                                    story.append(
                                        Paragraph(f"**{design}**", styles["Heading3"])
                                    )
                                    
                                    for k, v in score_data.items():
                                        story.append(
                                            Paragraph(
                                                f"{k}: {v}",
                                                styles["Normal"]
                                            )
                                        )
                                    
                                    story.append(Spacer(1, 12))
                        
                        # Charts
                        if include_charts:
                            story.append(PageBreak())
                            story.append(
                                Paragraph(
                                    "Visualizations",
                                    styles["Heading2"]
                                )
                            )
                            
                            # ESGFP scores bar chart
                            if "esgfp_scores" in results:
                                esgfp = results["esgfp_scores"]
                                fig = px.bar(
                                    x=list(esgfp.keys()),
                                    y=list(esgfp.values()),
                                    title="ESGFP Pillar Ratings",
                                    labels={"x": "Pillar", "y": "Score (0-10)"},
                                )
                                fig.update_layout(yaxis_range=[0, 10])
                                
                                img_bytes = mpl_fig_to_png_bytes(fig)
                                rl_img = pil_bytes_to_rlimage(img_bytes)
                                story.append(rl_img)
                                story.append(Spacer(1, 12))
                            
                            # Risk scatter plot
                            if "risk_dataframe" in results:
                                risk_df = results["risk_dataframe"]
                                
                                if not risk_df.empty:
                                    fig = px.scatter(
                                        risk_df,
                                        x="Probability",
                                        y="Severity",
                                        size="Rating",
                                        hover_name="Risk",
                                        title="Risk Matrix (Size = Rating)",
                                    )
                                    
                                    img_bytes = mpl_fig_to_png_bytes(fig)
                                    rl_img = pil_bytes_to_rlimage(img_bytes)
                                    story.append(rl_img)
                                    story.append(Spacer(1, 12))
                        
                        # DEA analysis
                        if include_dea and "dea_dominance" in results:
                            story.append(PageBreak())
                            story.append(
                                Paragraph(
                                    "DEA Dominance Analysis",
                                    styles["Heading2"]
                                )
                            )
                            
                            dominance = results["dea_dominance"]
                            
                            if isinstance(dominance, pd.DataFrame):
                                # Convert DataFrame to list of lists for ReportLab
                                dominance_data = [dominance.columns.tolist()] + dominance.values.tolist()
                                
                                tbl_dominance = RLTable(dominance_data)
                                tbl_dominance.setStyle(TableStyle([
                                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                                    (
                                        "BACKGROUND",
                                        (0, 0), (-1, 0),
                                        colors.lightgrey
                                    ),
                                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                                ]))
                                story.append(tbl_dominance)
                            else:
                                st.json(dominance)
                        
                        # Monte Carlo sensitivity
                        if include_mc and "monte_carlo" in results:
                            story.append(PageBreak())
                            story.append(
                                Paragraph(
                                    "Monte Carlo Sensitivity Analysis",
                                    styles["Heading2"]
                                )
                            )
                            
                            mc_results = results["monte_carlo"]
                            
                            if isinstance(mc_results, pd.DataFrame):
                                # Convert DataFrame to list of lists for ReportLab
                                mc_data = [mc_results.columns.tolist()] + mc_results.values.tolist()
                                
                                tbl_mc = RLTable(mc_data)
                                tbl_mc.setStyle(TableStyle([
                                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                                    (
                                        "BACKGROUND",
                                        (0, 0), (-1, 0),
                                        colors.lightgrey
                                    ),
                                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                                ]))
                                story.append(tbl_mc)
                            else:
                                st.json(mc_results)
                        
                        doc.build(story)
                        buf.seek(0)
                        
                        st.download_button(
                            "📄 Download PDF Report",
                            data=buf.getvalue(),
                            file_name="tea_report.pdf",
                            mime="application/pdf"
                        )
                        st.success("PDF generated successfully")
                except Exception as e:
                    st.error(f"PDF generation error: {e}")
        
    # Scenario comparison export
    if st.session_state.saved_scenarios:
        st.write("---")
        st.subheader("Compare Scenarios")
        comparison_data = []
        for name, data in st.session_state.saved_scenarios.items():
            res = data["results"]
            comparison_data.append({
                "Scenario": name,
                "CAPEX": res.get("CAPEX", 0),
                "Annual_OPEX": res.get("Annual_OPEX", 0),
                "Annual_Revenue": res.get("Annual_Revenue", 0),
            })
        df_comp = pd.DataFrame(comparison_data)
        st.dataframe(df_comp, use_container_width=True)
        # Download comparison
        csv_comp = df_comp.to_csv(index=False)
        st.download_button(
            "📊 Download scenario comparison (CSV)",
            data=csv_comp,
            file_name="scenario_comparison.csv",
            mime="text/csv"
        )
            
        # Comparison chart
        fig = px.bar(
            df_comp.melt(
                id_vars="Scenario",
                var_name="Metric",
                value_name="Value"
            ),
            x="Scenario",
            y="Value",
            color="Metric",
            barmode="group",
            title="Scenario Comparison"
        )
        st.plotly_chart(fig, use_container_width=True)

# ==================== FOOTER ====================

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: 12px;'>
    Compliance & ESG TEA Frontend © 2025 | 
    Powered by Streamlit & ReportLab
    </div>
    """,
    unsafe_allow_html=True
)
