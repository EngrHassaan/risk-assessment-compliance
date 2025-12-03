"""
compliance-frontend.py
Standalone Compliance & ESG Techno-Economic Analysis Frontend
Production-ready Streamlit app with full compliance.py functionality
"""

import json
import importlib.util
import os
import io
import sys
import tempfile
from datetime import datetime
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

# PDF tools
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image as RLImage,
    Table as RLTable,
    TableStyle,
    PageBreak,
    KeepTogether
)

# ----------------- CONFIG -----------------
COMPLIANCE_PATH = "model/compliance.py"
APP_TITLE = "Compliance & ESG Techno-Economic Analysis"
APP_VERSION = "1.0.0"

# ----------------- Helper Functions -----------------
def load_compliance_module(path: str):
    """Load compliance module dynamically."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Compliance script not found at {path}")
    spec = importlib.util.spec_from_file_location("compliance", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["compliance"] = module
    spec.loader.exec_module(module)
    return module

def mpl_fig_to_png_bytes(fig: Figure, dpi: int = 150) -> bytes:
    """Convert matplotlib figure to PNG bytes."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor='white')
    buf.seek(0)
    return buf.getvalue()

def plotly_fig_to_png_bytes(fig, width: int = 800, height: int = 500) -> bytes:
    """Convert plotly figure to PNG bytes."""
    try:
        img_bytes = fig.to_image(format="png", width=width, height=height, scale=2)
        return img_bytes
    except Exception as e:
        print(f"Plotly to PNG error: {e}")
        # Create a placeholder image
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.text(0.5, 0.5, "Chart could not be rendered", 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_axis_off()
        return mpl_fig_to_png_bytes(fig)

def pil_bytes_to_rlimage(png_bytes: bytes, max_width_mm: float = 160) -> RLImage:
    """Convert PIL bytes to ReportLab image with scaling."""
    img = Image.open(io.BytesIO(png_bytes))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    rl_img = RLImage(buf)
    max_w_pt = max_width_mm * mm
    if rl_img.drawWidth > max_w_pt:
        scale = max_w_pt / rl_img.drawWidth
        rl_img.drawWidth = rl_img.drawWidth * scale
        rl_img.drawHeight = rl_img.drawHeight * scale
    return rl_img

def build_pdf_report(title: str, sections: List[Dict[str, Any]], filename_prefix: str = "report") -> bytes:
    """Build comprehensive PDF report from sections."""
    buf = io.BytesIO()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    doc = SimpleDocTemplate(
        buf, 
        pagesize=A4,
        rightMargin=15*mm, 
        leftMargin=15*mm, 
        topMargin=15*mm, 
        bottomMargin=15*mm
    )
    
    styles = getSampleStyleSheet()
    
    # Custom styles
    styles.add(ParagraphStyle(
        name='CustomTitle',
        parent=styles['Title'],
        fontSize=16,
        spaceAfter=30,
        alignment=1
    ))
    
    styles.add(ParagraphStyle(
        name='SectionHeader',
        parent=styles['Heading2'],
        fontSize=14,
        spaceAfter=12,
        textColor=colors.HexColor('#2E4053')
    ))
    
    story = []
    
    # Cover page
    story.append(Paragraph(f"{title}", styles["CustomTitle"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Report Generated: {now}", styles["Normal"]))
    story.append(Paragraph(f"Version: {APP_VERSION}", styles["Normal"]))
    story.append(Spacer(1, 24))
    story.append(Paragraph("Techno-Economic Analysis with ESG Compliance", styles["Heading2"]))
    story.append(Spacer(1, 36))
    story.append(PageBreak())
    
    # Table of Contents
    story.append(Paragraph("Table of Contents", styles["Heading2"]))
    story.append(Spacer(1, 12))
    
    toc_items = []
    for sec in sections:
        if sec.get("heading"):
            toc_items.append(Paragraph(f"• {sec['heading']}", styles["Normal"]))
    story.extend(toc_items)
    story.append(PageBreak())
    
    # Add each section
    for sec_idx, sec in enumerate(sections):
        if sec.get("heading"):
            story.append(Paragraph(sec["heading"], styles["SectionHeader"]))
        
        if sec.get("text"):
            story.append(Paragraph(sec["text"], styles["Normal"]))
            story.append(Spacer(1, 6))
        
        # Add tables
        for tbl_data, caption in sec.get("tables", []):
            if tbl_data is None or (hasattr(tbl_data, 'empty') and tbl_data.empty):
                continue
            
            if caption:
                story.append(Paragraph(f"<b>{caption}</b>", styles["Italic"]))
                story.append(Spacer(1, 6))
            
            try:
                if isinstance(tbl_data, pd.DataFrame):
                    # Convert DataFrame to table
                    df = tbl_data.copy().round(4)
                    
                    # Handle column names
                    if hasattr(df, 'columns'):
                        header = list(df.columns)
                        if hasattr(df, 'index') and df.index.name:
                            data = [[df.index.name] + header]
                        else:
                            data = [header]
                        
                        # Add data rows
                        for idx, row in df.iterrows():
                            if isinstance(idx, tuple):
                                row_data = list(idx) + list(row)
                            else:
                                row_data = [idx] + list(row)
                            data.append(row_data)
                    else:
                        data = [["Data"]]  # Fallback
                    
                    # Create table
                    tbl = RLTable(data, hAlign="LEFT", repeatRows=1)
                    tbl.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, 0), 10),
                        ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
                        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                        ('FONTSIZE', (0, 1), (-1, -1), 9),
                        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ]))
                    
                    # Auto-adjust column widths
                    col_widths = [max(len(str(cell)) for cell in col) * 2 for col in zip(*data)]
                    total_width = sum(col_widths)
                    page_width = doc.width
                    
                    if total_width > page_width:
                        scale = page_width / total_width
                        col_widths = [w * scale for w in col_widths]
                    
                    tbl._argW = col_widths
                    story.append(tbl)
                
                story.append(Spacer(1, 12))
                
            except Exception as e:
                story.append(Paragraph(f"Table error: {str(e)[:100]}", styles["Normal"]))
        
        # Add images
        for img_bytes, img_caption in sec.get("images", []):
            try:
                if img_bytes:
                    rl_img = pil_bytes_to_rlimage(img_bytes)
                    story.append(rl_img)
                    if img_caption:
                        story.append(Paragraph(f"<i>{img_caption}</i>", styles["Italic"]))
                    story.append(Spacer(1, 12))
            except Exception as e:
                story.append(Paragraph(f"Image error: {str(e)[:100]}", styles["Normal"]))
        
        # Add page break between sections (except last one)
        if sec_idx < len(sections) - 1:
            story.append(PageBreak())
    
    # Build PDF
    doc.build(story)
    buf.seek(0)
    return buf.getvalue()

def generate_tea_results_summary(results: Dict, params: Dict) -> pd.DataFrame:
    """Generate summary table of TEA results."""
    summary_data = []
    
    # Financial Metrics
    summary_data.append(["NPV (Net Present Value)", f"${results.get('NPV', results.get('npv', 0)):,.0f}", "USD"])
    summary_data.append(["IRR (Internal Rate of Return)", f"{results.get('IRR', results.get('irr', 0))*100:.2f}%", "%"])
    summary_data.append(["LCOx (Levelized Cost)", f"${results.get('LCOx', results.get('lcox', 0)):,.2f}", "USD/ton"])
    
    payback = results.get('Payback_Period', results.get('payback_period', None))
    if payback:
        summary_data.append(["Payback Period", f"{payback} years", "years"])
    else:
        summary_data.append(["Payback Period", "> Project Life", "years"])
    
    # Cost Metrics
    summary_data.append(["Total CAPEX", f"${results.get('CAPEX', results.get('capex', 0)):,.0f}", "USD"])
    summary_data.append(["Annual OPEX", f"${results.get('Annual_OPEX', results.get('annual_opex', 0)):,.0f}", "USD/year"])
    
    annual_revenue = results.get('Annual_Revenue', results.get('annual_revenue', 0))
    if not annual_revenue:
        annual_revenue = params.get('Q_prod', 0) * params.get('P_prod', 0)
    summary_data.append(["Annual Revenue", f"${annual_revenue:,.0f}", "USD/year"])
    
    summary_data.append(["Carbon Cost", f"${results.get('Carbon_Cost', results.get('carbon_cost', 0)):,.0f}", "USD/year"])
    
    # Additional metrics if available
    bcr = results.get('BCR', results.get('bcr', None))
    if bcr and bcr != float('inf'):
        summary_data.append(["Benefit-Cost Ratio (BCR)", f"{bcr:.3f}", "ratio"])
    
    crf = results.get('CRF', results.get('crf', None))
    if crf:
        summary_data.append(["Capital Recovery Factor", f"{crf:.5f}", ""])
    
    return pd.DataFrame(summary_data, columns=["Metric", "Value", "Unit"])

# ----------------- Streamlit App -----------------
import streamlit as st

# Configure page
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------- Session State Initialization -----------------
if "compliance_module" not in st.session_state:
    st.session_state.compliance_module = None

if "uploaded_compliance" not in st.session_state:
    st.session_state.uploaded_compliance = None

if "tea_params" not in st.session_state:
    # Default parameters matching compliance.py structure
    st.session_state.tea_params = {
        "C_PE": 1e8,           # Process equipment cost
        "COL": 1e7,           # Operating labor cost
        "C_RM": 4e7,          # Raw materials cost
        "C_UT": 1.2e7,        # Utilities cost
        "C_CAT": 2e6,         # Catalysts cost
        "Q_prod": 5e5,        # Production rate (tons/year)
        "P_prod": 550.0,      # Product price (USD/ton)
        
        # CAPEX factors
        "f_ins": 0.30,        # Instrumentation
        "f_pipe": 0.45,       # Piping
        "f_elec": 0.10,       # Electrical
        "f_bldg": 0.05,       # Buildings
        "f_util": 0.06,       # Utilities
        "f_stor": 0.02,       # Storage
        "f_safe": 0.01,       # Safety
        "f_waste": 0.01,      # Waste
        "f_eng": 0.12,        # Engineering
        "f_cons": 0.10,       # Construction
        "f_licn": 0.00,       # Licensing
        "f_cont": 0.02,       # Contractor
        "f_contg": 0.0,       # Contingency
        "f_insur": 0.01,      # Insurance
        "f_own": 0.02,        # Owner's cost
        "f_start": 0.01,      # Startup
        
        # Financial parameters
        "N_project": 20,      # Project life
        "L_asset": 20,        # Asset life
        "salv_frac": 0.10,    # Salvage fraction
        "f_risk_op": 0.05,    # Operational risk factor
        "f_pack": 0.02,       # Packaging factor
        
        # ESG parameters
        "tau_CO2": 50.0,      # CO2 tax (USD/ton)
        "E_CO2": 200000.0,    # CO2 emissions (tons/year)
        "f_esg": 0.07,        # ESG compliance factor
        
        # Economic parameters
        "i_base": 0.08,       # Base discount rate
        "delta_risk": 0.03,   # Risk premium
        "dep_method": "SL"    # Depreciation method
    }

if "tea_results" not in st.session_state:
    st.session_state.tea_results = None

if "analysis_charts" not in st.session_state:
    st.session_state.analysis_charts = {}

if "scenario_data" not in st.session_state:
    st.session_state.scenario_data = {}

# ----------------- Sidebar -----------------
st.sidebar.title("💰 " + APP_TITLE)
st.sidebar.markdown("---")

# Module loading
st.sidebar.header("📦 Module Configuration")

# Try to load compliance module
if st.session_state.compliance_module is None:
    if os.path.exists(COMPLIANCE_PATH):
        try:
            comp_mod = load_compliance_module(COMPLIANCE_PATH)
            st.session_state.compliance_module = comp_mod
            st.sidebar.success("✅ Compliance module loaded!")
        except Exception as e:
            st.sidebar.error(f"❌ Failed to load: {str(e)[:100]}")

# File uploader for compliance.py
uploaded_file = st.sidebar.file_uploader("Upload compliance.py", type=["py"])
if uploaded_file is not None:
    try:
        # Save uploaded file to temp directory
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, "compliance_uploaded.py")
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())
        
        # Load the module
        comp_mod = load_compliance_module(temp_path)
        st.session_state.compliance_module = comp_mod
        st.session_state.uploaded_compliance = temp_path
        st.sidebar.success("✅ Uploaded module loaded!")
        st.rerun()
    except Exception as e:
        st.sidebar.error(f"❌ Failed to load: {str(e)[:100]}")

st.sidebar.markdown("---")
st.sidebar.header("📊 Quick Actions")

if st.sidebar.button("🔄 Reset Parameters"):
    st.session_state.tea_results = None
    st.session_state.analysis_charts = {}
    st.session_state.scenario_data = {}
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.info(f"""
**Version:** {APP_VERSION}

**Features:**
- Full TEA analysis
- ESG compliance integration
- Sensitivity analysis
- Multi-scenario comparison
- PDF reporting

**Requirements:**
- compliance.py module
- All dependencies installed
""")

# ----------------- Main App -----------------
st.title("💰 " + APP_TITLE)

# Check if module is loaded
if st.session_state.compliance_module is None:
    st.error("""
    ## Compliance Module Not Found!
    
    Please ensure `compliance.py` is available in the `model/` directory or upload it using the sidebar.
    
    The `compliance.py` module is required for all TEA calculations and analyses.
    """)
    
    # Show expected functions
    st.info("""
    **Expected compliance.py functions:**
    - `compute_TEA(params)`: Main TEA calculation
    - `run_esg_sweep_and_plots(params, design_label)`: ESG sensitivity analysis
    - `price_sweep(params)`: Price sensitivity analysis
    - `scenario_cba(params, design_label)`: Scenario CBA
    - `risk_analysis(params)`: Risk analysis
    - `base_params`: Default parameters dictionary
    """)
    
    # Create tabs for basic functionality even without module
    tabs = st.tabs(["⚙️ Parameters", "📊 Basic Analysis", "📤 Export"])
    
    with tabs[0]:
        st.header("⚙️ Parameter Configuration")
        st.warning("Compliance module not loaded. Parameters can be configured but analysis cannot run.")
        st.json(st.session_state.tea_params)
    
    st.stop()

# Module is loaded
comp_mod = st.session_state.compliance_module

# Check if module has required functions
required_funcs = ['compute_TEA', 'base_params']
missing_funcs = [f for f in required_funcs if not hasattr(comp_mod, f)]

if missing_funcs:
    st.error(f"Compliance module missing functions: {', '.join(missing_funcs)}")
    st.info("Please ensure your compliance.py has all required functions.")
    st.stop()

# Create main tabs
main_tabs = st.tabs([
    "🏠 Dashboard",
    "⚙️ Parameters", 
    "📊 TEA Analysis",
    "📈 Visualizations",
    "🔄 Sensitivity",
    "🔬 Advanced",
    "📤 Export"
])

# ----------------- Dashboard Tab -----------------
with main_tabs[0]:
    st.header("🏠 Dashboard")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Status",
            "Ready" if st.session_state.compliance_module else "Module Missing",
            delta="✅" if st.session_state.compliance_module else "❌"
        )
    
    with col2:
        param_count = len(st.session_state.tea_params)
        st.metric("Parameters Configured", f"{param_count}", "Total")
    
    with col3:
        results_status = "Available" if st.session_state.tea_results else "Not Run"
        st.metric("Analysis Results", results_status, "📊")
    
    st.markdown("---")
    
    # Quick analysis button
    if st.button("🚀 Run Quick Analysis", type="primary", use_container_width=True):
        with st.spinner("Running TEA analysis..."):
            try:
                results = comp_mod.compute_TEA(st.session_state.tea_params)
                if results:
                    st.session_state.tea_results = results
                    st.success("✅ Analysis complete! Go to TEA Analysis tab for results.")
                    st.rerun()
            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")
    
    st.markdown("---")
    
    # Recent analyses
    st.subheader("📋 Recent Analyses")
    if st.session_state.tea_results:
        col_a, col_b = st.columns(2)
        with col_a:
            npv = st.session_state.tea_results.get('NPV', st.session_state.tea_results.get('npv', 0))
            st.metric("NPV", f"${npv:,.0f}")
        
        with col_b:
            irr = st.session_state.tea_results.get('IRR', st.session_state.tea_results.get('irr', 0))
            st.metric("IRR", f"{irr*100:.2f}%")
    else:
        st.info("No analysis results yet. Run an analysis to see results here.")
    
    st.markdown("---")
    
    # Quick links
    st.subheader("🔗 Quick Navigation")
    col_links = st.columns(3)
    with col_links[0]:
        if st.button("Go to Parameters", use_container_width=True):
            st.switch_page("?tab=Parameters")
    with col_links[1]:
        if st.button("Go to Analysis", use_container_width=True):
            st.switch_page("?tab=TEA Analysis")
    with col_links[2]:
        if st.button("Go to Export", use_container_width=True):
            st.switch_page("?tab=Export")

# ----------------- Parameters Tab -----------------
with main_tabs[1]:
    st.header("⚙️ TEA Parameters Configuration")
    
    # Use base_params from compliance module if available
    if hasattr(comp_mod, 'base_params'):
        base_params = comp_mod.base_params.copy()
    else:
        base_params = st.session_state.tea_params.copy()
    
    # Create parameter groups in expanders
    with st.expander("💰 Capital Costs", expanded=True):
        col_cap1, col_cap2, col_cap3 = st.columns(3)
        
        with col_cap1:
            st.subheader("Major Equipment")
            base_params["C_PE"] = st.number_input(
                "Process Equipment (C_PE) USD", 
                value=float(base_params.get("C_PE", 1e8)),
                format="%.0f",
                help="Cost of major process equipment"
            )
        
        with col_cap2:
            st.subheader("Direct Cost Factors")
            base_params["f_ins"] = st.slider(
                "Instrumentation (f_ins)", 
                min_value=0.0, max_value=1.0, 
                value=float(base_params.get("f_ins", 0.30)),
                format="%.3f"
            )
            
            base_params["f_pipe"] = st.slider(
                "Piping (f_pipe)", 
                min_value=0.0, max_value=1.0, 
                value=float(base_params.get("f_pipe", 0.45)),
                format="%.3f"
            )
            
            base_params["f_elec"] = st.slider(
                "Electrical (f_elec)", 
                min_value=0.0, max_value=1.0, 
                value=float(base_params.get("f_elec", 0.10)),
                format="%.3f"
            )
        
        with col_cap3:
            st.subheader("Indirect Cost Factors")
            base_params["f_eng"] = st.slider(
                "Engineering (f_eng)", 
                min_value=0.0, max_value=1.0, 
                value=float(base_params.get("f_eng", 0.12)),
                format="%.3f"
            )
            
            base_params["f_cons"] = st.slider(
                "Construction (f_cons)", 
                min_value=0.0, max_value=1.0, 
                value=float(base_params.get("f_cons", 0.10)),
                format="%.3f"
            )
            
            base_params["f_contg"] = st.slider(
                "Contingency (f_contg)", 
                min_value=0.0, max_value=0.5, 
                value=float(base_params.get("f_contg", 0.0)),
                format="%.3f"
            )
    
    with st.expander("🏭 Operating Parameters", expanded=True):
        col_op1, col_op2, col_op3 = st.columns(3)
        
        with col_op1:
            st.subheader("Production")
            base_params["Q_prod"] = st.number_input(
                "Production Rate (Q_prod) tons/year", 
                value=float(base_params.get("Q_prod", 5e5)),
                format="%.0f"
            )
            
            base_params["P_prod"] = st.number_input(
                "Product Price (P_prod) USD/ton", 
                value=float(base_params.get("P_prod", 550.0)),
                format="%.2f"
            )
        
        with col_op2:
            st.subheader("Operating Costs")
            base_params["C_RM"] = st.number_input(
                "Raw Materials (C_RM) USD/year", 
                value=float(base_params.get("C_RM", 4e7)),
                format="%.0f"
            )
            
            base_params["COL"] = st.number_input(
                "Operating Labor (COL) USD/year", 
                value=float(base_params.get("COL", 1e7)),
                format="%.0f"
            )
        
        with col_op3:
            base_params["C_UT"] = st.number_input(
                "Utilities (C_UT) USD/year", 
                value=float(base_params.get("C_UT", 1.2e7)),
                format="%.0f"
            )
            
            base_params["C_CAT"] = st.number_input(
                "Catalysts (C_CAT) USD/year", 
                value=float(base_params.get("C_CAT", 2e6)),
                format="%.0f"
            )
    
    with st.expander("🌿 ESG Parameters", expanded=True):
        col_esg1, col_esg2, col_esg3 = st.columns(3)
        
        with col_esg1:
            st.subheader("Compliance Costs")
            base_params["f_esg"] = st.slider(
                "ESG Compliance Factor (f_esg)", 
                min_value=0.0, max_value=0.5, 
                value=float(base_params.get("f_esg", 0.07)),
                format="%.3f",
                help="Additional cost factor for ESG compliance"
            )
            
            base_params["f_risk_op"] = st.slider(
                "Operational Risk Factor", 
                min_value=0.0, max_value=0.2, 
                value=float(base_params.get("f_risk_op", 0.05)),
                format="%.3f"
            )
        
        with col_esg2:
            st.subheader("Carbon Costs")
            base_params["tau_CO2"] = st.number_input(
                "CO2 Tax (tau_CO2) USD/ton", 
                value=float(base_params.get("tau_CO2", 50.0)),
                format="%.2f"
            )
            
            base_params["E_CO2"] = st.number_input(
                "CO2 Emissions (E_CO2) tons/year", 
                value=float(base_params.get("E_CO2", 200000.0)),
                format="%.0f"
            )
        
        with col_esg3:
            st.subheader("Other ESG Factors")
            base_params["f_waste"] = st.slider(
                "Waste Handling", 
                min_value=0.0, max_value=0.1, 
                value=float(base_params.get("f_waste", 0.01)),
                format="%.3f"
            )
            
            base_params["f_safe"] = st.slider(
                "Safety Factor", 
                min_value=0.0, max_value=0.1, 
                value=float(base_params.get("f_safe", 0.01)),
                format="%.3f"
            )
    
    with st.expander("📈 Financial Parameters", expanded=True):
        col_fin1, col_fin2, col_fin3 = st.columns(3)
        
        with col_fin1:
            st.subheader("Project Life")
            base_params["N_project"] = st.number_input(
                "Project Life (N_project) years", 
                value=int(base_params.get("N_project", 20)),
                min_value=1, max_value=50
            )
            
            base_params["L_asset"] = st.number_input(
                "Asset Life (L_asset) years", 
                value=int(base_params.get("L_asset", 20)),
                min_value=1, max_value=50
            )
        
        with col_fin2:
            st.subheader("Economic Factors")
            base_params["i_base"] = st.slider(
                "Discount Rate (i_base)", 
                min_value=0.0, max_value=0.5, 
                value=float(base_params.get("i_base", 0.08)),
                format="%.3f"
            )
            
            base_params["delta_risk"] = st.slider(
                "Risk Premium (delta_risk)", 
                min_value=0.0, max_value=0.2, 
                value=float(base_params.get("delta_risk", 0.03)),
                format="%.3f"
            )
        
        with col_fin3:
            st.subheader("Depreciation & Salvage")
            base_params["salv_frac"] = st.slider(
                "Salvage Value Fraction", 
                min_value=0.0, max_value=1.0, 
                value=float(base_params.get("salv_frac", 0.10)),
                format="%.3f"
            )
            
            dep_methods = ["SL", "SYD", "DDB"]
            base_params["dep_method"] = st.selectbox(
                "Depreciation Method", 
                options=dep_methods,
                index=dep_methods.index(base_params.get("dep_method", "SL"))
            )
    
    # Additional parameters in collapsed expander
    with st.expander("⚙️ Additional Parameters"):
        col_add1, col_add2 = st.columns(2)
        
        with col_add1:
            base_params["f_bldg"] = st.slider("Buildings", 0.0, 0.2, float(base_params.get("f_bldg", 0.05)), 0.01)
            base_params["f_util"] = st.slider("Utilities", 0.0, 0.2, float(base_params.get("f_util", 0.06)), 0.01)
            base_params["f_stor"] = st.slider("Storage", 0.0, 0.1, float(base_params.get("f_stor", 0.02)), 0.01)
            base_params["f_licn"] = st.slider("Licensing", 0.0, 0.1, float(base_params.get("f_licn", 0.00)), 0.01)
        
        with col_add2:
            base_params["f_cont"] = st.slider("Contractor", 0.0, 0.1, float(base_params.get("f_cont", 0.02)), 0.01)
            base_params["f_insur"] = st.slider("Insurance", 0.0, 0.1, float(base_params.get("f_insur", 0.01)), 0.01)
            base_params["f_own"] = st.slider("Owner's Cost", 0.0, 0.1, float(base_params.get("f_own", 0.02)), 0.01)
            base_params["f_start"] = st.slider("Startup", 0.0, 0.1, float(base_params.get("f_start", 0.01)), 0.01)
            base_params["f_pack"] = st.slider("Packaging", 0.0, 0.1, float(base_params.get("f_pack", 0.02)), 0.01)
    
    # Save parameters
    st.session_state.tea_params = base_params
    
    st.markdown("---")
    
    # JSON Editor for advanced users
    with st.expander("🛠️ Advanced JSON Editor"):
        params_json = st.text_area(
            "Edit all parameters as JSON:",
            value=json.dumps(base_params, indent=2),
            height=300
        )
        
        col_json1, col_json2 = st.columns(2)
        with col_json1:
            if st.button("Apply JSON Configuration"):
                try:
                    new_params = json.loads(params_json)
                    st.session_state.tea_params.update(new_params)
                    st.success("Parameters updated successfully!")
                    st.rerun()
                except json.JSONDecodeError as e:
                    st.error(f"Invalid JSON format: {str(e)}")
        
        with col_json2:
            if st.button("Reset to Defaults"):
                if hasattr(comp_mod, 'base_params'):
                    st.session_state.tea_params = comp_mod.base_params.copy()
                    st.success("Reset to module defaults!")
                    st.rerun()
    
    # Save/Load parameters
    st.markdown("---")
    st.subheader("💾 Save/Load Parameters")
    
    col_save1, col_save2 = st.columns(2)
    
    with col_save1:
        if st.button("Download Parameters JSON"):
            params_str = json.dumps(st.session_state.tea_params, indent=2)
            st.download_button(
                "⬇️ Download JSON",
                data=params_str,
                file_name="tea_parameters.json",
                mime="application/json"
            )
    
    with col_save2:
        uploaded_params = st.file_uploader("Upload parameters JSON", type=["json"], key="param_upload")
        if uploaded_params:
            try:
                loaded_params = json.load(uploaded_params)
                st.session_state.tea_params.update(loaded_params)
                st.success("Parameters loaded successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Error loading parameters: {str(e)}")

# ----------------- TEA Analysis Tab -----------------
with main_tabs[2]:
    st.header("📊 TEA Analysis")
    
    col_run1, col_run2 = st.columns([2, 1])
    
    with col_run1:
        if st.button("🚀 Run Full TEA Analysis", type="primary", use_container_width=True):
            with st.spinner("Calculating TEA metrics..."):
                try:
                    # Clear previous charts
                    st.session_state.analysis_charts = {}
                    
                    # Run TEA analysis
                    results = comp_mod.compute_TEA(st.session_state.tea_params)
                    
                    if results:
                        st.session_state.tea_results = results
                        st.success("✅ TEA analysis complete!")
                        
                        # Generate charts
                        st.session_state.analysis_charts = generate_analysis_charts(results)
                        
                        st.rerun()
                    else:
                        st.error("❌ Analysis returned no results")
                        
                except Exception as e:
                    st.error(f"❌ Analysis error: {str(e)}")
    
    with col_run2:
        if st.button("🔄 Clear Results", use_container_width=True):
            st.session_state.tea_results = None
            st.session_state.analysis_charts = {}
            st.success("Results cleared!")
            st.rerun()
    
    st.markdown("---")
    
    # Display results if available
    if st.session_state.tea_results:
        results = st.session_state.tea_results
        
        # Key Metrics Cards
        st.subheader("📈 Key Financial Metrics")
        
        metric_cols = st.columns(4)
        
        with metric_cols[0]:
            npv = results.get('NPV', results.get('npv', 0))
            npv_color = "green" if npv > 0 else "red"
            st.markdown(f"""
            <div style='padding: 20px; background-color: #f8f9fa; border-radius: 10px; border-left: 5px solid {npv_color};'>
                <h3 style='margin: 0; color: {npv_color};'>${npv:,.0f}</h3>
                <p style='margin: 0; color: #666;'>Net Present Value (NPV)</p>
            </div>
            """, unsafe_allow_html=True)
        
        with metric_cols[1]:
            irr = results.get('IRR', results.get('irr', 0))
            discount_rate = st.session_state.tea_params.get('i_base', 0.08) + st.session_state.tea_params.get('delta_risk', 0.03)
            irr_color = "green" if irr > discount_rate else "red"
            st.markdown(f"""
            <div style='padding: 20px; background-color: #f8f9fa; border-radius: 10px; border-left: 5px solid {irr_color};'>
                <h3 style='margin: 0; color: {irr_color};'>{irr*100:.2f}%</h3>
                <p style='margin: 0; color: #666;'>Internal Rate of Return (IRR)</p>
            </div>
            """, unsafe_allow_html=True)
        
        with metric_cols[2]:
            lcox = results.get('LCOx', results.get('lcox', 0))
            product_price = st.session_state.tea_params.get('P_prod', 550.0)
            lcox_color = "green" if lcox < product_price else "red"
            st.markdown(f"""
            <div style='padding: 20px; background-color: #f8f9fa; border-radius: 10px; border-left: 5px solid {lcox_color};'>
                <h3 style='margin: 0; color: {lcox_color};'>${lcox:,.2f}</h3>
                <p style='margin: 0; color: #666;'>Levelized Cost (LCOx)</p>
            </div>
            """, unsafe_allow_html=True)
        
        with metric_cols[3]:
            payback = results.get('Payback_Period', results.get('payback_period', None))
            if payback:
                payback_color = "green" if payback < st.session_state.tea_params.get('N_project', 20)/2 else "orange"
                payback_text = f"{payback} years"
            else:
                payback_color = "red"
                payback_text = "> Project Life"
            
            st.markdown(f"""
            <div style='padding: 20px; background-color: #f8f9fa; border-radius: 10px; border-left: 5px solid {payback_color};'>
                <h3 style='margin: 0; color: {payback_color};'>{payback_text}</h3>
                <p style='margin: 0; color: #666;'>Payback Period</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Detailed Results Tables
        col_detail1, col_detail2 = st.columns(2)
        
        with col_detail1:
            st.subheader("💰 Cost Structure")
            
            cost_data = []
            
            # CAPEX breakdown
            capex = results.get('CAPEX', results.get('capex', 0))
            cost_data.append(["Total CAPEX", f"${capex:,.0f}"])
            
            # Annual costs
            annual_opex = results.get('Annual_OPEX', results.get('annual_opex', 0))
            cost_data.append(["Annual OPEX", f"${annual_opex:,.0f}/year"])
            
            # ESG and carbon costs
            carbon_cost = results.get('Carbon_Cost', results.get('carbon_cost', 0))
            cost_data.append(["Carbon Cost", f"${carbon_cost:,.0f}/year"])
            
            esg_cost = annual_opex * st.session_state.tea_params.get('f_esg', 0.07)
            cost_data.append(["ESG Compliance", f"${esg_cost:,.0f}/year"])
            
            # Revenue
            annual_revenue = results.get('Annual_Revenue', results.get('annual_revenue', 0))
            if not annual_revenue:
                annual_revenue = st.session_state.tea_params.get('Q_prod', 0) * st.session_state.tea_params.get('P_prod', 0)
            cost_data.append(["Annual Revenue", f"${annual_revenue:,.0f}/year"])
            
            cost_df = pd.DataFrame(cost_data, columns=["Cost Item", "Value"])
            st.dataframe(cost_df, use_container_width=True, hide_index=True)
        
        with col_detail2:
            st.subheader("📊 Additional Metrics")
            
            add_data = []
            
            # BCR if available
            bcr = results.get('BCR', results.get('bcr', None))
            if bcr and bcr != float('inf'):
                add_data.append(["Benefit-Cost Ratio (BCR)", f"{bcr:.3f}"])
            
            # CRF if available
            crf = results.get('CRF', results.get('crf', None))
            if crf:
                add_data.append(["Capital Recovery Factor", f"{crf:.5f}"])
            
            # Annualized CAPEX
            annualized_capex = results.get('Annualized_CAPEX', results.get('annualized_capex', None))
            if annualized_capex:
                add_data.append(["Annualized CAPEX", f"${annualized_capex:,.0f}/year"])
            
            # Production metrics
            q_prod = st.session_state.tea_params.get('Q_prod', 0)
            add_data.append(["Production Rate", f"{q_prod:,.0f} tons/year"])
            
            # Discount rate
            discount_rate = st.session_state.tea_params.get('i_base', 0.08) + st.session_state.tea_params.get('delta_risk', 0.03)
            add_data.append(["Discount Rate", f"{discount_rate*100:.2f}%"])
            
            if add_data:
                add_df = pd.DataFrame(add_data, columns=["Metric", "Value"])
                st.dataframe(add_df, use_container_width=True, hide_index=True)
        
        # Cash Flow Table
        st.markdown("---")
        st.subheader("💵 Cash Flow Analysis")
        
        # Get cash flow data
        cf = results.get('CF', results.get('cash_flow', []))
        if not cf:
            cf = [0] * (st.session_state.tea_params.get('N_project', 20) + 1)
        
        cumulative_cf = results.get('Cumulative_CF', results.get('cumulative_cf', []))
        if not cumulative_cf:
            cumulative_cf = np.cumsum(cf).tolist()
        
        discounted_cf = results.get('Discounted_CF', results.get('discounted_cf', []))
        if not discounted_cf:
            discount_rate = st.session_state.tea_params.get('i_base', 0.08) + st.session_state.tea_params.get('delta_risk', 0.03)
            discounted_cf = [cf[i] / ((1 + discount_rate) ** i) for i in range(len(cf))]
        
        # Create cash flow table
        cf_data = []
        for year in range(len(cf)):
            cf_data.append([
                year,
                f"${cf[year]:,.0f}",
                f"${discounted_cf[year]:,.0f}" if year < len(discounted_cf) else "$0",
                f"${cumulative_cf[year]:,.0f}" if year < len(cumulative_cf) else "$0"
            ])
        
        cf_df = pd.DataFrame(cf_data, columns=["Year", "Cash Flow", "Discounted CF", "Cumulative CF"])
        st.dataframe(cf_df, use_container_width=True, hide_index=True)
        
        # Download buttons
        st.markdown("---")
        st.subheader("📥 Download Results")
        
        col_dl1, col_dl2, col_dl3 = st.columns(3)
        
        with col_dl1:
            # CSV export
            summary_df = generate_tea_results_summary(results, st.session_state.tea_params)
            csv_summary = summary_df.to_csv(index=False).encode()
            st.download_button(
                "📊 Download Summary CSV",
                data=csv_summary,
                file_name="tea_summary.csv",
                mime="text/csv"
            )
        
        with col_dl2:
            # Cash flow CSV
            cf_df_csv = cf_df.to_csv(index=False).encode()
            st.download_button(
                "💵 Download Cash Flow CSV",
                data=cf_df_csv,
                file_name="cash_flow.csv",
                mime="text/csv"
            )
        
        with col_dl3:
            # JSON export
            exportable_results = {}
            for key, value in results.items():
                if isinstance(value, np.ndarray):
                    exportable_results[key] = value.tolist()
                elif isinstance(value, (int, float, str, list, dict, bool)) and value is not None:
                    exportable_results[key] = value
            
            exportable_results['parameters'] = st.session_state.tea_params
            json_str = json.dumps(exportable_results, indent=2, default=str)
            st.download_button(
                "📋 Download Full JSON",
                data=json_str,
                file_name="tea_full_results.json",
                mime="application/json"
            )
    
    else:
        st.info("👈 Run TEA analysis to see results here")
        st.image("https://images.unsplash.com/photo-1551288049-bebda4e38f71?ixlib=rb-4.0.3&auto=format&fit=crop&w=1000&q=80", 
                caption="Run analysis to generate financial metrics and visualizations")

def generate_analysis_charts(results: Dict) -> Dict:
    """Generate analysis charts from results."""
    charts = {}
    
    try:
        # Cash flow chart
        cf = results.get('CF', results.get('cash_flow', []))
        if cf:
            years = list(range(len(cf)))
            
            fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            
            # Annual cash flow
            bars = ax1.bar(years, cf, color=['red' if x < 0 else 'green' for x in cf])
            ax1.axhline(y=0, color='black', linewidth=0.5)
            ax1.set_xlabel("Year")
            ax1.set_ylabel("Cash Flow (USD)")
            ax1.set_title("Annual Cash Flow")
            ax1.grid(True, alpha=0.3)
            
            # Cumulative cash flow
            cumulative_cf = results.get('Cumulative_CF', results.get('cumulative_cf', []))
            if not cumulative_cf:
                cumulative_cf = np.cumsum(cf).tolist()
            
            ax2.plot(years, cumulative_cf, 'b-', linewidth=2, marker='o')
            ax2.axhline(y=0, color='red', linestyle='--', linewidth=1)
            ax2.fill_between(years, cumulative_cf, 0, where=np.array(cumulative_cf) >= 0, 
                           color='green', alpha=0.3, label='Profit')
            ax2.fill_between(years, cumulative_cf, 0, where=np.array(cumulative_cf) < 0, 
                           color='red', alpha=0.3, label='Loss')
            ax2.set_xlabel("Year")
            ax2.set_ylabel("Cumulative Cash Flow (USD)")
            ax2.set_title("Cumulative Cash Flow")
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            plt.tight_layout()
            charts['cash_flow'] = mpl_fig_to_png_bytes(fig1)
            plt.close(fig1)
        
        # Cost breakdown pie chart
        capex = results.get('CAPEX', results.get('capex', 0))
        annual_opex = results.get('Annual_OPEX', results.get('annual_opex', 0))
        carbon_cost = results.get('Carbon_Cost', results.get('carbon_cost', 0))
        esg_cost = annual_opex * st.session_state.tea_params.get('f_esg', 0.07)
        
        if capex > 0:
            fig2, ax = plt.subplots(figsize=(8, 6))
            
            costs = [capex, annual_opex, carbon_cost, esg_cost]
            labels = ['CAPEX', 'Annual OPEX', 'Carbon Cost', 'ESG Compliance']
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
            
            # Filter out zero costs
            filtered_costs = []
            filtered_labels = []
            filtered_colors = []
            
            for cost, label, color in zip(costs, labels, colors):
                if cost > 0:
                    filtered_costs.append(cost)
                    filtered_labels.append(label)
                    filtered_colors.append(color)
            
            if filtered_costs:
                wedges, texts, autotexts = ax.pie(filtered_costs, labels=filtered_labels, colors=filtered_colors,
                                                autopct='%1.1f%%', startangle=90)
                ax.set_title("Cost Structure Breakdown")
                charts['cost_breakdown'] = mpl_fig_to_png_bytes(fig2)
            
            plt.close(fig2)
        
        # NPV sensitivity chart (placeholder)
        fig3, ax = plt.subplots(figsize=(8, 5))
        ax.text(0.5, 0.5, "Run sensitivity analysis\nin Advanced tab", 
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_axis_off()
        ax.set_title("Sensitivity Analysis Results")
        charts['sensitivity_placeholder'] = mpl_fig_to_png_bytes(fig3)
        plt.close(fig3)
        
    except Exception as e:
        print(f"Chart generation error: {e}")
    
    return charts

# ----------------- Visualizations Tab -----------------
with main_tabs[3]:
    st.header("📈 TEA Visualizations")
    
    if not st.session_state.tea_results:
        st.info("👈 Run TEA analysis first to generate visualizations")
    else:
        results = st.session_state.tea_results
        
        # Regenerate charts if not in session state
        if not st.session_state.analysis_charts:
            st.session_state.analysis_charts = generate_analysis_charts(results)
        
        charts = st.session_state.analysis_charts
        
        # Display charts in tabs
        viz_tabs = st.tabs(["Cash Flow", "Cost Breakdown", "Financial Metrics", "Interactive"])
        
        with viz_tabs[0]:
            st.subheader("💵 Cash Flow Analysis")
            
            if 'cash_flow' in charts:
                st.image(charts['cash_flow'], caption="Annual and Cumulative Cash Flow")
            else:
                # Generate on the fly
                cf = results.get('CF', results.get('cash_flow', []))
                if cf:
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
                    
                    # Annual
                    ax1.bar(range(len(cf)), cf, color=['red' if x < 0 else 'green' for x in cf])
                    ax1.axhline(y=0, color='black', linewidth=0.5)
                    ax1.set_title("Annual Cash Flow")
                    ax1.set_xlabel("Year")
                    ax1.set_ylabel("USD")
                    ax1.grid(True, alpha=0.3)
                    
                    # Cumulative
                    cumulative_cf = results.get('Cumulative_CF', results.get('cumulative_cf', []))
                    if not cumulative_cf:
                        cumulative_cf = np.cumsum(cf)
                    
                    ax2.plot(range(len(cumulative_cf)), cumulative_cf, 'b-', linewidth=2, marker='o')
                    ax2.axhline(y=0, color='red', linestyle='--', linewidth=1)
                    ax2.set_title("Cumulative Cash Flow")
                    ax2.set_xlabel("Year")
                    ax2.set_ylabel("USD")
                    ax2.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
        
        with viz_tabs[1]:
            st.subheader("💰 Cost Structure")
            
            if 'cost_breakdown' in charts:
                st.image(charts['cost_breakdown'], caption="Cost Structure Breakdown")
            else:
                # Generate cost breakdown
                capex = results.get('CAPEX', results.get('capex', 0))
                annual_opex = results.get('Annual_OPEX', results.get('annual_opex', 0))
                carbon_cost = results.get('Carbon_Cost', results.get('carbon_cost', 0))
                esg_cost = annual_opex * st.session_state.tea_params.get('f_esg', 0.07)
                
                costs = [capex, annual_opex, carbon_cost, esg_cost]
                labels = ['CAPEX', 'Annual OPEX', 'Carbon Cost', 'ESG Compliance']
                
                # Filter zero costs
                filtered_data = [(cost, label) for cost, label in zip(costs, labels) if cost > 0]
                
                if filtered_data:
                    filtered_costs, filtered_labels = zip(*filtered_data)
                    
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                    
                    # Pie chart
                    colors = plt.cm.Set3(np.linspace(0, 1, len(filtered_costs)))
                    ax1.pie(filtered_costs, labels=filtered_labels, colors=colors, autopct='%1.1f%%')
                    ax1.set_title("Cost Distribution")
                    
                    # Bar chart
                    x_pos = np.arange(len(filtered_costs))
                    ax2.bar(x_pos, filtered_costs, color=colors)
                    ax2.set_xticks(x_pos)
                    ax2.set_xticklabels(filtered_labels, rotation=45, ha='right')
                    ax2.set_ylabel("USD")
                    ax2.set_title("Cost Magnitude")
                    ax2.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
        
        with viz_tabs[2]:
            st.subheader("📊 Financial Metrics Dashboard")
            
            # Create metric cards using Plotly
            metrics_data = [
                ("NPV", results.get('NPV', results.get('npv', 0)), "$"),
                ("IRR", results.get('IRR', results.get('irr', 0))*100, "%"),
                ("LCOx", results.get('LCOx', results.get('lcox', 0)), "$/ton"),
            ]
            
            # Create gauge charts for key metrics
            col_g1, col_g2, col_g3 = st.columns(3)
            
            with col_g1:
                # NPV gauge
                npv = results.get('NPV', results.get('npv', 0))
                fig_npv = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = npv,
                    title = {'text': "NPV"},
                    delta = {'reference': 0},
                    gauge = {
                        'axis': {'range': [min(npv*1.5, -abs(npv)), max(npv*1.5, abs(npv))]},
                        'bar': {'color': "green" if npv > 0 else "red"},
                        'steps': [
                            {'range': [min(npv*1.5, -abs(npv)), 0], 'color': "lightgray"},
                            {'range': [0, max(npv*1.5, abs(npv))], 'color': "lightgreen"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 0
                        }
                    }
                ))
                fig_npv.update_layout(height=250)
                st.plotly_chart(fig_npv, use_container_width=True)
            
            with col_g2:
                # IRR gauge
                irr = results.get('IRR', results.get('irr', 0))*100
                discount_rate = (st.session_state.tea_params.get('i_base', 0.08) + 
                               st.session_state.tea_params.get('delta_risk', 0.03)) * 100
                fig_irr = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = irr,
                    title = {'text': "IRR"},
                    delta = {'reference': discount_rate},
                    gauge = {
                        'axis': {'range': [0, max(50, irr*2)]},
                        'bar': {'color': "green" if irr > discount_rate else "red"},
                        'steps': [
                            {'range': [0, discount_rate], 'color': "lightgray"},
                            {'range': [discount_rate, max(50, irr*2)], 'color': "lightgreen"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': discount_rate
                        }
                    }
                ))
                fig_irr.update_layout(height=250)
                st.plotly_chart(fig_irr, use_container_width=True)
            
            with col_g3:
                # Payback indicator
                payback = results.get('Payback_Period', results.get('payback_period', None))
                project_life = st.session_state.tea_params.get('N_project', 20)
                
                if payback:
                    payback_percent = (payback / project_life) * 100
                    color = "green" if payback_percent < 50 else "orange" if payback_percent < 80 else "red"
                    
                    fig_payback = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = payback,
                        title = {'text': f"Payback (of {project_life} yrs)"},
                        gauge = {
                            'axis': {'range': [0, project_life]},
                            'bar': {'color': color},
                            'steps': [
                                {'range': [0, project_life/2], 'color': "lightgreen"},
                                {'range': [project_life/2, project_life*0.8], 'color': "lightyellow"},
                                {'range': [project_life*0.8, project_life], 'color': "lightpink"}
                            ]
                        }
                    ))
                    fig_payback.update_layout(height=250)
                    st.plotly_chart(fig_payback, use_container_width=True)
        
        with viz_tabs[3]:
            st.subheader("📈 Interactive Analysis")
            
            # Interactive cash flow chart
            cf = results.get('CF', results.get('cash_flow', []))
            if cf:
                years = list(range(len(cf)))
                
                df_interactive = pd.DataFrame({
                    'Year': years,
                    'Cash Flow': cf,
                    'Cumulative': np.cumsum(cf).tolist()
                })
                
                fig = px.line(df_interactive, x='Year', y=['Cash Flow', 'Cumulative'],
                            title="Interactive Cash Flow Analysis",
                            labels={'value': 'USD', 'variable': 'Metric'})
                fig.update_layout(hovermode='x unified')
                st.plotly_chart(fig, use_container_width=True)
            
            # Cost comparison chart
            cost_categories = ['CAPEX', 'OPEX', 'Carbon Cost', 'ESG Cost']
            cost_values = [
                results.get('CAPEX', results.get('capex', 0)),
                results.get('Annual_OPEX', results.get('annual_opex', 0)),
                results.get('Carbon_Cost', results.get('carbon_cost', 0)),
                results.get('Annual_OPEX', results.get('annual_opex', 0)) * st.session_state.tea_params.get('f_esg', 0.07)
            ]
            
            df_costs = pd.DataFrame({
                'Category': cost_categories,
                'Value': cost_values
            })
            
            fig2 = px.bar(df_costs, x='Category', y='Value', color='Category',
                         title="Cost Category Comparison",
                         labels={'Value': 'USD'})
            fig2.update_traces(texttemplate='%{y:.2e}', textposition='outside')
            st.plotly_chart(fig2, use_container_width=True)

# ----------------- Sensitivity Tab -----------------
with main_tabs[4]:
    st.header("🔄 Sensitivity Analysis")
    
    if not st.session_state.tea_results:
        st.info("👈 Run TEA analysis first to enable sensitivity analysis")
    else:
        st.subheader("Parameter Sensitivity")
        
        # Select parameter for sensitivity analysis
        param_options = ["P_prod", "Q_prod", "C_PE", "C_RM", "f_esg", "tau_CO2", "i_base"]
        selected_param = st.selectbox("Select parameter to analyze:", param_options)
        
        col_sense1, col_sense2, col_sense3 = st.columns(3)
        
        with col_sense1:
            base_value = st.session_state.tea_params.get(selected_param, 0)
            min_val = st.number_input(f"Min {selected_param}", 
                                    value=float(base_value) * 0.5,
                                    format="%.4f")
        
        with col_sense2:
            max_val = st.number_input(f"Max {selected_param}", 
                                    value=float(base_value) * 1.5,
                                    format="%.4f")
        
        with col_sense3:
            steps = st.number_input("Number of steps", 
                                  value=10, min_value=3, max_value=50)
        
        if st.button("Run Sensitivity Analysis"):
            with st.spinner(f"Running {selected_param} sensitivity..."):
                try:
                    param_values = np.linspace(min_val, max_val, int(steps))
                    npv_results = []
                    irr_results = []
                    
                    for value in param_values:
                        temp_params = st.session_state.tea_params.copy()
                        temp_params[selected_param] = float(value)
                        
                        temp_results = comp_mod.compute_TEA(temp_params)
                        
                        if temp_results:
                            npv_results.append(temp_results.get('NPV', temp_results.get('npv', 0)))
                            irr_val = temp_results.get('IRR', temp_results.get('irr', 0))
                            irr_results.append(irr_val if irr_val is not None else 0)
                    
                    if npv_results and irr_results:
                        # Create sensitivity charts
                        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
                        
                        # NPV sensitivity
                        ax1.plot(param_values, npv_results, 'b-', linewidth=2, marker='o')
                        ax1.axhline(y=0, color='r', linestyle='--', alpha=0.7)
                        ax1.axvline(x=base_value, color='g', linestyle=':', alpha=0.7, label=f'Base: {base_value}')
                        ax1.set_xlabel(selected_param)
                        ax1.set_ylabel("NPV (USD)")
                        ax1.set_title(f"NPV Sensitivity to {selected_param}")
                        ax1.grid(True, alpha=0.3)
                        ax1.legend()
                        
                        # IRR sensitivity
                        ax2.plot(param_values, np.array(irr_results)*100, 'g-', linewidth=2, marker='s')
                        discount_rate = (st.session_state.tea_params.get('i_base', 0.08) + 
                                       st.session_state.tea_params.get('delta_risk', 0.03)) * 100
                        ax2.axhline(y=discount_rate, color='r', linestyle='--', alpha=0.7, label=f'Hurdle: {discount_rate:.1f}%')
                        ax2.axvline(x=base_value, color='g', linestyle=':', alpha=0.7)
                        ax2.set_xlabel(selected_param)
                        ax2.set_ylabel("IRR (%)")
                        ax2.set_title(f"IRR Sensitivity to {selected_param}")
                        ax2.grid(True, alpha=0.3)
                        ax2.legend()
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Store chart for PDF export
                        buf = io.BytesIO()
                        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                        buf.seek(0)
                        st.session_state.analysis_charts['sensitivity'] = buf.getvalue()
                        
                        plt.close(fig)
                        
                        st.success(f"Sensitivity analysis for {selected_param} complete!")
                    else:
                        st.error("Could not calculate sensitivity results")
                        
                except Exception as e:
                    st.error(f"Sensitivity analysis failed: {str(e)}")
        
        st.markdown("---")
        
        # Multi-parameter sensitivity
        st.subheader("Multi-Parameter Sensitivity")
        
        if hasattr(comp_mod, 'price_sweep'):
            if st.button("Run Price Sensitivity Analysis (from module)"):
                with st.spinner("Running price sensitivity analysis..."):
                    try:
                        plt.close('all')
                        
                        # Run price sweep from compliance module
                        comp_mod.price_sweep(st.session_state.tea_params)
                        
                        # Display all generated figures
                        for fig_num in plt.get_fignums():
                            fig = plt.figure(fig_num)
                            st.pyplot(fig)
                            
                            # Store for PDF export
                            buf = io.BytesIO()
                            fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                            buf.seek(0)
                            st.session_state.analysis_charts[f'price_sweep_{fig_num}'] = buf.getvalue()
                        
                        st.success("Price sensitivity analysis complete!")
                    except Exception as e:
                        st.error(f"Price sweep failed: {str(e)}")
        else:
            st.info("Price sweep function not available in compliance module")

# ----------------- Advanced Tab -----------------
with main_tabs[5]:
    st.header("🔬 Advanced Analyses")
    
    if not hasattr(comp_mod, 'run_esg_sweep_and_plots'):
        st.warning("Advanced functions not available in compliance module")
        st.info("Ensure your compliance.py has the required advanced functions")
    else:
        st.subheader("🌿 ESG Compliance Analysis")
        
        col_adv1, col_adv2, col_adv3 = st.columns(3)
        
        with col_adv1:
            if st.button("ESG Factor Sweep", use_container_width=True):
                with st.spinner("Running ESG factor sweep..."):
                    try:
                        plt.close('all')
                        
                        comp_mod.run_esg_sweep_and_plots(
                            st.session_state.tea_params, 
                            design_label="Streamlit Analysis"
                        )
                        
                        # Display and store charts
                        for fig_num in plt.get_fignums():
                            fig = plt.figure(fig_num)
                            st.pyplot(fig)
                            
                            buf = io.BytesIO()
                            fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                            buf.seek(0)
                            st.session_state.analysis_charts[f'esg_sweep_{fig_num}'] = buf.getvalue()
                        
                        st.success("ESG factor sweep complete!")
                    except Exception as e:
                        st.error(f"ESG sweep failed: {str(e)}")
        
        with col_adv2:
            if st.button("Scenario CBA", use_container_width=True):
                with st.spinner("Running scenario CBA..."):
                    try:
                        plt.close('all')
                        
                        comp_mod.scenario_cba(
                            st.session_state.tea_params,
                            design_label="Streamlit Analysis"
                        )
                        
                        for fig_num in plt.get_fignums():
                            fig = plt.figure(fig_num)
                            st.pyplot(fig)
                            
                            buf = io.BytesIO()
                            fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                            buf.seek(0)
                            st.session_state.analysis_charts[f'scenario_cba_{fig_num}'] = buf.getvalue()
                        
                        st.success("Scenario CBA complete!")
                    except Exception as e:
                        st.error(f"Scenario CBA failed: {str(e)}")
        
        with col_adv3:
            if st.button("Risk Analysis", use_container_width=True):
                with st.spinner("Running risk analysis..."):
                    try:
                        plt.close('all')
                        
                        comp_mod.risk_analysis(st.session_state.tea_params)
                        
                        for fig_num in plt.get_fignums():
                            fig = plt.figure(fig_num)
                            st.pyplot(fig)
                            
                            buf = io.BytesIO()
                            fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                            buf.seek(0)
                            st.session_state.analysis_charts[f'risk_analysis_{fig_num}'] = buf.getvalue()
                        
                        st.success("Risk analysis complete!")
                    except Exception as e:
                        st.error(f"Risk analysis failed: {str(e)}")
        
        st.markdown("---")
        
        # Monte Carlo simulation
        st.subheader("🎲 Monte Carlo Simulation")
        
        mc_col1, mc_col2 = st.columns(2)
        
        with mc_col1:
            mc_iterations = st.number_input("Iterations", 
                                          min_value=100, 
                                          max_value=10000, 
                                          value=1000,
                                          help="Number of Monte Carlo simulations")
        
        with mc_col2:
            mc_variables = st.multiselect(
                "Variables to vary",
                options=["P_prod", "Q_prod", "C_PE", "C_RM", "f_esg"],
                default=["P_prod", "Q_prod"]
            )
        
        if st.button("Run Monte Carlo Simulation"):
            with st.spinner(f"Running {mc_iterations} Monte Carlo simulations..."):
                try:
                    # Simple Monte Carlo implementation
                    npv_samples = []
                    
                    for _ in range(int(mc_iterations)):
                        temp_params = st.session_state.tea_params.copy()
                        
                        # Add random variations
                        for var in mc_variables:
                            if var in temp_params:
                                base_val = temp_params[var]
                                # Add ±10% variation
                                variation = np.random.normal(0, 0.05)  # 5% std dev
                                temp_params[var] = base_val * (1 + variation)
                        
                        results = comp_mod.compute_TEA(temp_params)
                        if results:
                            npv_samples.append(results.get('NPV', results.get('npv', 0)))
                    
                    if npv_samples:
                        # Create Monte Carlo results
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                        
                        # Histogram
                        ax1.hist(npv_samples, bins=50, edgecolor='black', alpha=0.7)
                        ax1.axvline(x=np.mean(npv_samples), color='r', linestyle='--', 
                                  label=f'Mean: ${np.mean(npv_samples):,.0f}')
                        ax1.axvline(x=np.percentile(npv_samples, 5), color='orange', 
                                  linestyle=':', label=f'5%: ${np.percentile(npv_samples, 5):,.0f}')
                        ax1.axvline(x=np.percentile(npv_samples, 95), color='orange', 
                                  linestyle=':', label=f'95%: ${np.percentile(npv_samples, 95):,.0f}')
                        ax1.set_xlabel("NPV (USD)")
                        ax1.set_ylabel("Frequency")
                        ax1.set_title("Monte Carlo NPV Distribution")
                        ax1.legend()
                        ax1.grid(True, alpha=0.3)
                        
                        # Cumulative probability
                        sorted_npv = np.sort(npv_samples)
                        cum_prob = np.arange(1, len(sorted_npv) + 1) / len(sorted_npv)
                        ax2.plot(sorted_npv, cum_prob, 'b-', linewidth=2)
                        ax2.axvline(x=0, color='r', linestyle='--', alpha=0.7)
                        ax2.fill_betweenx(cum_prob, sorted_npv, 0, 
                                        where=sorted_npv >= 0, color='green', alpha=0.3)
                        ax2.fill_betweenx(cum_prob, sorted_npv, 0, 
                                        where=sorted_npv < 0, color='red', alpha=0.3)
                        ax2.set_xlabel("NPV (USD)")
                        ax2.set_ylabel("Cumulative Probability")
                        ax2.set_title("Cumulative Probability Distribution")
                        ax2.grid(True, alpha=0.3)
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Store for PDF
                        buf = io.BytesIO()
                        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                        buf.seek(0)
                        st.session_state.analysis_charts['monte_carlo'] = buf.getvalue()
                        
                        plt.close(fig)
                        
                        # Display statistics
                        st.subheader("📊 Monte Carlo Statistics")
                        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                        
                        with col_stat1:
                            st.metric("Mean NPV", f"${np.mean(npv_samples):,.0f}")
                        with col_stat2:
                            st.metric("Std Dev", f"${np.std(npv_samples):,.0f}")
                        with col_stat3:
                            st.metric("Probability NPV > 0", 
                                    f"{(np.sum(np.array(npv_samples) > 0) / len(npv_samples)) * 100:.1f}%")
                        with col_stat4:
                            st.metric("Range (5-95%)", 
                                    f"${np.percentile(npv_samples, 5):,.0f} - ${np.percentile(npv_samples, 95):,.0f}")
                        
                        st.success(f"Monte Carlo simulation with {mc_iterations} iterations complete!")
                    
                except Exception as e:
                    st.error(f"Monte Carlo simulation failed: {str(e)}")

# ----------------- Export Tab -----------------
with main_tabs[6]:
    st.header("📤 Export Results")
    
    if not st.session_state.tea_results:
        st.info("👈 Run TEA analysis first to export results")
    else:
        results = st.session_state.tea_params
        
        export_options = st.multiselect(
            "Select content to include in report:",
            options=[
                "Executive Summary",
                "Financial Metrics", 
                "Cost Breakdown",
                "Cash Flow Analysis",
                "Sensitivity Analysis",
                "Advanced Analyses",
                "All Parameters"
            ],
            default=["Executive Summary", "Financial Metrics", "Cost Breakdown"]
        )
        
        report_title = st.text_input("Report Title:", 
                                    value=f"TEA Analysis Report - {datetime.now().strftime('%Y-%m-%d')}")
        
        col_exp1, col_exp2, col_exp3 = st.columns(3)
        
        with col_exp1:
            if st.button("📄 Generate PDF Report", type="primary", use_container_width=True):
                with st.spinner("Building comprehensive PDF report..."):
                    try:
                        sections = []
                        
                        # Executive Summary
                        if "Executive Summary" in export_options:
                            sections.append({
                                "heading": "Executive Summary",
                                "text": f"""
                                This report presents the Techno-Economic Analysis (TEA) results including ESG compliance costs.
                                Analysis conducted on {datetime.now().strftime('%Y-%m-%d %H:%M')}.
                                
                                **Key Findings:**
                                - Project NPV: ${st.session_state.tea_results.get('NPV', st.session_state.tea_results.get('npv', 0)):,.0f}
                                - Internal Rate of Return: {st.session_state.tea_results.get('IRR', st.session_state.tea_results.get('irr', 0))*100:.2f}%
                                - Payback Period: {st.session_state.tea_results.get('Payback_Period', st.session_state.tea_results.get('payback_period', 'N/A'))} years
                                - ESG Compliance Cost: ${st.session_state.tea_results.get('Annual_OPEX', st.session_state.tea_results.get('annual_opex', 0)) * st.session_state.tea_params.get('f_esg', 0.07):,.0f}/year
                                """,
                                "tables": [],
                                "images": []
                            })
                        
                        # Financial Metrics
                        if "Financial Metrics" in export_options:
                            summary_df = generate_tea_results_summary(st.session_state.tea_results, st.session_state.tea_params)
                            sections.append({
                                "heading": "Financial Metrics",
                                "text": "Detailed financial performance metrics from the TEA analysis.",
                                "tables": [(summary_df, "Financial Metrics Summary")],
                                "images": []
                            })
                        
                        # Cost Breakdown
                        if "Cost Breakdown" in export_options:
                            # Generate cost breakdown data
                            cost_data = [
                                ["Total CAPEX", f"${st.session_state.tea_results.get('CAPEX', st.session_state.tea_results.get('capex', 0)):,.0f}"],
                                ["Annual OPEX", f"${st.session_state.tea_results.get('Annual_OPEX', st.session_state.tea_results.get('annual_opex', 0)):,.0f}/year"],
                                ["Carbon Cost", f"${st.session_state.tea_results.get('Carbon_Cost', st.session_state.tea_results.get('carbon_cost', 0)):,.0f}/year"],
                                ["ESG Compliance", f"${st.session_state.tea_results.get('Annual_OPEX', st.session_state.tea_results.get('annual_opex', 0)) * st.session_state.tea_params.get('f_esg', 0.07):,.0f}/year"],
                                ["Annual Revenue", f"${st.session_state.tea_results.get('Annual_Revenue', st.session_state.tea_results.get('annual_revenue', 0) or (st.session_state.tea_params.get('Q_prod', 0) * st.session_state.tea_params.get('P_prod', 0))):,.0f}/year"]
                            ]
                            cost_df = pd.DataFrame(cost_data, columns=["Cost Category", "Value"])
                            
                            sections.append({
                                "heading": "Cost Structure Analysis",
                                "text": "Breakdown of capital and operating costs including ESG compliance expenses.",
                                "tables": [(cost_df, "Cost Breakdown")],
                                "images": []
                            })
                        
                        # Cash Flow Analysis
                        if "Cash Flow Analysis" in export_options:
                            # Create cash flow table
                            cf = st.session_state.tea_results.get('CF', st.session_state.tea_results.get('cash_flow', []))
                            years = list(range(len(cf)))
                            
                            cf_data = []
                            for year in years:
                                cf_data.append([
                                    year,
                                    f"${cf[year]:,.0f}" if year < len(cf) else "$0",
                                    f"${np.cumsum(cf)[year]:,.0f}" if year < len(cf) else "$0"
                                ])
                            
                            cf_df = pd.DataFrame(cf_data, columns=["Year", "Cash Flow", "Cumulative"])
                            
                            sections.append({
                                "heading": "Cash Flow Analysis",
                                "text": "Annual and cumulative cash flow over the project life.",
                                "tables": [(cf_df.head(10), "Cash Flow Schedule (First 10 Years)")],
                                "images": []
                            })
                        
                        # Add images from analysis charts
                        if st.session_state.analysis_charts:
                            image_section = {
                                "heading": "Analysis Visualizations",
                                "text": "Charts and graphs from the TEA analysis.",
                                "tables": [],
                                "images": []
                            }
                            
                            # Add cash flow chart if available
                            if 'cash_flow' in st.session_state.analysis_charts:
                                image_section["images"].append(
                                    (st.session_state.analysis_charts['cash_flow'], "Cash Flow Analysis")
                                )
                            
                            # Add cost breakdown chart if available
                            if 'cost_breakdown' in st.session_state.analysis_charts:
                                image_section["images"].append(
                                    (st.session_state.analysis_charts['cost_breakdown'], "Cost Structure Breakdown")
                                )
                            
                            # Add sensitivity chart if available
                            if 'sensitivity' in st.session_state.analysis_charts:
                                image_section["images"].append(
                                    (st.session_state.analysis_charts['sensitivity'], "Parameter Sensitivity Analysis")
                                )
                            
                            # Add Monte Carlo chart if available
                            if 'monte_carlo' in st.session_state.analysis_charts:
                                image_section["images"].append(
                                    (st.session_state.analysis_charts['monte_carlo'], "Monte Carlo Simulation Results")
                                )
                            
                            if image_section["images"]:
                                sections.append(image_section)
                        
                        # All Parameters
                        if "All Parameters" in export_options:
                            # Convert parameters to DataFrame
                            param_data = [[k, str(v)] for k, v in st.session_state.tea_params.items()]
                            param_df = pd.DataFrame(param_data, columns=["Parameter", "Value"])
                            
                            sections.append({
                                "heading": "Analysis Parameters",
                                "text": "Complete set of parameters used in the TEA analysis.",
                                "tables": [(param_df, "TEA Analysis Parameters")],
                                "images": []
                            })
                        
                        # Build PDF
                        pdf_bytes = build_pdf_report(report_title, sections)
                        
                        # Create download button
                        filename = f"TEA_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                        
                        st.download_button(
                            label="⬇️ Download PDF Report",
                            data=pdf_bytes,
                            file_name=filename,
                            mime="application/pdf",
                            use_container_width=True
                        )
                        
                        st.success("✅ PDF report generated successfully!")
                        
                    except Exception as e:
                        st.error(f"❌ PDF generation failed: {str(e)}")
        
        with col_exp2:
            # Export data
            export_format = st.selectbox(
                "Data Format:",
                ["CSV", "JSON", "Excel"]
            )
            
            if st.button(f"Export Data as {export_format}", use_container_width=True):
                try:
                    if export_format == "CSV":
                        # Create comprehensive CSV
                        summary_df = generate_tea_results_summary(st.session_state.tea_results, st.session_state.tea_params)
                        csv_data = summary_df.to_csv(index=False).encode()
                        
                        st.download_button(
                            "⬇️ Download CSV",
                            data=csv_data,
                            file_name="tea_results.csv",
                            mime="text/csv"
                        )
                    
                    elif export_format == "JSON":
                        # Prepare JSON data
                        export_data = {
                            "metadata": {
                                "generated": datetime.now().isoformat(),
                                "title": report_title,
                                "version": APP_VERSION
                            },
                            "parameters": st.session_state.tea_params,
                            "results": {}
                        }
                        
                        # Add results (convert numpy arrays to lists)
                        for key, value in st.session_state.tea_results.items():
                            if isinstance(value, np.ndarray):
                                export_data["results"][key] = value.tolist()
                            elif isinstance(value, (int, float, str, list, dict, bool)):
                                export_data["results"][key] = value
                        
                        json_str = json.dumps(export_data, indent=2, default=str)
                        
                        st.download_button(
                            "⬇️ Download JSON",
                            data=json_str,
                            file_name="tea_results.json",
                            mime="application/json"
                        )
                    
                    elif export_format == "Excel":
                        # Create Excel file with multiple sheets
                        import io
                        from pandas import ExcelWriter
                        
                        output = io.BytesIO()
                        with pd.ExcelWriter(output, engine='openpyxl') as writer:
                            # Summary sheet
                            summary_df = generate_tea_results_summary(st.session_state.tea_results, st.session_state.tea_params)
                            summary_df.to_excel(writer, sheet_name='Summary', index=False)
                            
                            # Parameters sheet
                            param_df = pd.DataFrame(
                                [[k, v] for k, v in st.session_state.tea_params.items()],
                                columns=['Parameter', 'Value']
                            )
                            param_df.to_excel(writer, sheet_name='Parameters', index=False)
                            
                            # Cash flow sheet
                            cf = st.session_state.tea_results.get('CF', st.session_state.tea_results.get('cash_flow', []))
                            years = list(range(len(cf)))
                            cf_df = pd.DataFrame({
                                'Year': years,
                                'Cash_Flow': cf,
                                'Cumulative': np.cumsum(cf).tolist()
                            })
                            cf_df.to_excel(writer, sheet_name='Cash_Flow', index=False)
                        
                        excel_data = output.getvalue()
                        
                        st.download_button(
                            "⬇️ Download Excel",
                            data=excel_data,
                            file_name="tea_results.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                
                except Exception as e:
                    st.error(f"Data export failed: {str(e)}")
        
        with col_exp3:
            # Quick snapshot
            if st.button("📸 Capture Snapshot", use_container_width=True):
                try:
                    # Create a simple snapshot report
                    snapshot_data = {
                        "timestamp": datetime.now().isoformat(),
                        "npv": st.session_state.tea_results.get('NPV', st.session_state.tea_results.get('npv', 0)),
                        "irr": st.session_state.tea_results.get('IRR', st.session_state.tea_results.get('irr', 0)),
                        "capex": st.session_state.tea_results.get('CAPEX', st.session_state.tea_results.get('capex', 0)),
                        "opex": st.session_state.tea_results.get('Annual_OPEX', st.session_state.tea_results.get('annual_opex', 0))
                    }
                    
                    snapshot_str = f"""TEA Analysis Snapshot
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

Key Metrics:
• NPV: ${snapshot_data['npv']:,.0f}
• IRR: {snapshot_data['irr']*100:.2f}%
• CAPEX: ${snapshot_data['capex']:,.0f}
• Annual OPEX: ${snapshot_data['opex']:,.0f}

Parameters Used:
• Production: {st.session_state.tea_params.get('Q_prod', 0):,.0f} tons/year
• Price: ${st.session_state.tea_params.get('P_prod', 0):,.2f}/ton
• ESG Factor: {st.session_state.tea_params.get('f_esg', 0)*100:.1f}%
• CO2 Tax: ${st.session_state.tea_params.get('tau_CO2', 0):,.2f}/ton
"""
                    
                    st.download_button(
                        "⬇️ Download Snapshot",
                        data=snapshot_str,
                        file_name="tea_snapshot.txt",
                        mime="text/plain"
                    )
                    
                    st.success("Snapshot created!")
                    
                except Exception as e:
                    st.error(f"Snapshot failed: {str(e)}")
        
        st.markdown("---")
        
        # Report preview
        st.subheader("📋 Report Preview")
        
        if st.checkbox("Show report preview"):
            preview_col1, preview_col2 = st.columns(2)
            
            with preview_col1:
                st.info("**Report Contents:**")
                for option in export_options:
                    st.write(f"✓ {option}")
                
                if st.session_state.analysis_charts:
                    st.success(f"✓ {len(st.session_state.analysis_charts)} charts included")
            
            with preview_col2:
                st.info("**Estimated Report Size:**")
                # Estimate PDF size
                estimated_pages = len(export_options) * 1.5
                if st.session_state.analysis_charts:
                    estimated_pages += len(st.session_state.analysis_charts) * 0.5
                
                st.write(f"• Pages: ~{estimated_pages:.1f}")
                st.write(f"• Charts: {len(st.session_state.analysis_charts)}")
                st.write(f"• Tables: {len(export_options)}")
                st.write(f"• Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

# ----------------- Footer -----------------
st.markdown("---")
st.markdown(
    f"""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p><strong>{APP_TITLE}</strong> • Version {APP_VERSION} • © 2024</p>
        <p style='font-size: 0.9em;'>Compliance & ESG Techno-Economic Analysis Tool</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Handle any cleanup
if st.session_state.get("_cleanup_needed", False):
    st.session_state._cleanup_needed = False
    plt.close('all')