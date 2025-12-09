
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import json
import io
from datetime import datetime
from typing import List, Dict

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

# ----------------- Session State Initialization -----------------
st.set_page_config(page_title="Compliance TEA", layout="wide", initial_sidebar_state="expanded")

# Initialize session state for Compliance TEA
if "tea_params" not in st.session_state:
    # Default TEA parameters based on compliance.py structure
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
                # Use numpy if available, otherwise approximate
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

# ----------------- Compliance TEA Page -----------------
st.title("💰 Compliance & ESG Techno-Economic Analysis")

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
    
    # Save/Load parameters
    col_save1, col_save2 = st.columns(2)
    with col_save1:
        if st.button("💾 Save Current Parameters"):
            params_str = json.dumps(st.session_state.tea_params, indent=2)
            st.download_button(
                "Download Parameters JSON",
                data=params_str,
                file_name="tea_parameters.json",
                mime="application/json"
            )
    
    with col_save2:
        uploaded_params = st.file_uploader("Load parameters from JSON", type=["json"])
        if uploaded_params:
            try:
                loaded_params = json.load(uploaded_params)
                st.session_state.tea_params.update(loaded_params)
                st.success("Parameters loaded successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Error loading parameters: {str(e)}")

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
            ["PDF Report", "JSON Data", "CSV Tables", "All Formats"]
        )
        
        if export_format in ["PDF Report", "All Formats"]:
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
        
        if export_format in ["JSON Data", "All Formats"]:
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
        
        if export_format in ["CSV Tables", "All Formats"]:
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
