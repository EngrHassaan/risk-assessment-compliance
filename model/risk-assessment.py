"""
TEA Compliance & ESG Analysis Frontend
Streamlit interface for the commercial/ex-ante TEA tool with all calculations,
analysis, and visualizations from compliance.py.
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="TEA Compliance & ESG Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.8rem;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #3498db;
    }
    .sub-header {
        font-size: 1.4rem;
        color: #34495e;
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
    }
    .highlight {
        background-color: #e8f4f8;
        padding: 10px;
        border-radius: 5px;
        border-left: 4px solid #3498db;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #dee2e6;
        text-align: center;
        margin: 5px;
    }
    .download-btn {
        background-color: #28a745;
        color: white;
        padding: 8px 16px;
        border-radius: 4px;
        border: none;
        cursor: pointer;
    }
</style>
""", unsafe_allow_html=True)

# App title
st.markdown('<h1 class="main-header">📊 Commercial/Ex-ante TEA Compliance & ESG Analysis Tool</h1>', unsafe_allow_html=True)

# Initialize session state
if 'params' not in st.session_state:
    st.session_state.params = {}
if 'results' not in st.session_state:
    st.session_state.results = {}
if 'designs' not in st.session_state:
    st.session_state.designs = []
if 'current_design' not in st.session_state:
    st.session_state.current_design = None
if 'mode' not in st.session_state:
    st.session_state.mode = "commercial"
if 'esg_sweep_data' not in st.session_state:
    st.session_state.esg_sweep_data = None

# Sidebar navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox(
    "Select Mode",
    ["TEA Mode Selection", "Parameter Input", "TEA Analysis", "Sensitivity Analysis", 
     "Scenario Analysis", "ESG/Compliance Sweep", "Multi-Design Comparison", "Export Results"]
)

# ======================================================================
#  Core TEA Functions (from compliance.py)
# ======================================================================

def compute_TEA(p):
    """Core TEA computation function"""
    C_PE = p['C_PE']

    # Direct capital
    C_INS   = p['f_ins']   * C_PE
    C_PIPE  = p['f_pipe']  * C_PE
    C_ELEC  = p['f_elec']  * C_PE
    C_BLDG  = p['f_bldg']  * C_PE
    C_UTIL  = p['f_util']  * C_PE
    C_STOR  = p['f_stor']  * C_PE
    C_SAFE  = p['f_safe']  * C_PE
    C_WASTE = p['f_waste'] * C_PE

    DCC = C_PE + C_INS + C_PIPE + C_ELEC + C_BLDG + C_UTIL + C_STOR + C_SAFE + C_WASTE

    # Indirect capital
    C_ENG   = p['f_eng']   * DCC
    C_CONS  = p['f_cons']  * DCC
    C_LICN  = p['f_licn']  * DCC
    C_CONT  = p['f_cont']  * DCC
    C_CONTG = p['f_contg'] * DCC
    C_INSUR = p['f_insur'] * DCC
    C_OWN   = p['f_own']   * DCC
    C_START = p['f_start'] * DCC

    ICC = C_ENG + C_CONS + C_LICN + C_CONT + C_CONTG + C_INSUR + C_OWN + C_START
    CGR = DCC + ICC  # total CAPEX

    # Depreciation life / salvage
    N = p['N_project']
    L = int(round(p['L_asset']))
    dep_method = p['dep_method']
    salv_frac = p['salv_frac']
    S = salv_frac * CGR
    L_eff = max(1, L)

    # average straight-line depreciation
    C_DEP_avg = CGR / L_eff
    CM_avg = C_DEP_avg + C_INSUR

    # DOC / FOC / GMC heuristic
    DOC = (p['COL'] + 0.18*p['COL'] + p['C_RM'] + p['C_UT'] +
           0.06*CGR + 0.15*p['COL'] + p['C_CAT'] + p['f_pack']*p['C_RM'])

    FOC = (0.708*p['COL'] + 0.036*CGR + 0.032*CGR + p['f_insur']*CGR +
           C_DEP_avg + 0.13*CM_avg + 0.01*CGR)

    GMC = 0.177*p['COL'] + 0.009*CGR + 0.05*CM_avg + 0.11*CM_avg

    # risk + carbon contributions
    risk_cost = p['f_risk_op'] * DOC
    co2_cost  = p['tau_CO2'] * p['E_CO2']

    # ESG / compliance cost
    OC_base = DOC + FOC + GMC + risk_cost + co2_cost
    f_esg = p.get('f_esg', 0.0)
    esg_cost = f_esg * OC_base
    OC = OC_base + esg_cost

    R = p['Q_prod'] * p['P_prod']
    i_eff = p['i_base'] + p['delta_risk']

    # depreciation schedule
    dep = np.zeros(N+1)
    basis = CGR - S
    life = min(L_eff, N)

    method = dep_method.upper()
    if method not in ("SL", "SYD", "DDB"):
        method = "SL"

    if method == "SL":
        if life > 0:
            d = basis / life
            for t in range(1, life+1):
                dep[t] = d

    elif method == "SYD":
        n = life
        syd = n * (n + 1) / 2.0
        for j in range(1, life+1):
            numerator = n - (j - 1)
            dep[j] = basis * numerator / syd

    elif method == "DDB":
        rate = 2.0 / life
        BV = CGR
        for j in range(1, life+1):
            d = rate * BV
            if BV - d < S:
                d = BV - S
            dep[j] = max(0.0, d)
            BV -= d
            if BV <= S + 1e-6:
                break

    # cashflow with depreciation + salvage
    CF = []
    EBT_schedule = np.zeros(N+1)
    tax_schedule = np.zeros(N+1)

    for t in range(N+1):
        if t == 0:
            CF.append(-CGR)
            EBT_schedule[t] = 0.0
            tax_schedule[t] = 0.0
        else:
            D_t = dep[t] if t <= N else 0.0
            EBT = R - OC - D_t
            tax = p['tau_inc'] * max(0, EBT)
            CF_t = (EBT - tax) + D_t
            if t == N:
                CF_t += S
            CF.append(CF_t)
            EBT_schedule[t] = EBT
            tax_schedule[t] = tax

    # NPV calculation
    NPV = sum(CF[t] / ((1 + i_eff)**t) for t in range(len(CF)))

    # IRR calculation
    def IRR(cash):
        cf = np.array(cash, dtype=float)
        if not (np.any(cf < 0) and np.any(cf > 0)):
            return 0.0

        def npv(rate):
            rate = max(rate, -0.999999)
            rate = min(rate, 10.0)
            t = np.arange(cf.size, dtype=float)
            return np.sum(cf / (1.0 + rate)**t)

        r_low, r_high = -0.9, 1.0
        f_low, f_high = npv(r_low), npv(r_high)

        tries = 0
        while f_low * f_high > 0 and tries < 20 and r_high < 10.0:
            r_high += 0.5
            f_high = npv(r_high)
            tries += 1

        if f_low * f_high > 0:
            return 0.0

        for _ in range(80):
            r_mid = 0.5 * (r_low + r_high)
            f_mid = npv(r_mid)
            if abs(f_mid) < 1e-6:
                return r_mid
            if f_low * f_mid < 0:
                r_high, f_high = r_mid, f_mid
            else:
                r_low, f_low = r_mid, f_mid

        return r_mid

    irr_val = IRR(CF)

    # CRF and annualised CAPEX
    if i_eff > 0:
        CRF = i_eff * (1 + i_eff)**N / ((1 + i_eff)**N - 1)
    else:
        CRF = 1.0 / N if N > 0 else 0.0

    PV_capital_net = CGR - S / ((1 + i_eff)**N)
    Annual_CAPEX = PV_capital_net * CRF

    # Cost–Benefit Analysis
    PV_rev = 0.0
    PV_opex = 0.0
    if N > 0:
        for t in range(1, N+1):
            disc = (1 + i_eff)**t
            PV_rev  += R  / disc
            PV_opex += OC / disc

    PV_cost_total = PV_capital_net + PV_opex
    if PV_cost_total > 0:
        BCR = PV_rev / PV_cost_total
    else:
        BCR = 0.0

    # Payback calculation
    def compute_payback(cf):
        cum = np.cumsum(cf)
        for t in range(1, len(cum)):
            if cum[t] >= 0:
                dy = cum[t] - cum[t-1]
                if dy != 0:
                    frac = -cum[t-1] / dy
                    return (t - 1) + frac
                else:
                    return float(t)
        return None

    payback = compute_payback(CF)

    # ROI calculations
    years = np.arange(1, N+1)
    revenue = np.full_like(years, R, dtype=float)
    opex_array = np.full_like(years, OC, dtype=float)
    depreciation_array = dep[1:N+1]
    ebit = EBT_schedule[1:N+1]
    tax_array = tax_schedule[1:N+1]
    net_income = ebit - tax_array
    ebitda = revenue - opex_array

    CAPEX_tot = CGR
    total_net_income = float(np.sum(net_income))
    avg_net_income = float(np.mean(net_income))
    ROI_total = total_net_income / CAPEX_tot if CAPEX_tot != 0 else 0.0
    ROI_avg = avg_net_income / CAPEX_tot if CAPEX_tot != 0 else 0.0

    return {
        "CAPEX": CGR,
        "LCOx": OC / p['Q_prod'],
        "NPV": NPV,
        "IRR": irr_val,
        "CF": CF,
        "Salvage": S,
        "CRF": CRF,
        "Annual_CAPEX": Annual_CAPEX,
        "PV_revenue": PV_rev,
        "PV_cost_total": PV_cost_total,
        "BCR": BCR,
        "Payback": payback,
        "C_PE": C_PE,
        "DCC": DCC,
        "ICC": ICC,
        "DOC": DOC,
        "FOC": FOC,
        "GMC": GMC,
        "OC": OC,
        "OC_base": OC_base,
        "R": R,
        "risk_cost": risk_cost,
        "co2_cost": co2_cost,
        "esg_cost": esg_cost,
        "dep_schedule": dep,
        "EBT_schedule": EBT_schedule,
        "tax_schedule": tax_schedule,
        "ROI_total": ROI_total,
        "ROI_avg": ROI_avg,
        "net_income": net_income,
        "ebitda": ebitda,
        "ebit": ebit,
        "revenue": revenue,
        "opex_array": opex_array,
        "years": years
    }

def theoretical_irr_from_npv(cash, r_low=-0.9, r_high=1.0, max_iter=80, tol=1e-6):
    """Solve NPV(r) = 0 directly from cashflow using bisection"""
    cf = np.array(cash, dtype=float)
    if not (np.any(cf < 0) and np.any(cf > 0)):
        return 0.0

    def npv(rate):
        rate = max(rate, -0.999999)
        rate = min(rate, 10.0)
        t = np.arange(cf.size, dtype=float)
        return np.sum(cf / (1.0 + rate)**t)

    f_low = npv(r_low)
    f_high = npv(r_high)

    tries = 0
    while f_low * f_high > 0 and tries < 20 and r_high < 10.0:
        r_high += 0.5
        f_high = npv(r_high)
        tries += 1

    if f_low * f_high > 0:
        return 0.0

    for _ in range(max_iter):
        r_mid = 0.5 * (r_low + r_high)
        f_mid = npv(r_mid)
        if abs(f_mid) < tol:
            return r_mid
        if f_low * f_mid < 0:
            r_high, f_high = r_mid, f_mid
        else:
            r_low, f_low = r_mid, f_mid

    return r_mid

def price_for_target_irr(params, target_irr, tol=1e-4, max_iter=80):
    """Find selling price required to achieve target IRR"""
    base_params = params.copy()
    P0 = base_params.get("P_prod", 0.0)
    if P0 <= 0:
        return None, None

    def irr_at_price(price):
        if price <= 0:
            return -1.0
        p = base_params.copy()
        p["P_prod"] = price
        res = compute_TEA(p)
        return res["IRR"]

    # Initial bracket
    p_low = max(1e-6, 0.1 * P0)
    p_high = 5.0 * P0
    irr_low = irr_at_price(p_low)
    irr_high = irr_at_price(p_high)

    # Find bracket
    found = False
    for _ in range(40):
        if irr_low > irr_high:
            p_low, p_high = p_high, p_low
            irr_low, irr_high = irr_high, irr_low
        if irr_low <= target_irr <= irr_high:
            found = True
            break
        if irr_high < target_irr:
            p_high *= 1.5
            irr_high = irr_at_price(p_high)
        elif irr_low > target_irr:
            p_low *= 0.5
            irr_low = irr_at_price(p_low)

    if not found:
        return None, None

    # Bisection
    for _ in range(max_iter):
        p_mid = 0.5 * (p_low + p_high)
        irr_mid = irr_at_price(p_mid)
        if abs(irr_mid - target_irr) < tol:
            p_req = p_mid
            res_req = compute_TEA({**base_params, "P_prod": p_req})
            return p_req, res_req
        if irr_mid < target_irr:
            p_low, irr_low = p_mid, irr_mid
        else:
            p_high, irr_high = p_mid, irr_mid

    p_req = 0.5 * (p_low + p_high)
    res_req = compute_TEA({**base_params, "P_prod": p_req})
    return p_req, res_req

# ======================================================================
#  TEA Mode Selection
# ======================================================================
if app_mode == "TEA Mode Selection":
    st.markdown('<h2 class="section-header">TEA Mode Selection</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Commercial TEA")
        st.markdown("""
        <div class="highlight">
        Use for established processes with known cost data.
        <br><br>
        <strong>Features:</strong>
        <ul>
            <li>Direct input of annual costs</li>
            <li>Known equipment costs</li>
            <li>Mature technology risk assessment</li>
            <li>Detailed OPEX breakdown</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Select Commercial TEA", key="commercial", use_container_width=True):
            st.session_state.mode = "commercial"
            st.success("Commercial TEA mode selected!")
    
    with col2:
        st.markdown("### Ex-ante TEA")
        st.markdown("""
        <div class="highlight">
        Use for emerging technologies and scale-up projects.
        <br><br>
        <strong>Features:</strong>
        <ul>
            <li>Scaled from reference/pilot plants</li>
            <li>TRL-based risk assessment</li>
            <li>Cost intensity approach</li>
            <li>FOAK (First-of-a-kind) considerations</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Select Ex-ante TEA", key="exante", use_container_width=True):
            st.session_state.mode = "ex-ante"
            st.success("Ex-ante TEA mode selected!")
    
    st.markdown("---")
    st.markdown(f"**Current Mode:** {st.session_state.mode.upper()}")
    
    # Show parameter help
    with st.expander("Parameter Help & Definitions"):
        PARAM_HELP = {
            "C_PE": "Purchased Equipment Cost (USD). Base cost of major process equipment.",
            "COL": "Annual Operating Labour Cost (USD/year). Total yearly cost of operators.",
            "C_RM": "Annual Raw Material Cost (USD/year). Sum of all feedstock costs.",
            "C_UT": "Annual Utilities Cost (USD/year). Electricity, steam, cooling water, etc.",
            "C_CAT": "Annual Catalyst/Membrane/Sorbent Make-up Cost (USD/year).",
            "Q_prod": "Annual Product Output (ton/year). Net saleable product.",
            "P_prod": "Selling Price (USD/ton of product). Average selling price.",
            "f_esg": "ESG/compliance cost as fraction of base operating cost.",
            "L_asset": "Asset / Depreciation Life (years). Accounting life for depreciation.",
            "N_project": "Project Horizon / Plant Operating Life (years).",
            "tau_inc": "Income Tax Fraction (0–1). Corporate income tax rate.",
            "i_base": "Base Discount Rate (decimal). Firm-level WACC or hurdle rate.",
            "delta_risk": "Risk Premium Addition to Discount Rate (decimal)."
        }
        
        for param, desc in PARAM_HELP.items():
            st.markdown(f"**{param}**: {desc}")

# ======================================================================
#  Parameter Input
# ======================================================================
elif app_mode == "Parameter Input":
    st.markdown('<h2 class="section-header">Parameter Input</h2>', unsafe_allow_html=True)
    
    if st.session_state.mode == "commercial":
        st.info("📊 Commercial TEA Mode: Enter direct plant cost data")
        
        with st.form("commercial_params"):
            st.markdown("### Required Base Inputs")
            col1, col2 = st.columns(2)
            
            with col1:
                C_PE = st.number_input("Purchased Equipment Cost (USD)", 
                                     min_value=1e6, max_value=1e12, 
                                     value=1e8, format="%.0f")
                COL = st.number_input("Annual Operating Labour Cost (USD/year)", 
                                    min_value=1e5, max_value=1e10, 
                                    value=1e7, format="%.0f")
                C_RM = st.number_input("Raw Material Cost (USD/year)", 
                                     min_value=1e5, max_value=1e11, 
                                     value=4e7, format="%.0f")
                C_UT = st.number_input("Utilities Cost (USD/year)", 
                                     min_value=1e4, max_value=1e10, 
                                     value=1.2e7, format="%.0f")
            
            with col2:
                C_CAT = st.number_input("Catalyst Make-up Cost (USD/year)", 
                                      min_value=0.0, max_value=1e9, 
                                      value=2e6, format="%.0f")
                Q_prod = st.number_input("Annual Product Output (ton/year)", 
                                       min_value=1e3, max_value=1e8, 
                                       value=5e5, format="%.0f")
                P_prod = st.number_input("Selling Price (USD/ton)", 
                                       min_value=50.0, max_value=5000.0, 
                                       value=550.0, format="%.2f")
            
            st.markdown("### Capital Fractions")
            col3, col4 = st.columns(2)
            
            with col3:
                f_ins = st.slider("Installation Factor", 0.0, 3.0, 0.30, 0.05)
                f_pipe = st.slider("Piping Factor", 0.0, 3.0, 0.45, 0.05)
                f_elec = st.slider("Electrical Factor", 0.0, 1.0, 0.10, 0.02)
                f_bldg = st.slider("Buildings Factor", 0.0, 1.0, 0.15, 0.02)
                f_util = st.slider("Utilities Factor", 0.0, 3.0, 0.60, 0.05)
            
            with col4:
                f_stor = st.slider("Storage Factor", 0.0, 1.0, 0.10, 0.02)
                f_safe = st.slider("Safety Factor", 0.0, 1.0, 0.05, 0.01)
                f_waste = st.slider("Waste Treatment Factor", 0.0, 2.0, 0.10, 0.02)
            
            st.markdown("### Indirect Capital Fractions")
            col5, col6 = st.columns(2)
            
            with col5:
                f_eng = st.slider("Engineering Fraction", 0.0, 0.5, 0.20, 0.02)
                f_cons = st.slider("Construction Supervision", 0.0, 0.3, 0.10, 0.02)
                f_licn = st.slider("Licensing Fraction", 0.0, 0.1, 0.02, 0.01)
                f_cont = st.slider("Contractor Overhead+Profit", 0.0, 0.3, 0.10, 0.02)
            
            with col6:
                f_contg = st.slider("Contingency", 0.0, 0.5, 0.25, 0.02)
                f_insur = st.slider("Insurance", 0.0, 0.1, 0.02, 0.005)
                f_own = st.slider("Owner's Cost", 0.0, 0.2, 0.05, 0.01)
                f_start = st.slider("Start-up & Commissioning", 0.0, 0.3, 0.08, 0.01)
            
            st.markdown("### Finance, Tax & Asset Life")
            col7, col8 = st.columns(2)
            
            with col7:
                L_asset = st.number_input("Depreciation Life (years)", 
                                        min_value=1, max_value=50, value=20)
                tau_inc = st.slider("Income Tax Rate", 0.0, 1.0, 0.30, 0.01)
                i_base = st.slider("Base Discount Rate", 0.0, 0.5, 0.08, 0.01)
                delta_risk = st.slider("Risk Premium", 0.0, 0.2, 0.02, 0.005)
            
            with col8:
                N_project = st.number_input("Project Horizon (years)", 
                                          min_value=1, max_value=50, value=20)
                dep_method = st.selectbox("Depreciation Method", 
                                        ["SL", "SYD", "DDB"])
                salv_frac = st.slider("Salvage Fraction", 0.0, 1.0, 0.10, 0.01)
            
            st.markdown("### Risk & ESG Parameters")
            col9, col10 = st.columns(2)
            
            with col9:
                f_risk_op = st.slider("Operational Risk Factor", 0.0, 0.5, 0.05, 0.01)
                tau_CO2 = st.number_input("Carbon Tax (USD/ton CO₂)", 
                                        min_value=0.0, max_value=500.0, value=50.0)
                E_CO2 = st.number_input("Annual CO₂ Emissions (ton/year)", 
                                      min_value=0.0, max_value=1e7, value=200000.0)
                f_pack = st.slider("Packaging % of RM", 0.0, 0.2, 0.02, 0.005)
            
            with col10:
                st.markdown("#### ESG/Compliance Region")
                region = st.selectbox("Region", 
                                    ["North America (5-10% of OPEX)", 
                                     "Europe/UK (5-10% of OPEX)",
                                     "Asia-Pacific (>10% of OPEX)",
                                     "Other/Custom"])
                
                if "North America" in region:
                    default_esg = 0.07
                elif "Europe" in region:
                    default_esg = 0.07
                elif "Asia" in region:
                    default_esg = 0.12
                else:
                    default_esg = 0.05
                
                f_esg = st.slider("ESG/Compliance Cost Fraction", 
                                0.0, 0.5, default_esg, 0.01)
            
            # Design name
            design_name = st.text_input("Design Name", value="Design_1")
            
            submitted = st.form_submit_button("Save Parameters")
            
            if submitted:
                params = {
                    'C_PE': C_PE, 'COL': COL, 'C_RM': C_RM, 'C_UT': C_UT,
                    'C_CAT': C_CAT, 'Q_prod': Q_prod, 'P_prod': P_prod,
                    'f_ins': f_ins, 'f_pipe': f_pipe, 'f_elec': f_elec,
                    'f_bldg': f_bldg, 'f_util': f_util, 'f_stor': f_stor,
                    'f_safe': f_safe, 'f_waste': f_waste, 'f_eng': f_eng,
                    'f_cons': f_cons, 'f_licn': f_licn, 'f_cont': f_cont,
                    'f_contg': f_contg, 'f_insur': f_insur, 'f_own': f_own,
                    'f_start': f_start, 'L_asset': L_asset, 'tau_inc': tau_inc,
                    'i_base': i_base, 'delta_risk': delta_risk, 'N_project': N_project,
                    'dep_method': dep_method, 'salv_frac': salv_frac,
                    'f_risk_op': f_risk_op, 'tau_CO2': tau_CO2, 'E_CO2': E_CO2,
                    'f_pack': f_pack, 'f_esg': f_esg, 'mode': 'commercial'
                }
                
                st.session_state.params = params
                st.session_state.current_design = design_name
                st.success(f"Parameters saved for {design_name}!")
                
                # Add to designs list if not already there
                if design_name not in [d['name'] for d in st.session_state.designs]:
                    st.session_state.designs.append({
                        'name': design_name,
                        'mode': 'commercial',
                        'params': params
                    })
    
    else:  # Ex-ante mode
        st.info("🚀 Ex-ante TEA Mode: Scale from reference plant")
        
        with st.form("exante_params"):
            st.markdown("### Ex-ante CAPEX Scaling")
            col1, col2 = st.columns(2)
            
            with col1:
                Q_ref = st.number_input("Reference Plant Capacity (ton/year)", 
                                      min_value=1.0, max_value=1e9, 
                                      value=1e5, format="%.0f")
                C_PE_ref = st.number_input("Reference Equipment Cost (USD)", 
                                         min_value=1e6, max_value=1e11, 
                                         value=5e7, format="%.0f")
                n_capex = st.slider("CAPEX Scaling Exponent", 0.3, 1.0, 0.6, 0.05)
                foak_mult = st.slider("FOAK Novelty Factor", 1.0, 3.0, 1.2, 0.1)
            
            with col2:
                Q_design = st.number_input("Target Commercial Capacity (ton/year)", 
                                         min_value=1e3, max_value=1e8, 
                                         value=5e5, format="%.0f")
                
                # Calculate scaled C_PE
                C_PE_scaled = C_PE_ref * (Q_design / Q_ref)**n_capex * foak_mult
                st.metric("Scaled Equipment Cost (USD)", 
                         f"{C_PE_scaled:,.0f}")
            
            st.markdown("### Ex-ante OPEX Intensities (USD/ton)")
            col3, col4 = st.columns(2)
            
            with col3:
                c_lab = st.number_input("Labour Cost Intensity", 
                                      min_value=0.0, max_value=1000.0, 
                                      value=20.0, format="%.2f")
                c_rm = st.number_input("Raw Material Cost Intensity", 
                                     min_value=0.0, max_value=5000.0, 
                                     value=300.0, format="%.2f")
            
            with col4:
                c_ut = st.number_input("Utilities Cost Intensity", 
                                     min_value=0.0, max_value=1000.0, 
                                     value=50.0, format="%.2f")
                c_cat = st.number_input("Catalyst Cost Intensity", 
                                      min_value=0.0, max_value=500.0, 
                                      value=10.0, format="%.2f")
            
            P_prod = st.number_input("Expected Selling Price (USD/ton)", 
                                   min_value=50.0, max_value=5000.0, 
                                   value=550.0, format="%.2f")
            
            # Calculate derived values
            COL = c_lab * Q_design
            C_RM = c_rm * Q_design
            C_UT = c_ut * Q_design
            C_CAT = c_cat * Q_design
            
            st.markdown("### Technology Readiness Level (TRL)")
            TRL = st.slider("TRL (3-9)", 3, 9, 6, 1)
            
            # TRL-based risk premium
            if TRL <= 4:
                delta_risk = 0.08
            elif TRL <= 6:
                delta_risk = 0.05
            elif TRL <= 8:
                delta_risk = 0.03
            else:
                delta_risk = 0.01
            
            st.metric("TRL-based Risk Premium", f"{delta_risk:.3f}")
            
            # TRL-based operational risk
            if TRL <= 4:
                default_f_risk_op = 0.15
            elif TRL <= 6:
                default_f_risk_op = 0.10
            else:
                default_f_risk_op = 0.05
            
            st.markdown("### Capital Fractions (same as commercial)")
            # Include other parameters with defaults
            with st.expander("Capital & Financial Parameters"):
                col5, col6 = st.columns(2)
                
                with col5:
                    f_ins = st.slider("Installation Factor", 0.0, 3.0, 0.30, 0.05)
                    f_pipe = st.slider("Piping Factor", 0.0, 3.0, 0.45, 0.05)
                    f_elec = st.slider("Electrical Factor", 0.0, 1.0, 0.10, 0.02)
                    f_bldg = st.slider("Buildings Factor", 0.0, 1.0, 0.15, 0.02)
                    f_util = st.slider("Utilities Factor", 0.0, 3.0, 0.60, 0.05)
                    f_stor = st.slider("Storage Factor", 0.0, 1.0, 0.10, 0.02)
                    f_safe = st.slider("Safety Factor", 0.0, 1.0, 0.05, 0.01)
                    f_waste = st.slider("Waste Treatment Factor", 0.0, 2.0, 0.10, 0.02)
                
                with col6:
                    f_eng = st.slider("Engineering Fraction", 0.0, 0.5, 0.20, 0.02)
                    f_cons = st.slider("Construction Supervision", 0.0, 0.3, 0.10, 0.02)
                    f_licn = st.slider("Licensing Fraction", 0.0, 0.1, 0.02, 0.01)
                    f_cont = st.slider("Contractor Overhead+Profit", 0.0, 0.3, 0.10, 0.02)
                    f_contg = st.slider("Contingency", 0.0, 0.5, 0.25, 0.02)
                    f_insur = st.slider("Insurance", 0.0, 0.1, 0.02, 0.005)
                    f_own = st.slider("Owner's Cost", 0.0, 0.2, 0.05, 0.01)
                    f_start = st.slider("Start-up & Commissioning", 0.0, 0.3, 0.08, 0.01)
            
            st.markdown("### Emissions & ESG")
            col7, col8 = st.columns(2)
            
            with col7:
                e_CO2_int = st.number_input("CO₂ Emissions Intensity (ton/ton)", 
                                          min_value=0.0, max_value=10.0, 
                                          value=1.8, format="%.2f")
                E_CO2 = e_CO2_int * Q_design
                st.metric("Annual CO₂ Emissions (ton)", f"{E_CO2:,.0f}")
                
                tau_CO2 = st.number_input("Carbon Tax (USD/ton CO₂)", 
                                        min_value=0.0, max_value=500.0, 
                                        value=50.0)
                f_pack = st.slider("Packaging % of RM", 0.0, 0.2, 0.02, 0.005)
            
            with col8:
                f_risk_op = st.slider("Operational Risk Factor", 0.0, 0.5, 
                                    default_f_risk_op, 0.01)
                
                region = st.selectbox("ESG Region", 
                                    ["North America (5-10% of OPEX)", 
                                     "Europe/UK (5-10% of OPEX)",
                                     "Asia-Pacific (>10% of OPEX)",
                                     "Other/Custom"])
                
                if "North America" in region:
                    default_esg = 0.07
                elif "Europe" in region:
                    default_esg = 0.07
                elif "Asia" in region:
                    default_esg = 0.12
                else:
                    default_esg = 0.05
                
                f_esg = st.slider("ESG/Compliance Cost Fraction", 
                                0.0, 0.5, default_esg, 0.01)
            
            # Other financial parameters
            st.markdown("### Other Financial Parameters")
            col9, col10 = st.columns(2)
            
            with col9:
                L_asset = st.number_input("Depreciation Life (years)", 
                                        min_value=1, max_value=50, value=20)
                tau_inc = st.slider("Income Tax Rate", 0.0, 1.0, 0.30, 0.01)
                i_base = st.slider("Base Discount Rate", 0.0, 0.5, 0.08, 0.01)
            
            with col10:
                N_project = st.number_input("Project Horizon (years)", 
                                          min_value=1, max_value=50, value=20)
                dep_method = st.selectbox("Depreciation Method", 
                                        ["SL", "SYD", "DDB"])
                salv_frac = st.slider("Salvage Fraction", 0.0, 1.0, 0.10, 0.01)
            
            design_name = st.text_input("Design Name", value="ExAnte_1")
            
            submitted = st.form_submit_button("Save Parameters")
            
            if submitted:
                params = {
                    'C_PE': C_PE_scaled, 'COL': COL, 'C_RM': C_RM, 'C_UT': C_UT,
                    'C_CAT': C_CAT, 'Q_prod': Q_design, 'P_prod': P_prod,
                    'f_ins': f_ins, 'f_pipe': f_pipe, 'f_elec': f_elec,
                    'f_bldg': f_bldg, 'f_util': f_util, 'f_stor': f_stor,
                    'f_safe': f_safe, 'f_waste': f_waste, 'f_eng': f_eng,
                    'f_cons': f_cons, 'f_licn': f_licn, 'f_cont': f_cont,
                    'f_contg': f_contg, 'f_insur': f_insur, 'f_own': f_own,
                    'f_start': f_start, 'L_asset': L_asset, 'tau_inc': tau_inc,
                    'i_base': i_base, 'delta_risk': delta_risk, 'N_project': N_project,
                    'dep_method': dep_method, 'salv_frac': salv_frac,
                    'f_risk_op': f_risk_op, 'tau_CO2': tau_CO2, 'E_CO2': E_CO2,
                    'f_pack': f_pack, 'f_esg': f_esg, 'mode': 'ex-ante',
                    'TRL': TRL, 'Q_ref': Q_ref, 'C_PE_ref': C_PE_ref,
                    'n_capex': n_capex, 'foak_mult': foak_mult
                }
                
                st.session_state.params = params
                st.session_state.current_design = design_name
                st.success(f"Parameters saved for {design_name}!")
                
                if design_name not in [d['name'] for d in st.session_state.designs]:
                    st.session_state.designs.append({
                        'name': design_name,
                        'mode': 'ex-ante',
                        'params': params
                    })

# ======================================================================
#  TEA Analysis
# ======================================================================
elif app_mode == "TEA Analysis":
    st.markdown('<h2 class="section-header">TEA Analysis</h2>', unsafe_allow_html=True)
    
    if not st.session_state.params:
        st.warning("Please input parameters first in the Parameter Input section.")
        st.stop()
    
    params = st.session_state.params
    design_name = st.session_state.current_design or "Current Design"
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"### Analysis for: **{design_name}**")
        st.markdown(f"**Mode:** {params.get('mode', 'commercial').upper()}")
        
        if params.get('mode') == 'ex-ante':
            st.markdown(f"**TRL:** {params.get('TRL', 'N/A')}")
            st.markdown(f"**Reference Capacity:** {params.get('Q_ref', 0):,.0f} ton/yr")
            st.markdown(f"**Target Capacity:** {params.get('Q_prod', 0):,.0f} ton/yr")
    
    with col2:
        if st.button("Run TEA Analysis", type="primary", use_container_width=True):
            with st.spinner("Computing TEA..."):
                results = compute_TEA(params)
                st.session_state.results = results
                st.success("TEA analysis completed!")
    
    if st.session_state.results:
        results = st.session_state.results
        
        # Key metrics display
        st.markdown("### Key Metrics")
        metric_cols = st.columns(5)
        
        with metric_cols[0]:
            st.metric("CAPEX", f"${results['CAPEX']:,.0f}")
        with metric_cols[1]:
            st.metric("LCOx", f"${results['LCOx']:.2f}/ton")
        with metric_cols[2]:
            st.metric("NPV", f"${results['NPV']:,.0f}", 
                     delta_color="inverse" if results['NPV'] < 0 else "normal")
        with metric_cols[3]:
            st.metric("IRR", f"{results['IRR']*100:.2f}%")
        with metric_cols[4]:
            if results['Payback']:
                st.metric("Payback", f"{results['Payback']:.1f} years")
            else:
                st.metric("Payback", "N/A")
        
        # More metrics
        metric_cols2 = st.columns(4)
        with metric_cols2[0]:
            st.metric("BCR", f"{results['BCR']:.3f}")
        with metric_cols2[1]:
            st.metric("ROI Total", f"{results['ROI_total']*100:.1f}%")
        with metric_cols2[2]:
            st.metric("ROI Avg", f"{results['ROI_avg']*100:.1f}%")
        with metric_cols2[3]:
            st.metric("Salvage", f"${results['Salvage']:,.0f}")
        
        # Detailed results tabs
        tab1, tab2, tab3, tab4 = st.tabs(["Financial Summary", "Cash Flow Analysis", 
                                         "Income Statement", "Breakdown Analysis"])
        
        with tab1:
            st.markdown("#### Financial Summary")
            summary_data = {
                "Metric": ["CAPEX", "Annual Revenue", "Annual OPEX", "LCOx", 
                          "NPV", "IRR", "Payback", "BCR", "ROI Total", "ROI Avg"],
                "Value": [f"${results['CAPEX']:,.0f}", 
                         f"${results['R']:,.0f}/yr",
                         f"${results['OC']:,.0f}/yr",
                         f"${results['LCOx']:.2f}/ton",
                         f"${results['NPV']:,.0f}",
                         f"{results['IRR']*100:.2f}%",
                         f"{results['Payback']:.1f} years" if results['Payback'] else "N/A",
                         f"{results['BCR']:.3f}",
                         f"{results['ROI_total']*100:.1f}%",
                         f"{results['ROI_avg']*100:.1f}%"],
                "Description": ["Total capital expenditure", 
                              "Annual product revenue",
                              "Annual operating cost",
                              "Levelized cost of production",
                              "Net present value",
                              "Internal rate of return",
                              "Simple payback period",
                              "Benefit-cost ratio",
                              "Total return on investment",
                              "Average annual ROI"]
            }
            st.dataframe(pd.DataFrame(summary_data), use_container_width=True)
        
        with tab2:
            st.markdown("#### Cash Flow Analysis")
            
            # Annual cash flow chart
            fig, ax = plt.subplots(figsize=(10, 5))
            years = list(range(len(results['CF'])))
            colors = ['gray' if cf < 0 else 'green' for cf in results['CF']]
            ax.bar(years, results['CF'], color=colors, edgecolor='black')
            ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
            ax.set_xlabel('Year')
            ax.set_ylabel('Cash Flow (USD)')
            ax.set_title('Annual Cash Flow')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            # Cumulative cash flow
            fig, ax = plt.subplots(figsize=(10, 5))
            cum_cf = np.cumsum(results['CF'])
            ax.plot(years, cum_cf, marker='o', linewidth=2, color='blue')
            ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
            if results['Payback']:
                ax.axvline(results['Payback'], color='red', linestyle='--', 
                          label=f'Payback: {results["Payback"]:.1f} years')
            ax.set_xlabel('Year')
            ax.set_ylabel('Cumulative Cash Flow (USD)')
            ax.set_title('Cumulative Cash Flow')
            ax.grid(True, alpha=0.3)
            ax.legend()
            st.pyplot(fig)
        
        with tab3:
            st.markdown("#### Income Statement")
            
            # Create income statement DataFrame
            years = results['years']
            income_data = {
                'Year': years,
                'Revenue': results['revenue'],
                'OPEX': results['opex_array'],
                'EBITDA': results['ebitda'],
                'Depreciation': results['dep_schedule'][1:len(years)+1],
                'EBIT': results['ebit'],
                'Tax': results['tax_schedule'][1:len(years)+1],
                'Net Income': results['net_income']
            }
            income_df = pd.DataFrame(income_data)
            st.dataframe(income_df.style.format('{:,.0f}'), use_container_width=True)
            
            # Income statement chart
            fig, ax = plt.subplots(figsize=(12, 6))
            bar_width = 0.7
            x = np.arange(len(years))
            ax.bar(x, income_df['Revenue'], width=bar_width, alpha=0.5, 
                  label='Revenue', color='tab:blue')
            ax.bar(x, income_df['OPEX'], width=bar_width*0.7, alpha=0.6,
                  label='OPEX', color='tab:orange')
            ax.bar(x, income_df['Depreciation'], width=bar_width*0.4, alpha=0.7,
                  label='Depreciation', color='tab:red')
            ax.plot(x, income_df['Net Income'], marker='o', linewidth=2,
                   color='tab:green', label='Net Income')
            ax.set_xticks(x)
            ax.set_xticklabels(years)
            ax.set_xlabel('Year')
            ax.set_ylabel('USD/year')
            ax.set_title('Income Statement Profile')
            ax.grid(True, alpha=0.3)
            ax.legend()
            st.pyplot(fig)
        
        with tab4:
            st.markdown("#### Cost Breakdowns")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # CAPEX breakdown
                st.markdown("**CAPEX Breakdown**")
                cap_breakdown = {
                    'Component': ['Equipment', 'Installation', 'Piping', 'Electrical', 
                                 'Buildings', 'Utilities', 'Storage', 'Safety', 'Waste',
                                 'Engineering', 'Construction', 'Licensing', 'Contractor',
                                 'Contingency', 'Insurance', "Owner's", 'Start-up'],
                    'Cost': [
                        params['C_PE'],
                        params['f_ins'] * params['C_PE'],
                        params['f_pipe'] * params['C_PE'],
                        params['f_elec'] * params['C_PE'],
                        params['f_bldg'] * params['C_PE'],
                        params['f_util'] * params['C_PE'],
                        params['f_stor'] * params['C_PE'],
                        params['f_safe'] * params['C_PE'],
                        params['f_waste'] * params['C_PE'],
                        params['f_eng'] * results['DCC'],
                        params['f_cons'] * results['DCC'],
                        params['f_licn'] * results['DCC'],
                        params['f_cont'] * results['DCC'],
                        params['f_contg'] * results['DCC'],
                        params['f_insur'] * results['DCC'],
                        params['f_own'] * results['DCC'],
                        params['f_start'] * results['DCC']
                    ]
                }
                cap_df = pd.DataFrame(cap_breakdown)
                cap_df = cap_df.sort_values('Cost', ascending=False)
                
                fig, ax = plt.subplots(figsize=(8, 6))
                y_pos = np.arange(len(cap_df))
                ax.barh(y_pos, cap_df['Cost'], color='tab:blue', alpha=0.8)
                ax.set_yticks(y_pos)
                ax.set_yticklabels(cap_df['Component'])
                ax.set_xlabel('Cost (USD)')
                ax.set_title('CAPEX Breakdown')
                ax.invert_yaxis()
                ax.grid(axis='x', alpha=0.3)
                st.pyplot(fig)
            
            with col2:
                # OPEX breakdown
                st.markdown("**OPEX Breakdown**")
                opex_breakdown = {
                    'Component': ['DOC', 'FOC', 'GMC', 'Operational Risk', 
                                 'CO₂ Cost', 'ESG & Compliance'],
                    'Cost': [results['DOC'], results['FOC'], results['GMC'],
                            results['risk_cost'], results['co2_cost'], 
                            results.get('esg_cost', 0)]
                }
                opex_df = pd.DataFrame(opex_breakdown)
                
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.pie(opex_df['Cost'], labels=opex_df['Component'], 
                      autopct='%1.1f%%', startangle=90)
                ax.set_title('OPEX Breakdown')
                st.pyplot(fig)
        
        # Target IRR price calculation
        st.markdown("---")
        st.markdown("#### Required Selling Price for Target IRR")
        target_irr_pct = st.slider("Target IRR (%)", 0.0, 50.0, 15.0, 1.0)
        
        if st.button("Calculate Required Price"):
            target_irr = target_irr_pct / 100.0
            p_req, out_req = price_for_target_irr(params, target_irr)
            
            if p_req:
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Current Price", f"${params['P_prod']:.2f}/ton")
                with col_b:
                    st.metric("Required Price", f"${p_req:.2f}/ton")
                with col_c:
                    price_change = ((p_req - params['P_prod']) / params['P_prod']) * 100
                    st.metric("Change Required", f"{price_change:+.1f}%")
                
                st.markdown(f"**At ${p_req:.2f}/ton:**")
                st.markdown(f"- IRR: {out_req['IRR']*100:.2f}%")
                st.markdown(f"- LCOx: ${out_req['LCOx']:.2f}/ton")
                st.markdown(f"- NPV: ${out_req['NPV']:,.0f}")
            else:
                st.error("Could not find price that achieves target IRR.")

# ======================================================================
#  Sensitivity Analysis
# ======================================================================
elif app_mode == "Sensitivity Analysis":
    st.markdown('<h2 class="section-header">Sensitivity Analysis</h2>', unsafe_allow_html=True)
    
    if not st.session_state.params:
        st.warning("Please input parameters first.")
        st.stop()
    
    params = st.session_state.params
    
    # Sensitivity analysis options
    st.markdown("### Sensitivity Analysis Options")
    
    analysis_type = st.selectbox(
        "Select Analysis Type",
        ["Tornado Analysis", "Price Sweep", "Raw Material Cost Sweep", 
         "Pairwise Impact Analysis", "Monte Carlo Analysis"]
    )
    
    if analysis_type == "Tornado Analysis":
        st.markdown("#### Tornado Sensitivity Analysis")
        
        # Select parameters for sensitivity
        default_params = ['C_PE', 'C_RM', 'P_prod', 'f_esg', 'tau_CO2', 'E_CO2']
        selected_params = st.multiselect(
            "Select parameters for sensitivity analysis",
            ['C_PE', 'COL', 'C_RM', 'C_UT', 'C_CAT', 'Q_prod', 'P_prod', 
             'f_esg', 'tau_CO2', 'E_CO2', 'i_base', 'delta_risk'],
            default=default_params
        )
        
        swing = st.slider("Swing (±%)", 10, 50, 20, 5) / 100
        
        if st.button("Run Tornado Analysis") and selected_params:
            with st.spinner("Running tornado sensitivity..."):
                # Get base results
                base_results = compute_TEA(params)
                
                # Calculate sensitivity
                sensitivity_data = []
                low_NPV = []
                high_NPV = []
                low_LCOx = []
                high_LCOx = []
                
                for param in selected_params:
                    # Low value
                    p_low = params.copy()
                    p_low[param] = p_low[param] * (1 - swing)
                    res_low = compute_TEA(p_low)
                    
                    # High value
                    p_high = params.copy()
                    p_high[param] = p_high[param] * (1 + swing)
                    res_high = compute_TEA(p_high)
                    
                    sensitivity_data.append({
                        'Parameter': param,
                        'Base NPV': base_results['NPV'],
                        'Low NPV': res_low['NPV'],
                        'High NPV': res_high['NPV'],
                        'Base LCOx': base_results['LCOx'],
                        'Low LCOx': res_low['LCOx'],
                        'High LCOx': res_high['LCOx'],
                        'Base IRR': base_results['IRR'] * 100,
                        'Low IRR': res_low['IRR'] * 100,
                        'High IRR': res_high['IRR'] * 100
                    })
                    
                    low_NPV.append(res_low['NPV'])
                    high_NPV.append(res_high['NPV'])
                    low_LCOx.append(res_low['LCOx'])
                    high_LCOx.append(res_high['LCOx'])
                
                # Display results
                st.markdown("##### Sensitivity Results")
                sens_df = pd.DataFrame(sensitivity_data)
                # Just display without formatting
                st.dataframe(sens_df, use_container_width=True)
                
                # Plot tornado charts
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
                
                # NPV tornado
                y_pos = np.arange(len(selected_params))
                for i in range(len(selected_params)):
                    xl = min(low_NPV[i], high_NPV[i])
                    xh = max(low_NPV[i], high_NPV[i])
                    ax1.hlines(y_pos[i], xl, xh, linewidth=8, color='tab:orange')
                ax1.axvline(base_results['NPV'], linestyle='--', color='tab:blue')
                ax1.set_yticks(y_pos)
                ax1.set_yticklabels(selected_params)
                ax1.set_xlabel('NPV (USD)')
                ax1.set_title(f'NPV Sensitivity (±{int(swing*100)}%)')
                ax1.grid(axis='x', alpha=0.4)
                ax1.invert_yaxis()
                
                # LCOx tornado
                for i in range(len(selected_params)):
                    xl = min(low_LCOx[i], high_LCOx[i])
                    xh = max(low_LCOx[i], high_LCOx[i])
                    ax2.hlines(y_pos[i], xl, xh, linewidth=8, color='tab:green')
                ax2.axvline(base_results['LCOx'], linestyle='--', color='tab:red')
                ax2.set_yticks(y_pos)
                ax2.set_yticklabels(selected_params)
                ax2.set_xlabel('LCOx (USD/ton)')
                ax2.set_title(f'LCOx Sensitivity (±{int(swing*100)}%)')
                ax2.grid(axis='x', alpha=0.4)
                ax2.invert_yaxis()
                
                plt.tight_layout()
                st.pyplot(fig)
    
    elif analysis_type == "Price Sweep":
        st.markdown("#### Price Sensitivity Analysis")
        
        swing = st.slider("Price swing (±%)", 10, 50, 30, 5) / 100
        n_points = st.slider("Number of points", 10, 50, 25, 5)
        
        if st.button("Run Price Sweep"):
            P0 = params['P_prod']
            prices = np.linspace((1 - swing) * P0, (1 + swing) * P0, n_points)
            
            npv_list = []
            lcox_list = []
            irr_list = []
            
            for price in prices:
                p_temp = params.copy()
                p_temp['P_prod'] = price
                res = compute_TEA(p_temp)
                npv_list.append(res['NPV'])
                lcox_list.append(res['LCOx'])
                irr_list.append(res['IRR'] * 100)
            
            # Plot results
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            ax1.plot(prices, npv_list, marker='o', linewidth=2)
            ax1.axhline(0, linestyle='--', color='black', alpha=0.7)
            ax1.set_xlabel('Selling Price (USD/ton)')
            ax1.set_ylabel('NPV (USD)')
            ax1.set_title('NPV vs Selling Price')
            ax1.grid(True, alpha=0.3)
            
            ax2.plot(prices, irr_list, marker='s', linewidth=2, color='green')
            ax2.set_xlabel('Selling Price (USD/ton)')
            ax2.set_ylabel('IRR (%)')
            ax2.set_title('IRR vs Selling Price')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
    
    elif analysis_type == "Raw Material Cost Sweep":
        st.markdown("#### Raw Material Cost Sensitivity")
        
        swing = st.slider("Cost swing (±%)", 10, 50, 30, 5) / 100
        n_points = st.slider("Number of points", 10, 50, 25, 5)
        
        if st.button("Run RM Cost Sweep"):
            C0 = params['C_RM']
            costs = np.linspace((1 - swing) * C0, (1 + swing) * C0, n_points)
            
            npv_list = []
            lcox_list = []
            
            for cost in costs:
                p_temp = params.copy()
                p_temp['C_RM'] = cost
                res = compute_TEA(p_temp)
                npv_list.append(res['NPV'])
                lcox_list.append(res['LCOx'])
            
            # Plot results
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            ax1.plot(costs, npv_list, marker='o', linewidth=2)
            ax1.axhline(0, linestyle='--', color='black', alpha=0.7)
            ax1.set_xlabel('Raw Material Cost (USD/year)')
            ax1.set_ylabel('NPV (USD)')
            ax1.set_title('NPV vs Raw Material Cost')
            ax1.grid(True, alpha=0.3)
            
            ax2.plot(costs, lcox_list, marker='s', linewidth=2, color='red')
            ax2.set_xlabel('Raw Material Cost (USD/year)')
            ax2.set_ylabel('LCOx (USD/ton)')
            ax2.set_title('LCOx vs Raw Material Cost')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
    
    elif analysis_type == "Monte Carlo Analysis":
        st.markdown("#### Monte Carlo Uncertainty Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            n_simulations = st.number_input("Number of simulations", 
                                          min_value=100, max_value=10000, 
                                          value=1000, step=100)
            sigma_rm = st.slider("C_RM std dev (%)", 0, 50, 15, 5) / 100
        
        with col2:
            sigma_price = st.slider("Price std dev (%)", 0, 50, 10, 5) / 100
            sigma_co2 = st.slider("Emissions std dev (%)", 0, 100, 30, 5) / 100
        
        if st.button("Run Monte Carlo Analysis"):
            with st.spinner(f"Running {n_simulations} Monte Carlo simulations..."):
                L_list, N_list, I_list = [], [], []
                
                for _ in range(n_simulations):
                    p_temp = params.copy()
                    
                    # Add randomness to key parameters
                    p_temp['C_RM'] = p_temp['C_RM'] * np.random.normal(1.0, sigma_rm)
                    p_temp['P_prod'] = p_temp['P_prod'] * np.random.normal(1.0, sigma_price)
                    p_temp['E_CO2'] = p_temp['E_CO2'] * np.random.normal(1.0, sigma_co2)
                    
                    res = compute_TEA(p_temp)
                    L_list.append(res['LCOx'])
                    N_list.append(res['NPV'])
                    I_list.append(res['IRR'])
                
                L = np.array(L_list)
                N = np.array(N_list)
                I = np.array(I_list)
                
                # Calculate percentiles
                def bands(x):
                    return np.percentile(x, [5, 50, 95])
                
                L_p = bands(L)
                N_p = bands(N)
                I_p = bands(I)
                
                # Display results
                st.markdown("##### Monte Carlo Results (5%, 50%, 95% percentiles)")
                
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("LCOx (USD/ton)", 
                             f"{L_p[1]:.2f}", 
                             f"Range: {L_p[0]:.2f} - {L_p[2]:.2f}")
                with col_b:
                    st.metric("NPV (USD)", 
                             f"{N_p[1]:,.0f}", 
                             f"Range: {N_p[0]:,.0f} - {N_p[2]:,.0f}")
                with col_c:
                    st.metric("IRR (%)", 
                             f"{I_p[1]*100:.2f}%", 
                             f"Range: {I_p[0]*100:.2f}% - {I_p[2]*100:.2f}%")
                
                # Plot distributions
                fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
                
                ax1.hist(L, bins=50, edgecolor='black', alpha=0.7)
                for v in L_p:
                    ax1.axvline(v, linestyle='--', color='black', alpha=0.7)
                ax1.set_title('LCOx Distribution')
                ax1.set_xlabel('USD/ton')
                ax1.grid(True, alpha=0.3)
                
                ax2.hist(N, bins=50, edgecolor='black', alpha=0.7)
                for v in N_p:
                    ax2.axvline(v, linestyle='--', color='black', alpha=0.7)
                ax2.set_title('NPV Distribution')
                ax2.set_xlabel('USD')
                ax2.grid(True, alpha=0.3)
                
                ax3.hist(np.array(I)*100, bins=50, edgecolor='black', alpha=0.7)
                for v in I_p*100:
                    ax3.axvline(v, linestyle='--', color='black', alpha=0.7)
                ax3.set_title('IRR Distribution')
                ax3.set_xlabel('%')
                ax3.grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)

# ======================================================================
#  Scenario Analysis
# ======================================================================
elif app_mode == "Scenario Analysis":
    st.markdown('<h2 class="section-header">Scenario Analysis</h2>', unsafe_allow_html=True)
    
    if not st.session_state.params:
        st.warning("Please input parameters first.")
        st.stop()
    
    params = st.session_state.params
    
    st.markdown("### Scenario Definition")
    
    # Scenario parameters
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Base Scenario Factors")
        price_swing = st.slider("Selling price swing ±(%)", 0, 50, 20, 5)
        cost_swing = st.slider("Operating cost swing ±(%)", 0, 50, 20, 5)
        capex_swing = st.slider("CAPEX swing ±(%)", 0, 50, 15, 5)
    
    with col2:
        st.markdown("#### Risk & ESG Factors")
        emis_swing = st.slider("Emissions swing ±(%)", 0, 50, 20, 5)
        risk_swing = st.slider("Risk factor swing ±(%)", 0, 100, 50, 10)
        esg_swing = st.slider("ESG cost swing ±(%)", 0, 100, 30, 10)
    
    if st.button("Run Scenario Analysis"):
        # Convert to fractions
        ps = price_swing / 100.0
        cs = cost_swing / 100.0
        xs = capex_swing / 100.0
        es = emis_swing / 100.0
        rs = risk_swing / 100.0
        esg = esg_swing / 100.0
        
        # Base case
        base_results = compute_TEA(params)
        
        # Optimistic scenario
        p_opt = params.copy()
        p_opt['P_prod'] = params['P_prod'] * (1 + ps)
        p_opt['C_RM'] = params['C_RM'] * (1 - cs)
        p_opt['C_UT'] = params['C_UT'] * (1 - cs)
        p_opt['C_PE'] = params['C_PE'] * (1 - xs)
        p_opt['E_CO2'] = params['E_CO2'] * (1 - es)
        p_opt['f_risk_op'] = params['f_risk_op'] * (1 - rs)
        p_opt['delta_risk'] = params['delta_risk'] * (1 - rs)
        p_opt['f_esg'] = params['f_esg'] * (1 - esg)
        opt_results = compute_TEA(p_opt)
        
        # Moderate scenario
        p_mod = params.copy()
        p_mod['P_prod'] = params['P_prod'] * (1 + 0.5*ps)
        p_mod['C_RM'] = params['C_RM'] * (1 - 0.5*cs)
        p_mod['C_UT'] = params['C_UT'] * (1 - 0.5*cs)
        p_mod['C_PE'] = params['C_PE'] * (1 - 0.5*xs)
        p_mod['E_CO2'] = params['E_CO2'] * (1 - 0.5*es)
        p_mod['f_risk_op'] = params['f_risk_op'] * (1 - 0.5*rs)
        p_mod['delta_risk'] = params['delta_risk'] * (1 - 0.5*rs)
        p_mod['f_esg'] = params['f_esg'] * (1 - 0.5*esg)
        mod_results = compute_TEA(p_mod)
        
        # Pessimistic scenario
        p_pess = params.copy()
        p_pess['P_prod'] = params['P_prod'] * (1 - ps)
        p_pess['C_RM'] = params['C_RM'] * (1 + cs)
        p_pess['C_UT'] = params['C_UT'] * (1 + cs)
        p_pess['C_PE'] = params['C_PE'] * (1 + xs)
        p_pess['E_CO2'] = params['E_CO2'] * (1 + es)
        p_pess['f_risk_op'] = params['f_risk_op'] * (1 + rs)
        p_pess['delta_risk'] = params['delta_risk'] * (1 + rs)
        p_pess['f_esg'] = params['f_esg'] * (1 + esg)
        pess_results = compute_TEA(p_pess)
        
        # Prepare results table
        scenarios = {
            'Pessimistic': pess_results,
            'Moderate': mod_results,
            'Base': base_results,
            'Optimistic': opt_results
        }
        
        scenario_data = []
        for name, res in scenarios.items():
            scenario_data.append({
                'Scenario': name,
                'CAPEX (USD)': res['CAPEX'],
                'LCOx (USD/ton)': res['LCOx'],
                'NPV (USD)': res['NPV'],
                'IRR (%)': res['IRR'] * 100,
                'BCR': res['BCR'],
                'Payback (years)': res['Payback'] if res['Payback'] else np.nan,
                'ROI Total (%)': res['ROI_total'] * 100,
                'ROI Avg (%)': res['ROI_avg'] * 100
            })
        
        scenario_df = pd.DataFrame(scenario_data)
        st.markdown("### Scenario Results")
        st.dataframe(scenario_df, use_container_width=True)
        
        # Plot scenario comparisons
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # NPV comparison
        scenarios_order = ['Pessimistic', 'Moderate', 'Base', 'Optimistic']
        npv_vals = [scenarios[s]['NPV'] for s in scenarios_order]
        colors = ['tab:red', 'tab:orange', 'tab:blue', 'tab:green']
        ax1.bar(scenarios_order, npv_vals, color=colors)
        ax1.axhline(0, linestyle='--', color='black', alpha=0.7)
        ax1.set_ylabel('NPV (USD)')
        ax1.set_title('Scenario NPV')
        ax1.grid(axis='y', alpha=0.3)
        
        # IRR comparison
        irr_vals = [scenarios[s]['IRR'] * 100 for s in scenarios_order]
        ax2.bar(scenarios_order, irr_vals, color=colors)
        ax2.set_ylabel('IRR (%)')
        ax2.set_title('Scenario IRR')
        ax2.grid(axis='y', alpha=0.3)
        
        # LCOx comparison
        lcox_vals = [scenarios[s]['LCOx'] for s in scenarios_order]
        ax3.bar(scenarios_order, lcox_vals, color=colors)
        ax3.set_ylabel('LCOx (USD/ton)')
        ax3.set_title('Scenario LCOx')
        ax3.grid(axis='y', alpha=0.3)
        
        # BCR comparison
        bcr_vals = [scenarios[s]['BCR'] for s in scenarios_order]
        ax4.bar(scenarios_order, bcr_vals, color=colors)
        ax4.axhline(1, linestyle='--', color='black', alpha=0.7, label='BCR = 1')
        ax4.set_ylabel('Benefit-Cost Ratio')
        ax4.set_title('Scenario BCR')
        ax4.grid(axis='y', alpha=0.3)
        ax4.legend()
        
        plt.tight_layout()
        st.pyplot(fig)

# ======================================================================
#  ESG/Compliance Sweep
# ======================================================================
elif app_mode == "ESG/Compliance Sweep":
    st.markdown('<h2 class="section-header">ESG/Compliance Cost Sweep Analysis</h2>', unsafe_allow_html=True)
    
    if not st.session_state.params:
        st.warning("Please input parameters first.")
        st.stop()
    
    params = st.session_state.params
    
    st.markdown("### ESG Sweep Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        f_min = st.slider("Minimum f_esg (%)", 0, 20, 5, 1) / 100
        f_max = st.slider("Maximum f_esg (%)", 20, 100, 75, 5) / 100
        n_points = st.slider("Number of points", 5, 50, 15, 5)
    
    with col2:
        st.markdown("**Current ESG Fraction:**")
        st.metric("", f"{params.get('f_esg', 0)*100:.1f}%")
        
        if st.button("Run ESG Sweep Analysis", type="primary"):
            with st.spinner("Running ESG sweep analysis..."):
                # Get base results
                base_out = compute_TEA(params)
                base_IRR = base_out['IRR']
                
                # Perform sweep
                f_vals = np.linspace(f_min, f_max, n_points)
                
                sweep_data = []
                for f in f_vals:
                    p_temp = params.copy()
                    p_temp['f_esg'] = float(f)
                    
                    out = compute_TEA(p_temp)
                    
                    # Calculate ROI metrics
                    N = params['N_project']
                    years = np.arange(1, N+1)
                    ebit = out['EBT_schedule'][1:N+1]
                    tax = out['tax_schedule'][1:N+1]
                    net_income = ebit - tax
                    
                    CAPEX_tot = out['CAPEX']
                    total_net_income = float(np.sum(net_income))
                    avg_net_income = float(np.mean(net_income))
                    ROI_total = total_net_income / CAPEX_tot if CAPEX_tot != 0 else 0.0
                    ROI_avg = avg_net_income / CAPEX_tot if CAPEX_tot != 0 else 0.0
                    
                    # Calculate payback
                    cum_cf = np.cumsum(out['CF'])
                    payback = None
                    for t in range(1, len(cum_cf)):
                        if cum_cf[t] >= 0:
                            dy = cum_cf[t] - cum_cf[t-1]
                            if dy != 0:
                                frac = -cum_cf[t-1] / dy
                                payback = (t - 1) + frac
                                break
                    
                    # Calculate required price to hold base IRR
                    p_req, _ = price_for_target_irr(p_temp, base_IRR)
                    
                    sweep_data.append({
                        'f_esg': f,
                        'OPEX': out['OC'],
                        'IRR': out['IRR'],
                        'NPV': out['NPV'],
                        'Revenue': out['R'],
                        'EBITDA': float(np.mean(out['ebitda'])) if len(out['ebitda']) > 0 else 0,
                        'EBIT': float(np.mean(out['ebit'])) if len(out['ebit']) > 0 else 0,
                        'ROI_total': ROI_total,
                        'ROI_avg': ROI_avg,
                        'Payback': payback,
                        'P_req_hold_IRR': p_req if p_req else np.nan,
                        'ESG_cost': out.get('esg_cost', 0)
                    })
                
                st.session_state.esg_sweep_data = pd.DataFrame(sweep_data)
    
    if st.session_state.esg_sweep_data is not None:
        sweep_df = st.session_state.esg_sweep_data
        
        st.markdown("### ESG Sweep Results")
        st.dataframe(sweep_df.style.format('{:,.2f}'), use_container_width=True)
        
        # Create multiple visualization tabs
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "OPEX & IRR", "NPV & ROI", "Revenue & Profit", 
            "Payback", "Required Price", "ESG Share"
        ])
        
        with tab1:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            ax1.plot(sweep_df['f_esg']*100, sweep_df['OPEX'])
            ax1.set_xlabel('ESG fraction of base OPEX (%)')
            ax1.set_ylabel('OPEX (USD/yr)')
            ax1.set_title('OPEX vs ESG/Compliance Fraction')
            ax1.grid(True, alpha=0.3)
            
            ax2.plot(sweep_df['f_esg']*100, sweep_df['IRR']*100)
            ax2.set_xlabel('ESG fraction of base OPEX (%)')
            ax2.set_ylabel('IRR (%)')
            ax2.set_title('IRR vs ESG/Compliance Fraction')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
        
        with tab2:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            ax1.plot(sweep_df['f_esg']*100, sweep_df['NPV'])
            ax1.axhline(0, linestyle='--')
            ax1.set_xlabel('ESG fraction of base OPEX (%)')
            ax1.set_ylabel('NPV (USD)')
            ax1.set_title('NPV vs ESG/Compliance Fraction')
            ax1.grid(True, alpha=0.3)
            
            ax2.plot(sweep_df['f_esg']*100, sweep_df['ROI_total']*100, label='Total ROI')
            ax2.plot(sweep_df['f_esg']*100, sweep_df['ROI_avg']*100, label='Avg ROI')
            ax2.set_xlabel('ESG fraction of base OPEX (%)')
            ax2.set_ylabel('ROI (%)')
            ax2.set_title('ROI vs ESG/Compliance Fraction')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            plt.tight_layout()
            st.pyplot(fig)
        
        with tab3:
            fig, ax = plt.subplots(figsize=(10, 5))
            
            ax.plot(sweep_df['f_esg']*100, sweep_df['Revenue'], label='Revenue')
            ax.plot(sweep_df['f_esg']*100, sweep_df['EBITDA'], label='EBITDA')
            ax.plot(sweep_df['f_esg']*100, sweep_df['EBIT'], label='EBIT')
            ax.set_xlabel('ESG fraction of base OPEX (%)')
            ax.set_ylabel('USD/year')
            ax.set_title('Revenue/EBITDA/EBIT vs ESG/Compliance Fraction')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            plt.tight_layout()
            st.pyplot(fig)
        
        with tab4:
            fig, ax = plt.subplots(figsize=(10, 5))
            
            ax.plot(sweep_df['f_esg']*100, sweep_df['Payback'])
            ax.set_xlabel('ESG fraction of base OPEX (%)')
            ax.set_ylabel('Payback (years)')
            ax.set_title('Payback vs ESG/Compliance Fraction')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
        
        with tab5:
            fig, ax = plt.subplots(figsize=(10, 5))
            
            ax.plot(sweep_df['f_esg']*100, sweep_df['P_req_hold_IRR'])
            ax.set_xlabel('ESG fraction of base OPEX (%)')
            ax.set_ylabel('Required Price (USD/ton)')
            ax.set_title('Price to Hold Base IRR vs ESG/Compliance Fraction')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
        
        with tab6:
            fig, ax = plt.subplots(figsize=(10, 5))
            
            share = sweep_df['ESG_cost'] / sweep_df['OPEX']
            ax.plot(sweep_df['f_esg']*100, share*100)
            ax.set_xlabel('ESG fraction of base OPEX (%)')
            ax.set_ylabel('ESG Share of Total OPEX (%)')
            ax.set_title('ESG Share of OPEX vs ESG/Compliance Fraction')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)

# ======================================================================
#  Multi-Design Comparison
# ======================================================================
elif app_mode == "Multi-Design Comparison":
    st.markdown('<h2 class="section-header">Multi-Design Comparison</h2>', unsafe_allow_html=True)
    
    # Display current designs
    st.markdown("### Saved Designs")
    
    if not st.session_state.designs:
        st.info("No designs saved yet. Create designs in the Parameter Input section.")
    else:
        # Show designs table
        designs_data = []
        for i, design in enumerate(st.session_state.designs):
            # Calculate results for each design
            results = compute_TEA(design['params'])
            designs_data.append({
                'Name': design['name'],
                'Mode': design['mode'].upper(),
                'CAPEX (USD)': results['CAPEX'],
                'LCOx (USD/ton)': results['LCOx'],
                'NPV (USD)': results['NPV'],
                'IRR (%)': results['IRR'] * 100,
                'Payback (years)': results['Payback'] if results['Payback'] else np.nan
            })
        
        designs_df = pd.DataFrame(designs_data)
        # Apply formatting only to numeric columns
        display_df = designs_df.copy()
        for col in display_df.select_dtypes(include=[np.number]).columns:
            display_df[col] = display_df[col].apply(lambda x: f"{x:,.2f}" if pd.notnull(x) else 
        "")
        st.dataframe(display_df, use_container_width=True)
        
        # Allow selection of designs to compare
        design_names = [d['name'] for d in st.session_state.designs]
        selected_designs = st.multiselect(
            "Select designs to compare",
            design_names,
            default=design_names[:min(3, len(design_names))]
        )
        
        if selected_designs and st.button("Compare Selected Designs"):
            # Calculate results for selected designs
            comparison_data = []
            for name in selected_designs:
                design = next(d for d in st.session_state.designs if d['name'] == name)
                results = compute_TEA(design['params'])
                comparison_data.append({
                    'Design': name,
                    'CAPEX': results['CAPEX'],
                    'LCOx': results['LCOx'],
                    'NPV': results['NPV'],
                    'IRR': results['IRR'] * 100,
                    'BCR': results['BCR'],
                    'Payback': results['Payback'] if results['Payback'] else np.nan,
                    'ROI Total': results['ROI_total'] * 100,
                    'ROI Avg': results['ROI_avg'] * 100
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            
            # Display comparison table
            st.markdown("### Design Comparison")
            st.dataframe(comparison_df.set_index('Design').style.format('{:,.2f}'), 
                        use_container_width=True)
            
            # Create comparison charts
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
            
            designs = comparison_df['Design']
            idx = np.arange(len(designs))
            colors = plt.cm.Set3(np.linspace(0, 1, len(designs)))
            
            # CAPEX comparison
            ax1.bar(idx, comparison_df['CAPEX'], color=colors)
            ax1.set_xticks(idx)
            ax1.set_xticklabels(designs, rotation=45)
            ax1.set_ylabel('CAPEX (USD)')
            ax1.set_title('CAPEX Comparison')
            ax1.grid(axis='y', alpha=0.3)
            
            # NPV comparison
            ax2.bar(idx, comparison_df['NPV'], color=colors)
            ax2.axhline(0, linestyle='--', color='black', alpha=0.7)
            ax2.set_xticks(idx)
            ax2.set_xticklabels(designs, rotation=45)
            ax2.set_ylabel('NPV (USD)')
            ax2.set_title('NPV Comparison')
            ax2.grid(axis='y', alpha=0.3)
            
            # IRR comparison
            ax3.bar(idx, comparison_df['IRR'], color=colors)
            ax3.set_xticks(idx)
            ax3.set_xticklabels(designs, rotation=45)
            ax3.set_ylabel('IRR (%)')
            ax3.set_title('IRR Comparison')
            ax3.grid(axis='y', alpha=0.3)
            
            # LCOx comparison
            ax4.bar(idx, comparison_df['LCOx'], color=colors)
            ax4.set_xticks(idx)
            ax4.set_xticklabels(designs, rotation=45)
            ax4.set_ylabel('LCOx (USD/ton)')
            ax4.set_title('LCOx Comparison')
            ax4.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Additional metrics
            fig2, ((ax5, ax6), (ax7, ax8)) = plt.subplots(2, 2, figsize=(12, 10))
            
            # BCR comparison
            ax5.bar(idx, comparison_df['BCR'], color=colors)
            ax5.axhline(1, linestyle='--', color='black', alpha=0.7, label='BCR = 1')
            ax5.set_xticks(idx)
            ax5.set_xticklabels(designs, rotation=45)
            ax5.set_ylabel('Benefit-Cost Ratio')
            ax5.set_title('BCR Comparison')
            ax5.grid(axis='y', alpha=0.3)
            ax5.legend()
            
            # Payback comparison
            ax6.bar(idx, comparison_df['Payback'], color=colors)
            ax6.set_xticks(idx)
            ax6.set_xticklabels(designs, rotation=45)
            ax6.set_ylabel('Payback (years)')
            ax6.set_title('Payback Comparison')
            ax6.grid(axis='y', alpha=0.3)
            
            # ROI Total comparison
            ax7.bar(idx, comparison_df['ROI Total'], color=colors)
            ax7.set_xticks(idx)
            ax7.set_xticklabels(designs, rotation=45)
            ax7.set_ylabel('Total ROI (%)')
            ax7.set_title('Total ROI Comparison')
            ax7.grid(axis='y', alpha=0.3)
            
            # ROI Avg comparison
            ax8.bar(idx, comparison_df['ROI Avg'], color=colors)
            ax8.set_xticks(idx)
            ax8.set_xticklabels(designs, rotation=45)
            ax8.set_ylabel('Avg Annual ROI (%)')
            ax8.set_title('Average ROI Comparison')
            ax8.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig2)
    
    # Design management
    st.markdown("---")
    st.markdown("### Design Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Clear All Designs"):
            st.session_state.designs = []
            st.session_state.params = {}
            st.session_state.results = {}
            st.success("All designs cleared!")
            st.rerun()
    
    with col2:
        if st.session_state.designs:
            design_to_delete = st.selectbox(
                "Select design to delete",
                [d['name'] for d in st.session_state.designs]
            )
            if st.button("Delete Selected Design"):
                st.session_state.designs = [
                    d for d in st.session_state.designs 
                    if d['name'] != design_to_delete
                ]
                st.success(f"Deleted design: {design_to_delete}")
                st.rerun()

# ======================================================================
#  Export Results
# ======================================================================
elif app_mode == "Export Results":
    st.markdown('<h2 class="section-header">Export Results</h2>', unsafe_allow_html=True)
    
    if not st.session_state.results and not st.session_state.designs:
        st.warning("No results to export. Please run analyses first.")
        st.stop()
    
    st.markdown("### Export Options")
    
    export_format = st.selectbox(
        "Select export format",
        ["CSV", "Excel", "JSON", "PDF Report"]
    )
    
    if export_format == "CSV":
        if st.session_state.results:
            # Create comprehensive results DataFrame
            results = st.session_state.results
            params = st.session_state.params
            
            # Key metrics
            metrics_data = {
                'Metric': ['CAPEX', 'LCOx', 'NPV', 'IRR', 'Payback', 'BCR', 
                          'ROI Total', 'ROI Avg', 'Salvage', 'Annual Revenue',
                          'Annual OPEX', 'ESG Cost'],
                'Value': [results['CAPEX'], results['LCOx'], results['NPV'],
                         results['IRR']*100, results['Payback'] if results['Payback'] else np.nan,
                         results['BCR'], results['ROI_total']*100, 
                         results['ROI_avg']*100, results['Salvage'],
                         results['R'], results['OC'], results.get('esg_cost', 0)],
                'Unit': ['USD', 'USD/ton', 'USD', '%', 'years', '-', 
                        '%', '%', 'USD', 'USD/year', 'USD/year', 'USD/year']
            }
            metrics_df = pd.DataFrame(metrics_data)
            
            # Convert to CSV
            csv = metrics_df.to_csv(index=False)
            
            st.download_button(
                label="Download Metrics as CSV",
                data=csv,
                file_name="tea_metrics.csv",
                mime="text/csv"
            )
    
    elif export_format == "Excel":
        if st.session_state.results:
            # Create Excel file with multiple sheets
            import io
            from io import BytesIO
            
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                # Key metrics sheet
                metrics_data = {
                    'Metric': ['CAPEX', 'LCOx', 'NPV', 'IRR', 'Payback', 'BCR', 
                              'ROI Total', 'ROI Avg', 'Salvage', 'Annual Revenue',
                              'Annual OPEX', 'ESG Cost'],
                    'Value': [st.session_state.results['CAPEX'], 
                             st.session_state.results['LCOx'],
                             st.session_state.results['NPV'],
                             st.session_state.results['IRR']*100,
                             st.session_state.results['Payback'] if st.session_state.results['Payback'] else np.nan,
                             st.session_state.results['BCR'],
                             st.session_state.results['ROI_total']*100,
                             st.session_state.results['ROI_avg']*100,
                             st.session_state.results['Salvage'],
                             st.session_state.results['R'],
                             st.session_state.results['OC'],
                             st.session_state.results.get('esg_cost', 0)],
                    'Unit': ['USD', 'USD/ton', 'USD', '%', 'years', '-', 
                            '%', '%', 'USD', 'USD/year', 'USD/year', 'USD/year']
                }
                pd.DataFrame(metrics_data).to_excel(writer, sheet_name='Key Metrics', index=False)
                
                # Cash flow sheet
                cf_data = {
                    'Year': list(range(len(st.session_state.results['CF']))),
                    'Cash Flow': st.session_state.results['CF'],
                    'Cumulative CF': np.cumsum(st.session_state.results['CF'])
                }
                pd.DataFrame(cf_data).to_excel(writer, sheet_name='Cash Flow', index=False)
                
                # Income statement sheet
                if 'years' in st.session_state.results:
                    years = st.session_state.results['years']
                    is_data = {
                        'Year': years,
                        'Revenue': st.session_state.results['revenue'],
                        'OPEX': st.session_state.results['opex_array'],
                        'EBITDA': st.session_state.results['ebitda'],
                        'Depreciation': st.session_state.results['dep_schedule'][1:len(years)+1],
                        'EBIT': st.session_state.results['ebit'],
                        'Tax': st.session_state.results['tax_schedule'][1:len(years)+1],
                        'Net Income': st.session_state.results['net_income']
                    }
                    pd.DataFrame(is_data).to_excel(writer, sheet_name='Income Statement', index=False)
                
                # Parameters sheet
                params_df = pd.DataFrame(list(st.session_state.params.items()), 
                                       columns=['Parameter', 'Value'])
                params_df.to_excel(writer, sheet_name='Parameters', index=False)
            
            st.download_button(
                label="Download Excel Report",
                data=output.getvalue(),
                file_name="tea_analysis_report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    
    elif export_format == "JSON":
        if st.session_state.results:
            # Combine parameters and results
            export_data = {
                'parameters': st.session_state.params,
                'results': {k: (float(v) if isinstance(v, (np.ndarray, np.generic)) else v) 
                          for k, v in st.session_state.results.items() 
                          if not isinstance(v, (list, np.ndarray)) or 
                          (isinstance(v, np.ndarray) and v.size == 1)},
                'metadata': {
                    'export_date': datetime.now().isoformat(),
                    'design_name': st.session_state.current_design,
                    'mode': st.session_state.mode
                }
            }
            
            json_str = json.dumps(export_data, indent=2, default=str)
            
            st.download_button(
                label="Download JSON Data",
                data=json_str,
                file_name="tea_data.json",
                mime="application/json"
            )
    
    elif export_format == "PDF Report":
        st.info("PDF report generation requires additional setup. For now, use CSV or Excel export.")
        # Note: For PDF generation, you would need to implement report generation
        # using libraries like ReportLab or WeasyPrint
    
    # Design comparison export
    if st.session_state.designs and len(st.session_state.designs) > 1:
        st.markdown("---")
        st.markdown("### Export Design Comparison")
        
        if st.button("Export Design Comparison as Excel"):
            import io
            from io import BytesIO
            
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                # Collect all design results
                all_designs_data = []
                
                for design in st.session_state.designs:
                    results = compute_TEA(design['params'])
                    design_data = {
                        'Design': design['name'],
                        'Mode': design['mode'],
                        'CAPEX': results['CAPEX'],
                        'LCOx': results['LCOx'],
                        'NPV': results['NPV'],
                        'IRR': results['IRR'] * 100,
                        'BCR': results['BCR'],
                        'Payback': results['Payback'] if results['Payback'] else np.nan,
                        'ROI Total': results['ROI_total'] * 100,
                        'ROI Avg': results['ROI_avg'] * 100,
                        'Annual Revenue': results['R'],
                        'Annual OPEX': results['OC']
                    }
                    
                    if design['mode'] == 'ex-ante':
                        design_data.update({
                            'TRL': design['params'].get('TRL', ''),
                            'Reference Capacity': design['params'].get('Q_ref', ''),
                            'Target Capacity': design['params'].get('Q_prod', ''),
                            'FOAK Factor': design['params'].get('foak_mult', '')
                        })
                    
                    all_designs_data.append(design_data)
                
                comparison_df = pd.DataFrame(all_designs_data)
                comparison_df.to_excel(writer, sheet_name='Design Comparison', index=False)
                
                # Add parameters for each design
                for i, design in enumerate(st.session_state.designs):
                    params_df = pd.DataFrame(list(design['params'].items()), 
                                           columns=['Parameter', 'Value'])
                    sheet_name = f"{design['name']}"[:31]  # Excel sheet name limit
                    params_df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            st.download_button(
                label="Download Design Comparison Excel",
                data=output.getvalue(),
                file_name="design_comparison.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

# ======================================================================
#  Footer
# ======================================================================
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info("""
**TEA Compliance & ESG Analysis Tool**

This tool implements comprehensive Techno-Economic Analysis with:
- Commercial & Ex-ante TEA modes
- ESG/Compliance cost integration
- Sensitivity & scenario analysis
- Multi-design comparison

All calculations based on the compliance.py model.
""")

# Show current status
st.sidebar.markdown("### Current Status")
if st.session_state.params:
    st.sidebar.success(f"✅ Parameters loaded for {st.session_state.current_design}")
    st.sidebar.metric("Mode", st.session_state.mode.upper())
    
    if st.session_state.results:
        st.sidebar.metric("NPV", f"${st.session_state.results['NPV']:,.0f}")
        st.sidebar.metric("IRR", f"{st.session_state.results['IRR']*100:.1f}%")
    
    st.sidebar.metric("Designs Saved", len(st.session_state.designs))
else:
    st.sidebar.warning("⚠️ No parameters loaded")

# Reset button
if st.sidebar.button("Reset All Data", type="secondary"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()