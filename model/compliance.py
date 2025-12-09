# -*- coding: utf-8 -*-
"""
Created on Mon Dec  1 11:38:26 2025

@author: a1917785
"""

# -*- coding: utf-8 -*-
"""
Commercial / Ex-ante TEA tool with:
- Mode selection: Commercial TEA vs Ex-ante TEA
- Multi-design comparison
- Per-design TEA (CAPEX, LCOx, NPV, IRR, Payback, CRF, Salvage)
- Depreciation methods (SL, SYD, DDB) + salvage + CRF
- Cost–Benefit Analysis (CBA):
    * Discounted revenues vs discounted costs
    * Benefit–Cost Ratio (BCR)
- Scenario analysis (Optimistic / Moderate / Pessimistic) with visualisations
- Visualisations:
  * Annual cashflow (coloured bars)
  * Cumulative cashflow + payback marker
  * Discounted cashflow vs year (+ cumulative discounted CF)
  * CAPEX breakdown (direct & indirect)
  * OPEX breakdown pie (DOC / FOC / GMC / risk / CO₂ / ESG)
  * Tornado sensitivity (NPV & LCOx)
  * Monte Carlo distributions (LCOx, NPV, IRR)
  * Price sweep and RM-cost sweep
  * Critical multi-colour pairwise impact plots
  * Income Statement (per-year Revenue, OPEX, EBITDA, Depreciation, EBIT, Tax, Net Income)
- Interactive editing:
  * After each run, user can change any numeric parameter and re-run

ADDED:
- ROI calculation (total and average annual) and yearly ROI (%)
- EBIT & EBITDA graphs
- ROI graph
- ESG/compliance/regulatory OPEX cost (f_esg) with region defaults
- Scenario profit impact of ESG swings (Avg_Profit + ΔProfit%)
- NPV test vs discount rate, giving a "Theoretical IRR" (NPV = 0) vs "Project IRR" and plotting both.

NEW IN THIS VERSION:
- ESG/compliance sweep (5%→75% of base OPEX) with multiple visualisations:
  OPEX, IRR, NPV, SELLING PRICE (to hold base IRR), ROI (total & avg), Revenue, EBITDA, EBIT, Payback
  + Critical multi-colour %Δ scatter vs %Δ f_esg
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# ======================================================================
#  Beginner-friendly help text for key parameters
# ======================================================================
PARAM_HELP = {
    "C_PE": (
        "Purchased Equipment Cost (USD).\n"
        "- This is the base cost of major process equipment before installation.\n"
        "- Usually taken from your CAPEX sheet or vendor quotes.\n"
        "- All direct and indirect capital factors (installation, piping, buildings, etc.) "
        "are multiplied from this number."
    ),
    "COL": (
        "Annual Operating Labour Cost (USD/year).\n"
        "- Total yearly cost of operators, technicians, shift supervisors, etc.\n"
        "- Include salaries + benefits + overtime if known.\n"
        "- This feeds OPEX via DOC, FOC and GMC correlations."
    ),
    "C_RM": (
        "Annual Raw Material Cost (USD/year).\n"
        "- Sum of all feedstock costs (natural gas, CO₂, H₂, water, chemicals, etc.).\n"
        "- For ex-ante mode this is calculated from cost intensity × production."
    ),
    "C_UT": (
        "Annual Utilities Cost (USD/year).\n"
        "- Electricity, steam, cooling water, instrument air, nitrogen, etc.\n"
        "- For ex-ante mode this is cost intensity × production."
    ),
    "C_CAT": (
        "Annual Catalyst / Membrane / Sorbent Make-up Cost (USD/year).\n"
        "- Include periodic catalyst replacements, specialty membranes, sorbents, etc."
    ),
    "Q_prod": (
        "Annual Product Output (ton/year).\n"
        "- Net saleable product leaving the plant, after internal recycle/purge.\n"
        "- Used to compute revenue and LCOx = OPEX / Q_prod."
    ),
    "P_prod": (
        "Selling Price (USD/ton of product).\n"
        "- Realistic average selling price over the project horizon.\n"
        "- Used to calculate gross annual revenue: R = Q_prod × P_prod."
    ),
    "f_ins": (
        "Installation Factor (fraction of C_PE).\n"
        "- Covers foundations, supports, erection, hook-up, etc.\n"
        "- Typical range 0.2–0.5 depending on complexity."
    ),
    "f_pipe": (
        "Piping Factor (fraction of C_PE).\n"
        "- Includes piping, valves, fittings, painting, insulation.\n"
        "- Often 0.3–0.8 depending on how pipe-intensive the plant is."
    ),
    "f_elec": (
        "Electrical Factor (fraction of C_PE).\n"
        "- Motor control centres, cabling, lighting, etc."
    ),
    "f_bldg": (
        "Buildings Factor (fraction of C_PE).\n"
        "- Control rooms, admin buildings, workshop, etc."
    ),
    "f_util": (
        "Utilities Factor (fraction of C_PE).\n"
        "- Utility generation and distribution (boilers, chillers, etc.)."
    ),
    "f_stor": (
        "Storage Factor (fraction of C_PE).\n"
        "- Storage tanks, silos, bullets, etc. for product and feeds."
    ),
    "f_safe": (
        "Safety Factor (fraction of C_PE).\n"
        "- Firefighting, safety systems, ESD, alarms."
    ),
    "f_waste": (
        "Waste Treatment Factor (fraction of C_PE).\n"
        "- Off-gas treatment, wastewater treatment, solid waste handling equipment."
    ),
    "f_eng": (
        "Engineering Fraction (fraction of Direct Capital Cost).\n"
        "- Detail engineering, design, project management."
    ),
    "f_cons": (
        "Construction Supervision (fraction of DCC).\n"
        "- Site supervision, construction management."
    ),
    "f_licn": (
        "Licensing Fraction (fraction of DCC or CAPEX).\n"
        "- Technology license fees, patents, process know-how."
    ),
    "f_cont": (
        "Contractor Overhead + Profit (fraction of DCC).\n"
        "- EPC/EPCM contractor margin and indirect overhead."
    ),
    "f_contg": (
        "Contingency (fraction of DCC).\n"
        "- Covers unknowns, scope growth, early-stage uncertainty."
    ),
    "f_insur": (
        "Insurance (fraction of DCC).\n"
        "- Annual plant insurance used in OPEX and CAPEX correlations."
    ),
    "f_own": (
        "Owner’s Cost (fraction of DCC).\n"
        "- Owner’s internal project team, training, pre-ops, land, etc."
    ),
    "f_start": (
        "Start-up & Commissioning (fraction of DCC).\n"
        "- Costs of commissioning, start-up trials, initial consumables."
    ),
    "L_asset": (
        "Asset / Depreciation Life (years).\n"
        "- The accounting life over which CAPEX is depreciated.\n"
        "- Typical 15–25 years for large chemical plants.\n"
        "- Affects tax shield via depreciation schedule."
    ),
    "N_project": (
        "Project Horizon / Plant Operating Life (years).\n"
        "- Number of operating years used in NPV, IRR and CBA.\n"
        "- After N_project, the plant is assumed to stop and salvage is recovered."
    ),
    "tau_inc": (
        "Income Tax Fraction (0–1).\n"
        "- Corporate income tax rate on positive taxable earnings.\n"
        "- Only applied when Earnings Before Tax (EBT) > 0."
    ),
    "i_base": (
        "Base Discount Rate (decimal).\n"
        "- Firm-level weighted average cost of capital (WACC) or hurdle rate.\n"
        "- Does NOT include project-specific risk yet."
    ),
    "delta_risk": (
        "Risk Premium Addition to Discount Rate (decimal).\n"
        "- Extra project/technology/market risk on top of i_base.\n"
        "- Effective discount rate = i_base + delta_risk.\n"
        "- For FOAK or high-risk methanol projects, values like 0.02–0.08 are common."
    ),
    "dep_method": (
        "Depreciation Method (SL, SYD, DDB).\n"
        "- SL  = Straight-line (equal depreciation each year).\n"
        "- SYD = Sum-of-years-digits (front-loaded).\n"
        "- DDB = Double-declining-balance (strongly front-loaded)."
    ),
    "salv_frac": (
        "Salvage Value Fraction of CAPEX (0–1).\n"
        "- Fraction of total capital cost recovered at end of project.\n"
        "- Salvage is discounted and included in NPV and annualised CAPEX."
    ),
    "f_risk_op": (
        "Operational Risk Factor (fraction of DOC).\n"
        "- Models real-world under-performance vs nameplate: extra downtime,\n"
        "  start-up issues, more frequent maintenance, operator errors, etc.\n"
        "- Implemented as an additional cost = f_risk_op × DOC.\n"
        "- Typical ranges:\n"
        "   • Mature, stable plant: 0.02–0.05\n"
        "   • New / FOAK / complex process: 0.08–0.15 or higher."
    ),
    "tau_CO2": (
        "Carbon Tax (USD/ton CO₂).\n"
        "- Applies to annual CO₂ emissions (E_CO2).\n"
        "- Captures carbon pricing or emission trading costs in OPEX."
    ),
    "E_CO2": (
        "Annual CO₂ Emissions (ton/year).\n"
        "- Total direct scope 1 process and combustion CO₂ emissions.\n"
        "- In ex-ante mode, this is computed from specific intensity (tCO₂/t product)."
    ),
    "f_pack": (
        "Packaging as % of Raw Material Cost (fraction).\n"
        "- Approximates packaging cost using C_RM as base."
    ),
    "f_esg": (
        "Compliance / ESG / regulatory operating cost as a fraction of base operating cost.\n"
        "- C_ESG = f_esg × OPEX_base, where OPEX_base = DOC + FOC + GMC + risk_cost + CO₂ cost.\n"
        "- Region-typical heuristics (literature / industry):\n"
        "   • North America: ~5–10% of OPEX\n"
        "   • Europe / UK : ~5–10% of OPEX\n"
        "   • Asia-Pacific: often >10% of OPEX\n"
        "- Implemented as an explicit OPEX add-on and shown in the OPEX pie."
    ),
}


def print_param_help(name=None):
    """
    Print beginner-friendly help for one parameter, or list all.
    Use inside the interactive editor by typing:
      - 'help'          → list all documented parameters
      - 'help C_PE'     → show help for C_PE
    """
    if name is None or str(name).strip() == "":
        print("\nAvailable parameters with help (use 'help PARAM_NAME' inside the editor):")
        for k in sorted(PARAM_HELP.keys()):
            short = PARAM_HELP[k].split("\n", 1)[0]
            print(f"  {k:10s} - {short}")
        return

    key = str(name).strip()
    if key in PARAM_HELP:
        print(f"\n--- Help for {key} ---")
        print(PARAM_HELP[key])
    else:
        print(f"\nNo detailed help stored for '{key}'. "
              "Check spelling or use 'help' to list all documented parameters.")


# ======================================================================
#  Helper: numeric input with default and bounds
# ======================================================================
def ask_float(prompt, default=None, min_val=None, max_val=None):
    """
    Ask user for a float with optional default and [min, max] clamp.
    If user enters nothing and default is given, default is used.
    """
    while True:
        raw = input(prompt)
        if raw.strip() == "":
            if default is None:
                print("  Please enter a value.")
                continue
            val = default
        else:
            try:
                val = float(raw)
            except ValueError:
                print("  Invalid number, try again.")
                continue

        if min_val is not None and val < min_val:
            print(f"  Value below minimum ({min_val}), using min.")
            val = min_val
        if max_val is not None and val > max_val:
            print(f"  Value above maximum ({max_val}), using max.")
            val = max_val
        return val


def interactive_edit_params(params):
    """
    Allow user to change any numeric parameter and re-run.
    Extra: you can type 'help' or 'help PARAM_NAME' to see explanations.
    """
    print("\nEditable numeric parameters and current values:")
    for k, v in params.items():
        if isinstance(v, (int, float)):
            print(f"  {k:15s} = {v}")
    print("\nTip: inside this editor you can type 'help' or 'help PARAM_NAME' "
          "to see beginner-friendly explanations.")

    while True:
        key = input("\nEnter parameter name to change (or 'help', Enter to stop editing): ").strip()
        if key == "":
            break

        # Inline help inside the editor
        low = key.lower()
        if low == "help" or low == "h" or low == "?":
            print_param_help()
            continue
        if low.startswith("help "):
            parts = key.split(None, 1)
            if len(parts) == 2:
                print_param_help(parts[1])
            else:
                print_param_help()
            continue

        if key not in params:
            print("  Unknown parameter name. Please copy exactly from the list. "
                  "You can also type 'help' to see documented parameters.")
            continue
        if not isinstance(params[key], (int, float)):
            print("  Parameter is not numeric (or is internal); skipping.")
            continue

        current = params[key]
        val = ask_float(f"New value for {key} [current {current}]: ", default=current)
        if key in ["N_project", "L_asset"]:
            params[key] = int(round(val))
        else:
            params[key] = val

    return params


# ======================================================================
#  Core TEA engine (now with cost–benefit outputs & income-statement data)
# ======================================================================
def compute_TEA(p):
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

    CGR = DCC + ICC          # total CAPEX

    # Depreciation life / salvage
    N = p['N_project']
    L = int(round(p['L_asset']))
    dep_method = p['dep_method']     # 'SL', 'SYD', 'DDB'
    salv_frac  = p['salv_frac']      # fraction of CAPEX
    S = salv_frac * CGR              # salvage value at end of project
    L_eff = max(1, L)

    # average straight-line depreciation (for heuristic OPEX formulas)
    C_DEP_avg = CGR / L_eff
    CM_avg = C_DEP_avg + C_INSUR

    # DOC / FOC / GMC heuristic (Peters-style)
    DOC = (p['COL'] + 0.18*p['COL'] + p['C_RM'] + p['C_UT'] +
           0.06*CGR + 0.15*p['COL'] + p['C_CAT'] + p['f_pack']*p['C_RM'])

    FOC = (0.708*p['COL'] + 0.036*CGR + 0.032*CGR + p['f_insur']*CGR +
           C_DEP_avg + 0.13*CM_avg + 0.01*CGR)

    GMC = 0.177*p['COL'] + 0.009*CGR + 0.05*CM_avg + 0.11*CM_avg

    # risk + carbon contributions (for OPEX breakdown)
    risk_cost = p['f_risk_op'] * DOC
    co2_cost  = p['tau_CO2'] * p['E_CO2']

    # ---- ESG / compliance / regulatory cost as OPEX add-on ----
    OC_base = DOC + FOC + GMC + risk_cost + co2_cost
    f_esg = p.get('f_esg', 0.0)
    esg_cost = f_esg * OC_base
    OC = OC_base + esg_cost

    R = p['Q_prod'] * p['P_prod']
    i_eff = p['i_base'] + p['delta_risk']

    # ---- depreciation schedule for tax ----
    dep = np.zeros(N+1)   # index 0..N, year 0 = 0
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

    # ---- cashflow with depreciation + salvage ----
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
            CF_t = (EBT - tax) + D_t   # add back depreciation (non-cash)
            if t == N:
                CF_t += S              # salvage at end
            CF.append(CF_t)
            EBT_schedule[t] = EBT
            tax_schedule[t] = tax

    NPV = sum(CF[t] / ((1 + i_eff)**t) for t in range(len(CF)))

    # ---- safe IRR via bisection (Project IRR) ----
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

    # ---- CRF and annualised CAPEX (with salvage) ----
    if i_eff > 0:
        CRF = i_eff * (1 + i_eff)**N / ((1 + i_eff)**N - 1)
    else:
        CRF = 1.0 / N if N > 0 else 0.0

    PV_capital_net = CGR - S / ((1 + i_eff)**N)
    Annual_CAPEX = PV_capital_net * CRF

    # ---- Cost–Benefit Analysis (economic view, ignoring tax/depr) ----
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

    return {
        "CAPEX": CGR,
        "LCOx": OC / p['Q_prod'],
        "NPV": NPV,
        "IRR": irr_val,
        "CF": CF,
        "Salvage": S,
        "CRF": CRF,
        "Annual_CAPEX": Annual_CAPEX,
        # CBA metrics
        "PV_revenue": PV_rev,
        "PV_cost_total": PV_cost_total,
        "BCR": BCR,
        # extras for engineering visualisation
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
        # income-statement related
        "dep_schedule": dep,
        "EBT_schedule": EBT_schedule,
        "tax_schedule": tax_schedule,
    }


# ======================================================================
#  Payback time
# ======================================================================
def compute_payback(cf):
    """Simple (undiscounted) payback from cumulative cashflow."""
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


# ======================================================================
#  Theoretical IRR from NPV test (separate from project IRR)
# ======================================================================
def theoretical_irr_from_npv(cash, r_low=-0.9, r_high=1.0, max_iter=80, tol=1e-6):
    """
    Solve NPV(r) = 0 directly from the cashflow using bisection.
    This is the 'Theoretical IRR' from the textbook NPV definition:
      NPV = Σ_t CF_t / (1+r)^t = 0
    """
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


# ======================================================================
#  Helper: selling price required to hit a target IRR
# ======================================================================
def price_for_target_irr(params, target_irr, tol=1e-4, max_iter=80):
    """
    Find the selling price P_prod (USD/ton) required to achieve a target IRR.
    Returns (P_req, TEA_result_at_P_req) or (None, None) if no bracket is found.
    """
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

    # Initial bracket around current price
    p_low = max(1e-6, 0.1 * P0)
    p_high = 5.0 * P0
    irr_low = irr_at_price(p_low)
    irr_high = irr_at_price(p_high)

    # Try to find a bracket where irr_low <= target <= irr_high
    found = False
    for _ in range(40):
        if irr_low > irr_high:
            # ensure monotonic ordering
            p_low, p_high = p_high, p_low
            irr_low, irr_high = irr_high, irr_low
        if irr_low <= target_irr <= irr_high:
            found = True
            break
        if irr_high < target_irr:
            # need higher price
            p_high *= 1.5
            irr_high = irr_at_price(p_high)
        elif irr_low > target_irr:
            # need lower price
            p_low *= 0.5
            irr_low = irr_at_price(p_low)

    if not found:
        return None, None

    # Bisection on price
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

    # Fallback after max_iter: return mid
    p_req = 0.5 * (p_low + p_high)
    res_req = compute_TEA({**base_params, "P_prod": p_req})
    return p_req, res_req


# ======================================================================
#  Tornado sensitivity (user-defined swing)
# ======================================================================
def tornado_sensitivity(base_params, keys, swing=0.2):
    """
    Compute tornado-style sensitivity on NPV and LCOx vs chosen input keys.
    swing = fractional +/- (e.g., 0.2 = ±20%)
    Also prints a summary table of CAPEX, LCOx, NPV, IRR for base/low/high.
    """
    base = compute_TEA(base_params)
    base_CAPEX = base["CAPEX"]
    base_LCOx = base["LCOx"]
    base_NPV = base["NPV"]
    base_IRR = base["IRR"]

    rows = []

    low_NPV = []
    high_NPV = []
    low_LCOx = []
    high_LCOx = []

    for k in keys:
        p_low = base_params.copy()
        p_high = base_params.copy()

        p_low[k] = p_low[k] * (1 - swing)
        p_high[k] = p_high[k] * (1 + swing)

        r_low = compute_TEA(p_low)
        r_high = compute_TEA(p_high)

        low_NPV.append(r_low["NPV"])
        high_NPV.append(r_high["NPV"])
        low_LCOx.append(r_low["LCOx"])
        high_LCOx.append(r_high["LCOx"])

        rows.append({
            "Param": k,
            "Base_CAPEX": base_CAPEX,
            "Low_CAPEX": r_low["CAPEX"],
            "High_CAPEX": r_high["CAPEX"],
            "Base_LCOx": base_LCOx,
            "Low_LCOx": r_low["LCOx"],
            "High_LCOx": r_high["LCOx"],
            "Base_NPV": base_NPV,
            "Low_NPV": r_low["NPV"],
            "High_NPV": r_high["NPV"],
            "Base_IRR_%": base_IRR*100,
            "Low_IRR_%": r_low["IRR"]*100,
            "High_IRR_%": r_high["IRR"]*100
        })

    df = pd.DataFrame(rows)
    print("\n--- Tornado sensitivity summary (±{:.0f}% swing) ---".format(swing*100))
    with pd.option_context('display.max_columns', None):
        print(df.to_string(index=False,
                           float_format=lambda x: f"{x:,.3g}"))

    keys_str = [str(k) for k in keys]
    y_pos = np.arange(len(keys_str))

    # NPV tornado
    plt.figure(figsize=(8, 5), dpi=200)
    for i in range(len(keys_str)):
        xl = min(low_NPV[i], high_NPV[i])
        xh = max(low_NPV[i], high_NPV[i])
        plt.hlines(y_pos[i], xl, xh, linewidth=8, color="tab:orange")
    plt.axvline(base_NPV, linestyle="--", color="tab:blue")
    plt.yticks(y_pos, keys_str)
    plt.xlabel("NPV (USD)")
    plt.title(f"Tornado Sensitivity on NPV (±{int(swing*100)}%)")
    plt.grid(axis="x", alpha=0.4)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

    # LCOx tornado
    plt.figure(figsize=(8, 5), dpi=200)
    for i in range(len(keys_str)):
        xl = min(low_LCOx[i], high_LCOx[i])
        xh = max(low_LCOx[i], high_LCOx[i])
        plt.hlines(y_pos[i], xl, xh, linewidth=8, color="tab:green")
    plt.axvline(base_LCOx, linestyle="--", color="tab:red")
    plt.yticks(y_pos, keys_str)
    plt.xlabel("LCOx (USD/ton)")
    plt.title(f"Tornado Sensitivity on LCOx (±{int(swing*100)}%)")
    plt.grid(axis="x", alpha=0.4)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()


# ======================================================================
#  Monte-Carlo uncertainty
# ======================================================================
def monte_carlo_analysis(base_params, runs=1000, dist=None):
    """
    Monte Carlo on LCOx, NPV, IRR for a given distribution of inputs.
    dist = {param_name: sigma_fraction}
    """
    if dist is None:
        dist = {"C_RM": 0.15, "P_prod": 0.10, "E_CO2": 0.30}

    L_list, N_list, I_list = [], [], []

    for _ in range(runs):
        p = base_params.copy()
        for k, sigma in dist.items():
            p[k] = p[k] * np.random.normal(1.0, sigma)
        res = compute_TEA(p)
        L_list.append(res["LCOx"])
        N_list.append(res["NPV"])
        I_list.append(res["IRR"])

    L = np.array(L_list)
    N = np.array(N_list)
    I = np.array(I_list)

    def bands(x):
        return np.percentile(x, [5, 50, 95])

    L_p = bands(L)
    N_p = bands(N)
    I_p = bands(I)

    print("\nMonte-Carlo percentile bands (5%, 50%, 95%)")
    print(f"LCOx  : {L_p[0]:.2f}  {L_p[1]:.2f}  {L_p[2]:.2f}")
    print(f"NPV   : {N_p[0]:.2e}  {N_p[1]:.2e}  {N_p[2]:.2e}")
    print(f"IRR % : {I_p[0]*100:.2f}  {I_p[1]*100:.2f}  {I_p[2]*100:.2f}")

    fig, ax = plt.subplots(1, 3, figsize=(15, 4), dpi=200)

    ax[0].hist(L, bins=50)
    for v in L_p:
        ax[0].axvline(v, linestyle="--", color="black", alpha=0.7)
    ax[0].set_title("LCOx distribution")
    ax[0].grid(True, alpha=0.3)

    ax[1].hist(N, bins=50)
    for v in N_p:
        ax[1].axvline(v, linestyle="--", color="black", alpha=0.7)
    ax[1].set_title("NPV distribution")
    ax[1].grid(True, alpha=0.3)

    ax[2].hist(I, bins=50)
    for v in I_p:
        ax[2].axvline(v, linestyle="--", color="black", alpha=0.7)
    ax[2].set_title("IRR distribution")
    ax[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def interactive_monte_carlo(base_params):
    """
    Ask user for Monte-Carlo settings (runs and % sigmas) and run MC.
    """
    runs = int(ask_float("Number of Monte Carlo runs [default 1000, min 100]: ",
                         default=1000, min_val=100))
    sigma_rm = ask_float("Std dev for C_RM (%) [default 15]: ", default=15, min_val=0) / 100.0
    sigma_price = ask_float("Std dev for selling price (%) [default 10]: ", default=10, min_val=0) / 100.0
    sigma_co2 = ask_float("Std dev for emissions (E_CO2) (%) [default 30]: ", default=30, min_val=0) / 100.0

    dist = {"C_RM": sigma_rm, "P_prod": sigma_price, "E_CO2": sigma_co2}
    monte_carlo_analysis(base_params, runs=runs, dist=dist)


# ======================================================================
#  Extra visuals: price & raw-material sweeps
# ======================================================================
def price_sweep(base_params, swing=0.3, n=25):
    """
    Sweep selling price P_prod ± swing and plot NPV & LCOx vs price.
    """
    P0 = base_params['P_prod']
    vals = np.linspace((1 - swing)*P0, (1 + swing)*P0, n)
    npv_list = []
    lcox_list = []

    for pval in vals:
        p = base_params.copy()
        p['P_prod'] = pval
        out = compute_TEA(p)
        npv_list.append(out["NPV"])
        lcox_list.append(out["LCOx"])

    plt.figure(figsize=(8, 4), dpi=200)
    plt.plot(vals, npv_list, marker="o")
    plt.axhline(0.0, linestyle="--", color="black", alpha=0.7)
    plt.grid(True, alpha=0.3)
    plt.xlabel("Selling price (USD/ton)")
    plt.ylabel("NPV (USD)")
    plt.title("NPV vs Selling Price")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 4), dpi=200)
    plt.plot(vals, lcox_list, marker="s")
    plt.grid(True, alpha=0.3)
    plt.xlabel("Selling price (USD/ton)")
    plt.ylabel("LCOx (USD/ton)")
    plt.title("LCOx vs Selling Price")
    plt.tight_layout()
    plt.show()


def raw_material_sweep(base_params, swing=0.3, n=25):
    """
    Sweep raw-material cost C_RM ± swing and plot NPV & LCOx vs C_RM.
    """
    C0 = base_params['C_RM']
    vals = np.linspace((1 - swing)*C0, (1 + swing)*C0, n)
    npv_list = []
    lcox_list = []

    for cval in vals:
        p = base_params.copy()
        p['C_RM'] = cval
        out = compute_TEA(p)
        npv_list.append(out["NPV"])
        lcox_list.append(out["LCOx"])

    plt.figure(figsize=(8, 4), dpi=200)
    plt.plot(vals, npv_list, marker="o")
    plt.axhline(0.0, linestyle="--", color="black", alpha=0.7)
    plt.grid(True, alpha=0.3)
    plt.xlabel("Annual raw-material cost (USD/year)")
    plt.ylabel("NPV (USD)")
    plt.title("NPV vs Raw-material Cost")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 4), dpi=200)
    plt.plot(vals, lcox_list, marker="s")
    plt.grid(True, alpha=0.3)
    plt.xlabel("Annual raw-material cost (USD/year)")
    plt.ylabel("LCOx (USD/ton)")
    plt.title("LCOx vs Raw-material Cost")
    plt.tight_layout()
    plt.show()


# ======================================================================
#  Critical multi-colour pairwise impact visualisation
# ======================================================================
def pairwise_impact_visual(params, key_inputs, swing=0.3, n=9):
    """
    For each chosen key input:
      - Vary it from (1 - swing) to (1 + swing) in 'n' steps
      - Compute %Δ input vs %Δ outputs [CAPEX, LCOx, NPV, IRR]
    Plot as a grid:
      rows   = outputs
      columns= inputs
      each cell = scatter of %Δ output vs %Δ input (multi-colour, high dpi)
    """
    base = compute_TEA(params)
    outputs = ["CAPEX", "LCOx", "NPV", "IRR"]
    base_vals = {k: base[k] for k in outputs}
    facs = np.linspace(1 - swing, 1 + swing, n)

    n_out = len(outputs)
    n_in = len(key_inputs)

    colors = [
        "tab:blue", "tab:orange", "tab:green", "tab:red",
        "tab:purple", "tab:brown", "tab:pink", "tab:gray",
        "tab:olive", "tab:cyan"
    ]

    fig, axes = plt.subplots(
        n_out, n_in,
        figsize=(3.0 * n_in + 2, 2.8 * n_out + 2),
        dpi=200,
        squeeze=False
    )

    for j, inp in enumerate(key_inputs):
        col = colors[j % len(colors)]
        for i, out_name in enumerate(outputs):
            ax = axes[i, j]
            for f in facs:
                p2 = params.copy()
                p2[inp] = params[inp] * f
                res = compute_TEA(p2)

                d_in = (f - 1.0) * 100.0
                base_out = base_vals[out_name]
                if base_out != 0:
                    d_out = (res[out_name] - base_out) / base_out * 100.0
                else:
                    d_out = 0.0

                ax.scatter(d_in, d_out, color=col, alpha=0.8, s=30)

            ax.axhline(0.0, color="k", linewidth=0.8, linestyle="--", alpha=0.5)
            ax.axvline(0.0, color="k", linewidth=0.8, linestyle="--", alpha=0.5)

            if i == 0:
                ax.set_title(inp, fontsize=9)
            if j == 0:
                ax.set_ylabel(f"%Δ {out_name}", fontsize=9)

            ax.set_xlabel("%Δ input", fontsize=8)
            ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Pairwise impact of key cost drivers on TEA outputs (% change vs % change)",
        fontsize=12
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def interactive_pairwise_visual(params):
    """
    Wrapper that asks the user which inputs and what swing, then calls pairwise_impact_visual.
    """
    swing_pct = ask_float(
        "Pairwise sensitivity +/- percent (e.g., 30 for ±30%) [default 30]: ",
        default=30, min_val=0.0, max_val=300.0
    )
    swing = swing_pct / 100.0

    print("\nDefault key inputs for pairwise visualisation:")
    default_keys = ['C_PE', 'C_RM', 'C_UT', 'COL', 'P_prod', 'tau_CO2', 'E_CO2']
    print("  " + ", ".join(default_keys))

    use_default = input("Use default input list for pairwise plots? [Y/n]: ").strip().lower()
    if use_default in ("", "y"):
        keys = [k for k in default_keys if k in params]
    else:
        raw = input("Enter comma-separated parameter names (must be keys in params): ")
        cand = [x.strip() for x in raw.split(",") if x.strip() in params]
        if not cand:
            keys = [k for k in default_keys if k in params]
        else:
            keys = cand

    if not keys:
        print("No valid inputs found for pairwise visualisation; skipping.")
        return

    pairwise_impact_visual(params, keys, swing=swing, n=9)


# ======================================================================
#  >>> NEW: ESG/compliance sweep (5% -> 75%) with multi-visualisations <<<
# ======================================================================
def _roi_metrics_from_out(out):
    """
    Compute ROI_total, ROI_avg, EBITDA, EBIT (averages), Payback from a TEA result 'out'.
    Assumes constant per-year revenue and opex over the horizon.
    """
    N = int(len(out["dep_schedule"]) - 1)
    if N <= 0:
        return 0.0, 0.0, 0.0, 0.0, None

    years = np.arange(1, N+1)
    revenue = np.full_like(years, out["R"], dtype=float)
    opex = np.full_like(years, out["OC"], dtype=float)
    depreciation = out["dep_schedule"][1:N+1]
    ebit = out["EBT_schedule"][1:N+1]          # (no explicit interest)
    tax = out["tax_schedule"][1:N+1]
    net_income = ebit - tax
    ebitda = revenue - opex

    CAPEX_tot = out["CAPEX"]
    total_net_income = float(np.sum(net_income))
    avg_net_income = float(np.mean(net_income))
    ROI_total = total_net_income / CAPEX_tot if CAPEX_tot != 0 else 0.0
    ROI_avg   = avg_net_income   / CAPEX_tot if CAPEX_tot != 0 else 0.0
    avg_ebitda = float(np.mean(ebitda))
    avg_ebit   = float(np.mean(ebit))
    payback = compute_payback(out["CF"])

    return ROI_total, ROI_avg, avg_ebitda, avg_ebit, payback


def esg_sweep_metrics(params, f_min=0.05, f_max=0.75, n=15):
    """
    Sweep f_esg between [f_min, f_max] (fractions of OPEX) and compute:
    OPEX, IRR, NPV, Revenue, EBITDA, EBIT, ROI_total, ROI_avg, Payback,
    and the Required Selling Price to HOLD THE BASE IRR constant.
    Returns a pandas DataFrame.
    """
    p0 = params.copy()
    base_out = compute_TEA(p0)
    base_IRR = base_out["IRR"]
    base_f   = p0.get("f_esg", 0.0)

    f_vals = np.linspace(f_min, f_max, n)
    rows = []

    for f in f_vals:
        p = p0.copy()
        p["f_esg"] = float(f)

        out = compute_TEA(p)
        ROI_total, ROI_avg, avg_ebitda, avg_ebit, pb = _roi_metrics_from_out(out)

        # required selling price to hold the *base IRR* constant
        P_req, out_req = price_for_target_irr(p, target_irr=base_IRR)
        rows.append({
            "f_esg": f,
            "OPEX": out["OC"],
            "IRR": out["IRR"],
            "NPV": out["NPV"],
            "Revenue": out["R"],
            "EBITDA": avg_ebitda,
            "EBIT": avg_ebit,
            "ROI_total": ROI_total,
            "ROI_avg": ROI_avg,
            "Payback": pb if pb is not None else np.nan,
            "P_req_hold_base_IRR": P_req if P_req is not None else np.nan,
            "ESG_cost": out.get("esg_cost", 0.0),
            "Base_f_esg": base_f,
        })

    df = pd.DataFrame(rows)
    return df


def plot_esg_sweep_lines(df, design_label=""):
    """
    Simple line plots: metric vs f_esg.
    """
    # 1) OPEX
    plt.figure(figsize=(8, 4), dpi=200)
    plt.plot(df["f_esg"]*100, df["OPEX"])
    plt.xlabel("ESG/compliance fraction of base OPEX (%)")
    plt.ylabel("OPEX (USD/yr)")
    plt.title(f"OPEX vs ESG/compliance fraction{(' – ' + design_label) if design_label else ''}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # 2) NPV
    plt.figure(figsize=(8, 4), dpi=200)
    plt.plot(df["f_esg"]*100, df["NPV"])
    plt.axhline(0.0, linestyle="--")
    plt.xlabel("ESG/compliance fraction of base OPEX (%)")
    plt.ylabel("NPV (USD)")
    plt.title(f"NPV vs ESG/compliance fraction{(' – ' + design_label) if design_label else ''}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # 3) IRR
    plt.figure(figsize=(8, 4), dpi=200)
    plt.plot(df["f_esg"]*100, df["IRR"]*100.0)
    plt.xlabel("ESG/compliance fraction of base OPEX (%)")
    plt.ylabel("IRR (%)")
    plt.title(f"IRR vs ESG/compliance fraction{(' – ' + design_label) if design_label else ''}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # 4) ROI total & avg
    plt.figure(figsize=(8, 4), dpi=200)
    plt.plot(df["f_esg"]*100, df["ROI_total"]*100.0)
    plt.plot(df["f_esg"]*100, df["ROI_avg"]*100.0)
    plt.xlabel("ESG/compliance fraction of base OPEX (%)")
    plt.ylabel("ROI (%)")
    plt.title(f"ROI (total & avg) vs ESG/compliance fraction{(' – ' + design_label) if design_label else ''}")
    plt.grid(True, alpha=0.3)
    plt.legend(["Total ROI", "Avg annual ROI"])
    plt.tight_layout()
    plt.show()

    # 5) Revenue / EBITDA / EBIT
    plt.figure(figsize=(8, 4), dpi=200)
    plt.plot(df["f_esg"]*100, df["Revenue"])
    plt.plot(df["f_esg"]*100, df["EBITDA"])
    plt.plot(df["f_esg"]*100, df["EBIT"])
    plt.xlabel("ESG/compliance fraction of base OPEX (%)")
    plt.ylabel("USD/year")
    plt.title(f"Revenue / EBITDA / EBIT vs ESG/compliance fraction{(' – ' + design_label) if design_label else ''}")
    plt.grid(True, alpha=0.3)
    plt.legend(["Revenue", "EBITDA", "EBIT"])
    plt.tight_layout()
    plt.show()

    # 6) Payback
    plt.figure(figsize=(8, 4), dpi=200)
    plt.plot(df["f_esg"]*100, df["Payback"])
    plt.xlabel("ESG/compliance fraction of base OPEX (%)")
    plt.ylabel("Payback (years)")
    plt.title(f"Payback vs ESG/compliance fraction{(' – ' + design_label) if design_label else ''}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # 7) Required selling price to keep base IRR
    plt.figure(figsize=(8, 4), dpi=200)
    plt.plot(df["f_esg"]*100, df["P_req_hold_base_IRR"])
    plt.xlabel("ESG/compliance fraction of base OPEX (%)")
    plt.ylabel("Required selling price (USD/ton)")
    plt.title(f"Price to HOLD base IRR vs ESG/compliance fraction{(' – ' + design_label) if design_label else ''}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def esg_critical_scatter(df, outputs=None, design_label=""):
    """
    Critical multi-colour visualisation: %Δ(output) vs %Δ f_esg.
    """
    if outputs is None:
        outputs = ["OPEX", "IRR", "NPV", "Revenue", "EBITDA", "EBIT", "ROI_total", "ROI_avg", "Payback"]

    f0 = float(df["Base_f_esg"].iloc[0])
    idx0 = (df["f_esg"] - f0).abs().argmin()
    base_vals = {name: float(df.loc[idx0, name]) for name in outputs if name in df.columns}

    x = (df["f_esg"] - f0) / (f0 if abs(f0) > 1e-12 else 1.0) * 100.0

    chunk = 4
    for start in range(0, len(outputs), chunk):
        outs = outputs[start:start+chunk]
        n_rows = len(outs)
        fig, axes = plt.subplots(n_rows, 1, figsize=(7.5, 2.6*n_rows+1), dpi=200, squeeze=False)
        for i, out_name in enumerate(outs):
            ax = axes[i, 0]
            if out_name not in df.columns:
                continue
            y = df[out_name].astype(float)
            y0 = base_vals.get(out_name, y.iloc[0])
            y_rel = ((y - y0) / (abs(y0) if abs(y0) > 1e-12 else 1.0)) * 100.0
            ax.scatter(x, y_rel, s=30, alpha=0.9)
            ax.axhline(0.0, color="k", linewidth=0.8, linestyle="--", alpha=0.5)
            ax.axvline(0.0, color="k", linewidth=0.8, linestyle="--", alpha=0.5)
            ax.grid(True, alpha=0.3)
            ax.set_xlabel("%Δ ESG/compliance fraction (vs base)")
            ax.set_ylabel(f"%Δ {out_name}")
            if i == 0:
                ax.set_title(f"Critical %Δ plots vs %Δ f_esg{(' – ' + design_label) if design_label else ''}")
        plt.tight_layout()
        plt.show()


def run_esg_sweep_and_plots(params, design_label="", f_min=0.05, f_max=0.75, n=15):
    """
    Generates the sweep, prints a compact preview, and renders all visuals.
    """
    df = esg_sweep_metrics(params, f_min=f_min, f_max=f_max, n=n)

    # preview
    with pd.option_context('display.max_columns', None):
        print("\nESG/compliance sweep (head):")
        print(df.head().to_string(index=False, float_format=lambda x: f"{x:,.4g}"))
        print("\n... (tail):")
        print(df.tail().to_string(index=False, float_format=lambda x: f"{x:,.4g}"))

    # Lines
    plot_esg_sweep_lines(df, design_label=design_label)

    # Critical scatter
    esg_critical_scatter(df, design_label=design_label)

    # ESG share of OPEX evolution
    plt.figure(figsize=(8, 4), dpi=200)
    share = df["ESG_cost"] / df["OPEX"]
    plt.plot(df["f_esg"]*100.0, share*100.0)
    plt.xlabel("ESG/compliance fraction of base OPEX (%)")
    plt.ylabel("ESG share of total OPEX (%)")
    plt.title(f"ESG share of OPEX vs ESG/compliance fraction{(' – ' + design_label) if design_label else ''}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ======================================================================
#  Scenario cost–benefit analysis (Optimistic / Moderate / Pessimistic)
# ======================================================================
def scenario_cba(params, design_label):
    """
    Build 3 scenarios (Optimistic / Moderate / Pessimistic) around the base case
    and compute TEA + cost–benefit metrics, then visualise.

    Includes ESG/compliance swings (f_esg) and reports impact on average net profit.
    """
    base = compute_TEA(params)

    print("\n--- Scenario cost–benefit analysis for", design_label, "---")
    use_def = input("Use default percentage swings for scenarios? [Y/n]: ").strip().lower()

    if use_def in ("", "y"):
        price_swing_pct = 20.0   # ±20% on selling price
        cost_swing_pct  = 20.0   # ±20% on C_RM, C_UT
        capex_swing_pct = 15.0   # ±15% on C_PE
        emis_swing_pct  = 20.0   # ±20% on E_CO2
        risk_swing_pct  = 50.0   # ±50% on f_risk_op and delta_risk
        esg_swing_pct   = 30.0   # ±30% on f_esg
    else:
        price_swing_pct = ask_float("Selling price swing ±(%) [default 20]: ", default=20, min_val=0)
        cost_swing_pct  = ask_float("Operating cost swing ±(%) [default 20]: ", default=20, min_val=0)
        capex_swing_pct = ask_float("CAPEX swing ±(%) [default 15]: ", default=15, min_val=0)
        emis_swing_pct  = ask_float("CO₂ emissions swing ±(%) [default 20]: ", default=20, min_val=0)
        risk_swing_pct  = ask_float("Risk premium swing ±(%) [default 50]: ", default=50, min_val=0)
        esg_swing_pct   = ask_float("ESG/compliance OPEX swing ±(%) [default 30]: ", default=30, min_val=0)

    ps = price_swing_pct / 100.0
    cs = cost_swing_pct  / 100.0
    xs = capex_swing_pct / 100.0
    es = emis_swing_pct  / 100.0
    rs = risk_swing_pct  / 100.0
    esg = esg_swing_pct  / 100.0

    def apply_mult(base_p, mults):
        p = base_p.copy()
        for k, fac in mults.items():
            if k in p:
                p[k] = p[k] * fac
        return p

    scenarios = {}

    # Base
    scenarios["Base"] = (params.copy(), base)

    # Optimistic
    mult_opt = {
        "P_prod": 1 + ps,
        "C_RM":   1 - cs,
        "C_UT":   1 - cs,
        "C_PE":   1 - xs,
        "E_CO2":  1 - es,
        "f_risk_op": 1 - rs,
        "delta_risk": 1 - rs,
        "f_esg":  1 - esg,
    }
    p_opt = apply_mult(params, mult_opt)
    scenarios["Optimistic"] = (p_opt, compute_TEA(p_opt))

    # Moderate
    mult_mod = {
        "P_prod": 1 + 0.5*ps,
        "C_RM":   1 - 0.5*cs,
        "C_UT":   1 - 0.5*cs,
        "C_PE":   1 - 0.5*xs,
        "E_CO2":  1 - 0.5*es,
        "f_risk_op": 1 - 0.5*rs,
        "delta_risk": 1 - 0.5*rs,
        "f_esg":  1 - 0.5*esg,
    }
    p_mod = apply_mult(params, mult_mod)
    scenarios["Moderate"] = (p_mod, compute_TEA(p_mod))

    # Pessimistic
    mult_pess = {
        "P_prod": 1 - ps,
        "C_RM":   1 + cs,
        "C_UT":   1 + cs,
        "C_PE":   1 + xs,
        "E_CO2":  1 + es,
        "f_risk_op": 1 + rs,
        "delta_risk": 1 + rs,
        "f_esg":  1 + esg,
    }
    p_pess = apply_mult(params, mult_pess)
    scenarios["Pessimistic"] = (p_pess, compute_TEA(p_pess))

    # ---- Profit metrics per scenario (average annual net income) ----
    avg_profit = {}
    for name, (p_s, res_s) in scenarios.items():
        N = p_s['N_project']
        EBT_sched = res_s["EBT_schedule"]
        tax_sched = res_s["tax_schedule"]
        if N >= 1:
            net_income = EBT_sched[1:N+1] - tax_sched[1:N+1]
            avg_profit[name] = float(np.mean(net_income))
        else:
            avg_profit[name] = 0.0

    base_avg_profit = avg_profit.get("Base", list(avg_profit.values())[0])

    # ---- Tabular summary ----
    rows = []
    for name, (p_s, res) in scenarios.items():
        pb = compute_payback(res["CF"])
        avg_pi = avg_profit[name]
        if base_avg_profit != 0.0:
            delta_pi_pct = (avg_pi - base_avg_profit) / abs(base_avg_profit) * 100.0
        else:
            delta_pi_pct = 0.0
        rows.append({
            "Scenario": name,
            "CAPEX": res["CAPEX"],
            "LCOx": res["LCOx"],
            "NPV": res["NPV"],
            "IRR_%": res["IRR"]*100,
            "BCR": res["BCR"],
            "PV_revenue": res["PV_revenue"],
            "PV_cost_total": res["PV_cost_total"],
            "Payback": pb,
            "Avg_Profit": avg_pi,
            "Delta_Profit_%": delta_pi_pct,
        })

    df = pd.DataFrame(rows)
    print("\nScenario TEA + Cost–Benefit summary (incl. Avg Profit and ΔProfit%):")
    with pd.option_context('display.max_columns', None):
        print(df.to_string(index=False,
                           float_format=lambda x: f"{x:,.3g}"))

    # ---- Visualisations for scenarios (multi-colour) ----
    order = ["Pessimistic", "Moderate", "Base", "Optimistic"]
    order = [o for o in order if o in scenarios.keys()]

    caps = [scenarios[n][1]["CAPEX"] for n in order]
    lcoxs = [scenarios[n][1]["LCOx"] for n in order]
    npvs = [scenarios[n][1]["NPV"] for n in order]
    irrs = [scenarios[n][1]["IRR"]*100 for n in order]
    bcrs = [scenarios[n][1]["BCR"] for n in order]
    avg_pi_plot = [avg_profit[n] for n in order]

    idx = np.arange(len(order))
    colors = ["tab:red", "tab:orange", "tab:blue", "tab:green"]

    # NPV per scenario
    plt.figure(figsize=(8, 4), dpi=200)
    plt.bar(idx, npvs, color=[colors[i % len(colors)] for i in range(len(idx))])
    plt.axhline(0.0, linestyle="--", color="black", alpha=0.7)
    plt.xticks(idx, order)
    plt.ylabel("NPV (USD)")
    plt.title(f"Scenario NPV – {design_label}")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()

    # IRR per scenario
    plt.figure(figsize=(8, 4), dpi=200)
    plt.bar(idx, irrs, color=[colors[i % len(colors)] for i in range(len(idx))])
    plt.xticks(idx, order)
    plt.ylabel("IRR (%)")
    plt.title(f"Scenario IRR – {design_label}")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()

    # LCOx per scenario
    plt.figure(figsize=(8, 4), dpi=200)
    plt.bar(idx, lcoxs, color=[colors[i % len(colors)] for i in range(len(idx))])
    plt.xticks(idx, order)
    plt.ylabel("LCOx (USD/ton)")
    plt.title(f"Scenario LCOx – {design_label}")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()

    # BCR per scenario
    plt.figure(figsize=(8, 4), dpi=200)
    plt.bar(idx, bcrs, color=[colors[i % len(colors)] for i in range(len(idx))])
    plt.axhline(1.0, linestyle="--", color="black", alpha=0.7, label="BCR = 1")
    plt.xticks(idx, order)
    plt.ylabel("Benefit–Cost Ratio (-)")
    plt.title(f"Scenario Benefit–Cost Ratio – {design_label}")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Average Profit per scenario (incl. ESG)
    plt.figure(figsize=(8, 4), dpi=200)
    plt.bar(idx, avg_pi_plot, color=[colors[i % len(colors)] for i in range(len(idx))])
    plt.axhline(0.0, linestyle="--", color="black", alpha=0.7)
    plt.xticks(idx, order)
    plt.ylabel("Average annual net profit (USD/year)")
    plt.title(f"Scenario average net profit (incl. ESG cost) – {design_label}")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


# ======================================================================
#  Parameter input – Commercial TEA
# ======================================================================
def get_params_for_design_commercial(idx):
    print(f"\n================= DESIGN {idx} (Commercial TEA) =================")
    label = input(f"Name/label for this design (e.g., A, ATR, OxySteam) [default Design_{idx}]: ").strip()
    if label == "":
        label = f"Design_{idx}"

    params = {}

    # Required base inputs (direct annual plant numbers)
    params['C_PE']   = ask_float("Purchased Equipment Cost (USD) [default 1e8, min 1e6]: ",
                                 default=1e8, min_val=1e6)
    params['COL']    = ask_float("Annual Operating Labour Cost (USD/year) [default 1e7, min 1e5]: ",
                                 default=1e7, min_val=1e5)
    params['C_RM']   = ask_float("Raw Material Cost (USD/year) [default 4e7, min 1e5]: ",
                                 default=4e7, min_val=1e5)
    params['C_UT']   = ask_float("Utilities Cost (USD/year) [default 1.2e7, min 1e4]: ",
                                 default=1.2e7, min_val=1e4)
    params['C_CAT']  = ask_float("Catalyst Make-up Cost (USD/year) [default 2e6, min 0]: ",
                                 default=2e6, min_val=0.0)
    params['Q_prod'] = ask_float("Annual Product Output (ton/year) [default 5e5, min 1e3]: ",
                                 default=5e5, min_val=1e3)
    params['P_prod'] = ask_float("Selling Price (USD/ton) [default 550, min 50]: ",
                                 default=550, min_val=50)

    # Capital multipliers
    print("\n--- CAPITAL FRACTIONS (Typical Defaults & Bounds) ---")
    params['f_ins']   = ask_float("Installation Factor [default 0.30, min 0.0, max 3.0]: ",
                                  default=0.30, min_val=0.0, max_val=3.0)
    params['f_pipe']  = ask_float("Piping Factor [0.45, min 0.0, max 3.0]: ",
                                  default=0.45, min_val=0.0, max_val=3.0)
    params['f_elec']  = ask_float("Electrical Factor [0.10, min 0.0, max 1.0]: ",
                                  default=0.10, min_val=0.0, max_val=1.0)
    params['f_bldg']  = ask_float("Buildings Factor [0.15, min 0.0, max 1.0]: ",
                                  default=0.15, min_val=0.0, max_val=1.0)
    params['f_util']  = ask_float("Utilities Factor [0.60, min 0.0, max 3.0]: ",
                                  default=0.60, min_val=0.0, max_val=3.0)
    params['f_stor']  = ask_float("Storage Factor [0.10, min 0.0, max 1.0]: ",
                                  default=0.10, min_val=0.0, max_val=1.0)
    params['f_safe']  = ask_float("Safety Factor [0.05, min 0.0, max 1.0]: ",
                                  default=0.05, min_val=0.0, max_val=1.0)
    params['f_waste'] = ask_float("Waste Treatment Factor [0.10, min 0.0, max 2.0]: ",
                                  default=0.10, min_val=0.0, max_val=2.0)

    # Indirects
    params['f_eng']   = ask_float("Engineering Fraction [0.20, min 0.0, max 0.5]: ",
                                  default=0.20, min_val=0.0, max_val=0.5)
    params['f_cons']  = ask_float("Construction Supervision [0.10, min 0.0, max 0.3]: ",
                                  default=0.10, min_val=0.0, max_val=0.3)
    params['f_licn']  = ask_float("Licensing [% of CAPEX] [0.02, min 0.0, max 0.1]: ",
                                  default=0.02, min_val=0.0, max_val=0.1)
    params['f_cont']  = ask_float("Contractor Overhead+Profit [0.10, min 0.0, max 0.3]: ",
                                  default=0.10, min_val=0.0, max_val=0.3)
    params['f_contg'] = ask_float("Contingency [0.25, min 0.0, max 0.5]: ",
                                  default=0.25, min_val=0.0, max_val=0.5)
    params['f_insur'] = ask_float("Insurance [0.02, min 0.0, max 0.1]: ",
                                  default=0.02, min_val=0.0, max_val=0.1)
    params['f_own']   = ask_float("Owner’s Cost [0.05, min 0.0, max 0.2]: ",
                                  default=0.05, min_val=0.0, max_val=0.2)
    params['f_start'] = ask_float("Start-up & Commissioning [0.08, min 0.0, max 0.3]: ",
                                  default=0.08, min_val=0.0, max_val=0.3)

    # FINANCE, TAX, ASSET LIFE & PLANT LIFE
    print("\n--- FINANCE, TAX, ASSET LIFE & PLANT LIFE ---")

    params['L_asset'] = ask_float(
        "Asset / depreciation life (years) [default 20, min 1]: ",
        default=20, min_val=1
    )

    params['tau_inc'] = ask_float(
        "Income Tax Fraction (0-1) [default 0.30]: ",
        default=0.30, min_val=0.0, max_val=1.0
    )

    params['i_base'] = ask_float(
        "Base Discount Rate (decimal, e.g., 0.1=10%) [default 0.08, min 0, max 0.5]: ",
        default=0.08, min_val=0.0, max_val=0.5
    )

    params['delta_risk'] = ask_float(
        "Risk Premium Addition (extra on discount rate) [default 0.02, min 0, max 0.2]: ",
        default=0.02, min_val=0.0, max_val=0.2
    )

    params['N_project'] = int(ask_float(
        "Plant operating life / project horizon (years) [default 20, min 1]: ",
        default=20, min_val=1
    ))

    # Depreciation method(s)
    dep_raw = input(
        "Depreciation method(s) [SL=Straight Line, SYD=Sum-of-years, DDB=Double-declining, "
        "comma-separated or 'ALL'] (default SL): "
    )
    dep_raw_u = dep_raw.strip().upper()
    if dep_raw_u == "" or dep_raw_u is None:
        dep_methods = ["SL"]
    else:
        if dep_raw_u == "ALL":
            dep_methods = ["SL", "SYD", "DDB"]
        else:
            dep_methods = [m for m in [x.strip() for x in dep_raw_u.split(",")]
                           if m in ("SL", "SYD", "DDB")]
            if not dep_methods:
                dep_methods = ["SL"]
    params['dep_methods'] = dep_methods
    params['dep_method'] = dep_methods[0]  # primary method used for visuals

    # Salvage
    params['salv_frac'] = ask_float(
        "Salvage value at end of project (fraction of CAPEX, 0-1) [default 0.1]: ",
        default=0.10, min_val=0.0, max_val=1.0
    )

    # Risk & carbon (direct, plant-level)
    params['f_risk_op'] = ask_float(
        "Operational Risk Factor [default 0.05, min 0, max 0.5]: ",
        default=0.05, min_val=0.0, max_val=0.5
    )
    params['tau_CO2'] = ask_float(
        "Carbon Tax (USD/ton CO₂) [default 50, min 0, max 500]: ",
        default=50, min_val=0.0, max_val=500.0
    )
    params['E_CO2'] = ask_float(
        "Annual CO₂ Emissions (ton/year) [default 200000, min 0]: ",
        default=200000, min_val=0.0
    )
    params['f_pack'] = ask_float(
        "Packaging % of RM cost [Default 0.02, min 0, max 0.2]: ",
        default=0.02, min_val=0.0, max_val=0.2
    )

    # ESG / compliance / regulatory cost
    print("\n--- ESG / Compliance / Regulatory Operating Cost ---")
    print("Select region for default ESG/compliance cost as % of operating cost:")
    print("  1 = North America  (typical 5–10% of OPEX)")
    print("  2 = Europe / UK    (typical 5–10% of OPEX)")
    print("  3 = Asia-Pacific   (often >10% of OPEX)")
    print("  4 = Other / custom region")
    reg_choice = input("Region choice [1/2/3/4, default 3]: ").strip()
    if reg_choice == "1":
        region_name = "North America"
        default_esg_frac = 0.07
    elif reg_choice == "2":
        region_name = "Europe / UK"
        default_esg_frac = 0.07
    elif reg_choice == "4":
        region_name = "Other / custom"
        default_esg_frac = 0.05
    else:
        region_name = "Asia-Pacific"
        default_esg_frac = 0.12

    params['region_esg'] = region_name
    params['f_esg'] = ask_float(
        f"Compliance & ESG / regulatory cost as fraction of operating cost "
        f"for {region_name} [default {default_esg_frac:.2f}, 0–0.5]: ",
        default=default_esg_frac, min_val=0.0, max_val=0.5
    )

    params['mode'] = "commercial"
    return label, params


# ======================================================================
#  Parameter input – Ex-ante TEA
# ======================================================================
def get_params_for_design_exante(idx):
    print(f"\n================= DESIGN {idx} (Ex-ante TEA) =================")
    label = input(f"Name/label for this ex-ante design (e.g., Pilot→Commercial, FOAK) [default ExAnte_{idx}]: ").strip()
    if label == "":
        label = f"ExAnte_{idx}"

    params = {}

    # EX-ANTE CAPEX SCALING
    print("\n--- EX-ANTE CAPEX SCALING (from reference plant / pilot) ---")
    Q_ref = ask_float(
        "Reference plant capacity (ton/year) [default 1e5]: ",
        default=1e5, min_val=1.0
    )
    C_PE_ref = ask_float(
        "Reference purchased equipment cost at Q_ref (USD) [default 5e7]: ",
        default=5e7, min_val=1e6
    )
    n_capex = ask_float(
        "CAPEX scaling exponent (0.5–0.9; default 0.6): ",
        default=0.6, min_val=0.3, max_val=1.0
    )
    foak_mult = ask_float(
        "FOAK / novelty factor on CAPEX (e.g. 1.2 for +20%) [default 1.2]: ",
        default=1.2, min_val=1.0, max_val=3.0
    )

    Q_design = ask_float(
        "Target commercial capacity (ton/year) [default 5e5]: ",
        default=5e5, min_val=1e3
    )

    params['Q_ref'] = Q_ref
    params['C_PE_ref'] = C_PE_ref
    params['n_capex'] = n_capex
    params['foak_mult'] = foak_mult

    C_PE_scaled = C_PE_ref * (Q_design / Q_ref)**n_capex * foak_mult
    print(f"--> Ex-ante scaled C_PE ≈ {C_PE_scaled:,.2f} USD")

    params['C_PE'] = C_PE_scaled
    params['Q_prod'] = Q_design

    # EX-ANTE OPEX INTENSITIES
    print("\n--- EX-ANTE OPEX INTENSITIES (per ton of product) ---")
    c_lab = ask_float(
        "Labour cost intensity (USD/ton product) [default 20]: ",
        default=20, min_val=0.0
    )
    c_rm = ask_float(
        "Raw material cost intensity (USD/ton product) [default 300]: ",
        default=300, min_val=0.0
    )
    c_ut = ask_float(
        "Utilities cost intensity (USD/ton product) [default 50]: ",
        default=50, min_val=0.0
    )
    c_cat = ask_float(
        "Catalyst/membrane/etc. cost intensity (USD/ton product) [default 10]: ",
        default=10, min_val=0.0
    )

    P_prod = ask_float(
        "Expected selling price (USD/ton) [default 550]: ",
        default=550, min_val=50
    )

    params['COL']    = c_lab * params['Q_prod']
    params['C_RM']   = c_rm  * params['Q_prod']
    params['C_UT']   = c_ut  * params['Q_prod']
    params['C_CAT']  = c_cat * params['Q_prod']
    params['P_prod'] = P_prod

    # Capital multipliers
    print("\n--- CAPITAL FRACTIONS (Typical Defaults & Bounds) ---")
    params['f_ins']   = ask_float("Installation Factor [default 0.30, min 0.0, max 3.0]: ",
                                  default=0.30, min_val=0.0, max_val=3.0)
    params['f_pipe']  = ask_float("Piping Factor [0.45, min 0.0, max 3.0]: ",
                                  default=0.45, min_val=0.0, max_val=3.0)
    params['f_elec']  = ask_float("Electrical Factor [0.10, min 0.0, max 1.0]: ",
                                  default=0.10, min_val=0.0, max_val=1.0)
    params['f_bldg']  = ask_float("Buildings Factor [0.15, min 0.0, max 1.0]: ",
                                  default=0.15, min_val=0.0, max_val=1.0)
    params['f_util']  = ask_float("Utilities Factor [0.60, min 0.0, max 3.0]: ",
                                  default=0.60, min_val=0.0, max_val=3.0)
    params['f_stor']  = ask_float("Storage Factor [0.10, min 0.0, max 1.0]: ",
                                  default=0.10, min_val=0.0, max_val=1.0)
    params['f_safe']  = ask_float("Safety Factor [0.05, min 0.0, max 1.0]: ",
                                  default=0.05, min_val=0.0, max_val=1.0)
    params['f_waste'] = ask_float("Waste Treatment Factor [0.10, min 0.0, max 2.0]: ",
                                  default=0.10, min_val=0.0, max_val=2.0)

    # Indirects
    params['f_eng']   = ask_float("Engineering Fraction [0.20, min 0.0, max 0.5]: ",
                                  default=0.20, min_val=0.0, max_val=0.5)
    params['f_cons']  = ask_float("Construction Supervision [0.10, min 0.0, max 0.3]: ",
                                  default=0.10, min_val=0.0, max_val=0.3)
    params['f_licn']  = ask_float("Licensing [% of CAPEX] [0.02, min 0.0, max 0.1]: ",
                                  default=0.02, min_val=0.0, max_val=0.1)
    params['f_cont']  = ask_float("Contractor Overhead+Profit [0.10, min 0.0, max 0.3]: ",
                                  default=0.10, min_val=0.0, max_val=0.3)
    params['f_contg'] = ask_float("Contingency [0.25, min 0.0, max 0.5]: ",
                                  default=0.25, min_val=0.0, max_val=0.5)
    params['f_insur'] = ask_float("Insurance [0.02, min 0.0, max 0.1]: ",
                                  default=0.02, min_val=0.0, max_val=0.1)
    params['f_own']   = ask_float("Owner’s Cost [0.05, min 0.0, max 0.2]: ",
                                  default=0.05, min_val=0.0, max_val=0.2)
    params['f_start'] = ask_float("Start-up & Commissioning [0.08, min 0.0, max 0.3]: ",
                                  default=0.08, min_val=0.0, max_val=0.3)

    # FINANCE, TAX, ASSET LIFE & PLANT LIFE (EX-ANTE)
    print("\n--- FINANCE, TAX, ASSET LIFE & PLANT LIFE (EX-ANTE) ---")

    params['L_asset'] = ask_float(
        "Asset / depreciation life (years) [default 20, min 1]: ",
        default=20, min_val=1
    )

    params['tau_inc'] = ask_float(
        "Income Tax Fraction (0-1) [default 0.30]: ",
        default=0.30, min_val=0.0, max_val=1.0
    )

    print("\n--- TECHNOLOGY RISK (TRL-based for ex-ante) ---")
    TRL = ask_float(
        "Technology Readiness Level (3–9) [default 6]: ",
        default=6, min_val=3, max_val=9
    )
    params['TRL'] = TRL

    params['i_base'] = ask_float(
        "Base Discount Rate (firm-level, decimal) [default 0.08]: ",
        default=0.08, min_val=0.0, max_val=0.5
    )

    if TRL <= 4:
        delta_risk = 0.08
    elif TRL <= 6:
        delta_risk = 0.05
    elif TRL <= 8:
        delta_risk = 0.03
    else:
        delta_risk = 0.01

    print(f"--> Ex-ante risk premium added to discount rate based on TRL: {delta_risk:.3f}")
    params['delta_risk'] = delta_risk

    params['N_project'] = int(ask_float(
        "Plant operating life / project horizon (years) [default 20, min 1]: ",
        default=20, min_val=1
    ))

    # Depreciation method(s) for ex-ante
    dep_raw = input(
        "Depreciation method(s) [SL=Straight Line, SYD=Sum-of-years, DDB=Double-declining, "
        "comma-separated or 'ALL'] (default SL): "
    )
    dep_raw_u = dep_raw.strip().upper()
    if dep_raw_u == "" or dep_raw_u is None:
        dep_methods = ["SL"]
    else:
        if dep_raw_u == "ALL":
            dep_methods = ["SL", "SYD", "DDB"]
        else:
            dep_methods = [m for m in [x.strip() for x in dep_raw_u.split(",")]
                           if m in ("SL", "SYD", "DDB")]
            if not dep_methods:
                dep_methods = ["SL"]
    params['dep_methods'] = dep_methods
    params['dep_method'] = dep_methods[0]

    # Salvage
    params['salv_frac'] = ask_float(
        "Salvage value at end of project (fraction of CAPEX, 0-1) [default 0.1]: ",
        default=0.10, min_val=0.0, max_val=1.0
    )

    # EX-ANTE EMISSIONS & OPERATIONAL RISK
    print("\n--- EX-ANTE EMISSIONS & OPERATIONAL RISK ---")
    e_CO2_int = ask_float(
        "Specific emissions (ton CO₂ / ton product) [default 1.8]: ",
        default=1.8, min_val=0.0
    )
    params['E_CO2'] = e_CO2_int * params['Q_prod']

    params['tau_CO2'] = ask_float(
        "Carbon Tax (USD/ton CO₂) [default 50]: ",
        default=50, min_val=0.0, max_val=500.0
    )

    if TRL <= 4:
        default_f_risk_op = 0.15
    elif TRL <= 6:
        default_f_risk_op = 0.10
    else:
        default_f_risk_op = 0.05

    params['f_risk_op'] = ask_float(
        f"Operational Risk Factor (fraction of DOC) [default {default_f_risk_op:.2f}]: ",
        default=default_f_risk_op, min_val=0.0, max_val=0.5
    )

    params['f_pack'] = ask_float(
        "Packaging % of RM cost [Default 0.02, min 0, max 0.2]: ",
        default=0.02, min_val=0.0, max_val=0.2
    )

    # ESG / compliance / regulatory cost (ex-ante)
    print("\n--- ESG / Compliance / Regulatory Operating Cost (EX-ANTE) ---")
    print("Select region for default ESG/compliance cost as % of operating cost:")
    print("  1 = North America  (typical 5–10% of OPEX)")
    print("  2 = Europe / UK    (typical 5–10% of OPEX)")
    print("  3 = Asia-Pacific   (often >10% of OPEX)")
    print("  4 = Other / custom region")
    reg_choice = input("Region choice [1/2/3/4, default 3]: ").strip()
    if reg_choice == "1":
        region_name = "North America"
        default_esg_frac = 0.07
    elif reg_choice == "2":
        region_name = "Europe / UK"
        default_esg_frac = 0.07
    elif reg_choice == "4":
        region_name = "Other / custom"
        default_esg_frac = 0.05
    else:
        region_name = "Asia-Pacific"
        default_esg_frac = 0.12

    params['region_esg'] = region_name
    params['f_esg'] = ask_float(
        f"Compliance & ESG / regulatory cost as fraction of operating cost "
        f"for {region_name} [default {default_esg_frac:.2f}, 0–0.5]: ",
        default=default_esg_frac, min_val=0.0, max_val=0.5
    )

    params['mode'] = "ex-ante"
    return label, params


# ======================================================================
#  MAIN – multi-design runner
# ======================================================================
print("\n=========== COMMERCIAL / EX-ANTE TEA COMPARATOR ===========")
mode_raw = input("Choose TEA mode: [1] Commercial TEA, [2] Ex-ante TEA (scaled from reference) [default 1]: ").strip()
if mode_raw == "" or mode_raw == "1":
    mode = "commercial"
elif mode_raw == "2":
    mode = "ex-ante"
else:
    mode = "commercial"

n_designs = int(ask_float("How many different designs do you want to compare? [default 1, min 1]: ",
                          default=1, min_val=1))

all_results = []
all_params = []

for i in range(1, n_designs+1):
    if mode == "commercial":
        label, params = get_params_for_design_commercial(i)
    else:
        label, params = get_params_for_design_exante(i)

    while True:
        print("\n================= RUNNING TEA =================")
        # run with primary depreciation method for visuals
        params['dep_method'] = params.get('dep_method', params.get('dep_methods', ["SL"])[0])
        out = compute_TEA(params)
        payback = compute_payback(out["CF"])
        i_eff = params['i_base'] + params['delta_risk']

        # Theoretical IRR from explicit NPV test (same CF), for comparison
        theoretical_irr = theoretical_irr_from_npv(out["CF"])

        print(f"\nRESULTS for {label}")
        print(f"Mode    = {params.get('mode', 'n/a').upper()}")
        if params.get('mode') == "ex-ante":
            print(f"TRL     = {params.get('TRL', 'n/a')}")
            print(f"Q_ref   = {params.get('Q_ref', 'n/a')} ton/yr,  C_PE_ref = {params.get('C_PE_ref', 'n/a')} USD")
            print(f"Scale n = {params.get('n_capex', 'n/a')}, FOAK mult = {params.get('foak_mult', 'n/a')}")
        print(f"Primary depreciation method  : {params['dep_method']}")
        print(f"Region for ESG/compliance    : {params.get('region_esg', 'n/a')}")
        print(f"CAPEX   = {out['CAPEX']:.2f} USD")
        print(f"LCOx    = {out['LCOx']:.3f} USD/ton")
        print(f"NPV     = {out['NPV']:.2f} USD")
        print(f"Project IRR (from TEA CF)    = {out['IRR']*100:.2f}%")
        print(f"Theoretical IRR (NPV=0 test) = {theoretical_irr*100:.2f}%")
        print(f"IRR difference (Proj - Theo) = {(out['IRR']-theoretical_irr)*100:.3f} %-points")
        print(f"Salvage value (end of proj.) : {out['Salvage']:.2f} USD")
        print(f"Capital Recovery Factor (CRF): {out['CRF']:.5f}")
        print(f"Annualised CAPEX (via CRF)   : {out['Annual_CAPEX']:.2f} USD/yr")
        if payback is not None:
            print(f"Payback time ≈ {payback:.2f} years")
        else:
            print("Payback time: not reached within project horizon")
        print(f"PV of Revenues (no tax)      : {out['PV_revenue']:.2f} USD")
        print(f"PV of Total Costs            : {out['PV_cost_total']:.2f} USD")
        print(f"Benefit–Cost Ratio (BCR)     : {out['BCR']:.3f}")

        esg_cost = out.get("esg_cost", 0.0)
        oc_base = out.get("OC_base", out["OC"])
        esg_pct = (esg_cost / oc_base * 100.0) if oc_base > 0 else 0.0
        print(f"ESG/compliance cost (OPEX add-on): {esg_cost:.2f} USD/yr "
              f"(~{esg_pct:.2f}% of base OPEX)")

        # --- Extra check: required selling price for a target "good" IRR ---
        run_price_target = input(
            "\nCompute selling price required to reach a target IRR? [y/N]: "
        ).strip().lower()
        if run_price_target == "y":
            target_irr_pct = ask_float(
                "Target IRR (%) you consider 'good' [default 15]: ",
                default=15.0, min_val=-50.0, max_val=200.0
            )
            target_irr = target_irr_pct / 100.0
            p_req, out_req = price_for_target_irr(params, target_irr)
            if p_req is None or out_req is None:
                print("  Could not bracket a selling price that reaches the target IRR with current settings.")
            else:
                print(f"\nRequired selling price for IRR ≈ {target_irr_pct:.2f}%:")
                print(f"  P_prod ≈ {p_req:.3f} USD/ton")
                print(f"  -> IRR ≈ {out_req['IRR']*100:.2f}%")
                print(f"  -> LCOx ≈ {out_req['LCOx']:.3f} USD/ton")
                print(f"  -> NPV  ≈ {out_req['NPV']:.2f} USD")

        # ---- Depreciation-method comparison table (if multiple) ----
        dep_methods = params.get('dep_methods', [params['dep_method']])
        if len(dep_methods) > 1:
            rows_dm = []
            for m in dep_methods:
                p_m = params.copy()
                p_m['dep_method'] = m
                out_m = compute_TEA(p_m)
                pb_m = compute_payback(out_m["CF"])
                rows_dm.append({
                    "Dep_Method": m,
                    "NPV": out_m["NPV"],
                    "IRR_%": out_m["IRR"]*100.0,
                    "Payback_yrs": pb_m,
                    "Annual_CAPEX": out_m["Annual_CAPEX"]
                })
            df_dm = pd.DataFrame(rows_dm)
            print("\nDepreciation-method comparison (same design economics):")
            with pd.option_context('display.max_columns', None):
                print(df_dm.to_string(index=False,
                                      float_format=lambda x: f"{x:,.3g}"))

        # ---- Income Statement per year (using primary method) ----
        N = params['N_project']
        years = np.arange(1, N+1)
        dep_sched = out["dep_schedule"]
        EBT_sched = out["EBT_schedule"]
        tax_sched = out["tax_schedule"]

        revenue = np.full_like(years, out["R"], dtype=float)
        opex = np.full_like(years, out["OC"], dtype=float)
        depreciation = dep_sched[1:N+1]
        ebit = EBT_sched[1:N+1]           # EBIT = EBT here (no interest terms)
        tax = tax_sched[1:N+1]
        net_income = ebit - tax
        ebitda = revenue - opex           # EBITDA = Revenue - OPEX

        df_is = pd.DataFrame({
            "Year": years,
            "Revenue": revenue,
            "OPEX": opex,
            "EBITDA": ebitda,
            "Depreciation": depreciation,
            "EBIT": ebit,
            "Tax": tax,
            "Net_Income": net_income
        })

        print(f"\nIncome Statement (per operating year) – {label} "
              f"(primary depreciation method: {params['dep_method']})")
        with pd.option_context('display.max_rows', None,
                               'display.max_columns', None):
            print(df_is.to_string(index=False,
                                  float_format=lambda x: f"{x:,.3g}"))

        # ---- ROI CALCULATIONS ----
        CAPEX_tot = out["CAPEX"]
        total_net_income = float(np.sum(net_income))
        avg_net_income = float(np.mean(net_income))

        ROI_total = total_net_income / CAPEX_tot if CAPEX_tot != 0 else 0.0
        ROI_avg = avg_net_income / CAPEX_tot if CAPEX_tot != 0 else 0.0
        ROI_yearly_pct = (net_income / CAPEX_tot * 100.0) if CAPEX_tot != 0 else np.zeros_like(net_income)

        print(f"\nROI metrics (based on Net Income and total CAPEX):")
        print(f"  Total ROI over project life (Σ Net Income / CAPEX) : {ROI_total*100:.2f}%")
        print(f"  Average annual ROI (mean Net Income / CAPEX)       : {ROI_avg*100:.2f}%")

        # ---- Financial visual: multi-colour Income Statement ----
        plt.figure(figsize=(9, 5), dpi=200)
        bar_width = 0.7
        plt.bar(years, revenue, width=bar_width, alpha=0.5,
                label="Revenue", color="tab:blue", edgecolor="black", linewidth=0.4)
        plt.bar(years, opex, width=bar_width*0.7, alpha=0.6,
                label="OPEX", color="tab:orange", edgecolor="black", linewidth=0.4)
        plt.bar(years, depreciation, width=bar_width*0.4, alpha=0.7,
                label="Depreciation", color="tab:red", edgecolor="black", linewidth=0.4)
        plt.plot(years, net_income, marker="o", linewidth=2.0,
                 color="tab:green", label="Net Income")

        plt.axhline(0.0, linestyle="--", color="black", alpha=0.7)
        plt.xlabel("Year")
        plt.ylabel("USD/year")
        plt.title(f"Income Statement Profile – {label}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

        # ---- EBIT & EBITDA GRAPH ----
        plt.figure(figsize=(9, 5), dpi=200)
        plt.plot(years, ebitda, marker="o", linewidth=2.0,
                 label="EBITDA", color="tab:purple")
        plt.plot(years, ebit, marker="s", linewidth=2.0,
                 label="EBIT", color="tab:blue")
        plt.plot(years, net_income, marker="^", linewidth=2.0,
                 label="Net Income", color="tab:green")
        plt.axhline(0.0, linestyle="--", color="black", alpha=0.7)
        plt.xlabel("Year")
        plt.ylabel("USD/year")
        plt.title(f"EBITDA / EBIT / Net Income – {label}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

        # ---- ROI GRAPH ----
        plt.figure(figsize=(7, 4), dpi=200)
        plt.bar(years, ROI_yearly_pct, color="tab:purple", edgecolor="black", linewidth=0.4)
        plt.axhline(0.0, linestyle="--", color="black", alpha=0.7)
        plt.xlabel("Year")
        plt.ylabel("ROI per year (% of CAPEX)")
        plt.title(f"Yearly ROI (Net Income / CAPEX) – {label}")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        # ---- NPV vs Discount Rate with Project IRR vs Theoretical IRR ----
        CF_arr = np.array(out["CF"], dtype=float)
        r_min = -0.5
        r_max = max(0.5, out["IRR"]*3 + 0.1)
        r_grid = np.linspace(r_min, r_max, 300)

        def npv_at_rate(cf, r):
            r_eff = max(r, -0.9999)
            t = np.arange(cf.size, dtype=float)
            return np.sum(cf / (1.0 + r_eff)**t)

        npv_grid = [npv_at_rate(CF_arr, r) for r in r_grid]

        plt.figure(figsize=(8, 4), dpi=200)
        plt.plot(r_grid*100.0, npv_grid, label="NPV vs discount rate")
        plt.axhline(0.0, linestyle="--", color="black", alpha=0.7)
        plt.axvline(out["IRR"]*100.0, linestyle="-", color="tab:red",
                    label="Project IRR (from TEA)")
        plt.axvline(theoretical_irr*100.0, linestyle="--", color="tab:blue",
                    label="Theoretical IRR (NPV=0)")
        plt.xlabel("Discount rate (%)")
        plt.ylabel("NPV (USD)")
        plt.title(f"NPV vs Discount Rate – {label}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

        years_cf = np.arange(len(out["CF"]))

        # Annual CF (multi-colour bar: CAPEX, +CF, -CF)
        colors_cf = []
        for t, v in enumerate(out["CF"]):
            if t == 0:
                colors_cf.append("tab:gray")
            elif v >= 0:
                colors_cf.append("tab:green")
            else:
                colors_cf.append("tab:red")

        plt.figure(figsize=(7, 4), dpi=200)
        plt.bar(years_cf, out["CF"], color=colors_cf, edgecolor="black", linewidth=0.4)
        plt.axhline(0.0, linestyle="--", color="black", alpha=0.7)
        plt.grid(True, alpha=0.3)
        plt.title(f"Annual Cashflow – {label}")
        plt.xlabel("Year")
        plt.ylabel("Cashflow (USD)")
        plt.tight_layout()
        plt.show()

        # Cumulative CF + payback (engineering style)
        cum_cf = np.cumsum(out["CF"])
        plt.figure(figsize=(7, 4), dpi=200)
        plt.plot(years_cf, cum_cf, marker="o", label="Cumulative CF", color="tab:blue")
        plt.axhline(0.0, linestyle="--", color="black", alpha=0.7)
        if payback is not None:
            plt.axvline(payback, linestyle="--", color="tab:red", label=f"Payback ≈ {payback:.2f} yr")
        plt.grid(True, alpha=0.3)
        plt.title(f"Cumulative Cashflow – {label}")
        plt.xlabel("Year")
        plt.ylabel("Cumulative CF (USD)")
        if payback is not None:
            plt.legend()
        plt.tight_layout()
        plt.show()

        # Discounted CF and cumulative discounted CF
        disc_cf = [out["CF"][t] / ((1 + i_eff)**t) for t in range(len(out["CF"]))]
        cum_disc = np.cumsum(disc_cf)
        plt.figure(figsize=(7, 4), dpi=200)
        plt.plot(years_cf, disc_cf, marker="o", label="Discounted CF", color="tab:purple")
        plt.plot(years_cf, cum_disc, marker="s", label="Cum. discounted CF", color="tab:green")
        plt.axhline(0.0, linestyle="--", color="black", alpha=0.7)
        plt.grid(True, alpha=0.3)
        plt.title(f"Discounted Cashflow Profile – {label}")
        plt.xlabel("Year")
        plt.ylabel("USD")
        plt.legend()
        plt.tight_layout()
        plt.show()

        # --- CAPEX breakdown (engineering visual) ---
        C_PE = out["C_PE"]
        C_INS   = params['f_ins']   * C_PE
        C_PIPE  = params['f_pipe']  * C_PE
        C_ELEC  = params['f_elec']  * C_PE
        C_BLDG  = params['f_bldg']  * C_PE
        C_UTIL  = params['f_util']  * C_PE
        C_STOR  = params['f_stor']  * C_PE
        C_SAFE  = params['f_safe']  * C_PE
        C_WASTE = params['f_waste'] * C_PE

        DCC = C_PE + C_INS + C_PIPE + C_ELEC + C_BLDG + C_UTIL + C_STOR + C_SAFE + C_WASTE
        C_ENG   = params['f_eng']   * DCC
        C_CONS  = params['f_cons']  * DCC
        C_LICN  = params['f_licn']  * DCC
        C_CONT  = params['f_cont']  * DCC
        C_CONTG = params['f_contg'] * DCC
        C_INSUR = params['f_insur'] * DCC
        C_OWN   = params['f_own']   * DCC
        C_START = params['f_start'] * DCC

        cap_labels = [
            "C_PE", "Installation", "Piping", "Electrical", "Buildings", "Utilities",
            "Storage", "Safety", "Waste treat.",
            "Eng.", "Constr. Superv.", "Licensing", "Contractor O/H+Profit",
            "Contingency", "Insurance", "Owner’s cost", "Start-up"
        ]
        cap_vals = [
            C_PE, C_INS, C_PIPE, C_ELEC, C_BLDG, C_UTIL,
            C_STOR, C_SAFE, C_WASTE,
            C_ENG, C_CONS, C_LICN, C_CONT,
            C_CONTG, C_INSUR, C_OWN, C_START
        ]

        cap_pairs = sorted(zip(cap_vals, cap_labels), key=lambda z: z[0], reverse=True)
        sorted_vals, sorted_labels = zip(*cap_pairs)

        y_pos = np.arange(len(sorted_labels))
        plt.figure(figsize=(8, 6), dpi=200)
        plt.barh(y_pos, sorted_vals, color="tab:blue", alpha=0.8)
        plt.xlabel("Cost (USD)")
        plt.title(f"CAPEX breakdown – {label}")
        plt.yticks(y_pos, sorted_labels)
        plt.gca().invert_yaxis()
        plt.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        plt.show()

        # --- OPEX breakdown pie (DOC, FOC, GMC, risk, CO₂, ESG) ---
        DOC = out["DOC"]
        FOC = out["FOC"]
        GMC = out["GMC"]
        risk_cost = out["risk_cost"]
        co2_cost = out["co2_cost"]
        esg_cost = out.get("esg_cost", 0.0)

        opex_labels = ["DOC", "FOC", "GMC", "Op. risk", "CO₂ cost", "ESG & compliance"]
        opex_vals = [DOC, FOC, GMC, risk_cost, co2_cost, esg_cost]

        plt.figure(figsize=(6, 6), dpi=200)
        plt.pie(
            opex_vals,
            labels=opex_labels,
            autopct="%1.1f%%",
            startangle=90,
        )
        plt.title(f"OPEX breakdown – {label}")
        plt.tight_layout()
        plt.show()

        # === Sensitivity analysis (tornado, with user swing) ===
        run_sens = input("\nRun sensitivity (tornado) for this design? [y/N]: ").strip().lower()
        if run_sens == "y":
            swing_pct = ask_float("Sensitivity +/- percent (e.g., 20 for ±20%) [default 20]: ",
                                  default=20, min_val=0.0, max_val=200.0)
            swing = swing_pct / 100.0

            print("\nDefault sensitivity parameters:")
            default_keys = ['C_PE', 'C_RM', 'C_UT', 'COL', 'Q_prod', 'P_prod', 'tau_CO2', 'E_CO2', 'f_esg']
            print("  " + ", ".join(default_keys))
            use_default = input("Use default list? [Y/n]: ").strip().lower()
            if use_default in ("", "y"):
                sens_keys = [k for k in default_keys if k in params]
            else:
                raw_list = input("Enter comma-separated parameter names (must be keys in params): ")
                sens_keys = [x.strip() for x in raw_list.split(",") if x.strip() in params]
                if not sens_keys:
                    sens_keys = [k for k in default_keys if k in params]

            tornado_sensitivity(params, sens_keys, swing=swing)

        # === Monte Carlo ===
        run_mc = input("\nRun Monte-Carlo uncertainty for this design? [y/N]: ").strip().lower()
        if run_mc == "y":
            interactive_monte_carlo(params)

        # === Price / RM sweeps ===
        run_sweep = input("\nRun price & raw-material sweeps for this design? [y/N]: ").strip().lower()
        if run_sweep == "y":
            print("\n--- Price sweep (NPV & LCOx vs selling price) ---")
            price_sweep(params, swing=0.30, n=25)
            print("\n--- Raw-material cost sweep (NPV & LCOx vs C_RM) ---")
            raw_material_sweep(params, swing=0.30, n=25)

        # === Scenario cost–benefit analysis ===
        run_sc = input("\nRun scenario cost–benefit analysis (optimistic / moderate / pessimistic)? [y/N]: ").strip().lower()
        if run_sc == "y":
            scenario_cba(params, label)

        # === Critical multi-colour pairwise impact visualisation ===
        run_pair = input("\nRun critical multi-colour pairwise impact visualisation? [y/N]: ").strip().lower()
        if run_pair == "y":
            interactive_pairwise_visual(params)

        # === NEW: ESG/compliance sweep 5% -> 75% ===
        run_esg = input("\nRun ESG/compliance sweep 5%→75% (OPEX, IRR, NPV, Price-to-hold-IRR, ROI, Revenue, EBITDA, EBIT, Payback)? [y/N]: ").strip().lower()
        if run_esg == "y":
            # You can tweak n for smoothness (e.g., 15–25)
            run_esg_sweep_and_plots(params, design_label=label, f_min=0.05, f_max=0.75, n=17)

        # After all visuals, ask user if they want to edit/re-run
        print("\nOptions for this design:")
        print("  [1] Change one or more inputs and re-run")
        print("  [2] Accept this design and continue to next")
        choice = input("Enter choice [1/2, default 2]: ").strip()

        if choice == "1":
            params = interactive_edit_params(params)
            continue
        else:
            all_results.append({
                "label": label,
                "CAPEX": out["CAPEX"],
                "LCOx": out["LCOx"],
                "NPV": out["NPV"],
                "IRR": out["IRR"],
                "Payback": payback,
                "BCR": out["BCR"],
                "ROI_total": ROI_total,
                "ROI_avg": ROI_avg
            })
            all_params.append(params)
            break


# ======================================================================
#  Cross-design comparison plots
# ======================================================================
if n_designs > 1:
    labels = [r["label"] for r in all_results]
    idx = np.arange(len(labels))

    CAPEX_vals = [r["CAPEX"] for r in all_results]
    LCOx_vals  = [r["LCOx"] for r in all_results]
    NPV_vals   = [r["NPV"] for r in all_results]
    IRR_vals   = [r["IRR"]*100.0 for r in all_results]
    PB_vals    = [r["Payback"] if r["Payback"] is not None else np.nan
                  for r in all_results]
    BCR_vals   = [r["BCR"] for r in all_results]
    ROI_totals = [r.get("ROI_total", 0.0)*100.0 for r in all_results]
    ROI_avgs   = [r.get("ROI_avg", 0.0)*100.0 for r in all_results]

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",
              "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]

    # CAPEX
    plt.figure(figsize=(8, 4), dpi=200)
    plt.bar(idx, CAPEX_vals, color=[colors[i % len(colors)] for i in range(len(idx))])
    plt.xticks(idx, labels, rotation=45)
    plt.ylabel("CAPEX (USD)")
    plt.title("CAPEX comparison across designs")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()

    # LCOx
    plt.figure(figsize=(8, 4), dpi=200)
    plt.bar(idx, LCOx_vals, color=[colors[(i+2) % len(colors)] for i in range(len(idx))])
    plt.xticks(idx, labels, rotation=45)
    plt.ylabel("LCOx (USD/ton)")
    plt.title("Levelised Cost comparison across designs")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()

    # NPV
    plt.figure(figsize=(8, 4), dpi=200)
    plt.bar(idx, NPV_vals, color=[colors[(i+4) % len(colors)] for i in range(len(idx))])
    plt.xticks(idx, labels, rotation=45)
    plt.ylabel("NPV (USD)")
    plt.title("NPV comparison across designs")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()

    # IRR
    plt.figure(figsize=(8, 4), dpi=200)
    plt.bar(idx, IRR_vals, color=[colors[(i+6) % len(colors)] for i in range(len(idx))])
    plt.xticks(idx, labels, rotation=45)
    plt.ylabel("IRR (%)")
    plt.title("IRR comparison across designs")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Payback
    plt.figure(figsize=(8, 4), dpi=200)
    plt.bar(idx, PB_vals, color=[colors[(i+8) % len(colors)] for i in range(len(idx))])
    plt.xticks(idx, labels, rotation=45)
    plt.ylabel("Payback time (years)")
    plt.title("Payback comparison across designs")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()

    # BCR
    plt.figure(figsize=(8, 4), dpi=200)
    plt.bar(idx, BCR_vals, color=[colors[(i+1) % len(colors)] for i in range(len(idx))])
    plt.axhline(1.0, linestyle="--", color="black", alpha=0.7, label="BCR = 1")
    plt.xticks(idx, labels, rotation=45)
    plt.ylabel("Benefit–Cost Ratio (-)")
    plt.title("Benefit–Cost Ratio comparison across designs")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ROI total (Σ Net Income / CAPEX)
    plt.figure(figsize=(8, 4), dpi=200)
    plt.bar(idx, ROI_totals, color=[colors[(i+3) % len(colors)] for i in range(len(idx))])
    plt.xticks(idx, labels, rotation=45)
    plt.ylabel("Total ROI over life (%)")
    plt.title("Total ROI (Σ Net Income / CAPEX) across designs")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Average annual ROI
    plt.figure(figsize=(8, 4), dpi=200)
    plt.bar(idx, ROI_avgs, color=[colors[(i+5) % len(colors)] for i in range(len(idx))])
    plt.xticks(idx, labels, rotation=45)
    plt.ylabel("Average annual ROI (%)")
    plt.title("Average Annual ROI (mean Net Income / CAPEX) across designs")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()
