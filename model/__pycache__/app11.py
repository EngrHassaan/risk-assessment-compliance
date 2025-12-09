# comp-frontend.py
# Streamlit frontend for compliance.py TEA & ESG sweeps (Option 1 tabbed UI)
# Requires: streamlit, pandas, numpy, matplotlib, plotly, reportlab, pillow, kaleido
# Place this file in the same directory as compliance.py

import os
import io
import json
import importlib.util
from datetime import datetime
from typing import Dict, Any, List, Tuple

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import plotly.express as px
from PIL import Image

# ReportLab for PDF export
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table as RLTable, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

# ---------------- Load compliance module dynamically ----------------
MODEL_PATH = "compliance.py"  # file must exist in same dir

def load_compliance_module(path: str):
    if not os.path.exists(path):
        st.error(f"compliance model not found at: {path}. Please upload or place compliance.py here.")
        st.stop()
    spec = importlib.util.spec_from_file_location("compliance_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

mod = load_compliance_module(MODEL_PATH)

# ----------------- Helpers to capture figures & images -----------------
def mpl_fig_to_png_bytes(fig: Figure, dpi: int = 150) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()

def plotly_fig_to_png_bytes(fig, width: int = 900, height: int = 600) -> bytes:
    # requires kaleido installed
    try:
        img_bytes = fig.to_image(format="png", width=width, height=height, scale=2)
        return img_bytes
    except Exception as e:
        # fallback: save via HTML snapshot (not ideal). Raise for now.
        raise

def pil_bytes_to_rlimage(png_bytes: bytes, max_width_mm: float = 170) -> RLImage:
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

def build_pdf_report_basic(title: str, sections: List[Dict[str, Any]]) -> bytes:
    """
    Simple PDF builder using ReportLab.
    sections: [{"heading": str, "text": str, "tables": [(df, caption)], "images":[(png_bytes, caption)]}]
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, rightMargin=18*mm, leftMargin=18*mm, topMargin=18*mm, bottomMargin=18*mm)
    styles = getSampleStyleSheet()
    story = []

    # cover
    style_h = styles["Heading1"]
    style_h.alignment = 1
    story.append(Paragraph(title, style_h))
    story.append(Spacer(1, 6))
    story.append(Paragraph(f"Generated: {datetime.now().isoformat(' ', 'seconds')}", styles["Normal"]))
    story.append(Spacer(1, 12))

    for sec in sections:
        heading = sec.get("heading", "")
        if heading:
            story.append(Paragraph(heading, styles["Heading2"]))
        text = sec.get("text", None)
        if text:
            story.append(Paragraph(text, styles["Normal"]))
            story.append(Spacer(1, 6))

        for tbl, caption in sec.get("tables", []):
            if tbl is None or tbl.empty:
                continue
            story.append(Paragraph(caption or "Table", styles["Italic"]))
            df = tbl.copy()
            df = df.round(6)
            # convert df to list-of-lists including header
            data = [list(df.columns)]
            for row in df.itertuples(index=False):
                data.append([str(x) for x in row])
            # placeable table
            tbl_style = RLTable([list(df.columns)] + data, hAlign="LEFT")
            # fallback simple formatting
            tbl_style = RLTable([list(df.columns)] + data)
            tbl_style.setStyle(TableStyle([
                ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
                ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
                ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
                ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
            ]))
            story.append(tbl_style)
            story.append(Spacer(1, 8))

        for img_bytes, caption in sec.get("images", []):
            try:
                rl_img = pil_bytes_to_rlimage(img_bytes)
                story.append(rl_img)
                if caption:
                    story.append(Paragraph(caption, styles["Italic"]))
                story.append(Spacer(1, 8))
            except Exception as e:
                story.append(Paragraph(f"[Image could not be embedded: {e}]", styles["Normal"]))

        story.append(PageBreak())

    doc.build(story)
    buf.seek(0)
    return buf.getvalue()

# ----------------- Streamlit app state & helpers -----------------
st.set_page_config(page_title="Compliance / TEA Frontend", layout="wide")
st.title("Compliance / TEA Frontend — comp-frontend.py")
st.markdown("UI wrapper around `compliance.py` — TEA, ESG sweeps, sensitivity, Monte Carlo, and exports.")

if "designs" not in st.session_state:
    st.session_state.designs = {}  # label -> params dict
if "last_compute" not in st.session_state:
    st.session_state.last_compute = {}  # label -> latest TEA result dict
if "captured_figs" not in st.session_state:
    st.session_state.captured_figs = {}  # label -> list of (png_bytes, caption)

# Provide a helpful default params template (cover most keys used in compliance.compute_TEA)
def make_default_params() -> Dict[str, Any]:
    # These defaults are sensible guesses; user can edit in the UI
    return {
        "C_PE": 1_000_000.0,
        "f_ins": 0.3,
        "f_pipe": 0.4,
        "f_elec": 0.05,
        "f_bldg": 0.05,
        "f_util": 0.03,
        "f_stor": 0.02,
        "f_safe": 0.02,
        "f_waste": 0.01,
        "f_eng": 0.15,
        "f_cons": 0.05,
        "f_licn": 0.02,
        "f_cont": 0.10,
        "f_contg": 0.10,
        "f_insur": 0.01,
        "f_own": 0.02,
        "f_start": 0.03,
        "COL": 200_000.0,
        "C_RM": 500_000.0,
        "C_UT": 80_000.0,
        "C_CAT": 10_000.0,
        "f_pack": 0.02,
        "Q_prod": 10_000.0,
        "P_prod": 200.0,
        "f_risk_op": 0.05,
        "tau_CO2": 0.0,
        "E_CO2": 0.0,
        "f_esg": 0.05,
        "i_base": 0.08,
        "delta_risk": 0.02,
        "dep_method": "SL",
        "salv_frac": 0.05,
        "N_project": 20,
        "L_asset": 20,
        "tau_inc": 0.25,
    }

# UI helper: add design from JSON or manual edit
def add_design_from_json(json_obj: Dict[str, Any], label: str):
    st.session_state.designs[label] = json_obj
    st.success(f"Stored design: {label}")

# ----------------- Sidebar quick actions -----------------
st.sidebar.header("Quick actions")
if st.sidebar.button("Load example design"):
    label = f"Example Design {len(st.session_state.designs)+1}"
    add_design_from_json(make_default_params(), label)
if st.sidebar.button("Clear all designs"):
    st.session_state.designs = {}
    st.session_state.last_compute = {}
    st.session_state.captured_figs = {}
    st.experimental_rerun()

st.sidebar.markdown("---")
st.sidebar.write("Note: This frontend uses functions from `compliance.py` for calculations and plotting.")
st.sidebar.write("Dependencies: streamlit, pandas, numpy, matplotlib, plotly, reportlab, pillow, kaleido")

# ----------------- Tabs (Option 1) -----------------
tabs = st.tabs(["Input & Designs", "TEA Results", "Scenario Analysis", "Sensitivity", "Sweeps & ESG", "Export Reports"])

# ----------------- Tab 0: Input & Designs -----------------
with tabs[0]:
    st.header("Input & Designs")
    st.markdown("Add a new design (JSON upload / paste) or use the manual editor to create/edit a design.")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Add design from JSON file or paste JSON")
        uploaded = st.file_uploader("Upload design JSON file (single dict with keys) ", type=["json", "txt"])
        if uploaded:
            try:
                j = json.load(uploaded)
                label = st.text_input("Label for design", f"Design {len(st.session_state.designs)+1}")
                if st.button("Add uploaded design"):
                    add_design_from_json(j, label)
            except Exception as e:
                st.error(f"Failed to parse JSON: {e}")

        st.markdown("or paste JSON below (single dictionary):")
        pasted = st.text_area("Paste design JSON", height=160)
        if pasted.strip():
            label2 = st.text_input("Label for pasted design", f"Design {len(st.session_state.designs)+1}", key="label_paste")
            if st.button("Add pasted design"):
                try:
                    j2 = json.loads(pasted)
                    add_design_from_json(j2, label2)
                except Exception as e:
                    st.error(f"Invalid JSON: {e}")

        st.markdown("---")
        st.subheader("Manual editor (template) — create or edit design")
        name = st.text_input("Design label", f"Design {len(st.session_state.designs)+1}", key="manual_label")
        params = st.session_state.designs.get(name, None)
        if params is None:
            params = make_default_params()
        # break into two columns for compactness
        items = list(params.items())
        left_items = items[0:len(items)//2]
        right_items = items[len(items)//2:]
        with st.form(f"manual_form_{name}", clear_on_submit=False):
            cols = st.columns(2)
            new_params = {}
            for (k, v), col in zip([*left_items, *right_items], cols* ( (len(items)+1)//(2*len(cols)) )):
                pass
            # simpler: build two columns programmatically
            colA, colB = st.columns(2)
            idx = 0
            for k, v in params.items():
                if isinstance(v, (int, float)):
                    if k in ("dep_method",):
                        # will be handled below
                        pass
                idx += 1
            # we'll show most params in a scrollable form: numeric inputs when possible
            st.write("Numeric / string parameters (edit then Submit):")
            edited = {}
            colA, colB = st.columns(2)
            i = 0
            for k, v in params.items():
                target_col = colA if (i % 2 == 0) else colB
                if isinstance(v, (int, float)):
                    if k in ("N_project", "L_asset"):
                        edited[k] = int(target_col.number_input(k, value=int(v), step=1))
                    else:
                        edited[k] = target_col.number_input(k, value=float(v))
                else:
                    if k == "dep_method":
                        edited[k] = target_col.selectbox("dep_method", options=["SL", "SYD", "DDB"], index=["SL","SYD","DDB"].index(v) if v in ["SL","SYD","DDB"] else 0)
                    else:
                        try:
                            edited[k] = target_col.text_input(k, value=str(v))
                        except Exception:
                            edited[k] = v
                i += 1
            submitted = st.form_submit_button("Save design")
            if submitted:
                st.session_state.designs[name] = edited
                st.success(f"Saved design {name}.")

    with col2:
        st.subheader("Existing designs")
        if st.session_state.designs:
            # summary table
            rows = []
            for label, p in st.session_state.designs.items():
                row = {"Label": label, "P_prod": p.get("P_prod", ""), "Q_prod": p.get("Q_prod", ""), "C_PE": p.get("C_PE","")}
                rows.append(row)
            df_summary = pd.DataFrame(rows).set_index("Label")
            st.dataframe(df_summary)
            sel = st.selectbox("Select design to preview / delete / compute", options=list(st.session_state.designs.keys()))
            if st.button("Delete selected design"):
                del st.session_state.designs[sel]
                st.success(f"Deleted {sel}")
            st.markdown("Preview parameters (editable in manual editor):")
            st.json(st.session_state.designs.get(sel, {}))
        else:
            st.info("No designs yet. Upload JSON or use the manual editor or load example.")

# ----------------- Tab 1: TEA Results -----------------
with tabs[1]:
    st.header("TEA Results")
    if not st.session_state.designs:
        st.info("Add at least one design first (Input & Designs tab).")
    else:
        sel = st.selectbox("Select design for TEA & visuals", options=list(st.session_state.designs.keys()))
        params = st.session_state.designs.get(sel, {})
        st.subheader(f"Selected: {sel}")
        st.write("Key parameters snapshot:")
        key_cols = ["C_PE","Q_prod","P_prod","COL","C_RM","C_UT","f_esg","i_base","delta_risk"]
        snap = {k: params.get(k,"") for k in key_cols}
        st.json(snap)

        if st.button("Compute TEA (compute_TEA)"):
            try:
                # call compute_TEA from compliance.py
                out = mod.compute_TEA(params)
                st.session_state.last_compute[sel] = out
                st.success("TEA computed and stored.")
            except Exception as e:
                st.error(f"compute_TEA error: {e}")

        if sel in st.session_state.last_compute:
            out = st.session_state.last_compute[sel]
            st.subheader("Summary metrics")
            metrics = {
                "CAPEX": out.get("CAPEX"),
                "Annual_CAPEX": out.get("Annual_CAPEX"),
                "LCOx": out.get("LCOx"),
                "NPV": out.get("NPV"),
                "IRR": out.get("IRR"),
                "BCR": out.get("BCR"),
            }
            df_metrics = pd.Series(metrics).to_frame("Value")
            st.dataframe(df_metrics)

            st.subheader("Income statement & schedules")
            dep = out.get("dep_schedule")
            ebt = out.get("EBT_schedule")
            tax = out.get("tax_schedule")
            cf = out.get("CF")
            # tabular view
            df_sched = pd.DataFrame({
                "Year": list(range(len(cf))),
                "Cashflow": cf,
                "Depreciation": list(dep) if dep is not None else [None]*len(cf),
                "EBT": list(ebt) if ebt is not None else [None]*len(cf),
                "Tax": list(tax) if tax is not None else [None]*len(cf),
            }).set_index("Year")
            st.dataframe(df_sched.round(4))

            st.subheader("Plots (Matplotlib produced by module)")
            # many plotting funcs inside compliance.py call plt.show(); use pattern to capture new figures
            before = plt.get_fignums()
            try:
                # there is a function to generate many TEA visuals: maybe interactive functions - we'll try a few safe ones
                if hasattr(mod, "plot_esg_sweep_lines"):
                    # not always appropriate here; but compute_TEA may have generated internal visuals
                    pass
            except Exception:
                pass

            # Attempt to generate CAPEX / OPEX pie if compliance.py provides helper
            # 1) CAPEX breakdown (DCC, ICC)
            try:
                fig = plt.figure(figsize=(6,4))
                labels = ["DCC", "ICC"]
                values = [out.get("DCC", 0.0), out.get("ICC", 0.0)]
                plt.pie(values, labels=labels, autopct="%1.1f%%")
                plt.title("CAPEX breakdown")
                st.pyplot(fig)
                st.session_state.captured_figs.setdefault(sel, []).append((mpl_fig_to_png_bytes(fig), "CAPEX breakdown"))
            except Exception:
                pass

            # 2) OPEX pie (DOC, FOC, GMC, risk_cost, co2_cost, esg_cost)
            try:
                fig = plt.figure(figsize=(6,4))
                labels = ["DOC","FOC","GMC","risk_cost","co2_cost","esg_cost"]
                values = [out.get(k,0.0) for k in ["DOC","FOC","GMC","risk_cost","co2_cost","esg_cost"]]
                plt.pie(values, labels=labels, autopct="%1.1f%%")
                plt.title("OPEX breakdown")
                st.pyplot(fig)
                st.session_state.captured_figs.setdefault(sel, []).append((mpl_fig_to_png_bytes(fig), "OPEX breakdown"))
            except Exception:
                pass

            # 3) Cashflow bar & cumulative
            try:
                fig, ax = plt.subplots(figsize=(10,4))
                yrs = list(range(len(out.get("CF",[]))))
                cf_arr = np.array(out.get("CF",[]), dtype=float)
                ax.bar(yrs, cf_arr, color="tab:blue")
                ax.set_xlabel("Year"); ax.set_ylabel("Cashflow (USD)")
                ax.set_title("Annual Cashflow")
                st.pyplot(fig)
                st.session_state.captured_figs.setdefault(sel, []).append((mpl_fig_to_png_bytes(fig), "Annual cashflow"))
            except Exception:
                pass

            # provide small downloads
            if st.button("Download TEA table (CSV)"):
                buf = io.BytesIO()
                df_sched.to_csv(buf)
                buf.seek(0)
                st.download_button("Download CSV", data=buf, file_name=f"{sel}_schedule.csv", mime="text/csv")

        else:
            st.info("Compute TEA to see results and visuals.")

# ----------------- Tab 2: Scenario Analysis -----------------
with tabs[2]:
    st.header("Scenario Analysis (CBA & What-if)")
    st.markdown("Run scenario CBA (Optimistic / Moderate / Pessimistic) using `scenario_cba` from compliance.py if available.")

    if not st.session_state.designs:
        st.info("Add a design first (Input tab).")
    else:
        sel = st.selectbox("Select design for scenario CBA", options=list(st.session_state.designs.keys()))
        params = st.session_state.designs[sel]
        st.subheader(f"Scenario settings for {sel}")
        use_defaults = st.checkbox("Use default swings (±20% price & cost)", value=True)
        price_swing = st.number_input("Price swing % (±)", value=20.0) if not use_defaults else 20.0
        cost_swing = st.number_input("Cost swing % (±)", value=20.0) if not use_defaults else 20.0
        run_scen = st.button("Run scenario CBA")
        if run_scen:
            if hasattr(mod, "scenario_cba"):
                # many scenario functions in compliance.py are interactive; try to call with parameters that it expects
                try:
                    # call scenario_cba(params, design_label)
                    before = plt.get_fignums()
                    mod.scenario_cba(params, sel)
                    # capture any new figs
                    for num in plt.get_fignums():
                        if num not in before:
                            fig = plt.figure(num)
                            st.pyplot(fig)
                            st.session_state.captured_figs.setdefault(sel, []).append((mpl_fig_to_png_bytes(fig), f"Scenario visual {num}"))
                    st.success("Scenario CBA run (plots displayed).")
                except Exception as e:
                    st.error(f"scenario_cba raised an error: {e}")
            else:
                st.error("scenario_cba function not found in compliance.py")

# ----------------- Tab 3: Sensitivity (Tornado & Monte Carlo) -----------------
with tabs[3]:
    st.header("Sensitivity")
    st.markdown("Tornado sensitivity and Monte Carlo uncertainty. Uses functions `tornado_sensitivity` and `interactive_monte_carlo` (if present) from compliance.py.")

    if not st.session_state.designs:
        st.info("Add designs first.")
    else:
        sel = st.selectbox("Select design for sensitivity analysis", options=list(st.session_state.designs.keys()))
        params = st.session_state.designs[sel]

        st.subheader("Tornado sensitivity")
        default_swing = st.number_input("Tornado swing fraction (e.g., 0.2 for ±20%)", value=0.20, step=0.05)
        keys_input = st.text_input("Comma-separated parameter keys to test (leave blank for defaults)", value="")
        run_tornado = st.button("Run Tornado")
        if run_tornado:
            # choose sensible defaults if user didn't provide keys
            if keys_input.strip():
                keys = [k.strip() for k in keys_input.split(",") if k.strip()]
            else:
                # choose a small default set of keys present in params
                default_keys = ["C_PE","C_RM","COL","Q_prod","P_prod","f_esg","f_risk_op"]
                keys = [k for k in default_keys if k in params]
            if hasattr(mod, "tornado_sensitivity"):
                try:
                    before = plt.get_fignums()
                    mod.tornado_sensitivity(params, keys, swing=default_swing)
                    for num in plt.get_fignums():
                        if num not in before:
                            fig = plt.figure(num)
                            st.pyplot(fig)
                            st.session_state.captured_figs.setdefault(sel, []).append((mpl_fig_to_png_bytes(fig), f"Tornado {num}"))
                    st.success("Tornado plotted.")
                except Exception as e:
                    st.error(f"tornado_sensitivity error: {e}")
            else:
                st.error("tornado_sensitivity not available in compliance.py")

        st.subheader("Monte Carlo")
        mc_iters = st.number_input("Monte Carlo samples", value=2000, step=100)
        mc_noise = st.number_input("Parameter noise std (fraction)", value=0.05, step=0.01, format="%.3f")
        run_mc = st.button("Run Monte Carlo")
        if run_mc:
            if hasattr(mod, "interactive_monte_carlo"):
                try:
                    # interactive_monte_carlo likely expects params and maybe n; try common signature
                    before = plt.get_fignums()
                    try:
                        mod.interactive_monte_carlo(params, n=int(mc_iters), sigma=float(mc_noise))
                    except TypeError:
                        # fallback to single-arg call
                        mod.interactive_monte_carlo(params)
                    for num in plt.get_fignums():
                        if num not in before:
                            fig = plt.figure(num)
                            st.pyplot(fig)
                            st.session_state.captured_figs.setdefault(sel, []).append((mpl_fig_to_png_bytes(fig), f"MC {num}"))
                    st.success("Monte-Carlo executed (plots displayed).")
                except Exception as e:
                    st.error(f"interactive_monte_carlo error: {e}")
            else:
                st.error("interactive_monte_carlo not present in compliance.py")

# ----------------- Tab 4: Sweeps & ESG -----------------
with tabs[4]:
    st.header("Sweeps & ESG")
    st.markdown("Price sweep, raw-material sweep and ESG/compliance sweep (5%→75%).")

    if not st.session_state.designs:
        st.info("Add a design first.")
    else:
        sel = st.selectbox("Select design for sweeps", options=list(st.session_state.designs.keys()))
        params = st.session_state.designs[sel]

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Price sweep")
            price_swing = st.number_input("Price ± fraction (e.g., 0.3 = ±30%)", value=0.30)
            price_n = st.number_input("Points in sweep", value=25)
            if st.button("Run price sweep"):
                if hasattr(mod, "price_sweep"):
                    try:
                        before = plt.get_fignums()
                        mod.price_sweep(params, swing=float(price_swing), n=int(price_n))
                        for num in plt.get_fignums():
                            if num not in before:
                                fig = plt.figure(num)
                                st.pyplot(fig)
                                st.session_state.captured_figs.setdefault(sel, []).append((mpl_fig_to_png_bytes(fig), f"PriceSweep {num}"))
                        st.success("Price sweep done.")
                    except Exception as e:
                        st.error(f"price_sweep error: {e}")
                else:
                    st.error("price_sweep function not found in compliance.py")

        with col2:
            st.subheader("Raw-material sweep")
            rm_swing = st.number_input("RM cost ± fraction (e.g., 0.3)", value=0.30)
            rm_n = st.number_input("Points in sweep", value=25)
            if st.button("Run RM sweep"):
                if hasattr(mod, "raw_material_sweep"):
                    try:
                        before = plt.get_fignums()
                        mod.raw_material_sweep(params, swing=float(rm_swing), n=int(rm_n))
                        for num in plt.get_fignums():
                            if num not in before:
                                fig = plt.figure(num)
                                st.pyplot(fig)
                                st.session_state.captured_figs.setdefault(sel, []).append((mpl_fig_to_png_bytes(fig), f"RMSweep {num}"))
                        st.success("Raw-material sweep done.")
                    except Exception as e:
                        st.error(f"raw_material_sweep error: {e}")
                else:
                    st.error("raw_material_sweep function not found in compliance.py")

        st.markdown("---")
        st.subheader("ESG / compliance sweep (5% → 75%)")
        f_min = st.number_input("f_min (fraction)", value=0.05)
        f_max = st.number_input("f_max (fraction)", value=0.75)
        n_pts = st.number_input("Points in sweep", value=15)
        if st.button("Run ESG sweep & plots"):
            if hasattr(mod, "run_esg_sweep_and_plots"):
                try:
                    before = plt.get_fignums()
                    mod.run_esg_sweep_and_plots(params, design_label=sel, f_min=float(f_min), f_max=float(f_max), n=int(n_pts))
                    for num in plt.get_fignums():
                        if num not in before:
                            fig = plt.figure(num)
                            st.pyplot(fig)
                            st.session_state.captured_figs.setdefault(sel, []).append((mpl_fig_to_png_bytes(fig), f"ESG {num}"))
                    st.success("ESG sweep run and plotted.")
                except Exception as e:
                    st.error(f"run_esg_sweep_and_plots error: {e}")
            else:
                st.error("run_esg_sweep_and_plots not found in compliance.py")

# ----------------- Tab 5: Export Reports -----------------
with tabs[5]:
    st.header("Export Reports")
    st.markdown("Download TEA tables, captured plot images or build a simple PDF report.")

    if not st.session_state.designs:
        st.info("Add designs first.")
    else:
        sel = st.selectbox("Select design to export", options=list(st.session_state.designs.keys()))
        params = st.session_state.designs[sel]
        # Download parameters JSON
        if st.button("Download design JSON"):
            b = io.BytesIO()
            b.write(json.dumps(params, indent=2).encode("utf-8"))
            b.seek(0)
            st.download_button("Download design JSON", data=b, file_name=f"{sel}_params.json", mime="application/json")

        # Download captured figure images
        figs = st.session_state.captured_figs.get(sel, [])
        if figs:
            st.subheader("Captured figures")
            for i, (png_bytes, caption) in enumerate(figs):
                st.image(png_bytes, caption=caption, use_column_width=True)
                if st.download_button(f"Download figure {i+1} ({caption})", data=png_bytes, file_name=f"{sel}_fig_{i+1}.png", mime="image/png"):
                    st.success("Downloaded.")
        else:
            st.info("No captured figures yet. Run computations/plots to capture them automatically.")

        # Build simple PDF report
        if st.button("Build PDF report (basic)"):
            sections = []
            # summary
            text = f"Design: {sel}\nGenerated: {datetime.now().isoformat(sep=' ', timespec='seconds')}\n"
            text += "\nKey parameters snapshot:\n"
            for k,v in params.items():
                text += f"{k}: {v}\n"
            # TEA table inclusion
            tbls = []
            # include schedule if computed
            if sel in st.session_state.last_compute:
                out = st.session_state.last_compute[sel]
                cf = out.get("CF", [])
                dep = out.get("dep_schedule", [])
                ebt = out.get("EBT_schedule", [])
                tax = out.get("tax_schedule", [])
                df_sched = pd.DataFrame({
                    "Year": list(range(len(cf))),
                    "Cashflow": cf,
                    "Depreciation": list(dep) if dep is not None else [None]*len(cf),
                    "EBT": list(ebt) if ebt is not None else [None]*len(cf),
                    "Tax": list(tax) if tax is not None else [None]*len(cf),
                }).set_index("Year")
                tbls.append((df_sched, "Cashflow & schedules"))

            imgs = st.session_state.captured_figs.get(sel, [])
            sections.append({"heading": f"Design: {sel}", "text": text, "tables": tbls, "images": imgs})
            try:
                pdf_bytes = build_pdf_report_basic(f"TEA Report — {sel}", sections)
                st.download_button("Download PDF report", data=pdf_bytes, file_name=f"{sel}_report.pdf", mime="application/pdf")
                st.success("PDF ready.")
            except Exception as e:
                st.error(f"PDF build error: {e}")

st.markdown("---")
st.caption("comp-frontend.py — generated UI to run compliance.py functions. If a specific function isn't present in compliance.py, the UI shows an error message. Adjust design params in Input & Designs tab before computing.")
