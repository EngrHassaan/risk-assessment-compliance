# app.py
"""
Pro_DESG Materiality & Sustainability Tool Kit
Streamlit single-page (tabs) UI around risk_assessment.py with:
 - responsive layout
 - example CSV template
 - Plotly interactive charts + Matplotlib support
 - per-tab PDF export and combined full-report PDF (dynamic names)
"""

import json
from types import ModuleType

import importlib.util
import os
import io
import base64
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

# PDF tools
from reportlab.lib.pagesizes import A4, landscape
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
)

# ---------------- CONFIG ----------------
MODEL_PATH = "model/risk_assessment_model.py"  # new model file name in same directory
APP_TITLE = "Pro_DESG Materiality & Sustainability Tool Kit"

# ----------------- Utilities: dynamic import -----------------
def load_model_module(path: str):
    if not os.path.exists(path):
        st.error(f"Model file not found at: {path}\nUpload or correct MODEL_PATH.")
        st.stop()
    spec = importlib.util.spec_from_file_location("esgfp_model", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

mod = load_model_module(MODEL_PATH)

# ----------------- Helpers to convert figures to PNG bytes -----------------
def mpl_fig_to_png_bytes(fig: Figure, dpi: int = 150) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()

def plotly_fig_to_png_bytes(fig, width: int = 900, height: int = 600) -> bytes:
    # requires kaleido installed
    img_bytes = fig.to_image(format="png", width=width, height=height, scale=2)
    return img_bytes

def pil_bytes_to_rlimage(png_bytes: bytes, max_width_mm: float = 170) -> RLImage:
    # create tmp image and wrap for reportlab
    img = Image.open(io.BytesIO(png_bytes))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    rl_img = RLImage(buf)
    # scale to fit page width (max_width_mm)
    w_px, h_px = img.size
    # convert mm to points: 1 mm = 2.83465 points
    max_w_pt = max_width_mm * mm
    # current width in points (approx using 72 dpi)
    # but we'll scale relative to px -> points: pt = px * 72 / dpi. Since dpi varies, simpler: scale by ratio of desired width in pt to px.
    if rl_img.drawWidth > max_w_pt:
        scale = max_w_pt / rl_img.drawWidth
        rl_img.drawWidth = rl_img.drawWidth * scale
        rl_img.drawHeight = rl_img.drawHeight * scale
    return rl_img

# ----------------- PDF generation helpers -----------------
def build_pdf_report(title: str, sections: List[Dict], filename_prefix: str = "report") -> bytes:
    """
    sections: list of dicts:
      {"heading": str, "text": str, "tables": [(pandas.DataFrame, "Caption")], "images": [(png_bytes, "Caption")]}
    returns PDF bytes
    """
    buf = io.BytesIO()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    doc = SimpleDocTemplate(buf, pagesize=A4, rightMargin=18*mm, leftMargin=18*mm, topMargin=18*mm, bottomMargin=18*mm)
    styles = getSampleStyleSheet()
    story = []

    # Cover
    style_h = styles["Heading1"]
    style_h.alignment = 1  # center
    story.append(Paragraph(title, style_h))
    story.append(Spacer(1, 6))
    story.append(Paragraph(f"Generated: {now}", styles["Normal"]))
    story.append(Spacer(1, 12))

    for sec in sections:
        story.append(Paragraph(sec.get("heading", ""), styles["Heading2"]))
        if sec.get("text"):
            story.append(Paragraph(sec["text"], styles["Normal"]))
            story.append(Spacer(1, 6))

        # tables
        for tbl, caption in sec.get("tables", []):
            if tbl is None or tbl.empty:
                continue
            story.append(Paragraph(caption or "Table", styles["Italic"]))
            # convert DF to list-of-lists
            df = tbl.copy()
            df = df.round(6)
            data = [list(df.columns)]
            for r in df.itertuples(index=True):
                # include index as first column
                data.append([str(getattr(r, "Index"))] + [str(x) for x in r[1:]])
            # Add header for index
            header = ["Index"] + list(df.columns)
            data = [header] + [list(df.reset_index().iloc[i]) for i in range(len(df))]
            tbl_style = RLTable(data, hAlign="LEFT")
            tbl_style.setStyle(TableStyle([
                ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
                ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
                ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
                ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
            ]))
            story.append(tbl_style)
            story.append(Spacer(1, 8))

        # images
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

def make_download_button_bytes(bytes_data: bytes, filename: str, label: str):
    st.download_button(label, data=bytes_data, file_name=filename, mime="application/pdf")

# ----------------- Streamlit app state -----------------
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.markdown("Streamlined UI for running the ESGFP + Risk toolkit interactively. Upload CSVs, visualize charts, run scenarios and validation, and export PDFs.")

# initialize session state vars
if "risks" not in st.session_state:
    st.session_state.risks = []  # dict list: {"name","prob","sev"}
if "scores_by_tech" not in st.session_state:
    st.session_state.scores_by_tech = {}
if "pillars" not in st.session_state:
    st.session_state.pillars = dict(mod.DEFAULT_ESGFP)
if "pillar_avgs_df" not in st.session_state:
    st.session_state.pillar_avgs_df = pd.DataFrame()
if "last_weights" not in st.session_state:
    st.session_state.last_weights = None
if "last_methods" not in st.session_state:
    st.session_state.last_methods = None

# Sidebar quick actions & example CSV
st.sidebar.header("Quick Actions")
st.sidebar.write("- Use tabs to move: Risks → ESGFP → Scenarios → Validation.")
if st.sidebar.button("Load example dataset"):
    # build small example using default pillars and two techs
    pillars = dict(mod.DEFAULT_ESGFP)
    techs = {
        "Process Design A": {f"{p}:{sub}": round(5.0 * (1.0 + 0.25*(i%3)), 3) for i, (p, subs) in enumerate(pillars.items()) for sub in subs},
        "Process Design B": {f"{p}:{sub}": round(6.5 * (1.0 + 0.1*(i%2)), 3) for i, (p, subs) in enumerate(pillars.items()) for sub in subs},
    }
    st.session_state.pillars = pillars
    st.session_state.scores_by_tech = techs
    st.success("Example dataset loaded into session (two sample process designs).")

# Example CSV template generation (download)
def make_example_csv_bytes(pillars: Dict[str, List[str]]) -> bytes:
    cols = []
    for p, subs in pillars.items():
        for sub in subs:
            cols.append(f"{p}:{sub}")
    df = pd.DataFrame(columns=["Process Design"] + cols)
    # make two example rows
    row1 = ["Tech A"] + [5.0 for _ in cols]
    row2 = ["Tech B"] + [6.0 for _ in cols]
    df.loc[0] = row1
    df.loc[1] = row2
    b = io.BytesIO()
    df.to_csv(b, index=False)
    return b.getvalue()

st.sidebar.markdown("**Example CSV template**")
if st.sidebar.button("Download example CSV"):
    csv_bytes = make_example_csv_bytes(mod.DEFAULT_ESGFP)
    st.sidebar.download_button("Download CSV", csv_bytes, file_name="esgfp_example_template.csv", mime="text/csv")

st.sidebar.markdown("---")
st.sidebar.write("Dependencies:")
st.sidebar.write("streamlit, pandas, numpy, matplotlib, plotly, reportlab, pillow, kaleido")

# ----------------- Tabs -----------------
tabs = st.tabs(["Risk Assessment", "ESGFP Scoring", "Scenario Analysis (MCDA)", "Validation (DEA + Monte Carlo)", "Export Reports"])

# ----- Risk Assessment Tab -----
with tabs[0]:
    st.header("Risk Assessment")
    col_l, col_r = st.columns([2, 1])
    with col_l:
        st.subheader("Add / Edit Risks")
        with st.form("risk_form", clear_on_submit=True):
            rname = st.text_input("Risk name", "New Risk")
            prob = st.number_input("Probability (0 < p ≤ 1)", min_value=0.0001, max_value=1.0, value=0.1, format="%.4f")
            sev = st.number_input("Severity (1–10)", min_value=1.0, max_value=10.0, value=5.0, format="%.2f")
            add = st.form_submit_button("Add risk")
            if add:
                st.session_state.risks.append({"name": rname, "prob": float(prob), "sev": float(sev)})
                st.success(f"Added risk {rname}.")
        if st.session_state.risks:
            df_r = pd.DataFrame([
                {"Risk": r["name"], "Probability": r["prob"], "Severity": r["sev"], "Rating": r["prob"] * r["sev"]}
                for r in st.session_state.risks
            ])
            st.dataframe(df_r.round(4))
            if st.button("Clear risks"):
                st.session_state.risks = []
                st.session_state._do_rerun = True
        else:
            st.info("No risks added yet.")
    with col_r:
        st.subheader("Risk Visuals (Matplotlib + Plotly)")
        if st.session_state.risks:
            # convert to module Risk objects
            risks_objs = [mod.Risk(name=r["name"], probability=r["prob"], severity=r["sev"]) for r in st.session_state.risks]
            df_for_plot = mod.risk_dataframe(risks_objs)

            # Matplotlib visuals (use existing functions)
            before = plt.get_fignums()
            mod.plot_risk_views(df_for_plot)  # this generates matplotlib figures
            for num in plt.get_fignums():
                if num not in before:
                    fig = plt.figure(num)
                    st.pyplot(fig)

            # Plotly bubble chart (interactive)
            fig = px.scatter(
                df_for_plot,
                x="Probability", y="Severity", size="Rating", hover_name="Risk",
                title="Risk Bubble – Likelihood × Impact", labels={"Probability":"Probability", "Severity":"Severity"}
            )
            st.plotly_chart(fig, use_container_width=True)

            # small downloads
            csv_b = io.BytesIO()
            df_for_plot.to_csv(csv_b, index=False)
            csv_b.seek(0)
            st.download_button("Download risk table (CSV)", data=csv_b, file_name="risk_table.csv", mime="text/csv")
        else:
            st.info("Add risks to see visuals.")

# ----- ESGFP Scoring Tab -----
with tabs[1]:
    st.header("ESGFP Scoring")
    # two-column responsive: left inputs, right previews
    left, right = st.columns([2, 1])
    with left:
        st.subheader("Pillars & Key Issues (session)")
        st.write("You can add/remove key issues programmatically by uploading CSV with new columns or using the example dataset.")
        for p, subs in st.session_state.pillars.items():
            st.markdown(f"**{p}** — {', '.join(subs)}")

        st.write("---")
        st.subheader("Add Process Design & Scores")
        with st.expander("Manual add process design"):
            with st.form("add_tech_form", clear_on_submit=True):
                tech_name = st.text_input("Process Design name", f"Process Design {len(st.session_state.scores_by_tech)+1}")
                inputs = {}
                for p, subs in st.session_state.pillars.items():
                    st.markdown(f"**{p}**")
                    for sub in subs:
                        key = f"{p}:{sub}"
                        base = st.number_input(f"{p} → {sub} base (1–9)", min_value=1.0, max_value=9.0, value=5.0, key=f"{tech_name}_{key}")
                        exposure = st.slider(f"Exposure {key}", min_value=0.0, max_value=1.0, value=0.0, key=f"exp_{tech_name}_{key}")
                        inputs[key] = round(base * (1.0 + exposure), 4)
                added = st.form_submit_button("Add Process Design")
                if added:
                    label = tech_name
                    k = 2
                    while label in st.session_state.scores_by_tech:
                        label = f"{tech_name} ({k})"; k += 1
                    st.session_state.scores_by_tech[label] = inputs
                    st.success(f"Added {label}.")

        st.write("---")
        st.subheader("Upload ESGFP scores (CSV)")
        st.markdown("CSV should have first column `Tech`, other columns `Pillar:KeyIssue` (exposure already applied in values).")
        uploaded = st.file_uploader("Upload ESGFP CSV", type=["csv"])
        if uploaded:
            try:
                df_up = pd.read_csv(uploaded)
                if "Tech" not in df_up.columns and df_up.shape[1] >= 1:
                    df_up.columns.values[0] = "Tech"
                df_up = df_up.set_index("Tech")
                # update pillars with any new columns
                for col in df_up.columns:
                    if ":" in col:
                        p, sub = col.split(":", 1)
                        if p not in st.session_state.pillars:
                            st.session_state.pillars[p] = []
                        if sub not in st.session_state.pillars[p]:
                            st.session_state.pillars[p].append(sub)
                # add scores into session
                for tech, row in df_up.iterrows():
                    st.session_state.scores_by_tech[str(tech)] = row.dropna().to_dict()
                st.success("Uploaded and stored scores.")
            except Exception as e:
                st.error(f"CSV read error: {e}")

        st.write("---")
        st.subheader("Existing process designs")
        if st.session_state.scores_by_tech:
            st.dataframe(pd.DataFrame(st.session_state.scores_by_tech).round(4))
            if st.button("Clear process designs"):
                st.session_state.scores_by_tech = {}
                st.session_state._do_rerun = True
        else:
            st.info("No process designs in session. Add manually or upload CSV or load example dataset.")

    with right:
        st.subheader("Visualize & Compute Pillar Averages")
        if not st.session_state.scores_by_tech:
            st.info("Add process designs first.")
        else:
            pillar_avgs = mod.pillar_averages_multi(st.session_state.scores_by_tech, st.session_state.pillars)
            st.session_state.pillar_avgs_df = pillar_avgs.copy()
            st.write("Pillar Averages (raw):")
            st.dataframe(pillar_avgs.round(4))

            # Heatmap (matplotlib via module)
            before = plt.get_fignums()
            mod.plot_pillar_heatmaps(pillar_avgs)
            for num in plt.get_fignums():
                if num not in before:
                    st.pyplot(plt.figure(num))

            # Radar (matplotlib via module)
            before = plt.get_fignums()
            mod.plot_radar_profiles(pillar_avgs)
            for num in plt.get_fignums():
                if num not in before:
                    st.pyplot(plt.figure(num))

            # Small multiples
            before = plt.get_fignums()
            mod.plot_all_pillars_small_multiples(st.session_state.scores_by_tech, st.session_state.pillars)
            for num in plt.get_fignums():
                if num not in before:
                    st.pyplot(plt.figure(num))

            # Also provide a Plotly parallel coordinates for interactivity
            try:
                df_plotly = pillar_avgs.reset_index().melt(id_vars="Pillar", var_name="Tech", value_name="Score")
                df_wide = pillar_avgs.T  # alternatives x pillars
                fig_par = px.parallel_coordinates(df_wide.reset_index(), labels={c: c for c in df_wide.columns}, title="Parallel Coordinates (Pillar Profiles)")
                fig_par.update_layout(
                margin=dict(l=50, r=50, t=100, b=80),
                title=dict(y=0.95, x=0.5, xanchor='center', yanchor='top'),
                font=dict(size=13, color="black")
                )
                st.plotly_chart(fig_par, use_container_width=True)
            except Exception:
                pass

            csv_b = io.BytesIO()
            pillar_avgs.to_csv(csv_b)
            csv_b.seek(0)
            st.download_button("Download pillar averages (CSV)", data=csv_b, file_name="pillar_averages.csv", mime="text/csv")

# ----- Scenario Analysis Tab -----
# ----- Scenario Analysis Tab -----
with tabs[2]:
    st.header("Scenario Analysis (MCDA)")
    if st.session_state.pillar_avgs_df.empty:
        st.info("Create pillar averages in ESGFP tab first.")
    else:
        pillar_avgs = st.session_state.pillar_avgs_df.copy()
        st.write("Pillar Averages:")
        st.dataframe(pillar_avgs.round(4))

        st.subheader("Pillar Weights (sum to 100%)")
        weight_cols = st.columns(len(pillar_avgs.index))
        weights = {}
        for i, p in enumerate(pillar_avgs.index):
            with weight_cols[i]:
                w = st.number_input(
                    f"{p} (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=round(100.0 / len(pillar_avgs.index), 2),
                    key=f"w_{p}"
                )
                weights[p] = float(w)

        total = sum(weights.values())
        if abs(total - 100.0) > 1e-6:
            st.warning(f"Weights sum to {total:.2f}%. They must sum to 100%.")
            st.stop()

        methods_selected = st.multiselect(
            "Select MCDA Methods",
            options=["WEIGHTED", "WPM", "RANK", "TOPSIS", "VIKOR", "EDAS", "MAUT", "PCA"],
            default=["WEIGHTED"]
        )
        norm_flag = st.checkbox("Normalize outputs to 0–10 scale", value=True)

        # Ensure scenario history
        if "scenario_results" not in st.session_state:
            st.session_state.scenario_results = {}

        if st.button("Run scenarios"):
            per_method = {}
            if "WEIGHTED" in methods_selected:
                per_method["WEIGHTED"] = mod.method_weighted(pillar_avgs, weights)
            if "WPM" in methods_selected:
                per_method["WPM"] = mod.method_wpm(pillar_avgs, weights)
            if "RANK" in methods_selected:
                per_method["RANK"] = mod.method_rank(pillar_avgs, weights)
            if "TOPSIS" in methods_selected:
                per_method["TOPSIS"] = mod.method_topsis(pillar_avgs, weights)
            if "VIKOR" in methods_selected:
                per_method["VIKOR"] = mod.method_vikor(pillar_avgs, weights)
            if "EDAS" in methods_selected:
                per_method["EDAS"] = mod.method_edas(pillar_avgs, weights)
            if "MAUT" in methods_selected:
                per_method["MAUT"] = mod.method_maut(pillar_avgs, weights)
            if "PCA" in methods_selected:
                per_method["PCA"] = mod.method_pca(pillar_avgs)

            scenario_df = pd.DataFrame(per_method)
            scenario_df.index.name = "Alternative"

            # optional normalization
            if norm_flag:
                scaled_cols = {
                    col: mod._scale_series_by_method(scenario_df[col], col)
                    for col in scenario_df.columns
                }
                scenario_df_scaled = pd.DataFrame(scaled_cols, index=scenario_df.index)
            else:
                scenario_df_scaled = scenario_df.copy()

            # Save new scenario in session
            scenario_name = f"Scenario {len(st.session_state.scenario_results) + 1}"
            st.session_state.scenario_results[scenario_name] = {
                "weights": weights,
                "methods": methods_selected,
                "results": scenario_df_scaled,
            }

            st.session_state.last_weights = weights
            st.session_state.last_methods = methods_selected

        # Display all stored scenarios
        if st.session_state.scenario_results:
            st.subheader("Stored Scenarios")
            for name, sdata in st.session_state.scenario_results.items():
                st.markdown(f"**{name}** — Methods: {', '.join(sdata['methods'])}")
                st.dataframe(sdata["results"].round(4))

            # Plot aggregated comparison
            try:
                all_plot_data = []
                for name, sdata in st.session_state.scenario_results.items():
                    df = sdata["results"].copy()
                    df["Scenario"] = name
                    all_plot_data.append(df.reset_index().melt(
                        id_vars=["Alternative", "Scenario"],
                        var_name="Method",
                        value_name="Score"
                    ))
                plot_df = pd.concat(all_plot_data, ignore_index=True)
                fig = px.bar(
                    plot_df,
                    x="Alternative",
                    y="Score",
                    color="Method",
                    barmode="group",
                    facet_row="Scenario",
                    title="Scenario Comparisons by Method"
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Plot generation failed: {e}")

            # CSV download of last scenario
            last_key = list(st.session_state.scenario_results.keys())[-1]
            last_df = st.session_state.scenario_results[last_key]["results"]
            csv_b = io.BytesIO()
            last_df.to_csv(csv_b)
            csv_b.seek(0)
            st.download_button(
                "Download latest scenario (CSV)",
                data=csv_b,
                file_name=f"{last_key.replace(' ','_')}.csv",
                mime="text/csv"
            )

# ----- Validation Tab -----
with tabs[3]:
    st.header("Validation — DEA (approx) & Monte Carlo")
    if st.session_state.pillar_avgs_df.empty:
        st.info("Need pillar averages (from ESGFP tab).")
    else:
        pillar_avgs = st.session_state.pillar_avgs_df.copy()
        st.write("Pillar Averages:")
        st.dataframe(pillar_avgs.round(4))

        st.subheader("DEA settings")
        dea_samples = st.number_input("DEA convex-hull samples", min_value=100, max_value=200000, value=5000, step=100)
        peer_cut = st.number_input("Peer display cutoff", min_value=0.0, max_value=1.0, value=0.05, step=0.01)

        st.subheader("Monte Carlo settings")
        sims = st.number_input("Monte Carlo sims", min_value=100, max_value=200000, value=2000, step=100)
        alpha = st.number_input("Dirichlet alpha", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
        sigma = st.number_input("Score noise sigma", min_value=0.0, max_value=0.5, value=0.03, step=0.01)

        if st.button("Run validation suite"):
            with st.spinner("Running DEA diagnostics..."):
                dea_summary, peer_matrix, bottleneck_matrix, targets = mod.approx_dea_diagnostics(
                    pillar_avgs, samples=int(dea_samples), min_peer_lambda=float(peer_cut)
                )
            st.subheader("DEA Summary")
            st.dataframe(dea_summary.round(4))

            # visuals with matplotlib via module
            before = plt.get_fignums()
            mod.plot_pbest(dea_summary["FrontierProb"], "Approx DEA Frontier Probability")
            for num in plt.get_fignums():
                if num not in before:
                    st.pyplot(plt.figure(num))

            before = plt.get_fignums()
            mod.plot_dea_bottleneck_heatmap(bottleneck_matrix)
            for num in plt.get_fignums():
                if num not in before:
                    st.pyplot(plt.figure(num))

            before = plt.get_fignums()
            mod.plot_dea_peer_heatmap(peer_matrix)
            for num in plt.get_fignums():
                if num not in before:
                    st.pyplot(plt.figure(num))

            before = plt.get_fignums()
            mod.plot_dea_target_radars(pillar_avgs, targets)
            for num in plt.get_fignums():
                if num not in before:
                    st.pyplot(plt.figure(num))

            # Dominance matrix
            dom = mod.compute_dominance_matrix(pillar_avgs)
            st.subheader("Dominance matrix")
            st.dataframe(dom)

            # Monte-Carlo sensitivity
            with st.spinner("Running Monte-Carlo sensitivity..."):
                Pbest, MeanRank, StdRank, RankDist = mod.run_monte_carlo_sensitivity(
                    pillar_avgs,
                    st.session_state.last_weights if st.session_state.last_weights is not None else {p: 100.0/len(pillar_avgs.index) for p in pillar_avgs.index},
                    st.session_state.last_methods if st.session_state.last_methods is not None else ["WEIGHTED"],
                    sims=int(sims),
                    weight_alpha=float(alpha),
                    score_noise_sigma=float(sigma),
                )
            st.subheader("Monte-Carlo: P(Best)")
            st.dataframe(Pbest.round(4))

            # Plot P(best) for WEIGHTED if exists
            if "WEIGHTED" in Pbest.columns:
                fig = px.bar(Pbest.reset_index(), x="Tech", y="WEIGHTED", title="Monte-Carlo P(Best) — WEIGHTED")
                st.plotly_chart(fig, use_container_width=True)

            # rankograms for weighted (convert RankDist)
            if "WEIGHTED" in RankDist:
                try:
                    rd = RankDist["WEIGHTED"]
                    # create stacked area chart
                    fig = go.Figure()
                    ranks = list(rd.columns)
                    x = rd.index.tolist()
                    cum = np.zeros(len(x))
                    for col in ranks:
                        fig.add_trace(go.Bar(name=col, x=x, y=rd[col], offsetgroup=0))
                    fig.update_layout(barmode="stack", title="Rankogram — WEIGHTED")
                    st.plotly_chart(fig, use_container_width=True)
                except Exception:
                    pass

            st.subheader("Mean Rank & StdDev")
            st.dataframe(MeanRank.round(4))
            st.dataframe(StdRank.round(4))

# ----- Export Reports Tab -----
with tabs[4]:
    st.header("Export Reports (PDF)")

    def collect_risk_section():
        tables = []
        images = []
        if st.session_state.risks:
            df_r = pd.DataFrame([{"Risk": r["name"], "Probability": r["prob"], "Severity": r["sev"], "Rating": r["prob"] * r["sev"]} for r in st.session_state.risks])
            tables.append((df_r, "Risk table"))
            # Matplotlib figure snapshot using module
            before = plt.get_fignums()
            mod.plot_risk_views(df_r if not df_r.empty else pd.DataFrame())
            # capture any new fig
            for num in plt.get_fignums():
                if num not in before:
                    fig = plt.figure(num)
                    png = mpl_fig_to_png_bytes(fig)
                    images.append((png, "Risk visuals"))
        return {"heading":"Risk Assessment", "text":"Risk inputs & visuals", "tables": tables, "images": images}

    def collect_esgfp_section():
        tables = []
        images = []
        if st.session_state.scores_by_tech:
            df_scores = pd.DataFrame(st.session_state.scores_by_tech)
            # pivot to Tech x Pillar:Key
            tables.append((df_scores.round(6), "ESGFP scores (Techs x Pillar:Key)"))
        if not st.session_state.pillar_avgs_df.empty:
            pa = st.session_state.pillar_avgs_df
            tables.append((pa.round(6), "Pillar averages"))
            # plot heatmap & radar using module
            before = plt.get_fignums()
            try:
                mod.plot_pillar_heatmaps(pa)
                mod.plot_radar_profiles(pa)
            except Exception:
                pass
            for num in plt.get_fignums():
                if num not in before:
                    fig = plt.figure(num)
                    images.append((mpl_fig_to_png_bytes(fig), "Pillar visuals"))
        return {"heading":"ESGFP Scoring", "text":"Process designs, scores, and pillar averages", "tables": tables, "images": images}

    def collect_scenario_section():
        tables = []
        images = []
        # if scenario was last run (we saved nothing except last_weights), we will re-run with last settings if present
        if not st.session_state.pillar_avgs_df.empty and st.session_state.last_weights is not None and st.session_state.last_methods is not None:
            pa = st.session_state.pillar_avgs_df
            per_method = {}
            last_methods = st.session_state.last_methods
            weights = st.session_state.last_weights
            per_method["WEIGHTED"] = mod.method_weighted(pa, weights)
            for m in last_methods:
                if m == "WPM": per_method["WPM"] = mod.method_wpm(pa, weights)
                if m == "RANK": per_method["RANK"] = mod.method_rank(pa, weights)
                if m == "TOPSIS": per_method["TOPSIS"] = mod.method_topsis(pa, weights)
                if m == "VIKOR": per_method["VIKOR"] = mod.method_vikor(pa, weights)
                if m == "EDAS": per_method["EDAS"] = mod.method_edas(pa, weights)
                if m == "MAUT": per_method["MAUT"] = mod.method_maut(pa, weights)
                if m == "PCA": per_method["PCA"] = mod.method_pca(pa)
            scenario_df = pd.DataFrame(per_method)
            tables.append((scenario_df.round(6), "Scenario results"))
            # Plotbar via plotly and capture png
            try:
                fig = px.bar(scenario_df.reset_index().melt(id_vars="index", var_name="Method", value_name="Score"), x="index", y="Score", color="Method", barmode="group")
                png = plotly_fig_to_png_bytes(fig)
                images.append((png, "Scenario comparison"))
            except Exception:
                pass
        return {"heading":"Scenarios", "text":"MCDA scenario results (if available)", "tables": tables, "images": images}

    def collect_validation_section():
        tables = []
        images = []
        if not st.session_state.pillar_avgs_df.empty:
            pa = st.session_state.pillar_avgs_df
            try:
            # --- Run DEA diagnostics (sample) ---
                dea_summary, peer_matrix, bottleneck_matrix, targets = mod.approx_dea_diagnostics(
                    pa, samples=800, min_peer_lambda=0.05
                )
                tables.append((dea_summary.round(6), "DEA Summary (sample)"))
                tables.append((peer_matrix.round(6), "DEA Peer Matrix (sample)"))

                # --- Capture each DEA chart individually ---

                # 1️⃣ DEA Frontier Probability
                fig1_before = plt.get_fignums()
                mod.plot_pbest(dea_summary["FrontierProb"], "DEA Frontier Probability (sample)")
                for num in plt.get_fignums():
                    if num not in fig1_before:
                        fig = plt.figure(num)
                        images.append((mpl_fig_to_png_bytes(fig), "DEA Frontier Probability"))
                        plt.close(fig)

                # 2️⃣ DEA Bottleneck Heatmap
                fig2_before = plt.get_fignums()
                mod.plot_dea_bottleneck_heatmap(bottleneck_matrix)
                for num in plt.get_fignums():
                    if num not in fig2_before:
                        fig = plt.figure(num)
                        images.append((mpl_fig_to_png_bytes(fig), "DEA Bottleneck Heatmap"))
                        plt.close(fig)

                # 3️⃣ DEA Peer Heatmap
                fig3_before = plt.get_fignums()
                mod.plot_dea_peer_heatmap(peer_matrix)
                for num in plt.get_fignums():
                    if num not in fig3_before:
                        fig = plt.figure(num)
                        images.append((mpl_fig_to_png_bytes(fig), "DEA Peer Heatmap"))
                        plt.close(fig)

                # 4️⃣ DEA Target Radars
                fig4_before = plt.get_fignums()
                mod.plot_dea_target_radars(pa, targets)
                for num in plt.get_fignums():
                    if num not in fig4_before:
                        fig = plt.figure(num)
                        images.append((mpl_fig_to_png_bytes(fig), "DEA Target Radars"))
                        plt.close(fig)

                # --- Dominance Matrix ---
                dom = mod.compute_dominance_matrix(pa)
                tables.append((dom.round(6), "Dominance Matrix"))

                # --- Monte-Carlo (quick summary for PDF) ---
                try:
                    Pbest, MeanRank, StdRank, RankDist = mod.run_monte_carlo_sensitivity(
                        pa,
                        st.session_state.last_weights
                        if st.session_state.last_weights
                        else {p: 100.0 / len(pa.index) for p in pa.index},
                        st.session_state.last_methods
                        if st.session_state.last_methods
                        else ["WEIGHTED"],
                        sims=500,
                        weight_alpha=1.0,
                        score_noise_sigma=0.03,
                    )
                    tables.append((Pbest.round(6), "Monte Carlo – P(Best)"))

                    # P(Best) chart
                    if "WEIGHTED" in Pbest.columns:
                        fig = px.bar(
                            Pbest.reset_index(),
                            x="Tech",
                            y="WEIGHTED",
                            title="Monte Carlo P(Best) – WEIGHTED",
                        )
                        images.append((plotly_fig_to_png_bytes(fig), "Monte Carlo P(Best)"))

                    # Rankogram
                    if "WEIGHTED" in RankDist:
                        rd = RankDist["WEIGHTED"]
                        fig = go.Figure()
                        for col in rd.columns:
                            fig.add_trace(go.Bar(name=col, x=rd.index, y=rd[col]))
                        fig.update_layout(barmode="stack", title="Monte Carlo Rankogram – WEIGHTED")
                        images.append((plotly_fig_to_png_bytes(fig), "Monte Carlo Rankogram"))

                except Exception as e:
                    images.append(
                        (
                            mpl_fig_to_png_bytes(plt.figure()),
                            f"Monte Carlo generation failed: {e}",
                        )
                    )

            except Exception as e:
                images.append(
                    (mpl_fig_to_png_bytes(plt.figure()), f"DEA generation failed: {e}")
                )

        return {
        "heading": "Validation",
        "text": "DEA diagnostics and Monte Carlo sensitivity visuals (if pillar averages exist).",
        "tables": tables,
        "images": images,
    }

    # Buttons to generate per-tab PDFs
    cols = st.columns(3)
    with cols[0]:
        if st.button("Export Risk tab to PDF"):
            sec = collect_risk_section()
            pdf_bytes = build_pdf_report(APP_TITLE, [sec], filename_prefix="Risk")
            fname = f"{APP_TITLE.replace(' ','_')}_Risk_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            make_download_button_bytes(pdf_bytes, fname, f"Download Risk PDF ({fname})")
    with cols[1]:
        if st.button("Export ESGFP tab to PDF"):
            sec = collect_esgfp_section()
            pdf_bytes = build_pdf_report(APP_TITLE, [sec], filename_prefix="ESGFP")
            fname = f"{APP_TITLE.replace(' ','_')}_ESGFP_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            make_download_button_bytes(pdf_bytes, fname, f"Download ESGFP PDF ({fname})")
    with cols[2]:
        if st.button("Export Scenarios tab to PDF"):
            sec = collect_scenario_section()
            pdf_bytes = build_pdf_report(APP_TITLE, [sec], filename_prefix="Scenarios")
            fname = f"{APP_TITLE.replace(' ','_')}_Scenarios_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            make_download_button_bytes(pdf_bytes, fname, f"Download Scenarios PDF ({fname})")

    st.write("---")
    cols2 = st.columns(2)
    with cols2[0]:
        if st.button("Export Validation tab to PDF"):
            sec = collect_validation_section()
            pdf_bytes = build_pdf_report(APP_TITLE, [sec], filename_prefix="Validation")
            fname = f"{APP_TITLE.replace(' ','_')}_Validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            make_download_button_bytes(pdf_bytes, fname, f"Download Validation PDF ({fname})")
    with cols2[1]:
        if st.button("Export Full Combined Report PDF"):
            sections = []
            sections.append(collect_risk_section())
            sections.append(collect_esgfp_section())
            sections.append(collect_scenario_section())
            sections.append(collect_validation_section())
            pdf_bytes = build_pdf_report(APP_TITLE, sections, filename_prefix="FullReport")
            fname = f"{APP_TITLE.replace(' ','_')}_FullReport_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            make_download_button_bytes(pdf_bytes, fname, f"Download Full Report PDF ({fname})")

st.info("Tip: Use 'Export Reports' to create PDFs of the current session. The files are generated dynamically and include charts and tables present in the session.")
if st.session_state.get("_do_rerun", False):
    st.session_state._do_rerun = False
    st.rerun()
