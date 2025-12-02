# risk_assessment.py
# Combined Risk Assessment + ESGFP + MCDA + Validation + Compliance/ESG TEA
# Generated in 5 chunks — paste them sequentially into a single file.

from dataclasses import dataclass
from typing import Dict, List, Tuple, Sequence, Optional, Any
import sys
import math
import re
import random
import os
import io
import json
import tempfile
import importlib.util
from types import ModuleType
from datetime import datetime

import streamlit as st
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
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage,
    Table as RLTable, TableStyle, PageBreak
)

# ----------------------- Plot style (professional) ------------------
def apply_pro_style() -> None:
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "figure.titlesize": 16,
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
        "axes.axisbelow": True,
        "figure.figsize": (10, 6),
    })

# ----------------------- Data models -----------------------
@dataclass
class Risk:
    name: str
    probability: float  # (0,1]
    severity: float     # [1,10]

    @property
    def rating(self) -> float:
        return self.probability * self.severity

# ----------------------- Constants -------------------------
DEFAULT_ESGFP: Dict[str, List[str]] = {
    "Environmental": ["GHG Emissions", "Water Use", "Waste Management"],
    "Social": ["Labor Safety", "Community Impact"],
    "Governance": ["Compliance", "Transparency"],
    "Financial": ["CAPEX", "OPEX", "ROI"],
    "Process": ["Efficiency", "Flexibility", "Scalability"],
}
EXPOSURE_MAP: Dict[str, float] = {
    "high": 1.0,
    "moderate": 0.75,
    "moderate to lower": 0.5,
    "lower": 0.25,
    "1": 1.0,
    "0.75": 0.75,
    "0.5": 0.5,
    "0.25": 0.25,
}

# Display scaling
OUTPUT_SCALE = 10.0
WEIGHTED_THEORETICAL_MAX = 18.0  # e.g., max sub-issue 9 × (1 + 1.0 exposure)
MIN_DISPLAY_SCORE = 1.0          # score floor (avoid zeros in plots)

# ----------------------- Helper plotting conversions -----------------
def mpl_fig_to_png_bytes(fig: Figure, dpi: int = 150) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()

def plotly_fig_to_png_bytes(fig, width: int = 900, height: int = 600) -> bytes:
    # requires kaleido; if not present, this will raise — used only for PDF embedding if available
    try:
        img_bytes = fig.to_image(format="png", width=width, height=height, scale=2)
        return img_bytes
    except Exception:
        # fallback: render to PNG via static HTML snapshot not available offline
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
        rl_img.drawWidth = rl_img.drawWidth * scale
        rl_img.drawHeight = rl_img.drawHeight * scale
    return rl_img

# ----------------------- Load compliance module dynamically -----------------
DEFAULT_COMPLIANCE_PATH = "model/compliance.py"

def load_compliance_module(path: str) -> ModuleType:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Compliance script not found at {path}")
    spec = importlib.util.spec_from_file_location("compliance_teawork", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod

# ----------------------- Risk section ----------------------
def risk_dataframe(risks: List[Risk]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"Risk": r.name, "Probability": r.probability, "Severity": r.severity, "Rating": r.rating} for r in risks]
    )

def plot_risk_views(df: pd.DataFrame) -> None:
    if df.empty:
        return
    attrs = df.set_index("Risk")[["Probability", "Severity", "Rating"]]

    # Heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(attrs.values)
    ax.set_xticks(range(attrs.shape[1]))
    ax.set_xticklabels(attrs.columns)
    ax.set_yticks(range(attrs.shape[0]))
    ax.set_yticklabels(attrs.index)
    for i in range(attrs.shape[0]):
        for j in range(attrs.shape[1]):
            ax.text(j, i, f"{attrs.values[i, j]:.2f}", ha="center", va="center")
    ax.set_title("Risk Attributes Heatmap")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    plt.show()

    # Bar
    ax = df.plot(
        x="Risk", y="Rating", kind="bar", rot=45, figsize=(10, 6), title="Risk Rating Bar Plot", legend=False
    )
    ax.set_ylabel("Risk Rating")
    ax.grid(True)
    plt.tight_layout()
    plt.show()

    # Bubble
    x_min, x_max = 0.0, 1.0
    y_min, y_max = 1.0, 10.0
    fig, ax = plt.subplots(figsize=(16, 9))
    nx, ny = 500, 500
    gx = np.linspace(x_min, x_max, nx)
    gy = np.linspace(y_min, y_max, ny)
    gy01 = (gy - y_min) / (y_max - y_min)
    Z = gy01[:, None] + gx[None, :]
    Z = (Z - Z.min()) / (Z.max() - Z.min())
    ax.imshow(
        Z, extent=[x_min, x_max, y_min, y_max], origin="lower",
        cmap="RdYlGn_r", alpha=0.85, aspect="auto", interpolation="bilinear",
    )
    x = df["Probability"].to_numpy()
    y = df["Severity"].to_numpy()
    r = df["Rating"].to_numpy()
    labels = df["Risk"].astype(str).to_list()
    sizes = np.clip(r, 0.05, None) * 600.0
    ax.scatter(x, y, s=sizes, edgecolors="black", linewidths=0.8, alpha=0.95, zorder=3)
    for xi, yi, lab in zip(x, y, labels):
        offset_x = 18 if xi < 0.7 else -18
        ha = "left" if xi < 0.7 else "right"
        ax.annotate(
            lab, xy=(xi, yi), xytext=(offset_x, 10), textcoords="offset points",
            ha=ha, va="bottom", fontsize=13, color="black",
            arrowprops=dict(arrowstyle="-", color="gray", lw=0.9, alpha=0.9),
            zorder=4,
        )
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("Probability (Likelihood)")
    ax.set_ylabel("Severity (Impact)")
    ax.set_title("Materiality Assessment Diagram – Likelihood × Impact")
    ax.grid(False)
    plt.tight_layout()
    plt.show()

# ----------------------- ESGFP scoring (N technologies) ----
def _sanitize_filename(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)

def _norm0_10(values: Sequence[float], base: float = WEIGHTED_THEORETICAL_MAX) -> List[float]:
    return [(float(v) / base) * OUTPUT_SCALE for v in values]

def collect_esgfp_scores_for_tech(tech_label: str, pillars: Dict[str, List[str]]) -> Dict[str, float]:
    # CLI helper retained for compatibility; not used in Streamlit automation.
    print(f"\n📥 ESGFP Scores for {tech_label}")
    scores: Dict[str, float] = {}
    for pillar, subs in pillars.items():
        for sub in subs:
            # placeholder interactive CLI fallback
            s = 5.0
            g = 0.0
            final = s * (1.0 + g)
            scores[f"{pillar}:{sub}"] = round(final, 4)
    return scores

def pillar_averages_multi(scores_by_tech: Dict[str, Dict[str, float]], pillars: Dict[str, List[str]]) -> pd.DataFrame:
    techs = list(scores_by_tech.keys())
    rows = []
    for pillar, subs in pillars.items():
        row = {"Pillar": pillar}
        for tech in techs:
            vals = [scores_by_tech[tech].get(f"{pillar}:{sub}", 0.0) for sub in subs]
            row[tech] = float(np.mean(vals)) if vals else 0.0
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=techs).set_index(pd.Index([], name="Pillar"))
    return pd.DataFrame(rows).set_index("Pillar")[techs]

def plot_pillar_key_issue_comparisons(scores_by_tech: Dict[str, Dict[str, float]], pillars: Dict[str, List[str]]) -> None:
    techs = list(scores_by_tech.keys())
    for pillar, subs in pillars.items():
        if not subs:
            continue
        x = np.arange(len(subs))
        width = 0.8 / max(len(techs), 1)
        fig, ax = plt.subplots(figsize=(12, 6))
        for idx, tech in enumerate(techs):
            vals = [scores_by_tech[tech].get(f"{pillar}:{sub}", 0.0) for sub in subs]
            y = _norm0_10(vals)
            ax.bar(x + (idx - (len(techs) - 1) / 2) * width, y, width=width, label=tech)
        ax.set_xticks(x)
        ax.set_xticklabels(subs, rotation=0)
        ax.set_ylabel("Normalized Score (0–10)")
        ax.set_title(f"{pillar} – Key Issue Comparison (0–10 normalized)")
        ax.legend()
        ax.grid(True, axis="y", linestyle="--", linewidth=0.5)
        plt.tight_layout()
        plt.show()

def plot_all_pillars_small_multiples(scores_by_tech: Dict[str, Dict[str, float]], pillars: Dict[str, List[str]]) -> None:
    techs = list(scores_by_tech.keys())
    if not pillars:
        return
    r = math.ceil(len(pillars) / 3)
    c = 3
    fig, axes = plt.subplots(r, c, figsize=(18, 3 * r), sharey=True)
    axes = np.array(axes).reshape(-1)
    for i, (pillar, subs) in enumerate(pillars.items()):
        ax = axes[i]
        x = np.arange(len(subs))
        width = 0.8 / max(len(techs), 1)
        for idx, tech in enumerate(techs):
            vals = [scores_by_tech[tech].get(f"{pillar}:{sub}", 0.0) for sub in subs]
            y = _norm0_10(vals)
            ax.bar(x + (idx - (len(techs) - 1) / 2) * width, y, width=width, label=tech if i == 0 else None)
        ax.set_title(pillar)
        ax.set_xticks(x)
        ax.set_xticklabels(subs, rotation=0)
        ax.grid(True, axis="y", linestyle="--", linewidth=0.5)
        if i % 3 == 0:
            ax.set_ylabel("0–10")
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(len(techs), 5))
    fig.suptitle("Key Issue Comparison per Pillar (0–10 normalized)")
    fig.tight_layout(rect=[0, 0.06, 1, 0.96])
    plt.show()

def plot_pillar_heatmaps(pillar_avgs: pd.DataFrame) -> None:
    if pillar_avgs.empty:
        return
    data = pillar_avgs.copy()
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(data.values, aspect="auto")
    ax.set_xticks(range(data.shape[1]))
    ax.set_xticklabels(data.columns, rotation=45, ha="right")
    ax.set_yticks(range(data.shape[0]))
    ax.set_yticklabels(data.index)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data.values[i, j]:.2f}", ha="center", va="center", fontsize=9)
    ax.set_title("Pillar Averages – Raw")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Score")
    fig.tight_layout()
    plt.show()

def plot_radar_profiles(pillar_avgs: pd.DataFrame) -> None:
    if pillar_avgs.empty:
        return
    pillars = pillar_avgs.index.tolist()
    techs = pillar_avgs.columns.tolist()
    norm = (pillar_avgs / WEIGHTED_THEORETICAL_MAX) * OUTPUT_SCALE
    ax, angles = _radar_axes(len(pillars))
    for tech in techs:
        vals = norm[tech].tolist()
        vals += vals[:1]
        ax.plot(angles, vals, linewidth=2, alpha=0.9, label=tech)
        ax.fill(angles, vals, alpha=0.1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(pillars)
    ax.set_yticklabels([])
    ax.set_ylim(0, 10)
    ax.set_title("Pillar Profiles (Radar, 0–10)")
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), frameon=False)
    plt.show()

def plot_parallel_coordinates(pillar_avgs: pd.DataFrame) -> None:
    if pillar_avgs.empty:
        return
    data = pillar_avgs.copy()
    lo = data.min(axis=1)
    hi = data.max(axis=1)
    denom = (hi - lo).replace(0, 1.0)
    norm = data.sub(lo, axis=0).div(denom, axis=0)
    x = np.arange(len(data.index))
    fig, ax = plt.subplots(figsize=(12, 6))
    for tech in norm.columns:
        ax.plot(x, norm[tech].values, marker="o", linewidth=2, alpha=0.9, label=tech)
    ax.set_xticks(x)
    ax.set_xticklabels(data.index)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Normalized (0–1)")
    ax.set_title("Parallel Coordinates – Pillar Profiles")
    ax.legend(loc="upper right", ncol=2, frameon=False)
    plt.show()

def plot_tradeoff_scatter(pillar_avgs: pd.DataFrame) -> None:
    if pillar_avgs.empty:
        return
    pairs = [
        ("Financial", "Environmental"),
        ("Governance", "Process"),
        ("Financial", "Social"),
    ]
    for xlab, ylab in pairs:
        if xlab not in pillar_avgs.index or ylab not in pillar_avgs.index:
            continue
        x = pillar_avgs.loc[xlab, :]
        y = pillar_avgs.loc[ylab, :]
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(x, y, s=120, alpha=0.85, edgecolor="black", linewidth=0.7)
        for tech in pillar_avgs.columns:
            ax.annotate(tech, (x[tech], y[tech]), xytext=(6, 6), textcoords="offset points")
        ax.set_xlabel(f"{xlab} (pillar mean)")
        ax.set_ylabel(f"{ylab} (pillar mean)")
        ax.set_title(f"{ylab} vs {xlab} – Trade-off View")
        plt.show()

# ----------------------- MCDA utilities & Methods -------------------
def _weights_vector(pillars: Sequence[str], weights_pct: Dict[str, float]) -> np.ndarray:
    w = np.array([float(weights_pct.get(p, 0.0)) for p in pillars], dtype=float)
    s = w.sum()
    return w / (s if s != 0 else 1.0)

def _decision_matrix(pillar_avgs: pd.DataFrame) -> Tuple[np.ndarray, List[str], List[str]]:
    alts = list(pillar_avgs.columns)
    crits = list(pillar_avgs.index)
    A = pillar_avgs.to_numpy().T  # (m_alts, n_criteria)
    return A, alts, crits

def method_weighted(pillar_avgs: pd.DataFrame, weights_pct: Dict[str, float]) -> pd.Series:
    A, alts, crits = _decision_matrix(pillar_avgs)
    w = _weights_vector(crits, weights_pct)
    s = A @ w
    return pd.Series(s, index=alts).sort_values(ascending=False)

def method_wpm(pillar_avgs: pd.DataFrame, weights_pct: Dict[str, float]) -> pd.Series:
    A, alts, crits = _decision_matrix(pillar_avgs)
    w = _weights_vector(crits, weights_pct)
    col_max = A.max(axis=0)
    col_max[col_max == 0.0] = 1.0
    N = A / col_max
    N[N <= 0.0] = 1e-12
    log_scores = (np.log(N) * w).sum(axis=1)
    scores = np.exp(log_scores)
    return pd.Series(scores, index=alts).sort_values(ascending=False)

def method_rank(pillar_avgs: pd.DataFrame, weights_pct: Dict[str, float]) -> pd.Series:
    alts = list(pillar_avgs.columns)
    crits = list(pillar_avgs.index)
    w = _weights_vector(crits, weights_pct)
    m = len(alts)
    points = np.zeros(m)
    for j, _ in enumerate(crits):
        ranks = pillar_avgs.iloc[j, :].rank(ascending=False, method="average")
        pts = (m - ranks + 1.0).to_numpy()
        points += w[j] * pts
    return pd.Series(points, index=alts).sort_values(ascending=False)

def method_topsis(pillar_avgs: pd.DataFrame, weights_pct: Dict[str, float]) -> pd.Series:
    A, alts, crits = _decision_matrix(pillar_avgs)
    w = _weights_vector(crits, weights_pct)
    norm = np.linalg.norm(A, axis=0)
    norm[norm == 0.0] = 1.0
    R = A / norm
    V = R * w
    ideal_best = V.max(axis=0)
    ideal_worst = V.min(axis=0)
    d_best = np.linalg.norm(V - ideal_best, axis=1)
    d_worst = np.linalg.norm(V - ideal_worst, axis=1)
    score = d_worst / (d_best + d_worst + 1e-12)
    return pd.Series(score, index=alts).sort_values(ascending=False)

def method_vikor(pillar_avgs: pd.DataFrame, weights_pct: Dict[str, float], v: float = 0.5) -> pd.Series:
    A, alts, crits = _decision_matrix(pillar_avgs)
    w = _weights_vector(crits, weights_pct)
    f_star = A.max(axis=0)
    f_minus = A.min(axis=0)
    denom = f_star - f_minus
    denom[denom == 0.0] = 1.0
    gap = (f_star - A) / denom
    S = (gap * w).sum(axis=1)
    R = (gap * w).max(axis=1)
    S_star, S_minus = S.min(), S.max()
    R_star, R_minus = R.min(), R.max()
    QS = (S - S_star) / (S_minus - S_star + 1e-12)
    QR = (R - R_star) / (R_minus - R_star + 1e-12)
    Q = v * QS + (1 - v) * QR
    score = 1.0 - Q
    return pd.Series(score, index=alts).sort_values(ascending=False)

def method_edas(pillar_avgs: pd.DataFrame, weights_pct: Dict[str, float]) -> pd.Series:
    A, alts, crits = _decision_matrix(pillar_avgs)
    w = _weights_vector(crits, weights_pct)
    avg = A.mean(axis=0)
    avg[avg == 0.0] = 1.0
    PDA = np.maximum(0.0, (A - avg) / avg)
    NDA = np.maximum(0.0, (avg - A) / avg)
    SP = PDA @ w
    SN = NDA @ w
    NSP = SP / (SP.max() + 1e-12)
    NSN = SN / (SN.max() + 1e-12)
    AS = (NSP + (1.0 - NSN)) / 2.0
    return pd.Series(AS, index=alts).sort_values(ascending=False)

def method_maut(pillar_avgs: pd.DataFrame, weights_pct: Dict[str, float]) -> pd.Series:
    A, alts, crits = _decision_matrix(pillar_avgs)
    w = _weights_vector(crits, weights_pct)
    lo = A.min(axis=0)
    hi = A.max(axis=0)
    denom = hi - lo
    denom[denom == 0.0] = 1.0
    U = (A - lo) / denom
    score = U @ w
    return pd.Series(score, index=alts).sort_values(ascending=False)

def method_pca(pillar_avgs: pd.DataFrame) -> pd.Series:
    A, alts, _ = _decision_matrix(pillar_avgs)
    if A.shape[0] < 2 or A.shape[1] < 1:
        return pd.Series(np.ones(A.shape[0]), index=alts)
    X = A - A.mean(axis=0)
    std = A.std(axis=0, ddof=1)
    std[std == 0.0] = 1.0
    X = X / std
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    pc1_scores = U[:, 0] * S[0]
    mean_profile = X.mean(axis=1)
    if np.corrcoef(pc1_scores, mean_profile)[0, 1] < 0:
        pc1_scores = -pc1_scores
    lo, hi = pc1_scores.min(), pc1_scores.max()
    if hi - lo <= 1e-12:
        scaled = np.full_like(pc1_scores, 0.5)
    else:
        scaled = (pc1_scores - lo) / (hi - lo)
    return pd.Series(scaled, index=alts).sort_values(ascending=False)

# ----------------------- Normalization (0–10 display) ----
def _scale_series_by_method(s: pd.Series, method: str) -> pd.Series:
    if s.empty:
        return s
    m = method.upper()
    if m == "WEIGHTED":
        base = WEIGHTED_THEORETICAL_MAX
        return ((s / base) * OUTPUT_SCALE).clip(lower=MIN_DISPLAY_SCORE)
    if m in {"TOPSIS", "VIKOR", "EDAS", "MAUT", "PCA"}:
        return (s * OUTPUT_SCALE).clip(lower=MIN_DISPLAY_SCORE)
    if m == "RANK":
        lo, hi = float(s.min()), float(s.max())
        if math.isclose(hi, lo):
            return pd.Series([OUTPUT_SCALE / 2.0] * len(s), index=s.index)
        return (((s - lo) / (hi - lo)) * OUTPUT_SCALE).clip(lower=MIN_DISPLAY_SCORE)
    if m == "WPM":
        return (s * OUTPUT_SCALE).clip(lower=MIN_DISPLAY_SCORE)
    lo, hi = float(s.min()), float(s.max())
    if math.isclose(hi, lo):
        return pd.Series([OUTPUT_SCALE / 2.0] * len(s), index=s.index)
    return (((s - lo) / (hi - lo)) * OUTPUT_SCALE).clip(lower=MIN_DISPLAY_SCORE)

# ----------------------- DEA & Monte-Carlo ----------------
def _dirichlet(alpha: np.ndarray) -> np.ndarray:
    samples = np.random.gamma(shape=alpha, scale=1.0)
    s = samples.sum()
    if s <= 0:
        return np.ones_like(alpha) / len(alpha)
    return samples / s

def compute_dominance_matrix(pillar_avgs: pd.DataFrame) -> pd.DataFrame:
    alts = list(pillar_avgs.columns)
    A = pillar_avgs.to_numpy().T
    m = len(alts)
    dom = np.zeros((m, m), dtype=int)
    for i in range(m):
        for j in range(m):
            if i == j:
                continue
            ge_all = np.all(A[i, :] >= A[j, :])
            gt_any = np.any(A[i, :] > A[j, :])
            dom[i, j] = int(ge_all and gt_any)
    return pd.DataFrame(dom, index=alts, columns=alts)

def approx_dea_diagnostics(
    pillar_avgs: pd.DataFrame,
    samples: int = 5000,
    seed: Optional[int] = None,
    min_peer_lambda: float = 0.05,
    eps: float = 1e-9,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    alts = list(pillar_avgs.columns)
    pillars = list(pillar_avgs.index)
    Y = pillar_avgs.to_numpy().T
    m, n = Y.shape
    frontier_hits = np.zeros(m, dtype=int)
    rmax = np.ones(m, dtype=float)
    best_lambda_store = [None] * m
    bottleneck_counts = np.zeros((n, m), dtype=int)
    for i in range(m):
        others = [k for k in range(m) if k != i]
        if not others:
            continue
        Y_others = Y[others, :]
        best_r = 1.0
        best_lamb = None
        for _ in range(samples):
            lamb = _dirichlet(np.ones(len(others)))
            mix = lamb @ Y_others
            denom = np.maximum(Y[i, :], eps)
            ratios = mix / denom
            r = float(np.min(ratios))
            bottleneck_idx = int(np.argmin(ratios))
            bottleneck_counts[bottleneck_idx, i] += 1
            if r >= 1.0:
                frontier_hits[i] += 1
                if r > best_r:
                    best_r = r
                    best_lamb = lamb
        rmax[i] = max(best_r, 1.0)
        best_lambda_store[i] = best_lamb
    phi_hat = rmax
    eff_out = 1.0 / phi_hat
    frontier_prob = frontier_hits / float(samples)
    top_b_idx = bottleneck_counts.argmax(axis=0)
    top_b_share = bottleneck_counts.max(axis=0) / np.maximum(bottleneck_counts.sum(axis=0), 1)
    dea_summary = pd.DataFrame({
        "Tech": alts,
        "FrontierProb": frontier_prob,
        "PhiHat": phi_hat,
        "EffOut": eff_out,
        "Bottleneck": [pillars[idx] for idx in top_b_idx],
        "BottleneckShare": top_b_share,
    }).set_index("Tech")
    peer = np.zeros((m, m), dtype=float)
    for i in range(m):
        others = [k for k in range(m) if k != i]
        lamb = best_lambda_store[i]
        if lamb is None:
            continue
        for w, k in zip(lamb, others):
            if w >= min_peer_lambda:
                peer[i, k] = w
    peer_matrix = pd.DataFrame(peer, index=alts, columns=alts)
    bmat = bottleneck_counts / np.maximum(bottleneck_counts.sum(axis=0, keepdims=True), 1)
    bottleneck_matrix = pd.DataFrame(bmat, index=pillars, columns=alts)
    targets = pillar_avgs.copy()
    for idx, tech in enumerate(alts):
        targets[tech] = pillar_avgs[tech] * phi_hat[idx]
    return dea_summary, peer_matrix, bottleneck_matrix, targets

def run_monte_carlo_sensitivity(
    pillar_avgs: pd.DataFrame,
    weights_pct: Dict[str, float],
    methods: Sequence[str],
    sims: int = 2000,
    weight_alpha: float = 1.0,
    score_noise_sigma: float = 0.03,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame]]:
    methods = [m.upper() for m in methods]
    if "WEIGHTED" not in methods:
        methods = ["WEIGHTED"] + methods
    alts = pillar_avgs.columns.tolist()
    pillars = pillar_avgs.index.tolist()
    m = len(alts)
    best_counts: Dict[str, Dict[str, int]] = {meth: {a: 0 for a in alts} for meth in methods}
    rank_sum: Dict[str, np.ndarray] = {meth: np.zeros(m, dtype=float) for meth in methods}
    rank_sqsum: Dict[str, np.ndarray] = {meth: np.zeros(m, dtype=float) for meth in methods}
    rank_counts: Dict[str, np.ndarray] = {meth: np.zeros((m, m), dtype=int) for meth in methods}
    A0 = pillar_avgs.to_numpy().T
    alpha_vec = np.full(len(pillars), float(weight_alpha))
    for _ in range(sims):
        w = _dirichlet(alpha_vec)
        noise = np.random.normal(0.0, score_noise_sigma, size=A0.shape)
        A = np.clip(A0 + noise, a_min=0.0, a_max=None)
        dfA = pd.DataFrame(A.T, index=pillars, columns=alts)
        per_method: Dict[str, pd.Series] = {}
        per_method["WEIGHTED"] = method_weighted(dfA, {p: w[i]*100 for i,p in enumerate(pillars)})
        if "WPM" in methods:
            per_method["WPM"] = method_wpm(dfA, {p: w[i]*100 for i,p in enumerate(pillars)})
        if "RANK" in methods:
            per_method["RANK"] = method_rank(dfA, {p: w[i]*100 for i,p in enumerate(pillars)})
        if "TOPSIS" in methods:
            per_method["TOPSIS"] = method_topsis(dfA, {p: w[i]*100 for i,p in enumerate(pillars)})
        if "VIKOR" in methods:
            per_method["VIKOR"] = method_vikor(dfA, {p: w[i]*100 for i,p in enumerate(pillars)})
        if "EDAS" in methods:
            per_method["EDAS"] = method_edas(dfA, {p: w[i]*100 for i,p in enumerate(pillars)})
        if "MAUT" in methods:
            per_method["MAUT"] = method_maut(dfA, {p: w[i]*100 for i,p in enumerate(pillars)})
        if "PCA" in methods:
            per_method["PCA"] = method_pca(dfA)
        for meth, s in per_method.items():
            s_sorted = s.sort_values(ascending=False)
            winner = s_sorted.index[0]
            best_counts[meth][winner] += 1
            ranks = s.rank(ascending=False, method="average")
            rank_vals = ranks.loc[alts].to_numpy(dtype=float)
            for i_alt, rk in enumerate(rank_vals.astype(int)):
                rank_counts[meth][i_alt, rk-1] += 1
            rank_sum[meth] += rank_vals
            rank_sqsum[meth] += rank_vals ** 2
    pbest_rows = []
    for meth, d in best_counts.items():
        for a in alts:
            pbest_rows.append({"Method": meth, "Tech": a, "P(Best)": d[a] / float(sims)})
    Pbest = pd.DataFrame(pbest_rows).pivot(index="Tech", columns="Method", values="P(Best)").fillna(0.0)
    mean_rows = []
    std_rows = []
    RankDist: Dict[str, pd.DataFrame] = {}
    for meth in methods:
        mu = rank_sum[meth] / float(sims)
        var = (rank_sqsum[meth] / float(sims)) - (mu ** 2)
        var = np.maximum(var, 0.0)
        sd = np.sqrt(var)
        for i, a in enumerate(alts):
            mean_rows.append({"Method": meth, "Tech": a, "MeanRank": mu[i]})
            std_rows.append({"Method": meth, "Tech": a, "StdRank": sd[i]})
        rd = (rank_counts[meth] / float(sims))
        RankDist[meth] = pd.DataFrame(rd, index=alts, columns=[f"rank_{k}" for k in range(1, m+1)])
    MeanRank = pd.DataFrame(mean_rows).pivot(index="Tech", columns="Method", values="MeanRank")
    StdRank = pd.DataFrame(std_rows).pivot(index="Tech", columns="Method", values="StdRank")
    return Pbest, MeanRank, StdRank, RankDist

# ----------------------- Validation visuals ----------------
def plot_pbest(series: pd.Series, title: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    series.sort_values(ascending=False).plot(kind="bar", ax=ax, title=title)
    ax.set_ylabel("Probability")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.show()

# ----------------------- PDF report builder ----------------
def build_pdf_report(title: str, sections: List[Dict[str, Any]]) -> bytes:
    buf = io.BytesIO()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    doc = SimpleDocTemplate(buf, pagesize=A4, rightMargin=18*mm, leftMargin=18*mm, topMargin=18*mm, bottomMargin=18*mm)
    styles = getSampleStyleSheet()
    story = []
    style_h = styles["Heading1"]
    style_h.alignment = 1
    story.append(Paragraph(title, style_h))
    story.append(Spacer(1, 6))
    story.append(Paragraph(f"Generated: {now}", styles["Normal"]))
    story.append(Spacer(1, 12))
    for sec in sections:
        story.append(Paragraph(sec.get("heading", ""), styles["Heading2"]))
        if sec.get("text"):
            story.append(Paragraph(sec["text"], styles["Normal"]))
            story.append(Spacer(1, 6))
        for tbl, caption in sec.get("tables", []):
            if tbl is None or tbl.empty:
                continue
            story.append(Paragraph(caption or "Table", styles["Italic"]))
            df = tbl.copy()
            df = df.round(6)
            header = list(df.reset_index().columns)
            data = [header] + df.reset_index().values.tolist()
            tbl_style = RLTable(data, hAlign="LEFT")
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
                story.append(Paragraph(f"[Image error: {e}]", styles["Normal"]))
        story.append(PageBreak())
    doc.build(story)
    buf.seek(0)
    return buf.getvalue()

# ----------------------- Streamlit UI ---------------------
apply_pro_style()
st.set_page_config(page_title="Risk Assessment Toolkit", layout="wide")
st.title("Risk Assessment Toolkit")
st.markdown("Use tabs to manage Risks, ESGFP scoring, Scenarios, Validation and Compliance TEA.")

# Session state initialization
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

# Sidebar quick actions
st.sidebar.header("Quick Actions")
if st.sidebar.button("Load example dataset"):
    pillars = dict(DEFAULT_ESGFP)
    techs = {
        "Process Design A": {f"{p}:{sub}": round(5.0 * (1.0 + 0.25*(i%3)), 3) for i, (p, subs) in enumerate(pillars.items()) for sub in subs},
        "Process Design B": {f"{p}:{sub}": round(6.5 * (1.0 + 0.1*(i%2)), 3) for i, (p, subs) in enumerate(pillars.items()) for sub in subs},
    }
    st.session_state.pillars = pillars
    st.session_state.scores_by_tech = techs
    st.success("Example dataset loaded.")

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

st.sidebar.download_button("Download example CSV", make_example_csv_bytes(st.session_state.pillars), file_name="esgfp_example_template.csv", mime="text/csv")
st.sidebar.markdown("---")
st.sidebar.write("Dependencies: streamlit, pandas, numpy, matplotlib, plotly, reportlab, pillow")

# Tabs
tabs = st.tabs(["Risk Assessment", "ESGFP Scoring", "Scenario Analysis (MCDA)", "Validation (DEA + Monte Carlo)", "Export Reports", "Compliance / ESG TEA"])

# ----- Tab 0: Risk Assessment -----
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
            df_r = pd.DataFrame([{"Risk": r["name"], "Probability": r["prob"], "Severity": r["sev"], "Rating": r["prob"] * r["sev"]} for r in st.session_state.risks])
            st.dataframe(df_r.round(4))
            if st.button("Clear risks"):
                st.session_state.risks = []
        else:
            st.info("No risks added yet.")
    with col_r:
        st.subheader("Risk Visuals")
        if st.session_state.risks:
            risks_objs = [Risk(r["name"], r["prob"], r["sev"]) for r in st.session_state.risks]
            df_for_plot = risk_dataframe(risks_objs)
            before = plt.get_fignums()
            plot_risk_views(df_for_plot)
            for num in plt.get_fignums():
                if num not in before:
                    st.pyplot(plt.figure(num))
            fig = px.scatter(df_for_plot, x="Probability", y="Severity", size="Rating", hover_name="Risk", title="Risk Bubble – Likelihood × Impact")
            st.plotly_chart(fig, use_container_width=True)
            csv_b = io.BytesIO()
            df_for_plot.to_csv(csv_b, index=False)
            csv_b.seek(0)
            st.download_button("Download risk table (CSV)", data=csv_b, file_name="risk_table.csv", mime="text/csv")
        else:
            st.info("Add risks to see visuals.")

# ----- Tab 1: ESGFP Scoring -----
with tabs[1]:
    st.header("ESGFP Scoring")
    left, right = st.columns([2, 1])
    with left:
        st.subheader("Pillars & Key Issues (session)")
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
        uploaded = st.file_uploader("Upload ESGFP CSV", type=["csv"], key="esg_upload")
        if uploaded:
            try:
                df_up = pd.read_csv(uploaded)
                if "Tech" not in df_up.columns and df_up.shape[1] >= 1:
                    df_up.columns.values[0] = "Tech"
                df_up = df_up.set_index("Tech")
                for col in df_up.columns:
                    if ":" in col:
                        p, sub = col.split(":", 1)
                        if p not in st.session_state.pillars:
                            st.session_state.pillars[p] = []
                        if sub not in st.session_state.pillars[p]:
                            st.session_state.pillars[p].append(sub)
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
        else:
            st.info("No process designs in session.")

    with right:
        st.subheader("Visualize & Compute Pillar Averages")
        if not st.session_state.scores_by_tech:
            st.info("Add process designs first.")
        else:
            pillar_avgs = pillar_averages_multi(st.session_state.scores_by_tech, st.session_state.pillars)
            st.session_state.pillar_avgs_df = pillar_avgs.copy()
            st.write("Pillar Averages (raw):")
            st.dataframe(pillar_avgs.round(4))
            before = plt.get_fignums()
            plot_pillar_heatmaps(pillar_avgs)
            for num in plt.get_fignums():
                if num not in before:
                    st.pyplot(plt.figure(num))
            before = plt.get_fignums()
            plot_radar_profiles(pillar_avgs)
            for num in plt.get_fignums():
                if num not in before:
                    st.pyplot(plt.figure(num))
            before = plt.get_fignums()
            plot_parallel_coordinates(pillar_avgs)
            for num in plt.get_fignums():
                if num not in before:
                    st.pyplot(plt.figure(num))
            try:
                df_wide = pillar_avgs.T
                fig_par = px.parallel_coordinates(df_wide.reset_index(), labels={c: c for c in df_wide.columns}, title="Parallel Coordinates (Pillar Profiles)")
                st.plotly_chart(fig_par, use_container_width=True)
            except Exception:
                pass
            csv_b = io.BytesIO()
            pillar_avgs.to_csv(csv_b)
            csv_b.seek(0)
            st.download_button("Download pillar averages (CSV)", data=csv_b, file_name="pillar_averages.csv", mime="text/csv")

# ----- Tab 2, 3, 4 handled in next chunk -----

# ----------------------- Tab 2: Scenario Analysis (MCDA) ----------------
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
                w = st.number_input(f"{p} (%)", min_value=0.0, max_value=100.0, value=round(100.0 / len(pillar_avgs.index), 2), key=f"w_{p}")
                weights[p] = float(w)
        total = sum(weights.values())
        if abs(total - 100.0) > 1e-6:
            st.warning(f"Weights sum to {total:.2f}%. They must sum to 100%.")
        methods_selected = st.multiselect("Select MCDA Methods", options=["WEIGHTED", "WPM", "RANK", "TOPSIS", "VIKOR", "EDAS", "MAUT", "PCA"], default=["WEIGHTED"], key="methods_select")
        norm_flag = st.checkbox("Normalize outputs to 0–10 scale", value=True, key="norm_flag")
        if st.button("Run scenarios", key="run_scenarios_btn"):
            per_method = {}
            if "WEIGHTED" in methods_selected:
                per_method["WEIGHTED"] = method_weighted(pillar_avgs, weights)
            if "WPM" in methods_selected:
                per_method["WPM"] = method_wpm(pillar_avgs, weights)
            if "RANK" in methods_selected:
                per_method["RANK"] = method_rank(pillar_avgs, weights)
            if "TOPSIS" in methods_selected:
                per_method["TOPSIS"] = method_topsis(pillar_avgs, weights)
            if "VIKOR" in methods_selected:
                per_method["VIKOR"] = method_vikor(pillar_avgs, weights)
            if "EDAS" in methods_selected:
                per_method["EDAS"] = method_edas(pillar_avgs, weights)
            if "MAUT" in methods_selected:
                per_method["MAUT"] = method_maut(pillar_avgs, weights)
            if "PCA" in methods_selected:
                per_method["PCA"] = method_pca(pillar_avgs)
            scenario_df = pd.DataFrame(per_method)
            scenario_df.index.name = "Alternative"
            if norm_flag:
                scaled_cols = {col: _scale_series_by_method(scenario_df[col], col) for col in scenario_df.columns}
                scenario_df_scaled = pd.DataFrame(scaled_cols, index=scenario_df.index)
            else:
                scenario_df_scaled = scenario_df.copy()
            scenario_name = f"Scenario {len(st.session_state.scenario_results) + 1}"
            st.session_state.scenario_results[scenario_name] = {"weights": weights, "methods": methods_selected, "results": scenario_df_scaled}
            st.session_state.last_weights = weights
            st.session_state.last_methods = methods_selected
        if st.session_state.scenario_results:
            st.subheader("Stored Scenarios")
            for name, sdata in st.session_state.scenario_results.items():
                st.markdown(f"**{name}** — Methods: {', '.join(sdata['methods'])}")
                st.dataframe(sdata["results"].round(4))
            try:
                all_plot_data = []
                for name, sdata in st.session_state.scenario_results.items():
                    df = sdata["results"].copy()
                    df["Scenario"] = name
                    all_plot_data.append(df.reset_index().melt(id_vars=["Alternative", "Scenario"], var_name="Method", value_name="Score"))
                plot_df = pd.concat(all_plot_data, ignore_index=True)
                fig = px.bar(plot_df, x="Alternative", y="Score", color="Method", barmode="group", facet_row="Scenario", title="Scenario Comparisons by Method")
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                pass
            last_key = list(st.session_state.scenario_results.keys())[-1]
            last_df = st.session_state.scenario_results[last_key]["results"]
            csv_b = io.BytesIO()
            last_df.to_csv(csv_b)
            csv_b.seek(0)
            st.download_button("Download latest scenario (CSV)", data=csv_b, file_name=f"{last_key.replace(' ','_')}.csv", mime="text/csv")

# ----------------------- Tab 3: Validation ----------------
with tabs[3]:
    st.header("Validation — DEA (approx) & Monte Carlo")
    if st.session_state.pillar_avgs_df.empty:
        st.info("Need pillar averages (from ESGFP tab).")
    else:
        pillar_avgs = st.session_state.pillar_avgs_df.copy()
        st.write("Pillar Averages:")
        st.dataframe(pillar_avgs.round(4))
        dea_samples = st.number_input("DEA convex-hull samples", min_value=100, max_value=200000, value=5000, step=100, key="dea_samples")
        peer_cut = st.number_input("Peer display cutoff", min_value=0.0, max_value=1.0, value=0.05, step=0.01, key="peer_cut")
        sims = st.number_input("Monte Carlo sims", min_value=100, max_value=200000, value=2000, step=100, key="mc_sims")
        alpha = st.number_input("Dirichlet alpha", min_value=0.1, max_value=10.0, value=1.0, step=0.1, key="mc_alpha")
        sigma = st.number_input("Score noise sigma", min_value=0.0, max_value=0.5, value=0.03, step=0.01, key="mc_sigma")
        if st.button("Run validation suite", key="run_validation_btn"):
            with st.spinner("Running DEA diagnostics."):
                dea_summary, peer_matrix, bottleneck_matrix, targets = approx_dea_diagnostics(
                    pillar_avgs, samples=int(dea_samples), min_peer_lambda=float(peer_cut)
                )
            st.subheader("DEA Summary")
            st.dataframe(dea_summary.round(4))
            # Bottleneck heatmap
            try:
                fig, ax = plt.subplots(figsize=(10,6))
                im = ax.imshow(bottleneck_matrix.values, aspect="auto", cmap="OrRd")
                ax.set_xticks(range(bottleneck_matrix.shape[1])); ax.set_xticklabels(bottleneck_matrix.columns, rotation=45, ha="right")
                ax.set_yticks(range(bottleneck_matrix.shape[0])); ax.set_yticklabels(bottleneck_matrix.index)
                for i in range(bottleneck_matrix.shape[0]):
                    for j in range(bottleneck_matrix.shape[1]):
                        ax.text(j, i, f"{bottleneck_matrix.values[i,j]:.2f}", ha="center", va="center", fontsize=9)
                ax.set_title("DEA Bottleneck Frequency by Pillar")
                st.pyplot(fig)
            except Exception:
                pass
            # Peer heatmap
            try:
                fig, ax = plt.subplots(figsize=(10,8))
                im = ax.imshow(peer_matrix.values, aspect="auto", cmap="Blues")
                ax.set_xticks(range(peer_matrix.shape[1])); ax.set_xticklabels(peer_matrix.columns, rotation=45, ha="right")
                ax.set_yticks(range(peer_matrix.shape[0])); ax.set_yticklabels(peer_matrix.index)
                for i in range(peer_matrix.shape[0]):
                    for j in range(peer_matrix.shape[1]):
                        val = peer_matrix.values[i, j]
                        if val > 0:
                            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9)
                ax.set_title("DEA Peer Reference Weights (avg best mix)")
                st.pyplot(fig)
            except Exception:
                pass
            dom = compute_dominance_matrix(pillar_avgs)
            st.subheader("Dominance matrix")
            st.dataframe(dom)
            with st.spinner("Running Monte-Carlo sensitivity."):
                Pbest, MeanRank, StdRank, RankDist = run_monte_carlo_sensitivity(
                    pillar_avgs,
                    st.session_state.last_weights if st.session_state.last_weights is not None else {p: 100.0/len(pillar_avgs.index) for p in pillar_avgs.index},
                    st.session_state.last_methods if st.session_state.last_methods is not None else ["WEIGHTED"],
                    sims=int(sims),
                    weight_alpha=float(alpha),
                    score_noise_sigma=float(sigma),
                )
            st.subheader("Monte-Carlo: P(Best)")
            st.dataframe(Pbest.round(4))
            if "WEIGHTED" in Pbest.columns:
                fig = px.bar(Pbest.reset_index(), x="Tech", y="WEIGHTED", title="Monte-Carlo P(Best) — WEIGHTED")
                st.plotly_chart(fig, use_container_width=True)
            if "WEIGHTED" in RankDist:
                try:
                    rd = RankDist["WEIGHTED"]
                    fig = go.Figure()
                    ranks = list(rd.columns)
                    x = rd.index.tolist()
                    for col in ranks:
                        fig.add_trace(go.Bar(name=col, x=x, y=rd[col], offsetgroup=0))
                    fig.update_layout(barmode="stack", title="Rankogram — WEIGHTED")
                    st.plotly_chart(fig, use_container_width=True)
                except Exception:
                    pass
            st.subheader("Mean Rank & StdDev")
            st.dataframe(MeanRank.round(4))
            st.dataframe(StdRank.round(4))

# ----------------------- Tab 4: Export Reports ----------------
with tabs[4]:
    st.header("Export Reports (PDF)")
    def collect_risk_section():
        tables = []
        images = []
        if st.session_state.risks:
            df_r = pd.DataFrame([{"Risk": r["name"], "Probability": r["prob"], "Severity": r["sev"], "Rating": r["prob"] * r["sev"]} for r in st.session_state.risks])
            tables.append((df_r, "Risk table"))
            try:
                before = plt.get_fignums()
                plot_risk_views(df_r)
                for num in plt.get_fignums():
                    if num not in before:
                        fig = plt.figure(num)
                        png = mpl_fig_to_png_bytes(fig)
                        images.append((png, "Risk visuals"))
            except Exception:
                pass
        return {"heading":"Risk Assessment", "text":"Risk inputs & visuals", "tables": tables, "images": images}

    def collect_esgfp_section():
        tables = []
        images = []
        if st.session_state.scores_by_tech:
            df_scores = pd.DataFrame(st.session_state.scores_by_tech)
            tables.append((df_scores.round(6), "ESGFP scores (Techs x Pillar:Key)"))
        if not st.session_state.pillar_avgs_df.empty:
            pa = st.session_state.pillar_avgs_df
            tables.append((pa.round(6), "Pillar averages"))
            try:
                before = plt.get_fignums()
                plot_pillar_heatmaps(pa)
                plot_radar_profiles(pa)
                for num in plt.get_fignums():
                    if num not in before:
                        fig = plt.figure(num)
                        images.append((mpl_fig_to_png_bytes(fig), "Pillar visuals"))
            except Exception:
                pass
        return {"heading":"ESGFP Scoring", "text":"Process designs, scores, and pillar averages", "tables": tables, "images": images}

    def collect_scenario_section():
        tables = []
        images = []
        if not st.session_state.pillar_avgs_df.empty and st.session_state.last_weights is not None and st.session_state.last_methods is not None:
            pa = st.session_state.pillar_avgs_df
            per_method = {}
            last_methods = st.session_state.last_methods
            weights = st.session_state.last_weights
            per_method["WEIGHTED"] = method_weighted(pa, weights)
            for m in last_methods:
                if m == "WPM": per_method["WPM"] = method_wpm(pa, weights)
                if m == "RANK": per_method["RANK"] = method_rank(pa, weights)
                if m == "TOPSIS": per_method["TOPSIS"] = method_topsis(pa, weights)
                if m == "VIKOR": per_method["VIKOR"] = method_vikor(pa, weights)
                if m == "EDAS": per_method["EDAS"] = method_edas(pa, weights)
                if m == "MAUT": per_method["MAUT"] = method_maut(pa, weights)
                if m == "PCA": per_method["PCA"] = method_pca(pa)
            scenario_df = pd.DataFrame(per_method)
            tables.append((scenario_df.round(6), "Scenario results"))
            try:
                fig = px.bar(scenario_df.reset_index().melt(id_vars="index", var_name="Method", value_name="Score"), x="index", y="Score", color="Method", barmode="group")
                img = fig.to_image(format="png")
                images.append((img, "Scenario comparison"))
            except Exception:
                pass
        return {"heading":"Scenarios", "text":"MCDA scenario results (if available)", "tables": tables, "images": images}

    sections = []
    sections.append(collect_risk_section())
    sections.append(collect_esgfp_section())
    sections.append(collect_scenario_section())
    pdf_bytes = build_pdf_report("Risk Assessment Toolkit Report", sections)
    if pdf_bytes:
        st.download_button("Download combined report (PDF)", data=pdf_bytes, file_name="pro_desg_report.pdf", mime="application/pdf")

# ----------------------- Tab 5: Compliance / ESG TEA ----------------
with tabs[5]:
    st.header("Compliance / ESG TEA")
    st.write("Run the TEA/compliance module and view results and plots. Upload your compliance script (.py) or use the default uploaded one.")
    user_path = st.text_input("Compliance script path (local)", DEFAULT_COMPLIANCE_PATH, key="comp_path")
    uploaded_file = st.file_uploader("(Optional) Upload a compliance script to use instead", type=["py"], key="comp_upload")
    module_path = None
    if uploaded_file is not None:
        try:
            temp_dir = tempfile.gettempdir()
            tmp_path = os.path.join(temp_dir, "compliance_uploaded.py")
            with open(tmp_path, "wb") as fh:
                fh.write(uploaded_file.read())
            module_path = tmp_path
            st.success(f"Uploaded script saved to: {tmp_path}")
        except Exception as e:
            st.error(f"Could not save uploaded script: {e}")
            module_path = user_path
    else:
        module_path = user_path

    comp_mod = None
    try:
        comp_mod = load_compliance_module(module_path)
    except Exception as e:
        st.error(f"Could not load compliance module: {e}")
        comp_mod = None

    if comp_mod is not None:
        if not hasattr(comp_mod, "compute_TEA"):
            st.error("Loaded compliance script does not expose `compute_TEA(params)` function.")
        else:
            # Build defaults and UI for many TEA params
            st.subheader("TEA Parameters (edit values or use JSON editor)")
            base_params = {
                "C_PE": 1e8, "COL": 1e7, "C_RM": 4e7, "C_UT": 1.2e7, "C_CAT": 2e6,
                "Q_prod": 5e5, "P_prod": 550.0, "f_ins": 0.30, "f_pipe": 0.45, "f_elec": 0.10,
                "f_bldg": 0.05, "f_util": 0.06, "f_stor": 0.02, "f_safe": 0.01, "f_waste": 0.01,
                "f_eng": 0.12, "f_cons": 0.10, "f_licn": 0.00, "f_cont": 0.02, "f_contg": 0.0,
                "f_insur": 0.01, "f_own": 0.02, "f_start": 0.01,
                "N_project": 20, "L_asset": 20, "salv_frac": 0.10, "f_risk_op": 0.05,
                "tau_CO2": 50.0, "E_CO2": 200000.0, "f_pack": 0.02, "f_esg": 0.07,
                "i_base": 0.08, "delta_risk": 0.03, "dep_method": "SL"
            }
            if hasattr(comp_mod, "base_params") and isinstance(getattr(comp_mod, "base_params"), dict):
                try:
                    base_params.update(getattr(comp_mod, "base_params"))
                except Exception:
                    pass

            c1, c2, c3 = st.columns(3)
            with c1:
                base_params["C_PE"] = st.number_input("C_PE (USD)", value=float(base_params["C_PE"]), format="%.2f", key="C_PE")
                base_params["C_RM"] = st.number_input("C_RM (USD/yr)", value=float(base_params["C_RM"]), format="%.2f", key="C_RM")
                base_params["P_prod"] = st.number_input("P_prod (USD/ton)", value=float(base_params["P_prod"]), format="%.2f", key="P_prod")
            with c2:
                base_params["Q_prod"] = st.number_input("Q_prod (ton/yr)", value=float(base_params["Q_prod"]), format="%.2f", key="Q_prod")
                base_params["COL"] = st.number_input("COL (USD/yr)", value=float(base_params["COL"]), format="%.2f", key="COL")
                base_params["f_esg"] = st.number_input("f_esg (fraction of OPEX)", value=float(base_params["f_esg"]), format="%.4f", key="f_esg")
            with c3:
                base_params["N_project"] = st.number_input("N_project (yrs)", value=int(base_params["N_project"]), min_value=1, key="N_project")
                base_params["i_base"] = st.number_input("i_base (discount rate)", value=float(base_params["i_base"]), format="%.4f", key="i_base")
                base_params["dep_method"] = st.selectbox("dep_method", options=["SL","SYD","DDB"], index=0, key="dep_method")

            st.markdown("**Advanced parameters (JSON editor)** — edit freely then press *Apply JSON*")
            params_json = st.text_area("params JSON", value=json.dumps(base_params, indent=2), height=240, key="params_json")

            if st.button("Apply JSON", key="apply_json_btn"):
                try:
                    parsed = json.loads(params_json)
                    if isinstance(parsed, dict):
                        base_params.update(parsed)
                        st.success("Applied JSON parameters.")
                    else:
                        st.error("JSON must encode a dictionary of parameters.")
                except Exception as e:
                    st.error(f"Invalid JSON: {e}")

            run_col, save_col = st.columns([1,1])
            if run_col.button("Run TEA (compute_TEA)", key="run_tea_btn"):
                with st.spinner("Running TEA..."):
                    try:
                        out = comp_mod.compute_TEA(base_params)
                    except Exception as e:
                        st.exception(f"compute_TEA raised an exception: {e}")
                        out = None
                if out is not None:
                    st.success("TEA computed.")
                    keys = ["CAPEX", "LCOx", "NPV", "IRR", "Salvage", "CRF", "Annual_CAPEX", "PV_revenue", "PV_cost_total", "BCR"]
                    st.subheader("Key TEA Results")
                    for k in keys:
                        if k in out:
                            try:
                                if k == "IRR":
                                    st.write(f"{k}: {float(out[k])*100:.3f}%")
                                else:
                                    st.write(f"{k}: {out[k]}")
                            except Exception:
                                st.write(f"{k}: {out[k]}")
                    others = {k: v for k, v in out.items() if k not in keys and (isinstance(v, (int, float, str)) or (isinstance(v, (list,tuple, np.ndarray)) and len(v)<=100))}
                    if others:
                        st.subheader("Other outputs (sample)")
                        st.write(others)
                    if "CF" in out:
                        try:
                            cf = list(out["CF"])
                            years = list(range(len(cf)))
                            fig, ax = plt.subplots(figsize=(8,4))
                            ax.bar(years, cf)
                            ax.set_xlabel("Year"); ax.set_ylabel("Cash Flow (USD)"); ax.set_title("Annual Cash Flow")
                            ax.grid(True, alpha=0.3)
                            st.pyplot(fig)
                            cum = np.cumsum(cf)
                            fig2, ax2 = plt.subplots(figsize=(8,4))
                            ax2.plot(years, cum, marker="o"); ax2.axhline(0.0, linestyle="--", color="k", alpha=0.6)
                            ax2.set_xlabel("Year"); ax2.set_ylabel("Cumulative Cash Flow (USD)"); ax2.set_title("Cumulative Cash Flow (Payback)")
                            ax2.grid(True, alpha=0.3)
                            st.pyplot(fig2)
                        except Exception:
                            pass
                    try:
                        out_serializable = {}
                        for k, v in out.items():
                            if isinstance(v, np.ndarray):
                                out_serializable[k] = v.tolist()
                            else:
                                out_serializable[k] = v
                        out_json = json.dumps(out_serializable, indent=2)
                        st.download_button("Download TEA results (JSON)", out_json, file_name="tea_results.json", mime="application/json", key="download_tea_json")
                    except Exception:
                        st.info("Could not create download (some outputs may be non-serializable).")
                else:
                    st.error("TEA run failed; see error above.")

            if save_col.button("Save params JSON to file", key="save_params_btn"):
                try:
                    fname = os.path.join(tempfile.gettempdir(), "tea_params.json")
                    with open(fname, "w") as fh:
                        fh.write(json.dumps(base_params, indent=2))
                    st.success(f"Saved params to {fname}")
                except Exception as e:
                    st.error(f"Could not save params file: {e}")

            st.subheader("Advanced analyses (if provided by script)")
            ac1, ac2, ac3 = st.columns([1,1,1])
            if ac1.button("Run ESG sweep visuals", key="esg_sweep_btn"):
                if hasattr(comp_mod, "run_esg_sweep_and_plots"):
                    try:
                        comp_mod.run_esg_sweep_and_plots(base_params, design_label="Streamlit run")
                        for num in plt.get_fignums():
                            st.pyplot(plt.figure(num))
                        st.success("ESG sweep completed.")
                    except Exception as e:
                        st.exception(f"ESG sweep failed: {e}")
                else:
                    st.warning("run_esg_sweep_and_plots not found in compliance script.")
            if ac2.button("Price sweep (NPV & LCOx)", key="price_sweep_btn"):
                if hasattr(comp_mod, "price_sweep"):
                    try:
                        comp_mod.price_sweep(base_params)
                        for num in plt.get_fignums():
                            st.pyplot(plt.figure(num))
                        st.success("Price sweep completed.")
                    except Exception as e:
                        st.exception(f"price_sweep failed: {e}")
                else:
                    st.warning("price_sweep not found in compliance script.")
            if ac3.button("Scenario CBA (3 scenarios)", key="scenario_cba_btn"):
                if hasattr(comp_mod, "scenario_cba"):
                    try:
                        comp_mod.scenario_cba(base_params, design_label="Streamlit_run")
                        for num in plt.get_fignums():
                            st.pyplot(plt.figure(num))
                        st.success("Scenario CBA completed.")
                    except Exception as e:
                        st.exception(f"scenario_cba failed: {e}")
                else:
                    st.warning("scenario_cba not found in compliance script.")
