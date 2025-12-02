# -*- coding: utf-8 -*-
"""
Created on Thu Oct  9 16:45:28 2025

@author: a1917785
"""

# -*- coding: utf-8 -*-
"""
Interactive ESGFP + Risk assessment with edit/re-run menus, pro visuals,
user-extendable pillar key issues, and validation (DEA extended + Monte Carlo).

NOTE: This version does NOT save any files. All outputs display on screen only.

What’s inside:
- Risk section with full edit loop (add/edit/rename/delete/re-run/proceed)
- ESGFP section (N technologies) with:
  • Add/edit tech scores
  • Add new key issues to any pillar (applies across all technologies)
  • Re-run visuals or proceed to scenarios
- Scenarios: user-selectable MCDA methods (Weighted, WPM, Rank, TOPSIS, VIKOR, EDAS, MAUT, PCA)
- Validation: DEA (extended, solver-free) + Monte-Carlo sensitivity
- Professional visuals for sustainability/engineering audiences

Type 'help' at any prompt for guidance. Press Enter to accept defaults when offered.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Sequence, Union, Optional
import sys
import math
import re
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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


# ----------------------- Help / Cheat sheets ----------------
def method_cheatsheet() -> None:
    print(r"""
🧭 MCDA Quick Course

Core here:
• WSM / WLC (Weighted Sum / Weighted Linear Combination)
  - Weight source: Manual / expert
  - Aggregation: Additive
  - Suitable for: ESG, policy, design ranking
  - Enter pillar weights summing to 100%.

Also available:
• WPM – Multiplicative (ratio tradeoffs), tech comparison
• Rank-based – Additive over ranks, early-stage screening
• TOPSIS / VIKOR / EDAS – Distance/compromise-based
• MAUT – Utility-weighted (risk–benefit, diminishing returns)
• PCA – Statistical, objective (correlated criteria)

Validation:
• DEA (approximate convex-hull, output-oriented, VRS)
  - P(frontier), φ̂ (radial expansion), EffOut=1/φ̂, bottlenecks, peer mix, targets.
• Monte-Carlo sensitivity
  - Random weights (Dirichlet) + mild noise → P(Best), rank stability.
""")


def validation_help() -> None:
    print(r"""
🔎 Validation — quick help

DEA (approximate, output-oriented, VRS):
• A tech is “efficient” if no convex mix of other techs can match/exceed it on all pillars.
• We simulate many peer mixes and estimate:
  - Frontier probability: share of mixes that envelope the tech (1=often efficient).
  - Radial expansion φ̂: scale factor where φ̂·y_i is still matched by peers; EffOut = 1/φ̂.
  - Bottleneck pillar: pillar that most often limits φ̂.
  - Peers: average reference weights in the best mix.
  - Targets: projected pillar vector ≈ φ̂·current.

Monte-Carlo sensitivity:
• Randomize weights (Dirichlet α, default 1.0) and add mild score noise (σ).
• Re-run your chosen methods and summarize P(Best), mean/σ of ranks, rankograms.
""")


# ----------------------- Input helpers ---------------------
def _is_help(s: str) -> bool:
    return s.strip().lower() in {"help", "?", "h"}


def prompt_int(prompt: str, min_val: int, max_val: int) -> int:
    while True:
        raw = input(prompt).strip()
        if _is_help(raw):
            print(f"Hint: enter an integer in [{min_val}, {max_val}].")
            continue
        try:
            v = int(raw)
            if min_val <= v <= max_val:
                return v
            print(f"❌ Enter integer in [{min_val}, {max_val}]")
        except ValueError:
            print("❌ Invalid integer. Type 'help' for guidance.")


def prompt_str_nonempty(prompt: str, default: str) -> str:
    raw = input(f"{prompt} [default {default}]: ").strip()
    return raw or default


def prompt_float(
    prompt: str,
    lo: float,
    hi: float,
    inclusive_low: bool = True,
    inclusive_high: bool = True,
) -> float:
    while True:
        raw = input(prompt).strip()
        if _is_help(raw):
            print(
                f"Hint: enter a number in {'[' if inclusive_low else '('}{lo}, {hi}{']' if inclusive_high else ')'}"
            )
            continue
        try:
            v = float(raw)
            ok_lo = v >= lo if inclusive_low else v > lo
            ok_hi = v <= hi if inclusive_high else v < hi
            if ok_lo and ok_hi:
                return v
            bounds = f"{'[' if inclusive_low else '('}{lo}, {hi}{']' if inclusive_high else ')'}"
            print(f"❌ Enter number in {bounds}")
        except ValueError:
            print("❌ Invalid number. Type 'help' for guidance.")


def prompt_float_or_default(prompt: str, default: float, lo: float, hi: float) -> float:
    while True:
        raw = input(f"{prompt} [default {default}]: ").strip()
        if raw == "":
            return default
        if _is_help(raw):
            print(f"Hint: press Enter to use default. Otherwise enter a number in [{lo}, {hi}].")
            continue
        try:
            v = float(raw)
            if lo <= v <= hi:
                return v
            print(f"❌ Enter number in [{lo}, {hi}]")
        except ValueError:
            print("❌ Invalid number. Type 'help' or press Enter for default.")


def prompt_exposure(prompt: str) -> float:
    """Accept label shortcuts or any numeric in [0,1]."""
    while True:
        raw = input(prompt).strip().lower()
        if _is_help(raw):
            print("Hint: type high/moderate/moderate to lower/lower or any number in [0,1].")
            continue
        if raw in EXPOSURE_MAP:
            return float(EXPOSURE_MAP[raw])
        try:
            if raw.startswith("."):
                raw = "0" + raw
            v = float(raw)
            if 0.0 <= v <= 1.0:
                return v
        except ValueError:
            pass
        print("❌ Use: high/moderate/moderate to lower/lower or numeric ∈ [0,1].")


# ----------------------- Risk section ----------------------
def risk_dataframe(risks: List[Risk]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"Risk": r.name, "Probability": r.probability, "Severity": r.severity, "Rating": r.rating} for r in risks]
    )


def plot_risk_views(df: pd.DataFrame) -> None:
    if df.empty:
        print("(No risks to plot.)")
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


def risk_section() -> List[Risk]:
    """Interactive risk section with full edit/re-run loop."""
    risks: List[Risk] = []

    print("\n🔍 MATERIALITY ASSESSMENT & RISK IDENTIFICATION")
    print("   Probability: (0,1]; Severity: [1,10]. Rating = Probability × Severity.")
    while True:
        cmd = input(
            "\n[Risks] Choose: [A]dd  [E]dit  Re[n]ame  [D]elete  [C]ompute visuals  [L]ist  [H]elp  [P]roceed → "
        ).strip().lower()

        if cmd in {"h", "help", "?"}:
            print("Help: Add risks, then Compute to see visuals. Edit/rename/delete as needed, then Proceed.")
            continue
        if cmd in {"l", "list"}:
            if not risks:
                print("(no risks yet)")
            else:
                for i, r in enumerate(risks, 1):
                    print(f"{i:2d}. {r.name} | P={r.probability:.3f}, S={r.severity:.2f}, R={r.rating:.2f}")
            continue
        if cmd in {"a", "add"}:
            k = prompt_int("   How many risks to add? ", 1, 1000)
            for _ in range(k):
                name = prompt_str_nonempty("   - Risk name", "New Risk")
                prob = prompt_float("     Probability (0–1], e.g., 0.25: ", 0.0, 1.0, inclusive_low=False, inclusive_high=True)
                sev = prompt_float("     Severity [1–10]: ", 1.0, 10.0)
                risks.append(Risk(name=name, probability=prob, severity=sev))
            continue
        if cmd in {"e", "edit"}:
            if not risks:
                print("No risks to edit.")
                continue
            idx = prompt_int("   Edit which risk #? ", 1, len(risks)) - 1
            r = risks[idx]
            print(f"   Editing '{r.name}' (press Enter to keep current)")
            raw = input(f"     Probability current={r.probability:.3f}: ").strip()
            if raw:
                try:
                    v = float(raw); assert 0.0 < v <= 1.0
                    r.probability = v
                except Exception:
                    print("     ❌ ignored (must be (0,1])")
            raw = input(f"     Severity current={r.severity:.2f}: ").strip()
            if raw:
                try:
                    v = float(raw); assert 1.0 <= v <= 10.0
                    r.severity = v
                except Exception:
                    print("     ❌ ignored (must be [1,10])")
            continue
        if cmd in {"n", "rename"}:
            if not risks:
                print("No risks to rename.")
                continue
            idx = prompt_int("   Rename which risk #? ", 1, len(risks)) - 1
            r = risks[idx]
            new_name = prompt_str_nonempty(f"   New name for '{r.name}'", r.name)
            r.name = new_name
            continue
        if cmd in {"d", "delete"}:
            if not risks:
                print("No risks to delete.")
                continue
            idx = prompt_int("   Delete which risk #? ", 1, len(risks)) - 1
            del risks[idx]
            print("   Deleted.")
            continue
        if cmd in {"c", "compute"}:
            df = risk_dataframe(risks)
            print("\n📊 Risk Ratings:")
            print(df)
            plot_risk_views(df)
            # After visuals, offer re-run choices
            choice = input("Re-run risks? [E]dit / [A]dd / [D]elete / [P]roceed → ").strip().lower()
            if choice in {"p", "proceed", ""}:
                break
            # Otherwise loop continues for edits
            continue
        if cmd in {"p", "proceed", ""}:
            if not risks:
                print("Proceeding with 0 risks (you can still score ESGFP).")
            break
        print("❌ Unknown option. Type 'help' for guidance.")
    return risks


# ----------------------- ESGFP scoring (N technologies) ----
def _sanitize_filename(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def _norm0_10(values: Sequence[float], base: float = WEIGHTED_THEORETICAL_MAX) -> List[float]:
    return [(float(v) / base) * OUTPUT_SCALE for v in values]


def collect_esgfp_scores_for_tech(tech_label: str, pillars: Dict[str, List[str]]) -> Dict[str, float]:
    print(f"\n📥 ESGFP Scores for {tech_label}")
    print("   Sub-metric score in [1–9]. Geographic Exposure multiplies score by (1 + exposure).")
    print("   Exposure: high(1.0), moderate(0.75), moderate to lower(0.5), lower(0.25) or any numeric in [0,1].")
    scores: Dict[str, float] = {}
    for pillar, subs in pillars.items():
        for sub in subs:
            s = prompt_float(f" - {pillar} → {sub} (1–9): ", 1.0, 9.0)
            g = prompt_exposure("   - Geographic Exposure: ")
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
        print("(No pillar averages to plot.)")
        return

    data = pillar_avgs.copy()

    # Raw
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

    # Z-score per pillar (row-wise)
    mean = data.mean(axis=1)
    std = data.std(axis=1).replace(0, 1.0)
    z = data.sub(mean, axis=0).div(std, axis=0)

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(z.values, aspect="auto", cmap="coolwarm")
    ax.set_xticks(range(z.shape[1]))
    ax.set_xticklabels(z.columns, rotation=45, ha="right")
    ax.set_yticks(range(z.shape[0]))
    ax.set_yticklabels(z.index)
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            ax.text(j, i, f"{z.values[i, j]:.2f}", ha="center", va="center", fontsize=9)
    ax.set_title("Pillar Averages – Z-score by Pillar")
    fig.colorbar(im, ax=ax, shrink=0.8, label="z")
    fig.tight_layout()
    plt.show()


def _radar_axes(num_vars: int, frame: str = "circle"):
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    ax = plt.subplot(111, polar=True)
    if frame == "circle":
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
    return ax, angles


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


def esgfp_section() -> Tuple[Dict[str, Dict[str, float]], Dict[str, List[str]]]:
    """
    Full ESGFP interactive section:
      - Add/edit/delete technologies & scores
      - Add new key issues to any pillar
      - Re-run visuals or proceed
    Returns scores_by_tech and the (possibly extended) pillars dict.
    """
    # Start from defaults (make a deep-ish copy)
    pillars: Dict[str, List[str]] = {p: list(subs) for p, subs in DEFAULT_ESGFP.items()}
    scores_by_tech: Dict[str, Dict[str, float]] = {}

    print("\n📦 ESGFP SECTION — manage technologies, scores, and key issues.")
    print("   Tip: Add technologies first, then you can add new key issues per pillar as needed.")

    while True:
        cmd = input(
            "\n[ESGFP] Choose: [T]ech add  [E]dit score  [K]ey issue add  [R]emove key issue  "
            "[V]isualize  [L]ist  [H]elp  [P]roceed → "
        ).strip().lower()

        if cmd in {"h", "help", "?"}:
            print("Help:\n"
                  " - Tech add: add one or more technologies (we'll collect all scores).\n"
                  " - Edit score: modify a specific tech/pillar/key-issue score.\n"
                  " - Key issue add: extend a pillar (e.g., add 'Water Stress' in Environmental) "
                  "   → prompts for scores for ALL technologies.\n"
                  " - Remove key issue: drop an issue from a pillar (optional).\n"
                  " - Visualize: pillar averages, radars, heatmaps, key-issue comparisons.\n"
                  " - Proceed: go to decision scenarios.")
            continue

        if cmd in {"l", "list"}:
            if not scores_by_tech:
                print("(no technologies yet)")
            else:
                print("Technologies:", ", ".join(scores_by_tech.keys()))
            print("Pillars & key issues:")
            for p, subs in pillars.items():
                print(f"  - {p}: {', '.join(subs) if subs else '(none)'}")
            continue

        if cmd in {"t", "tech", "add"}:
            n = prompt_int("   How many technologies to add now? ", 1, 50)
            used = set(scores_by_tech.keys())
            for i in range(n):
                default_name = f"Technology {len(scores_by_tech)+1}"
                name = prompt_str_nonempty("   Tech name", default_name)
                base = name.strip() or default_name
                label = base
                k = 2
                while label in used:
                    label = f"{base} ({k})"
                    k += 1
                used.add(label)
                scores_by_tech[label] = collect_esgfp_scores_for_tech(label, pillars)
            continue

        if cmd in {"e", "edit"}:
            if not scores_by_tech:
                print("No technologies to edit.")
                continue
            techs = list(scores_by_tech.keys())
            for i, t in enumerate(techs, 1):
                print(f"{i:2d}. {t}")
            ti = prompt_int("   Edit which technology #? ", 1, len(techs)) - 1
            tech = techs[ti]

            pillars_list = list(pillars.keys())
            for i, p in enumerate(pillars_list, 1):
                print(f"{i:2d}. {p}")
            pi = prompt_int("   Pillar #? ", 1, len(pillars_list)) - 1
            pillar = pillars_list[pi]
            subs = pillars[pillar]
            if not subs:
                print(f"'{pillar}' has no key issues. Add key issues first.")
                continue
            for i, s in enumerate(subs, 1):
                print(f"{i:2d}. {s}")
            si = prompt_int("   Key issue #? ", 1, len(subs)) - 1
            issue = subs[si]
            key = f"{pillar}:{issue}"

            # show current
            cur = scores_by_tech[tech].get(key, 0.0)
            print(f"   Current score for {tech} → {pillar}:{issue} = {cur:.2f}")
            s = prompt_float("   New base score (1–9): ", 1.0, 9.0)
            g = prompt_exposure("   Geographic Exposure (0–1 or label): ")
            final = s * (1.0 + g)
            scores_by_tech[tech][key] = round(final, 4)
            print("   Updated.")
            continue

        if cmd in {"k", "key", "key issue", "keyissue"}:
            # Add new key issue to a pillar
            pillars_list = list(pillars.keys())
            for i, p in enumerate(pillars_list, 1):
                print(f"{i:2d}. {p}")
            pi = prompt_int("   Add key issue to which pillar #? ", 1, len(pillars_list)) - 1
            pillar = pillars_list[pi]
            new_issue = prompt_str_nonempty("   New key issue name", "New Issue")
            if new_issue in pillars[pillar]:
                print("   Already exists. No changes.")
                continue
            pillars[pillar].append(new_issue)
            # collect for all technologies
            for tech in scores_by_tech.keys():
                print(f"   Entering scores for {tech} → {pillar}:{new_issue}")
                s = prompt_float("     Base score (1–9): ", 1.0, 9.0)
                g = prompt_exposure("     Geographic Exposure (0–1 or label): ")
                final = s * (1.0 + g)
                scores_by_tech[tech][f"{pillar}:{new_issue}"] = round(final, 4)
            print("   Key issue added and scored across technologies.")
            continue

        if cmd in {"r", "remove"}:
            pillars_list = list(pillars.keys())
            for i, p in enumerate(pillars_list, 1):
                print(f"{i:2d}. {p}")
            pi = prompt_int("   Remove key issue from pillar #? ", 1, len(pillars_list)) - 1
            pillar = pillars_list[pi]
            subs = pillars[pillar]
            if not subs:
                print(f"No key issues in '{pillar}'.")
                continue
            for i, s in enumerate(subs, 1):
                print(f"{i:2d}. {s}")
            si = prompt_int("   Key issue # to remove? ", 1, len(subs)) - 1
            issue = subs.pop(si)
            key = f"{pillar}:{issue}"
            for tech in scores_by_tech.keys():
                scores_by_tech[tech].pop(key, None)
            print("   Removed key issue from all technologies.")
            continue

        if cmd in {"v", "visualize"}:
            if not scores_by_tech:
                print("Add technologies first.")
                continue
            # Per-pillar key-issue visuals
            plot_pillar_key_issue_comparisons(scores_by_tech, pillars)
            plot_all_pillars_small_multiples(scores_by_tech, pillars)

            # Pillar aggregation
            pillar_avgs_df = pillar_averages_multi(scores_by_tech, pillars)
            print("\n📊 Pillar Averages:")
            print(pillar_avgs_df)

            # Overview visuals
            plot_pillar_heatmaps(pillar_avgs_df)
            plot_radar_profiles(pillar_avgs_df)
            plot_parallel_coordinates(pillar_avgs_df)
            plot_tradeoff_scatter(pillar_avgs_df)

            choice = input("Re-run ESGFP edits? [E]dit / [K]ey issue / [P]roceed → ").strip().lower()
            if choice in {"p", "proceed", ""}:
                return scores_by_tech, pillars
            # else loop again for edits
            continue

        if cmd in {"p", "proceed", ""}:
            if not scores_by_tech:
                print("Proceeding without technologies is not useful. Add at least 1 technology.")
                continue
            # Do a quick aggregate and return
            pillar_avgs_df = pillar_averages_multi(scores_by_tech, pillars)
            print("\n📊 Pillar Averages:")
            print(pillar_avgs_df)
            return scores_by_tech, pillars

        print("❌ Unknown option. Type 'help' for guidance.")


# ----------------------- MCDA utilities -------------------
def _weights_vector(pillars: Sequence[str], weights_pct: Dict[str, float]) -> np.ndarray:
    w = np.array([float(weights_pct.get(p, 0.0)) for p in pillars], dtype=float)
    s = w.sum()
    return w / (s if s != 0 else 1.0)


def _decision_matrix(pillar_avgs: pd.DataFrame) -> Tuple[np.ndarray, List[str], List[str]]:
    alts = list(pillar_avgs.columns)
    crits = list(pillar_avgs.index)
    A = pillar_avgs.to_numpy().T  # (m_alts, n_criteria)
    return A, alts, crits


# ----------------------- Methods ---------------------------
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
    N[N <= 0.0] = 1e-12  # avoid log(0)
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


# ----------------------- Scenarios (methods) ---------------
def prompt_pillar_weights(pillars: List[str]) -> Dict[str, float]:
    while True:
        weights: Dict[str, float] = {}
        print("\n🔢 Enter pillar weights (sum must equal 100%). Tip: 40/30/20/10; avoid 0%.")
        for p in pillars:
            w = prompt_float(f" - {p} (%): ", 0.0, 100.0)
            weights[p] = w
        total = sum(weights.values())
        if abs(total - 100.0) < 1e-6:
            return weights
        print(f"❌ Total = {total}. Must be exactly 100. Try again.")


def prompt_methods() -> List[str]:
    print("\n🧰 Select decision methods (comma-separated). Options:")
    print("   weighted, wpm, rank, topsis, vikor, edas, maut, pca")
    print("   - Press Enter for 'weighted' only. Type 'help' to view the quick course.")
    while True:
        raw = input("   Methods: ").strip().lower()
        if raw == "":
            return ["WEIGHTED"]
        if _is_help(raw):
            method_cheatsheet()
            continue
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        allowed = {
            "weighted": "WEIGHTED",
            "wpm": "WPM",
            "rank": "RANK",
            "topsis": "TOPSIS",
            "vikor": "VIKOR",
            "edas": "EDAS",
            "maut": "MAUT",
            "pca": "PCA",
        }
        chosen: List[str] = []
        for p in parts:
            if p in allowed:
                tag = allowed[p]
                if tag not in chosen:
                    chosen.append(tag)
        if chosen:
            if "WEIGHTED" not in chosen:
                chosen.insert(0, "WEIGHTED")
            return chosen
        print("❌ Please enter valid methods (or press Enter). Type 'help' for the course.")


def run_scenarios_with_methods(pillar_avgs: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float], List[str]]:
    while True:
        print("\n🎯 SCENARIOS — build and view results; you can re-run as needed.")
        pillars = pillar_avgs.index.tolist()
        alts = pillar_avgs.columns.tolist()
        weights_pct = prompt_pillar_weights(pillars)
        methods = prompt_methods()

        per_method: Dict[str, pd.Series] = {}
        per_method["WEIGHTED"] = method_weighted(pillar_avgs, weights_pct)
        if "WPM" in methods:
            per_method["WPM"] = method_wpm(pillar_avgs, weights_pct)
        if "RANK" in methods:
            per_method["RANK"] = method_rank(pillar_avgs, weights_pct)
        if "TOPSIS" in methods:
            per_method["TOPSIS"] = method_topsis(pillar_avgs, weights_pct)
        if "VIKOR" in methods:
            per_method["VIKOR"] = method_vikor(pillar_avgs, weights_pct, v=0.5)
        if "EDAS" in methods:
            per_method["EDAS"] = method_edas(pillar_avgs, weights_pct)
        if "MAUT" in methods:
            per_method["MAUT"] = method_maut(pillar_avgs, weights_pct)
        if "PCA" in methods:
            per_method["PCA"] = method_pca(pillar_avgs)

        scenario_df = pd.DataFrame(per_method)
        scenario_df.index.name = "Alternative"

        norm_ans = input("   Normalize outputs to 0–10 for display? [Y/n]: ").strip().lower()
        do_norm = norm_ans not in {"n", "no"}
        if do_norm:
            scaled_cols = {col: _scale_series_by_method(scenario_df[col], col) for col in scenario_df.columns}
            scenario_df_scaled = pd.DataFrame(scaled_cols, index=scenario_df.index)
        else:
            scenario_df_scaled = scenario_df.copy()

        print("\n🏁 Scenario Results" + (" (0–10 scaled)" if do_norm else " (raw)") + ":")
        print(scenario_df_scaled)

        # Main methods bar
        cols_for_plot = list(scenario_df_scaled.columns)[:6]
        ax = scenario_df_scaled[cols_for_plot].plot(
            kind="bar", figsize=(12, 6),
            title="Scenario – Selected Methods" + (" (0–10)" if do_norm else "")
        )
        ax.set_ylabel("Score (0–10)" if do_norm else "Score")
        ax.grid(True)
        plt.tight_layout()
        plt.show()

        # Weight donut
        plot_weight_donut(weights_pct)

        ans = input("Re-run scenarios? [Y/n], or [P]roceed to validation → ").strip().lower()
        if ans in {"n", "no", "p", "proceed", ""}:
            return scenario_df_scaled, weights_pct, methods
        # else loop to re-run with new weights/methods


def plot_weight_donut(weights_pct: Dict[str, float]) -> None:
    labels = list(weights_pct.keys())
    sizes = [weights_pct[k] for k in labels]
    fig, ax = plt.subplots(figsize=(6, 6))
    wedges, _ = ax.pie(sizes, wedgeprops=dict(width=0.4), startangle=90, counterclock=False)
    ax.set_title("Pillar Weights (%)")
    ax.legend(wedges, [f"{l} – {s:.1f}%" for l, s in zip(labels, sizes)],
              title="Pillars", loc="center left", bbox_to_anchor=(1, 0.5), frameon=False)
    plt.show()


# ----------------------- Validation (DEA + Monte-Carlo) ----
def _dirichlet(alpha: np.ndarray) -> np.ndarray:
    samples = np.random.gamma(shape=alpha, scale=1.0)
    s = samples.sum()
    if s <= 0:
        return np.ones_like(alpha) / len(alpha)
    return samples / s


def compute_dominance_matrix(pillar_avgs: pd.DataFrame) -> pd.DataFrame:
    """Dom[i,j]=1 if i dominates j (>= on all pillars and > on at least one)."""
    alts = list(pillar_avgs.columns)
    A = pillar_avgs.to_numpy().T  # (m,n)
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
    """
    Extended approximate DEA (output-oriented, VRS) via convex-hull Monte-Carlo.

    Returns:
      - dea_summary: Tech, FrontierProb, PhiHat (>=1), EffOut (<=1), Bottleneck, BottleneckShare
      - peer_matrix: (tech x tech) average peer weights for reference mix (best sample)
      - bottleneck_matrix: (pillar x tech) frequency that pillar is the limiting ratio
      - targets: (pillar x tech) projected DEA targets ≈ PhiHat * current
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    alts = list(pillar_avgs.columns)
    pillars = list(pillar_avgs.index)
    Y = pillar_avgs.to_numpy().T  # (m, n)
    m, n = Y.shape

    frontier_hits = np.zeros(m, dtype=int)
    rmax = np.ones(m, dtype=float)           # best min_j mix_j / y_ij
    best_lambda_store = [None] * m           # store best reference mix
    bottleneck_counts = np.zeros((n, m), dtype=int)  # pillars x tech

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

    # Summary
    phi_hat = rmax  # output expansion factor
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

    # Peer matrix
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

    # Bottleneck matrix (normalize to probabilities)
    bmat = bottleneck_counts / np.maximum(bottleneck_counts.sum(axis=0, keepdims=True), 1)
    bottleneck_matrix = pd.DataFrame(bmat, index=pillars, columns=alts)

    # Targets (output orientation: target ≈ φ̂·y)
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
    n = len(pillars)

    best_counts: Dict[str, Dict[str, int]] = {meth: {a: 0 for a in alts} for meth in methods}
    rank_sum: Dict[str, np.ndarray] = {meth: np.zeros(m, dtype=float) for meth in methods}
    rank_sqsum: Dict[str, np.ndarray] = {meth: np.zeros(m, dtype=float) for meth in methods}
    rank_counts: Dict[str, np.ndarray] = {meth: np.zeros((m, m), dtype=int) for meth in methods}

    A0 = pillar_avgs.to_numpy().T  # (m, n)
    alpha_vec = np.full(n, float(weight_alpha))

    for _ in range(sims):
        w = _dirichlet(alpha_vec)
        noise = np.random.normal(0.0, score_noise_sigma, size=A0.shape)
        A = np.clip(A0 + noise, a_min=0.0, a_max=None)
        dfA = pd.DataFrame(A.T, index=pillars, columns=alts)

        per_method: Dict[str, pd.Series] = {}
        w_pct = {p: w[i]*100 for i, p in enumerate(pillars)}
        per_method["WEIGHTED"] = method_weighted(dfA, w_pct)
        if "WPM" in methods:
            per_method["WPM"] = method_wpm(dfA, w_pct)
        if "RANK" in methods:
            per_method["RANK"] = method_rank(dfA, w_pct)
        if "TOPSIS" in methods:
            per_method["TOPSIS"] = method_topsis(dfA, w_pct)
        if "VIKOR" in methods:
            per_method["VIKOR"] = method_vikor(dfA, w_pct, v=0.5)
        if "EDAS" in methods:
            per_method["EDAS"] = method_edas(dfA, w_pct)
        if "MAUT" in methods:
            per_method["MAUT"] = method_maut(dfA, w_pct)
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


def plot_rankogram(rankdist: pd.DataFrame, method: str) -> None:
    df = rankdist.copy()
    df = df.loc[:, sorted(df.columns, key=lambda c: int(c.split("_")[1]))]
    fig, ax = plt.subplots(figsize=(12, 6))
    bottom = np.zeros(df.shape[0])
    x = np.arange(df.shape[0])
    for col in df.columns:
        ax.bar(x, df[col].values, bottom=bottom, width=0.8, label=col.replace("_", " ").title())
        bottom += df[col].values
    ax.set_xticks(x)
    ax.set_xticklabels(df.index, rotation=0)
    ax.set_ylabel("Probability")
    ax.set_title(f"Rankogram – {method}")
    ax.legend(ncol=min(df.shape[1], 5), frameon=False)
    plt.show()


def plot_mc_heatmaps(Pbest: pd.DataFrame, MeanRank: pd.DataFrame) -> None:
    for name, mat, cmap, label in [
        ("P(Best)", Pbest, "Greens", "Probability"),
        ("Mean Rank", MeanRank, "RdYlGn_r", "Rank (lower=better)"),
    ]:
        data = mat.copy()
        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(data.values, aspect="auto", cmap=cmap)
        ax.set_xticks(range(data.shape[1]))
        ax.set_xticklabels(data.columns, rotation=45, ha="right")
        ax.set_yticks(range(data.shape[0]))
        ax.set_yticklabels(data.index)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                ax.text(j, i, f"{data.values[i, j]:.2f}", ha="center", va="center", fontsize=9)
        ax.set_title(f"Monte-Carlo – {name}")
        fig.colorbar(im, ax=ax, shrink=0.8, label=label)
        fig.tight_layout()
        plt.show()


def plot_dea_pca_scatter(pillar_avgs: pd.DataFrame, frontier_prob: pd.Series) -> None:
    A, alts, _ = _decision_matrix(pillar_avgs)
    if A.shape[0] < 2:
        print("(Not enough alternatives for PCA scatter.)")
        return
    X = A - A.mean(axis=0)
    std = A.std(axis=0, ddof=1)
    std[std == 0.0] = 1.0
    X = X / std
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    coords = U[:, :2] * S[:2]
    fig, ax = plt.subplots(figsize=(9, 7))
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=frontier_prob.loc[alts].to_numpy(),
                    s=160, edgecolor="black", linewidth=0.8, cmap="viridis")
    for i, a in enumerate(alts):
        ax.annotate(a, (coords[i, 0], coords[i, 1]), xytext=(6, 6), textcoords="offset points")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("DEA Frontier Probability (PCA Projection)")
    cb = fig.colorbar(sc, ax=ax, shrink=0.8)
    cb.set_label("Frontier Probability (1=efficient)")
    plt.show()


def plot_dea_bottleneck_heatmap(bottleneck_matrix: pd.DataFrame) -> None:
    data = bottleneck_matrix.copy()
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(data.values, aspect="auto", cmap="OrRd")
    ax.set_xticks(range(data.shape[1]))
    ax.set_xticklabels(data.columns, rotation=45, ha="right")
    ax.set_yticks(range(data.shape[0]))
    ax.set_yticklabels(data.index)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data.values[i, j]:.2f}", ha="center", va="center", fontsize=9)
    ax.set_title("DEA Bottleneck Frequency by Pillar")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Frequency")
    fig.tight_layout()
    plt.show()


def plot_dea_peer_heatmap(peer_matrix: pd.DataFrame) -> None:
    data = peer_matrix.copy()
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(data.values, aspect="auto", cmap="Blues")
    ax.set_xticks(range(data.shape[1]))
    ax.set_xticklabels(data.columns, rotation=45, ha="right")
    ax.set_yticks(range(data.shape[0]))
    ax.set_yticklabels(data.index)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data.values[i, j]
            if val > 0:
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9)
    ax.set_title("DEA Peer Reference Weights (avg best mix)")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Weight")
    fig.tight_layout()
    plt.show()


def plot_dea_target_radars(pillar_avgs: pd.DataFrame, targets: pd.DataFrame) -> None:
    """One radar per technology: current vs projected target (0–10 normalized)."""
    if pillar_avgs.empty:
        return
    pillars = list(pillar_avgs.index)
    techs = list(pillar_avgs.columns)
    for tech in techs:
        ax, angles = _radar_axes(len(pillars))
        cur = (pillar_avgs[tech] / WEIGHTED_THEORETICAL_MAX * OUTPUT_SCALE).tolist()
        tgt = (targets[tech] / WEIGHTED_THEORETICAL_MAX * OUTPUT_SCALE).tolist()
        cur += cur[:1]
        tgt += tgt[:1]
        ax.plot(angles, cur, linewidth=2, alpha=0.9, label=f"{tech} – Current")
        ax.fill(angles, cur, alpha=0.08)
        ax.plot(angles, tgt, linewidth=2, alpha=0.9, linestyle="--", label=f"{tech} – DEA Target")
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(pillars)
        ax.set_yticklabels([])
        ax.set_ylim(0, 10)
        ax.set_title(f"DEA Projection – {tech} (0–10)")
        ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), frameon=False)
        plt.show()


def plot_dominance_heatmap(dom: pd.DataFrame) -> None:
    data = dom.copy()
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(data.values, aspect="auto", cmap="Greys")
    ax.set_xticks(range(data.shape[1]))
    ax.set_xticklabels(data.columns, rotation=45, ha="right")
    ax.set_yticks(range(data.shape[0]))
    ax.set_yticklabels(data.index)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if data.values[i, j] == 1:
                ax.text(j, i, "✓", ha="center", va="center", fontsize=11)
    ax.set_title("Dominance Matrix (✓ = row dominates column)")
    fig.tight_layout()
    plt.show()


def run_validation_suite_interactive(
    pillar_avgs: pd.DataFrame,
    last_weights_pct: Dict[str, float],
    last_methods: Sequence[str],
) -> None:
    print("\n🔎 Validation Suite — DEA (extended) & Monte-Carlo")
    raw = input("   Show validation help? [y/N]: ").strip().lower()
    if raw in {"y", "yes", "help", "h", "?"}:
        validation_help()

    while True:
        # DEA settings
        dea_samples = int(prompt_float_or_default("   DEA convex-hull samples", 5000, 500, 200000))
        peer_cut = prompt_float_or_default("   Peer weight display cutoff", 0.05, 0.0, 1.0)

        # MC settings
        sims = int(prompt_float_or_default("   Monte-Carlo simulations", 2000, 100, 200000))
        alpha = prompt_float_or_default("   Dirichlet alpha (1.0=uniform)", 1.0, 0.1, 10.0)
        sigma = prompt_float_or_default("   Score noise sigma (0.0–0.2)", 0.03, 0.0, 0.5)

        print("\n▶ DEA (extended) — computing diagnostics...")
        dea_summary, peer_matrix, bottleneck_matrix, targets = approx_dea_diagnostics(
            pillar_avgs, samples=dea_samples, min_peer_lambda=peer_cut
        )
        print("\nDEA Summary (EffOut=1/φ̂; higher is better; FrontierProb≈efficiency frequency):")
        print(dea_summary.sort_values(["EffOut", "FrontierProb"], ascending=[False, False]).round(3))

        # Visuals for DEA
        plot_pbest(dea_summary["FrontierProb"], "Approx. DEA Frontier Probability")
        plot_dea_bottleneck_heatmap(bottleneck_matrix)
        plot_dea_peer_heatmap(peer_matrix)
        plot_dea_target_radars(pillar_avgs, targets)

        # Dominance (structural check)
        dom = compute_dominance_matrix(pillar_avgs)
        plot_dominance_heatmap(dom)

        # PCA scatter colored by frontier prob
        plot_dea_pca_scatter(pillar_avgs, dea_summary["FrontierProb"])

        # Monte-Carlo sensitivity
        print("\n▶ Monte-Carlo sensitivity — running simulations...")
        Pbest, MeanRank, StdRank, RankDist = run_monte_carlo_sensitivity(
            pillar_avgs,
            last_weights_pct,
            last_methods,
            sims=sims,
            weight_alpha=alpha,
            score_noise_sigma=sigma,
        )
        print("\nMonte-Carlo — P(Best):")
        print(Pbest.fillna(0.0))
        if "WEIGHTED" in Pbest.columns:
            plot_pbest(Pbest["WEIGHTED"], "Monte-Carlo P(Best) — WEIGHTED")
            plot_rankogram(RankDist["WEIGHTED"], "WEIGHTED")
        plot_mc_heatmaps(Pbest, MeanRank)
        print("\nMonte-Carlo — Mean Rank (lower is better):")
        print(MeanRank.round(3))
        print("\nMonte-Carlo — Rank Std Dev (lower = more stable):")
        print(StdRank.round(3))

        again = input("\nRe-run validation with different settings? [y/N] → ").strip().lower()
        if again not in {"y", "yes"}:
            break


# ----------------------- Main ------------------------------
def main() -> None:
    apply_pro_style()
    method_cheatsheet()

    # 1) Risk section with edit loop
    risks = risk_section()
    if risks:
        df = risk_dataframe(risks)
        print("\nFinalized Risk Table:")
        print(df)

    # Main outer loop: allow repeating ESGFP blocks if user wants
    while True:
        # 2) ESGFP section (add/edit/key issues/visualize)
        scores_by_tech, pillars = esgfp_section()

        # Compute pillar averages and overview visuals again (if not visualized already)
        pillar_avgs_df = pillar_averages_multi(scores_by_tech, pillars)
        if pillar_avgs_df.empty:
            print("No pillar averages to proceed with. Returning to ESGFP section.")
            continue
        print("\n📊 Pillar Averages (final before scenarios):")
        print(pillar_avgs_df)

        # Overview visuals (useful before scenarios)
        plot_pillar_heatmaps(pillar_avgs_df)
        plot_radar_profiles(pillar_avgs_df)
        plot_parallel_coordinates(pillar_avgs_df)
        plot_tradeoff_scatter(pillar_avgs_df)

        # 3) Scenarios (user-selectable methods) with re-run option
        combined, last_weights, last_methods = run_scenarios_with_methods(pillar_avgs_df)

        # 4) Validation (DEA + Monte Carlo) with re-run option
        ans = input("\n🧪 Run validation (DEA extended + Monte-Carlo) now? [Y/n]: ").strip().lower()
        if ans not in {"n", "no"}:
            run_validation_suite_interactive(pillar_avgs_df, last_weights, last_methods)

        # 5) Exit or continue with a fresh ESGFP cycle
        again = input("\n✅ Do you want to exit or continue with NEW ESGFP Pillar calculations? "
                      "[E]xit / [C]ontinue → ").strip().lower()
        if again in {"e", "exit"}:
            print("Goodbye!")
            break
        # else loop again for a new ESGFP run (pillars reset to defaults; inputs start fresh)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⛔ Interrupted by user.")
        sys.exit(1)
