"""
generate_graphical_abstract.py
==============================
Creates a publication-quality graphical abstract for the Applied Energy manuscript.
Features:
  - Embedded mini-bar chart: MAE comparison across model families
  - Embedded mini-line chart: Cumulative PnL trajectories
  - Embedded coverage gauge: CQR vs QRF PICP
  - Modern gradient headers, iconographic elements, subtle shadows
  - Elsevier-compliant 16:9 aspect ratio at 300 DPI
"""

import os
import sys
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
import config

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Arc, Circle, Wedge
from matplotlib.collections import PatchCollection
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors
from matplotlib import cm

# ── Style ──
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

FIG_DIR = os.path.join(config.REPORT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ── Color Palette ──
C = {
    "bg":          "#0F172A",      # Dark navy background
    "bg_card":     "#1E293B",      # Slightly lighter card bg
    "bg_card2":    "#334155",      # Mid card
    "accent":      "#38BDF8",      # Sky blue accent
    "accent2":     "#818CF8",      # Indigo accent
    "green":       "#34D399",      # Emerald green (Trees)
    "green_dark":  "#059669",
    "orange":      "#FB923C",      # Orange (Chronos/FM)
    "orange_dark": "#EA580C",
    "red":         "#F87171",      # Red (failure)
    "red_dark":    "#DC2626",
    "blue":        "#60A5FA",      # Blue (DL)
    "blue_dark":   "#2563EB",
    "purple":      "#A78BFA",      # Purple (UQ)
    "yellow":      "#FBBF24",      # Gold (PnL)
    "text":        "#F8FAFC",      # Near-white text
    "text_dim":    "#94A3B8",      # Muted text
    "text_mid":    "#CBD5E1",      # Mid brightness text
    "border":      "#475569",      # Border gray
    "white":       "#FFFFFF",
}


def draw_rounded_rect(ax, xy, w, h, r=0.3, fc="#1E293B", ec=None, lw=1.5, alpha=1.0, zorder=1):
    """Draw a rounded rectangle with optional glow."""
    rect = FancyBboxPatch(xy, w, h, boxstyle=f"round,pad={r}",
                          facecolor=fc, edgecolor=ec or fc, linewidth=lw, alpha=alpha, zorder=zorder)
    ax.add_patch(rect)
    return rect


def draw_gradient_header(ax, xy, w, h, c_left, c_right, text, fontsize=13, zorder=10):
    """Simulate a horizontal gradient header using thin vertical strips."""
    x0, y0 = xy
    n_strips = 80
    strip_w = w / n_strips
    r1, g1, b1 = mcolors.to_rgb(c_left)
    r2, g2, b2 = mcolors.to_rgb(c_right)
    for i in range(n_strips):
        t = i / n_strips
        c = (r1 + t*(r2-r1), g1 + t*(g2-g1), b1 + t*(b2-b1))
        ax.add_patch(plt.Rectangle((x0 + i*strip_w, y0), strip_w + 0.01, h,
                                    facecolor=c, edgecolor="none", zorder=zorder))
    # Rounded corners overlay
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.15",
                                facecolor="none", edgecolor="none", zorder=zorder+1))
    ax.text(x0 + w/2, y0 + h/2, text, ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color=C["white"], zorder=zorder+2,
            path_effects=[pe.withStroke(linewidth=2, foreground="#00000044")])


def draw_stat_box(ax, x, y, value, label, color, fontsize_val=18, fontsize_lbl=7.5):
    """Draw a prominent statistic with label."""
    ax.text(x, y, value, ha="center", va="center",
            fontsize=fontsize_val, fontweight="bold", color=color, zorder=15,
            path_effects=[pe.withStroke(linewidth=1.5, foreground="#00000033")])
    ax.text(x, y - 0.35, label, ha="center", va="center",
            fontsize=fontsize_lbl, color=C["text_dim"], zorder=15)


def draw_section_label(ax, x, y, icon, text, color):
    """Draw a section label with a colored dot icon."""
    ax.plot(x, y, "o", color=color, markersize=8, zorder=15)
    ax.text(x + 0.2, y, text, ha="left", va="center",
            fontsize=9.5, fontweight="bold", color=C["text"], zorder=15)


def draw_mini_bar_chart(ax_sub, data, colors, labels, title):
    """Draw a compact horizontal bar chart. Best (lowest) at top."""
    ax_sub.set_facecolor(C["bg_card"])
    # Reverse so best (lowest MAE) is at top
    data = list(reversed(data))
    colors = list(reversed(colors))
    labels = list(reversed(labels))
    y_pos = np.arange(len(data))
    bars = ax_sub.barh(y_pos, data, height=0.55, color=colors, alpha=0.9, edgecolor="none")

    for i, (v, lbl) in enumerate(zip(data, labels)):
        ax_sub.text(0.15, i, lbl, ha="left", va="center", fontsize=7,
                   color=C["white"], fontweight="bold", zorder=10)
        ax_sub.text(v + 0.08, i, f"{v:.2f}", ha="left", va="center", fontsize=6.5,
                   color=C["text_mid"], fontweight="bold", zorder=10)

    ax_sub.set_xlim(0, max(data) * 1.15)
    ax_sub.set_yticks([])
    ax_sub.set_xticks([])
    ax_sub.set_title(title, fontsize=8, color=C["text_mid"], fontweight="bold", pad=4)
    ax_sub.spines["left"].set_visible(False)
    ax_sub.spines["bottom"].set_visible(False)
    ax_sub.tick_params(left=False, bottom=False)


def draw_coverage_gauges(ax_sub):
    """Draw PICP gauge comparison."""
    ax_sub.set_facecolor(C["bg_card"])
    ax_sub.set_xlim(-1.5, 1.5)
    ax_sub.set_ylim(-0.6, 1.1)
    ax_sub.set_aspect("equal")
    ax_sub.axis("off")

    # Draw gauges for CQR and QRF
    for i, (label, picp, target, color) in enumerate([
        ("CQR", 80.1, 90, C["red"]),
        ("QRF", 90.2, 90, C["green"]),
    ]):
        cx = -0.7 + i * 1.4
        # Background arc
        theta1, theta2 = 0, 180
        arc_bg = Arc((cx, 0), 1.1, 1.1, angle=0, theta1=theta1, theta2=theta2,
                     color=C["bg_card2"], lw=10, zorder=5)
        ax_sub.add_patch(arc_bg)

        # Fill arc proportional to coverage
        fill_angle = (picp / 100) * 180
        arc_fill = Arc((cx, 0), 1.1, 1.1, angle=0, theta1=0, theta2=fill_angle,
                       color=color, lw=10, zorder=6)
        ax_sub.add_patch(arc_fill)

        # Target line
        target_angle = np.radians((target / 100) * 180)
        tx = cx + 0.55 * np.cos(target_angle)
        ty = 0.55 * np.sin(target_angle)
        ax_sub.plot([cx, tx], [0, ty], color=C["yellow"], lw=2, ls="--", zorder=7)

        # Value text
        ax_sub.text(cx, 0.2, f"{picp}%", ha="center", va="center",
                   fontsize=13, fontweight="bold", color=color, zorder=10)
        ax_sub.text(cx, -0.12, label, ha="center", va="center",
                   fontsize=9, color=C["text_dim"], fontweight="bold", zorder=10)
        # Status indicator
        status = "\u2713" if picp >= target else "\u2717"
        status_color = C["green"] if picp >= target else C["red"]
        ax_sub.text(cx, -0.42, f"{status} 90% target", ha="center", va="center",
                   fontsize=7, color=status_color, fontweight="bold", zorder=10)

    # No title here — it overlaps with the section label on the main axes


def draw_pnl_sparklines(ax_sub):
    """Draw PnL trajectory sparklines."""
    ax_sub.set_facecolor(C["bg_card"])

    pnl_path = os.path.join(config.REPORT_DIR, "pnl_daily_pjm.csv")
    if not os.path.exists(pnl_path):
        ax_sub.text(0.5, 0.5, "PnL data\nnot found", ha="center", va="center",
                   fontsize=7, color=C["text_dim"], transform=ax_sub.transAxes)
        return

    pnl = pd.read_csv(pnl_path, index_col=0, parse_dates=True)

    strategies = {
        "LightGBM":      (C["green"],  "-",  1.8),
        "Seasonal Naive": (C["text_dim"], "--", 0.8),
        "Risk-Aware CQR": (C["red"],    "-",  1.0),
    }

    for col, (color, ls, lw) in strategies.items():
        if col in pnl.columns:
            cumulative = pnl[col].cumsum()
            x = np.linspace(0, 1, len(cumulative))
            ax_sub.plot(x, cumulative.values, color=color, ls=ls, lw=lw, alpha=0.9)

    ax_sub.axhline(0, color=C["border"], lw=0.5, ls="-")
    ax_sub.set_xlim(0, 1)
    ax_sub.set_xticks([])

    # Minimal y-axis
    ax_sub.yaxis.set_major_locator(plt.MaxNLocator(3))
    ax_sub.tick_params(axis="y", labelsize=5, colors=C["text_dim"], length=2)
    ax_sub.spines["left"].set_color(C["border"])
    ax_sub.spines["left"].set_linewidth(0.5)
    ax_sub.spines["bottom"].set_visible(False)
    ax_sub.set_title("Cumulative PnL (PJM)", fontsize=7.5, color=C["text_mid"],
                    fontweight="bold", pad=4)

    # Legend
    for i, (name, (color, _, _)) in enumerate(strategies.items()):
        ax_sub.text(0.02, 0.88 - i*0.12, f"\u25cf {name}", transform=ax_sub.transAxes,
                   fontsize=5.5, color=color, va="top")


def create_graphical_abstract():
    """Create an advanced, publication-quality graphical abstract."""
    print("Generating advanced Graphical Abstract...")

    fig = plt.figure(figsize=(16, 9))
    fig.patch.set_facecolor(C["bg"])

    # Master axes for text and shapes
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")
    ax.set_facecolor(C["bg"])

    # ══════════════════════════════════════════════════════════
    # HEADER: Title Banner with Gradient
    # ══════════════════════════════════════════════════════════
    draw_gradient_header(ax, (0.3, 8.0), 15.4, 0.85,
                         "#1E40AF", "#7C3AED",
                         "Disentangling Architecture, Strategy, and Adaptation in Day-Ahead Electricity Price Forecasting",
                         fontsize=13)

    # Subtitle / Authors
    ax.text(8.0, 7.65, "M. Q. Ansari & M. Q. Ansari  •  PJM & ERCOT  •  2019–2025  •  19 Models  •  Applied Energy",
            ha="center", va="center", fontsize=8, color=C["text_dim"], zorder=15)

    # ══════════════════════════════════════════════════════════
    # ROW 1: Pipeline Flow (4 cards)
    # ══════════════════════════════════════════════════════════
    pipeline_y = 6.2
    card_w, card_h = 3.35, 1.2

    pipeline_cards = [
        {
            "x": 0.4, "title": "DATA", "color": C["accent"],
            "items": ["PJM + ERCOT (2 US markets)", "61K hours × 50 features", "4 volatility regimes"],
            "icon": "◆"
        },
        {
            "x": 4.15, "title": "19 CONFIGURATIONS", "color": C["orange"],
            "items": ["4 Stat · 3 Trees · 6 DL · 2 FM", "Chronos-Bolt v1 & v2+Cov", "Rolling retraining (24 mo)"],
            "icon": "◆"
        },
        {
            "x": 7.9, "title": "TRIPLE EVALUATION", "color": C["green"],
            "items": ["Point: MAE, RMSE, sMAPE", "Probabilistic: CQR, QRF, MC-DO", "Economic: PnL, Sharpe, Win%"],
            "icon": "◆"
        },
        {
            "x": 11.65, "title": "STATISTICAL RIGOR", "color": C["purple"],
            "items": ["DM test + HLN correction", "Benjamini–Hochberg FDR", "24-window robustness"],
            "icon": "◆"
        },
    ]

    for card in pipeline_cards:
        # Card background
        draw_rounded_rect(ax, (card["x"], pipeline_y), card_w, card_h,
                         r=0.15, fc=C["bg_card"], ec=card["color"], lw=1.2, alpha=0.85, zorder=5)

        # Top accent line
        ax.plot([card["x"] + 0.3, card["x"] + card_w - 0.3], [pipeline_y + card_h - 0.08]*2,
                color=card["color"], lw=2.5, alpha=0.8, zorder=6, solid_capstyle="round")

        # Title
        ax.text(card["x"] + card_w/2, pipeline_y + card_h - 0.28,
                card["title"], ha="center", va="center",
                fontsize=10.5, fontweight="bold", color=card["color"], zorder=10)

        # Items
        for j, item in enumerate(card["items"]):
            ax.text(card["x"] + card_w/2, pipeline_y + card_h - 0.55 - j*0.23,
                    item, ha="center", va="center",
                    fontsize=7.5, color=C["text_mid"], zorder=10)

    # Arrows between pipeline cards
    arrow_y = pipeline_y + card_h/2
    for i in range(3):
        x_start = pipeline_cards[i]["x"] + card_w
        x_end = pipeline_cards[i+1]["x"]
        ax.annotate("", xy=(x_end + 0.05, arrow_y), xytext=(x_start - 0.05, arrow_y),
                    arrowprops=dict(arrowstyle="-|>", color=C["accent"], lw=1.5,
                                   mutation_scale=12), zorder=10)

    # ══════════════════════════════════════════════════════════
    # ROW 2: Key Findings with Embedded Charts
    # ══════════════════════════════════════════════════════════

    # Divider
    ax.text(8.0, 5.85, "─── KEY FINDINGS ───", ha="center", va="center",
            fontsize=10, fontweight="bold", color=C["text_dim"], zorder=15,
            family="monospace")

    # ── Finding 1: MAE Bar Chart ──
    draw_rounded_rect(ax, (0.4, 2.85), 3.75, 2.75, r=0.15,
                     fc=C["bg_card"], ec=C["blue"], lw=1.0, alpha=0.9, zorder=5)

    draw_section_label(ax, 0.7, 5.3, "●", "BiLSTM Dominates", C["blue"])

    # Embedded bar chart — raised and shortened to avoid overlapping stat text
    ax_bar = fig.add_axes([0.04, 0.40, 0.2, 0.21])  # [left, bottom, width, height]

    mae_data = [4.57, 6.69, 6.69, 7.30, 7.47, 7.71]
    mae_labels = ["BiLSTM", "LGBM (rolling)", "PatchTST", "Chronos+Cov", "iTransformer", "LGBM (static)"]
    mae_colors = [C["blue"], C["green"], C["blue"], C["orange"], C["blue"], C["green"]]

    draw_mini_bar_chart(ax_bar, mae_data, mae_colors, mae_labels, r"MAE (\$/MWh) — PJM")

    # Stats below chart — lowered to y=3.10 for clearance
    draw_stat_box(ax, 2.27, 3.10, "4.57", r"MAE \$/MWh (PJM)", C["blue"], fontsize_val=16)

    # ── Finding 2: Feature Engineering > Architecture ──
    draw_rounded_rect(ax, (4.45, 2.85), 3.45, 2.75, r=0.15,
                     fc=C["bg_card"], ec=C["orange"], lw=1.0, alpha=0.9, zorder=5)

    draw_section_label(ax, 4.75, 5.3, "●", "Covariates > Scale", C["orange"])

    # Chronos improvement waterfall
    ax.text(6.17, 4.95, "Chronos Architecture Upgrade", ha="center", va="center",
            fontsize=6.5, color=C["text_dim"], zorder=15)
    ax.text(6.17, 4.65, "v1 → v2:  −2.3%", ha="center", va="center",
            fontsize=9, fontweight="bold", color=C["text_dim"], zorder=15)

    # Divider line
    ax.plot([4.9, 7.4], [4.35, 4.35], color=C["border"], lw=0.5, zorder=10)

    ax.text(6.17, 4.1, "Adding Covariates", ha="center", va="center",
            fontsize=6.5, color=C["text_dim"], zorder=15)
    ax.text(6.17, 3.8, "v2 → v2+Cov:  +6.5%", ha="center", va="center",
            fontsize=11, fontweight="bold", color=C["red"], zorder=15)

    # Arrow emphasis
    ax.annotate("", xy=(6.17, 3.45), xytext=(6.17, 3.6),
                arrowprops=dict(arrowstyle="-|>", color=C["red"], lw=2, mutation_scale=15),
                zorder=10)
    ax.text(6.17, 3.2, r"6.85 → 7.30 \$/MWh", ha="center", va="center",
            fontsize=8, fontweight="bold", color=C["red"], zorder=15)

    # ── Finding 3: PnL Trajectory ──
    draw_rounded_rect(ax, (8.2, 2.85), 3.55, 2.75, r=0.15,
                     fc=C["bg_card"], ec=C["yellow"], lw=1.0, alpha=0.9, zorder=5)

    draw_section_label(ax, 8.5, 5.3, "●", "Economic Value", C["yellow"])

    # Embedded PnL chart — position higher to avoid overlapping stats
    ax_pnl = fig.add_axes([0.535, 0.39, 0.19, 0.22])
    draw_pnl_sparklines(ax_pnl)

    # Key stats — placed below the chart
    ax.text(9.97, 3.15, r"\$19,107", ha="center", va="center",
            fontsize=14, fontweight="bold", color=C["yellow"], zorder=15)
    ax.text(9.97, 2.93, "LGBM PnL  •  Sharpe 14.67  •  93.7% Win", ha="center", va="center",
            fontsize=7, color=C["text_dim"], zorder=15)

    # ── Finding 4: UQ / Coverage Gauges ──
    draw_rounded_rect(ax, (12.05, 2.85), 3.55, 2.75, r=0.15,
                     fc=C["bg_card"], ec=C["red"], lw=1.0, alpha=0.9, zorder=5)

    draw_section_label(ax, 12.35, 5.3, "●", "Conformal Failure", C["red"])
    ax.text(13.82, 5.05, "Prediction Interval Coverage (ERCOT)", ha="center", va="center",
            fontsize=7, color=C["text_dim"], zorder=15)

    # Embedded gauge — shifted down for clearance from label
    ax_gauge = fig.add_axes([0.775, 0.345, 0.19, 0.22])
    draw_coverage_gauges(ax_gauge)

    # Bottom stat
    ax.text(13.82, 3.1, "Exchangeability violated", ha="center", va="center",
            fontsize=7, color=C["red"], fontweight="bold", zorder=15)

    # ══════════════════════════════════════════════════════════
    # BOTTOM: Conclusion Banner
    # ══════════════════════════════════════════════════════════
    # Dark gradient banner
    draw_gradient_header(ax, (0.3, 0.4), 15.4, 2.15,
                         "#064E3B", "#1E3A5F",
                         "", fontsize=1)  # Empty text; we'll add custom text

    # Main conclusion
    ax.text(8.0, 2.05,
            r"Static BiLSTM:  MAE = 4.57 \$/MWh (PJM)  │  3.64 \$/MWh (ERCOT)",
            ha="center", va="center", fontsize=15, fontweight="bold",
            color=C["white"], zorder=15,
            path_effects=[pe.withStroke(linewidth=2, foreground="#00000055")])

    ax.text(8.0, 1.6,
            "Outperforming all 17 alternatives including Rolling LightGBM, Chronos-Bolt & PatchTST  │  p < 0.01 (DM + BH)",
            ha="center", va="center", fontsize=9, color=C["text_mid"], zorder=15)

    # 4 stat pills across the bottom
    pill_y = 0.95
    pill_data = [
        ("-41%", "BiLSTM advantage", C["blue"]),
        ("-6.5%", "Covariate gain", C["orange"]),
        (r"\$19,107", "Trading PnL (LGBM)", C["yellow"]),
        ("90.2%", "QRF coverage", C["purple"]),
    ]

    pill_positions = [2.3, 5.8, 9.8, 13.5]
    for (val, lbl, col), px in zip(pill_data, pill_positions):
        draw_rounded_rect(ax, (px - 1.2, pill_y - 0.32), 2.4, 0.65, r=0.12,
                         fc="#0F172A", ec=col, lw=1.0, alpha=0.7, zorder=12)
        ax.text(px, pill_y + 0.05, val, ha="center", va="center",
                fontsize=12, fontweight="bold", color=col, zorder=15)
        ax.text(px, pill_y - 0.22, lbl, ha="center", va="center",
                fontsize=6.5, color=C["text_dim"], zorder=15)

    # ── Save ──
    for ext in ["png", "pdf"]:
        fig.savefig(os.path.join(FIG_DIR, f"Graphical_Abstract.{ext}"),
                   dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor(),
                   edgecolor="none")
    plt.close(fig)
    print("✅ Saved: Graphical_Abstract.png / .pdf")


if __name__ == "__main__":
    create_graphical_abstract()
