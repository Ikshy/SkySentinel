from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401  (registers 3-D projection)




BG      = "#060c18"
PANEL   = "#0b1526"
CARD    = "#0f1e35"
BORDER  = "#1a3050"
ACCENT  = "#00d4ff"
ACCENT2 = "#0080ff"
GRID    = "#112240"
TEXT    = "#cce8ff"
DIM     = "#4a7a9b"
GREEN   = "#00ff88"
ORANGE  = "#ff6b35"
YELLOW  = "#ffcc00"
RED     = "#ff4444"
PURPLE  = "#c77dff"

THREAT_COLORS = {
    "ballistic_missile":   RED,
    "cruise_missile":      ORANGE,
    "drone":               YELLOW,
    "fighter_jet":         PURPLE,
    "commercial_aircraft": GREEN,
    "unknown":             DIM,
}

ALERT_COLORS = {
    "critical": RED,
    "high":     ORANGE,
    "medium":   YELLOW,
    "low":      ACCENT,
    "none":     GREEN,
}

# Custom inferno-style colormap for radar heatmaps
_RADAR_CMAP = LinearSegmentedColormap.from_list(
    "radar", ["#060c18", "#0a2050", "#0050a0", "#00aaff",
              "#00ffcc", "#ffcc00", "#ff6600", "#ff0000"]
)


def _apply_dark_style(ax, title: str = "", xlabel: str = "", ylabel: str = "") -> None:
    """Apply the SkySentinel dark theme to a single Axes object."""
    ax.set_facecolor(BG)
    ax.tick_params(colors=TEXT, labelsize=8)
    for sp in ax.spines.values():
        sp.set_color(BORDER)
    ax.set_title(title, color=ACCENT, fontsize=9, fontfamily="monospace")
    ax.set_xlabel(xlabel, color=DIM, fontsize=8)
    ax.set_ylabel(ylabel, color=DIM, fontsize=8)
    ax.grid(color=GRID, linewidth=0.5, linestyle="--")


def _new_fig(
    nrows: int = 1, ncols: int = 1,
    figsize: tuple = (12, 6),
    title: str = "",
) -> tuple[plt.Figure, any]:
    """Create a new dark-themed figure."""
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, facecolor=BG)
    if title:
        fig.suptitle(title, color=ACCENT, fontsize=12,
                     fontweight="bold", fontfamily="monospace", y=0.98)
    return fig, axes




def plot_radar_data_cube(
    clean_cube:      np.ndarray,
    compressed_cube: np.ndarray,
    rd_map:          np.ndarray,
    prf_hz:          float = 2000.0,
    save_path:       Optional[Path] = None,
) -> plt.Figure:

    fig = plt.figure(figsize=(16, 5), facecolor=BG)
    fig.suptitle("SKYSENTINEL — RADAR DATA CUBE ANALYSIS",
                 color=ACCENT, fontsize=11, fontfamily="monospace", y=1.01)
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    # ── Panel 1: Raw range profile ────────────────────────────────────────
    ax0 = fig.add_subplot(gs[0])
    ax0.plot(np.abs(clean_cube[0]), color=ACCENT, lw=0.8, label="Clean")
    _apply_dark_style(ax0, "Raw Range Profile  (Pulse 0)",
                      "Range Gate", "|Amplitude|")
    ax0.legend(fontsize=7, facecolor=PANEL, labelcolor=TEXT)

    # ── Panel 2: Pulse-compressed profile ────────────────────────────────
    ax1 = fig.add_subplot(gs[1])
    ax1.plot(np.abs(compressed_cube[0]), color=ORANGE, lw=0.8,
             label="Pulse Compressed")
    _apply_dark_style(ax1, "Matched-Filter Output  (Pulse 0)",
                      "Range Gate", "|Amplitude|")
    ax1.legend(fontsize=7, facecolor=PANEL, labelcolor=TEXT)

    # ── Panel 3: Range-Doppler map ────────────────────────────────────────
    ax2 = fig.add_subplot(gs[2])
    rd_db = 20 * np.log10(np.abs(rd_map) + 1e-12)
    vmin  = float(np.percentile(rd_db, 5))
    vmax  = float(np.percentile(rd_db, 99.5))
    im    = ax2.imshow(
        rd_db, aspect="auto", origin="lower",
        cmap=_RADAR_CMAP, vmin=vmin, vmax=vmax,
        extent=[0, rd_map.shape[1], -prf_hz/2, prf_hz/2],
    )
    _apply_dark_style(ax2, "Range-Doppler Map  (dB)",
                      "Range Gate", "Doppler Freq (Hz)")
    cb = fig.colorbar(im, ax=ax2, pad=0.02, fraction=0.04)
    cb.set_label("Power (dB)", color=DIM, fontsize=7)
    cb.ax.yaxis.set_tick_params(color=TEXT, labelsize=7)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=TEXT)

    fig.patch.set_facecolor(BG)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG)
    return fig




def plot_track_history_2d(
    tracks: list,
    classifications: dict,
    title: str = "MULTI-TARGET TRACKING — BIRD'S-EYE VIEW",
    save_path: Optional[Path] = None,
) -> plt.Figure:

    fig, ax = plt.subplots(figsize=(10, 10), facecolor=BG)
    ax.set_facecolor(BG)

    # ── Radar range rings ─────────────────────────────────────────────────
    theta = np.linspace(0, 2 * np.pi, 300)
    for r_km in [100, 200, 300, 400]:
        ax.plot(r_km * np.cos(theta), r_km * np.sin(theta),
                color=GRID, lw=0.7, ls=":", alpha=0.6)
        ax.text(r_km * 0.707, r_km * 0.707,
                f"{r_km} km", color=DIM, fontsize=7, va="center")

    # ── Compass axes ─────────────────────────────────────────────────────
    for ang_deg, label in [(90, "E"), (0, "N"), (270, "W"), (180, "S")]:
        ang = math.radians(ang_deg)
        ax.plot([0, 410 * math.cos(ang)], [0, 410 * math.sin(ang)],
                color=BORDER, lw=0.5, ls="--")
        ax.text(420 * math.cos(ang), 420 * math.sin(ang),
                label, color=DIM, fontsize=8, ha="center", va="center")

    # ── Track histories + current positions ───────────────────────────────
    for track in tracks:
        if not track.positions:
            continue
        cl    = classifications.get(track.track_id)
        t_type = track.threat_type if hasattr(track, "threat_type") else "unknown"
        color  = THREAT_COLORS.get(t_type, DIM)
        alert  = getattr(track, "alert_level", "none")
        a_col  = ALERT_COLORS.get(alert, DIM)

        pos_arr = np.array(track.positions) / 1000.0
        ax.plot(pos_arr[:, 0], pos_arr[:, 1],
                color=color, lw=1.2, alpha=0.7)

        # Arrowhead at current position
        if len(pos_arr) >= 2:
            dx = pos_arr[-1, 0] - pos_arr[-2, 0]
            dy = pos_arr[-1, 1] - pos_arr[-2, 1]
            ax.annotate(
                "", xy=pos_arr[-1, :2],
                xytext=pos_arr[-2, :2],
                arrowprops=dict(arrowstyle="->", color=a_col, lw=1.5),
            )

        # Label
        ax.scatter(pos_arr[-1, 0], pos_arr[-1, 1],
                   s=60, color=a_col, zorder=5,
                   edgecolors=BG, linewidths=0.8)
        ax.text(pos_arr[-1, 0] + 3, pos_arr[-1, 1] + 3,
                f"{track.track_id}\n{t_type[:8]}",
                color=TEXT, fontsize=7, fontfamily="monospace",
                va="bottom")

    # ── Radar site ────────────────────────────────────────────────────────
    ax.scatter(0, 0, s=120, color=GREEN, marker="*", zorder=10)
    ax.text(5, 5, "RADAR", color=GREEN, fontsize=8, fontfamily="monospace")

    # ── Legend ────────────────────────────────────────────────────────────
    legend_patches = [
        mpatches.Patch(color=c, label=t.replace("_", " ").title())
        for t, c in THREAT_COLORS.items()
    ]
    ax.legend(handles=legend_patches, loc="upper right",
              fontsize=8, facecolor=PANEL, labelcolor=TEXT,
              framealpha=0.8, edgecolor=BORDER)

    _apply_dark_style(ax, title, "East (km)", "North (km)")
    ax.set_xlim(-450, 450)
    ax.set_ylim(-450, 450)
    ax.set_aspect("equal")
    fig.patch.set_facecolor(BG)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG)
    return fig



def plot_track_history_3d(
    tracks: list,
    predictions: list,
    title: str = "3-D TRACK HISTORY & PREDICTIONS",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """3-D Matplotlib plot of track histories with LSTM prediction overlays."""
    fig = plt.figure(figsize=(12, 9), facecolor=BG)
    ax  = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(BG)
    fig.patch.set_facecolor(BG)

    colours = [ACCENT, ORANGE, "#a8ff78", YELLOW, PURPLE,
               "#f7c59f", RED, GREEN]

    pred_dict = {p.track_id: p for p in predictions}

    for i, track in enumerate(tracks):
        if not track.positions:
            continue
        col = colours[i % len(colours)]
        pos = np.array(track.positions) / 1000.0   # km

        ax.plot(pos[:, 0], pos[:, 1], pos[:, 2],
                "-o", color=col, lw=1.5, ms=2, alpha=0.8,
                label=f"{track.track_id} [{track.threat_type[:3].upper()}]")

        # Current position (larger marker)
        ax.scatter(*pos[-1], s=60, color=col, edgecolors=BG, lw=0.8, zorder=5)

        # Vertical drop line
        ax.plot([pos[-1, 0], pos[-1, 0]],
                [pos[-1, 1], pos[-1, 1]],
                [0, pos[-1, 2]],
                color=col, lw=0.5, ls=":", alpha=0.3)

        # Prediction overlay
        if track.track_id in pred_dict:
            pp = pred_dict[track.track_id].pred_positions / 1000.0
            ax.plot(pp[:, 0], pp[:, 1], pp[:, 2],
                    "--s", color=YELLOW, lw=1.5, ms=3, alpha=0.7)

    # Radar site
    ax.scatter(0, 0, 0, s=100, color=GREEN, marker="*", zorder=10)

    # Style
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.pane.fill = False
        axis.pane.set_edgecolor(BORDER)
        axis._axinfo["grid"]["color"] = GRID

    ax.set_xlabel("East (km)", color=DIM, fontsize=8)
    ax.set_ylabel("North (km)", color=DIM, fontsize=8)
    ax.set_zlabel("Altitude (km)", color=DIM, fontsize=8)
    ax.tick_params(colors=TEXT, labelsize=7)
    ax.set_title(title, color=ACCENT, fontsize=10, fontfamily="monospace")
    ax.legend(loc="upper left", fontsize=7, facecolor=PANEL, labelcolor=TEXT,
              framealpha=0.8)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG)
    return fig




def plot_trajectory_prediction(
    prediction,          # PredictedTrajectory
    track,               # Track
    ground_truth: Optional[np.ndarray] = None,   # (H, 3) in metres
    save_path: Optional[Path] = None,
) -> plt.Figure:

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor=BG)
    fig.suptitle(
        f"TRAJECTORY PREDICTION — Track {prediction.track_id}  "
        f"[{track.threat_type.replace('_', ' ').upper()}]",
        color=ACCENT, fontsize=10, fontfamily="monospace",
    )

    hist = np.array(track.positions) / 1000.0        # km
    pred = prediction.pred_positions / 1000.0
    std  = prediction.pred_std / 1000.0

    proj_pairs = [
        (0, 1, "East (km)", "North (km)"),
        (1, 2, "North (km)", "Altitude (km)"),
        (0, 2, "East (km)", "Altitude (km)"),
    ]

    for ax, (xi, yi, xl, yl) in zip(axes, proj_pairs):
        # History
        ax.plot(hist[:, xi], hist[:, yi], "o-",
                color=ACCENT, lw=1.5, ms=3, label="Track History")
        # Prediction mean
        ax.plot(pred[:, xi], pred[:, yi], "s--",
                color=ORANGE, lw=2, ms=4, label="LSTM Prediction")
        # ±2σ uncertainty band
        ax.fill_between(
            pred[:, xi],
            pred[:, yi] - 2 * std[:, yi],
            pred[:, yi] + 2 * std[:, yi],
            color=ORANGE, alpha=0.15, label="±2σ Band",
        )
        # Ground truth if provided
        if ground_truth is not None:
            gt = ground_truth / 1000.0
            ax.plot(gt[:, xi], gt[:, yi], "^-",
                    color=GREEN, lw=1.5, ms=4, label="Ground Truth")

        # Connection from last history to first prediction
        ax.plot([hist[-1, xi], pred[0, xi]],
                [hist[-1, yi], pred[0, yi]],
                color=DIM, lw=1, ls=":")

        _apply_dark_style(ax, f"{xl.split('(')[0].strip()} vs "
                          f"{yl.split('(')[0].strip()}", xl, yl)
        ax.legend(fontsize=7, facecolor=PANEL, labelcolor=TEXT)

    fig.patch.set_facecolor(BG)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG)
    return fig




def plot_threat_distribution(
    classifications: dict,
    save_path: Optional[Path] = None,
) -> plt.Figure:

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 6), facecolor=BG)
    fig.suptitle("THREAT CLASSIFICATION SUMMARY",
                 color=ACCENT, fontsize=11, fontfamily="monospace")

    results = list(classifications.values())
    if not results:
        for ax in (ax0, ax1):
            _apply_dark_style(ax, "No data")
        return fig

    # ── Donut chart ───────────────────────────────────────────────────────
    from collections import Counter
    type_counts = Counter(r.predicted_class for r in results)

    labels = list(type_counts.keys())
    sizes  = list(type_counts.values())
    colors = [THREAT_COLORS.get(l, DIM) for l in labels]

    wedges, texts, autotexts = ax0.pie(
        sizes, labels=labels, autopct="%1.0f%%",
        colors=colors,
        textprops={"color": TEXT, "fontsize": 9, "fontfamily": "monospace"},
        wedgeprops={"edgecolor": BG, "linewidth": 2.5},
        pctdistance=0.78,
        startangle=90,
    )
    for at in autotexts:
        at.set_color(BG)
        at.set_fontsize(8)
    # Draw inner circle (donut hole)
    centre_circle = plt.Circle((0, 0), 0.55, fc=BG)
    ax0.add_artist(centre_circle)
    ax0.text(0, 0, f"{len(results)}\nTRACKS", ha="center", va="center",
             color=ACCENT, fontsize=11, fontfamily="monospace", fontweight="bold")
    ax0.set_facecolor(BG)
    ax0.set_title("THREAT TYPE DISTRIBUTION",
                  color=ACCENT, fontsize=9, fontfamily="monospace", pad=10)

    # ── Confidence bar chart ──────────────────────────────────────────────
    track_ids = [r.track_id for r in results]
    confs     = [r.confidence for r in results]
    bar_colors = [ALERT_COLORS.get(r.alert_level, DIM) for r in results]

    y_pos = range(len(track_ids))
    bars  = ax1.barh(y_pos, confs, color=bar_colors, height=0.6, alpha=0.85)

    for bar, r in zip(bars, results):
        ax1.text(
            bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
            f"{r.predicted_class.replace('_', ' ')[:14]}  {r.confidence:.0%}",
            va="center", ha="left", color=TEXT, fontsize=8,
            fontfamily="monospace",
        )

    ax1.set_yticks(list(y_pos))
    ax1.set_yticklabels(track_ids, color=TEXT, fontsize=9, fontfamily="monospace")
    ax1.set_xlim(0, 1.45)
    ax1.axvline(0.5, color=BORDER, lw=0.8, ls="--")
    _apply_dark_style(ax1, "CLASSIFICATION CONFIDENCE PER TRACK",
                      "Confidence Score", "Track ID")
    ax1.invert_yaxis()

    fig.patch.set_facecolor(BG)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG)
    return fig




def plot_noise_comparison(
    clean_cube:    np.ndarray,
    degraded_cube: np.ndarray,
    labels:        tuple[str, str] = ("Clean Signal", "Degraded Signal"),
    save_path:     Optional[Path]  = None,
) -> plt.Figure:

    fig = plt.figure(figsize=(14, 8), facecolor=BG)
    fig.suptitle("NOISE & INTERFERENCE ANALYSIS",
                 color=ACCENT, fontsize=11, fontfamily="monospace")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    def rd(cube):
        return np.abs(np.fft.fftshift(np.fft.fft(cube, axis=0), axes=0))

    def rddb(cube):
        return 20 * np.log10(rd(cube) + 1e-12)

    for col, (cube, label, color) in enumerate([
        (clean_cube,    labels[0], ACCENT),
        (degraded_cube, labels[1], ORANGE),
    ]):
        # Row 0: range profiles
        ax_r = fig.add_subplot(gs[0, col])
        ax_r.plot(np.abs(cube[0]), color=color, lw=0.8)
        _apply_dark_style(ax_r, f"{label} — Range Profile (Pulse 0)",
                          "Range Gate", "|Amplitude|")

        # Row 1: RD map
        ax_d = fig.add_subplot(gs[1, col])
        rdb = rddb(cube)
        vmin, vmax = float(np.percentile(rdb, 5)), float(np.percentile(rdb, 99))
        im = ax_d.imshow(
            rdb, aspect="auto", origin="lower",
            cmap=_RADAR_CMAP, vmin=vmin, vmax=vmax,
        )
        _apply_dark_style(ax_d, f"{label} — Range-Doppler Map",
                          "Range Gate", "Doppler Bin")
        cb = fig.colorbar(im, ax=ax_d, pad=0.02, fraction=0.04)
        cb.set_label("dB", color=DIM, fontsize=7)
        cb.ax.yaxis.set_tick_params(color=TEXT, labelsize=7)
        plt.setp(cb.ax.yaxis.get_ticklabels(), color=TEXT)

    fig.patch.set_facecolor(BG)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG)
    return fig



def save_all_plots(
    output_dir: Path,
    clean_cube: Optional[np.ndarray] = None,
    compressed_cube: Optional[np.ndarray] = None,
    rd_map: Optional[np.ndarray] = None,
    degraded_cube: Optional[np.ndarray] = None,
    tracks: Optional[list] = None,
    predictions: Optional[list] = None,
    classifications: Optional[dict] = None,
) -> list[Path]:

    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    matplotlib.use("Agg")   # ensure non-interactive backend

    if all(x is not None for x in [clean_cube, compressed_cube, rd_map]):
        p = output_dir / "radar_data_cube.png"
        plot_radar_data_cube(clean_cube, compressed_cube, rd_map, save_path=p)
        plt.close("all")
        saved.append(p)

    if clean_cube is not None and degraded_cube is not None:
        p = output_dir / "noise_comparison.png"
        plot_noise_comparison(clean_cube, degraded_cube, save_path=p)
        plt.close("all")
        saved.append(p)

    if tracks:
        p = output_dir / "track_history_2d.png"
        plot_track_history_2d(tracks, classifications or {}, save_path=p)
        plt.close("all")
        saved.append(p)

        p = output_dir / "track_history_3d.png"
        plot_track_history_3d(tracks, predictions or [], save_path=p)
        plt.close("all")
        saved.append(p)

    if classifications:
        p = output_dir / "threat_distribution.png"
        plot_threat_distribution(classifications, save_path=p)
        plt.close("all")
        saved.append(p)

    print(f"  Saved {len(saved)} plots to {output_dir}")
    return saved