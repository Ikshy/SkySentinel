from __future__ import annotations

import json
import math
import sys
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State, ctx
import dash_bootstrap_components as dbc

# ── Add project root to path ─────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from radar_simulation.generate_signals import (
    RadarParameters, generate_cpi, generate_lfm_pulse, TargetState,
)
from radar_simulation.noise_model import (
    NoisePipelineConfig, ThermalNoiseConfig, ClutterConfig, JammingConfig,
    apply_noise_pipeline,
)
from radar_simulation.filters import (
    run_dsp_pipeline, DSPPipelineConfig, rd_map_axes, Detection,
)
from multi_target_tracking.kalman_tracker import (
    TrackManager, TrackManagerConfig, Track, TrackStatus,
    detections_to_measurements,
)
from ml_models.threat_classifier import (
    ThreatClassifier, ClassifierConfig, ClassificationResult,
    ALERT_COLORS, THREAT_CLASSES,
)
from ml_models.trajectory_predictor import (
    TrajectoryPredictorInference, ModelConfig, TrainConfig,
    PredictedTrajectory,
)


#

SCAN_INTERVAL_MS   = 1500     # Milliseconds between dashboard refreshes
MAX_TRAIL_POINTS   = 20       # Max historical positions shown per track
PRED_HORIZON_SHOW  = 8        # Prediction steps shown on 3-D plot
MAX_RANGE_DISPLAY  = 420      # km, radius of radar scope display

# Colour palette — dark military aesthetic
PALETTE = dict(
    bg_dark   = "#060c18",
    bg_panel  = "#0b1526",
    bg_card   = "#0f1e35",
    border    = "#1a3050",
    accent    = "#00d4ff",
    accent2   = "#0080ff",
    grid      = "#112240",
    text      = "#cce8ff",
    text_dim  = "#4a7a9b",
    success   = "#00ff88",
    warning   = "#ffcc00",
    danger    = "#ff4444",
    scan_ring = "#0a3060",
)

# Alert level → Plotly marker symbol
ALERT_SYMBOLS = {
    "critical": "diamond",
    "high":     "triangle-up",
    "medium":   "circle",
    "low":      "circle-open",
    "none":     "circle-dot",
}

# Threat type → marker size
THREAT_SIZES = {
    "ballistic_missile":  18,
    "cruise_missile":     14,
    "drone":              10,
    "fighter_jet":        16,
    "commercial_aircraft": 13,
    "unknown":            11,
}



class SimulationEngine:

    def __init__(self) -> None:
        self.scan_count = 0
        self.timestamp  = 0.0
        self._rng       = np.random.default_rng(seed=42)

        # Radar
        self.radar = RadarParameters(
            n_pulses_per_cpi=64,
            max_range_m=400_000,
            prf_hz=2000,
        )
        self.ref_pulse = generate_lfm_pulse(self.radar, self._rng)

        # Noise pipeline
        self.noise_cfg = NoisePipelineConfig(
            enable_thermal=True,
            thermal=ThermalNoiseConfig(noise_power_dbw=-108.0),
            enable_clutter=True,
            clutter=ClutterConfig(clutter_to_noise_ratio_db=28.0,
                                  max_range_m=80_000.0),
            enable_jamming=False,
            jamming=JammingConfig(jam_type="barrage", jammer_to_noise_ratio_db=12.0),
        )

        # DSP
        self.dsp_cfg = DSPPipelineConfig(enable_mti=True, enable_cfar=True)

        # Tracker
        self.tracker = TrackManager(TrackManagerConfig(
            sigma_a=15.0, dt_sec=1.0,
            confirm_m_hits=3, confirm_n_scans=5, max_coast_scans=4,
        ))

        # Classifier (trained on demand)
        print("  [SkySentinel] Training threat classifier …")
        self.classifier = ThreatClassifier(
            ClassifierConfig(n_per_class=500, n_estimators=100)
        ).train(verbose=False, evaluate=False)

        # Trajectory predictor
        print("  [SkySentinel] Training trajectory predictor …")
        m_cfg = ModelConfig(hidden_dim=64, seq_len=10, pred_horizon=PRED_HORIZON_SHOW)
        t_cfg = TrainConfig(epochs=20, n_train_seqs=1500, patience=6,
                            batch_size=64, device="auto")
        self.predictor = TrajectoryPredictorInference.train_and_build(
            m_cfg, t_cfg, verbose=False
        )

        # Latest results (updated each tick)
        self.latest_rd_map: Optional[np.ndarray] = None
        self.latest_detections: list[Detection]  = []
        self.latest_predictions: list[PredictedTrajectory] = []
        self.latest_classifications: dict[str, ClassificationResult] = {}
        self.scenario = "mixed_threat"
        self.jamming_enabled = False
        print("  [SkySentinel] Engine ready.\n")

    def tick(self) -> list[Track]:
        """Run one radar scan and update all downstream modules."""
        self.scan_count += 1
        self.timestamp   = float(self.scan_count)

        # Toggle jamming periodically for demo
        if self.scan_count % 20 == 10:
            self.noise_cfg.enable_jamming = self.jamming_enabled

        # Build data cube
        cube, _, _ = generate_cpi(
            scenario=self.scenario,
            radar=self.radar,
            seed=self.scan_count,
            verbose=False,
        )
        self.ref_pulse = generate_lfm_pulse(self.radar, self._rng)

        # Add noise / interference
        noisy_cube, _ = apply_noise_pipeline(
            cube, self.noise_cfg, rng=self._rng,
            range_gate_length_m=self.radar.range_gate_length_m,
            reference_pulse=self.ref_pulse,
            noise_power=self.radar.thermal_noise_power(),
        )

        # DSP pipeline
        _, rd_map, detections = run_dsp_pipeline(
            noisy_cube, self.ref_pulse,
            prf_hz=self.radar.prf_hz,
            range_gate_length_m=self.radar.range_gate_length_m,
            wavelength_m=self.radar.wavelength_m,
            config=self.dsp_cfg,
        )

        self.latest_rd_map    = rd_map
        self.latest_detections = detections

        # Tracking
        meas   = detections_to_measurements(detections)
        tracks = self.tracker.update(meas, self.timestamp)

        # Classification for confirmed tracks
        confirmed = self.tracker.confirmed_tracks
        if confirmed:
            results = self.classifier.classify_batch(confirmed)
            for t, r in zip(confirmed, results):
                self.latest_classifications[t.track_id] = r
                t.threat_type       = r.predicted_class
                t.threat_confidence = r.confidence
                t.alert_level       = r.alert_level

        # Trajectory prediction
        self.latest_predictions = self.predictor.predict_batch(
            confirmed, current_timestamp=self.timestamp
        )

        return tracks

    def set_scenario(self, scenario: str) -> None:
        self.scenario = scenario
        # Reset tracker when scenario changes
        self.tracker = TrackManager(TrackManagerConfig(
            sigma_a=15.0, dt_sec=1.0,
            confirm_m_hits=3, confirm_n_scans=5, max_coast_scans=4,
        ))
        self.latest_classifications.clear()

    def toggle_jamming(self, enabled: bool) -> None:
        self.jamming_enabled = enabled
        self.noise_cfg.enable_jamming = enabled


# Module-level singleton
ENGINE = SimulationEngine()



def _fig_layout(fig: go.Figure, title: str = "", height: int = 400) -> go.Figure:
    """Apply consistent dark theme to any figure."""
    fig.update_layout(
        title=dict(text=title, font=dict(color=PALETTE["accent"],
                   size=11, family="monospace"), x=0.01, xanchor="left"),
        paper_bgcolor=PALETTE["bg_panel"],
        plot_bgcolor=PALETTE["bg_dark"],
        font=dict(color=PALETTE["text"], family="monospace", size=10),
        margin=dict(l=10, r=10, t=35, b=10),
        height=height,
    )
    return fig


def build_radar_scope_3d(
    tracks:      list[Track],
    predictions: list[PredictedTrajectory],
    detections:  list[Detection],
) -> go.Figure:

    fig = go.Figure()

    # ── Radar range rings (every 100 km) ──────────────────────────────────
    theta = np.linspace(0, 2 * np.pi, 120)
    for r_km in [100, 200, 300, 400]:
        fig.add_trace(go.Scatter3d(
            x=r_km * np.cos(theta), y=r_km * np.sin(theta),
            z=np.zeros(120),
            mode="lines",
            line=dict(color=PALETTE["scan_ring"], width=1),
            showlegend=False, hoverinfo="skip",
        ))

    # ── Radar axes ────────────────────────────────────────────────────────
    for axis_xy in [(400, 0), (-400, 0), (0, 400), (0, -400)]:
        fig.add_trace(go.Scatter3d(
            x=[0, axis_xy[0]], y=[0, axis_xy[1]], z=[0, 0],
            mode="lines",
            line=dict(color=PALETTE["border"], width=1, dash="dot"),
            showlegend=False, hoverinfo="skip",
        ))

    # ── Detection scatter (range-plane projection) ────────────────────────
    if detections:
        det_ranges = [d.range_m / 1000.0 for d in detections]
        det_vels   = [d.velocity_mps for d in detections]
        # Project detections to North axis for simple display
        fig.add_trace(go.Scatter3d(
            x=[0] * len(detections),
            y=det_ranges,
            z=[0] * len(detections),
            mode="markers",
            marker=dict(size=3, color=PALETTE["accent2"],
                        opacity=0.4, symbol="circle"),
            name="Raw Detections",
            hovertemplate="Range: %{y:.1f} km<extra></extra>",
        ))

    # ── Prediction trajectories (before track trails so trails are on top) ─
    pred_dict = {p.track_id: p for p in predictions}
    for pred in predictions:
        pp = pred.pred_positions / 1000.0   # → km
        # Dashed prediction line
        fig.add_trace(go.Scatter3d(
            x=pp[:, 0], y=pp[:, 1], z=pp[:, 2],
            mode="lines+markers",
            line=dict(color="#ff9944", width=2, dash="dash"),
            marker=dict(size=2, color="#ff9944", opacity=0.6),
            name=f"Pred {pred.track_id}",
            hovertemplate=(
                f"Track {pred.track_id} (predicted)<br>"
                "X: %{x:.1f} km  Y: %{y:.1f} km  Z: %{z:.1f} km<extra></extra>"
            ),
        ))

        # Uncertainty tube (1-σ) — rendered as error bars on last pred point
        last = pp[-1]
        std  = pred.pred_std[-1] / 1000.0
        for dim, (dx, dy, dz) in enumerate([
            (std[0], 0, 0), (0, std[1], 0), (0, 0, std[2])
        ]):
            fig.add_trace(go.Scatter3d(
                x=[last[0] - dx, last[0] + dx],
                y=[last[1] - dy, last[1] + dy],
                z=[last[2] - dz, last[2] + dz],
                mode="lines",
                line=dict(color="#ff6622", width=3, dash="solid"),
                showlegend=False, hoverinfo="skip", opacity=0.5,
            ))

    # ── Track trails + current markers ────────────────────────────────────
    for track in tracks:
        if not track.positions:
            continue

        cl_result = ENGINE.latest_classifications.get(track.track_id)
        alert     = cl_result.alert_level if cl_result else "none"
        color     = ALERT_COLORS.get(alert, PALETTE["text_dim"])
        symbol    = ALERT_SYMBOLS.get(alert, "circle")
        t_type    = track.threat_type
        msize     = THREAT_SIZES.get(t_type, 11)

        # History trail
        trail = np.array(track.positions[-MAX_TRAIL_POINTS:]) / 1000.0
        fig.add_trace(go.Scatter3d(
            x=trail[:, 0], y=trail[:, 1], z=trail[:, 2],
            mode="lines",
            line=dict(color=color, width=1.5),
            opacity=0.55,
            showlegend=False, hoverinfo="skip",
        ))

        # Current position marker
        pos = track.position_3d()
        fig.add_trace(go.Scatter3d(
            x=[pos[0] / 1000.0],
            y=[pos[1] / 1000.0],
            z=[pos[2] / 1000.0],
            mode="markers+text",
            marker=dict(size=msize, color=color, symbol=symbol,
                        line=dict(color=PALETTE["bg_dark"], width=1)),
            text=[f" {track.track_id}"],
            textfont=dict(color=PALETTE["text"], size=9),
            textposition="middle right",
            name=f"{t_type} ({track.track_id})",
            hovertemplate=(
                f"<b>{t_type}</b> [{track.track_id}]<br>"
                f"Alert: <b>{alert.upper()}</b><br>"
                "X: %{x:.1f} km  Y: %{y:.1f} km  Z: %{z:.1f} km<br>"
                f"Speed: {track.speed_mps():.0f} m/s<br>"
                f"Confidence: {track.threat_confidence:.0%}<extra></extra>"
            ),
        ))

        # Vertical drop line to ground plane
        fig.add_trace(go.Scatter3d(
            x=[pos[0]/1000.0, pos[0]/1000.0],
            y=[pos[1]/1000.0, pos[1]/1000.0],
            z=[0, pos[2]/1000.0],
            mode="lines",
            line=dict(color=color, width=0.5, dash="dot"),
            opacity=0.3,
            showlegend=False, hoverinfo="skip",
        ))

    # ── Radar site marker ─────────────────────────────────────────────────
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode="markers+text",
        marker=dict(size=12, color=PALETTE["success"],
                    symbol="diamond", line=dict(color=PALETTE["bg_dark"], width=2)),
        text=["RADAR"],
        textfont=dict(color=PALETTE["success"], size=9, family="monospace"),
        textposition="top center",
        name="Radar Site",
        hovertemplate="Radar Site<extra></extra>",
    ))

    # ── Layout ────────────────────────────────────────────────────────────
    fig.update_layout(
        scene=dict(
            xaxis=dict(title="East (km)", range=[-MAX_RANGE_DISPLAY, MAX_RANGE_DISPLAY],
                       gridcolor=PALETTE["grid"], zerolinecolor=PALETTE["border"],
                       backgroundcolor=PALETTE["bg_dark"],
                       titlefont=dict(color=PALETTE["text_dim"], size=9)),
            yaxis=dict(title="North (km)", range=[-MAX_RANGE_DISPLAY, MAX_RANGE_DISPLAY],
                       gridcolor=PALETTE["grid"], zerolinecolor=PALETTE["border"],
                       backgroundcolor=PALETTE["bg_dark"],
                       titlefont=dict(color=PALETTE["text_dim"], size=9)),
            zaxis=dict(title="Alt (km)", range=[0, 60],
                       gridcolor=PALETTE["grid"], zerolinecolor=PALETTE["border"],
                       backgroundcolor=PALETTE["bg_dark"],
                       titlefont=dict(color=PALETTE["text_dim"], size=9)),
            bgcolor=PALETTE["bg_dark"],
            camera=dict(eye=dict(x=1.4, y=1.4, z=0.9)),
            aspectmode="cube",
        ),
        legend=dict(
            font=dict(color=PALETTE["text"], size=9),
            bgcolor=PALETTE["bg_panel"],
            bordercolor=PALETTE["border"], borderwidth=1,
            x=0.0, y=1.0,
        ),
        paper_bgcolor=PALETTE["bg_panel"],
        margin=dict(l=0, r=0, t=35, b=0),
        height=480,
        title=dict(
            text=f"◉ RADAR SCOPE — {ENGINE.scan_count} SCANS",
            font=dict(color=PALETTE["accent"], size=11, family="monospace"),
            x=0.01,
        ),
    )
    return fig


def build_rd_map_figure(rd_map: Optional[np.ndarray]) -> go.Figure:
    """Range-Doppler heatmap with matched-filter output."""
    fig = go.Figure()

    if rd_map is not None:
        n_dopp, n_gates = rd_map.shape
        vel_ax, rng_ax = rd_map_axes(
            n_dopp, n_gates,
            ENGINE.radar.prf_hz,
            ENGINE.radar.range_gate_length_m,
            ENGINE.radar.wavelength_m,
        )
        rd_db = 20 * np.log10(np.abs(rd_map) + 1e-12)
        vmin  = float(np.percentile(rd_db, 5))
        vmax  = float(np.percentile(rd_db, 99))

        fig.add_trace(go.Heatmap(
            z=rd_db,
            x=rng_ax,
            y=vel_ax,
            colorscale="Inferno",
            zmin=vmin, zmax=vmax,
            showscale=True,
            colorbar=dict(
                title=dict(text="dB", font=dict(color=PALETTE["text_dim"], size=9)),
                tickfont=dict(color=PALETTE["text_dim"], size=8),
                len=0.8,
            ),
            hovertemplate="Range: %{x:.1f} km<br>Doppler: %{y:.0f} m/s<br>Power: %{z:.1f} dB<extra></extra>",
        ))

    fig.update_layout(
        xaxis=dict(title="Range (km)", gridcolor=PALETTE["grid"],
                   color=PALETTE["text_dim"], titlefont=dict(size=9)),
        yaxis=dict(title="Radial Velocity (m/s)", gridcolor=PALETTE["grid"],
                   color=PALETTE["text_dim"], titlefont=dict(size=9)),
        paper_bgcolor=PALETTE["bg_panel"],
        plot_bgcolor=PALETTE["bg_dark"],
        font=dict(color=PALETTE["text"], family="monospace", size=9),
        margin=dict(l=50, r=10, t=35, b=40),
        height=260,
        title=dict(text="RANGE-DOPPLER MAP",
                   font=dict(color=PALETTE["accent"], size=10, family="monospace"),
                   x=0.01),
    )
    return fig


def build_timeline_figure(tracks: list[Track]) -> go.Figure:
    """SNR history line chart per confirmed track."""
    fig = go.Figure()
    colours = [PALETTE["accent"], "#ff6b35", "#a8ff78", "#f7c59f",
               "#e84393", "#b8d8d8", "#ffd700", "#c77dff"]

    confirmed = [t for t in tracks if t.status == TrackStatus.CONFIRMED]
    for i, track in enumerate(confirmed[:6]):
        snr_vals = [m.get("snr_db", 0) for m in track.measurements_history
                    if "snr_db" in m][-30:]
        if not snr_vals:
            continue
        col = colours[i % len(colours)]
        fig.add_trace(go.Scatter(
            x=list(range(len(snr_vals))),
            y=snr_vals,
            mode="lines",
            line=dict(color=col, width=1.5),
            name=f"{track.track_id} [{track.threat_type[:3].upper()}]",
            hovertemplate="%{y:.1f} dB<extra></extra>",
        ))

    fig.add_hline(y=0, line_color=PALETTE["danger"], line_dash="dot",
                  line_width=1, opacity=0.6)
    fig.update_layout(
        xaxis=dict(title="Scan (recent)", gridcolor=PALETTE["grid"],
                   color=PALETTE["text_dim"], titlefont=dict(size=9)),
        yaxis=dict(title="SNR (dB)", gridcolor=PALETTE["grid"],
                   color=PALETTE["text_dim"], titlefont=dict(size=9)),
        paper_bgcolor=PALETTE["bg_panel"],
        plot_bgcolor=PALETTE["bg_dark"],
        font=dict(color=PALETTE["text"], family="monospace", size=9),
        margin=dict(l=50, r=10, t=35, b=40),
        height=260,
        legend=dict(font=dict(size=8, color=PALETTE["text"]),
                    bgcolor=PALETTE["bg_panel"], x=0, y=1),
        title=dict(text="TRACK SNR HISTORY",
                   font=dict(color=PALETTE["accent"], size=10, family="monospace"),
                   x=0.01),
    )
    return fig


def build_threat_cards(tracks: list[Track]) -> list:
    """Build Bootstrap alert cards for the threat panel."""
    cards = []
    confirmed = sorted(
        [t for t in tracks if t.status == TrackStatus.CONFIRMED],
        key=lambda t: ALERT_THRESHOLDS_ORDER(t.alert_level),
        reverse=True,
    )

    for track in confirmed[:8]:
        cl = ENGINE.latest_classifications.get(track.track_id)
        alert = track.alert_level
        color = ALERT_COLORS.get(alert, PALETTE["text_dim"])

        # Probability bar for top-3 classes
        prob_bars = []
        if cl:
            for cls_name, prob in cl.top_classes(3):
                bar_w = f"{prob * 100:.0f}%"
                prob_bars.append(
                    html.Div([
                        html.Span(cls_name.replace("_", " "),
                                  style={"fontSize": "9px", "color": PALETTE["text_dim"],
                                         "display": "inline-block", "width": "110px"}),
                        html.Div(style={
                            "display": "inline-block", "height": "6px",
                            "width": bar_w, "background": color,
                            "borderRadius": "3px", "marginLeft": "4px",
                            "verticalAlign": "middle",
                        }),
                        html.Span(f" {prob*100:.0f}%",
                                  style={"fontSize": "9px", "color": PALETTE["text"],
                                         "marginLeft": "4px"}),
                    ], style={"marginBottom": "2px"})
                )

        pred_info = ""
        for p in ENGINE.latest_predictions:
            if p.track_id == track.track_id and p.impact_point_m:
                x, y, _ = p.impact_point_m
                pred_info = f"⚠ CPA: ({x/1e3:.0f}, {y/1e3:.0f}) km"
                break

        card = html.Div([
            # Header row
            html.Div([
                html.Span(f"◈ {track.track_id}", style={
                    "color": color, "fontWeight": "bold",
                    "fontSize": "11px", "fontFamily": "monospace",
                }),
                html.Span(f" {alert.upper()}", style={
                    "color": color, "fontSize": "9px",
                    "border": f"1px solid {color}",
                    "borderRadius": "3px", "padding": "1px 4px",
                    "marginLeft": "6px",
                }),
            ], style={"marginBottom": "5px"}),

            # Type + kinematics
            html.Div([
                html.Span(track.threat_type.replace("_", " ").title(),
                          style={"color": PALETTE["text"], "fontSize": "10px"}),
                html.Span(f"  {track.speed_mps():.0f} m/s",
                          style={"color": PALETTE["text_dim"], "fontSize": "9px",
                                 "marginLeft": "8px"}),
                html.Span(f"  {track.position_3d()[2]/1e3:.1f} km alt",
                          style={"color": PALETTE["text_dim"], "fontSize": "9px",
                                 "marginLeft": "8px"}),
            ], style={"marginBottom": "5px"}),

            # Range + radial velocity
            html.Div([
                html.Span(f"R: {track.range_m()/1e3:.1f} km",
                          style={"color": PALETTE["text_dim"], "fontSize": "9px",
                                 "marginRight": "12px"}),
                html.Span(f"v_r: {track.radial_velocity_mps():.0f} m/s",
                          style={"color": PALETTE["text_dim"], "fontSize": "9px"}),
            ], style={"marginBottom": "4px"}),

            # Probability bars
            *prob_bars,

            # CPA / impact info
            html.Div(pred_info, style={
                "color": PALETTE["warning"], "fontSize": "9px",
                "marginTop": "4px", "fontFamily": "monospace",
            }) if pred_info else html.Div(),

        ], style={
            "border": f"1px solid {color}",
            "borderLeft": f"3px solid {color}",
            "borderRadius": "4px",
            "padding": "8px 10px",
            "marginBottom": "8px",
            "background": PALETTE["bg_card"],
        })
        cards.append(card)

    if not cards:
        cards.append(html.Div(
            "NO CONFIRMED TRACKS",
            style={"color": PALETTE["text_dim"], "fontSize": "11px",
                   "fontFamily": "monospace", "padding": "20px",
                   "textAlign": "center"},
        ))
    return cards


def build_track_table(tracks: list[Track]) -> html.Table:
    """Compact data table of all active tracks."""
    headers = ["ID", "Type", "Alert", "Range(km)", "Speed(m/s)",
               "Alt(km)", "Az(°)", "Conf%", "Status"]
    rows = []
    for t in sorted(tracks, key=lambda x: x.range_m())[:12]:
        cl    = ENGINE.latest_classifications.get(t.track_id)
        conf  = f"{t.threat_confidence*100:.0f}" if t.threat_confidence > 0 else "—"
        color = ALERT_COLORS.get(t.alert_level, PALETTE["text_dim"])
        rows.append(html.Tr([
            html.Td(t.track_id, style={"color": PALETTE["accent"]}),
            html.Td(t.threat_type.replace("_", " ")[:12]),
            html.Td(t.alert_level.upper(),
                    style={"color": color, "fontWeight": "bold"}),
            html.Td(f"{t.range_m()/1e3:.1f}"),
            html.Td(f"{t.speed_mps():.0f}"),
            html.Td(f"{t.position_3d()[2]/1e3:.1f}"),
            html.Td(f"{t.azimuth_deg():.0f}"),
            html.Td(conf),
            html.Td(t.status.name[:4]),
        ], style={"borderBottom": f"1px solid {PALETTE['border']}",
                  "fontSize": "9px", "fontFamily": "monospace"}))

    return html.Table(
        [html.Thead(html.Tr([
            html.Th(h, style={"color": PALETTE["text_dim"], "fontSize": "9px",
                              "fontFamily": "monospace", "fontWeight": "normal",
                              "padding": "4px 6px", "borderBottom":
                              f"1px solid {PALETTE['border']}"})
            for h in headers
        ]))] + [html.Tbody(rows)],
        style={"width": "100%", "borderCollapse": "collapse"},
    )


def ALERT_THRESHOLDS_ORDER(level: str) -> int:
    order = {"critical": 5, "high": 4, "medium": 3, "low": 2, "none": 1}
    return order.get(level, 0)




external_stylesheets = [dbc.themes.SLATE]

app = dash.Dash(
    __name__,
    external_stylesheets=external_stylesheets,
    title="SkySentinel — Air Defense AI",
    suppress_callback_exceptions=True,
    update_title=None,
)

HEADER = html.Div([
    # Logo + title
    html.Div([
        html.Div("◈", style={"color": PALETTE["accent"], "fontSize": "28px",
                              "marginRight": "10px", "lineHeight": "1"}),
        html.Div([
            html.Div("SKYSENTINEL", style={
                "color": PALETTE["accent"], "fontSize": "18px",
                "fontWeight": "bold", "fontFamily": "monospace",
                "letterSpacing": "6px", "lineHeight": "1.1",
            }),
            html.Div("AI AIR DEFENSE THREAT PREDICTION SYSTEM", style={
                "color": PALETTE["text_dim"], "fontSize": "8px",
                "fontFamily": "monospace", "letterSpacing": "3px",
            }),
        ]),
    ], style={"display": "flex", "alignItems": "center"}),

    # Status + controls cluster
    html.Div([
        # Scenario selector
        html.Div([
            html.Span("SCENARIO  ", style={"color": PALETTE["text_dim"],
                      "fontSize": "9px", "fontFamily": "monospace"}),
            dcc.Dropdown(
                id="scenario-select",
                options=[
                    {"label": "Mixed Threat",      "value": "mixed_threat"},
                    {"label": "Drone Swarm",        "value": "drone_swarm"},
                    {"label": "Single Missile",     "value": "single_missile"},
                    {"label": "Saturation Attack",  "value": "saturation_attack"},
                ],
                value="mixed_threat",
                clearable=False,
                style={"width": "160px", "fontSize": "10px",
                       "color": PALETTE["bg_dark"], "display": "inline-block"},
            ),
        ], style={"marginRight": "20px", "display": "flex",
                  "alignItems": "center", "gap": "6px"}),

        # Jamming toggle
        html.Div([
            html.Span("ECM JAM  ", style={"color": PALETTE["text_dim"],
                       "fontSize": "9px", "fontFamily": "monospace"}),
            dbc.Switch(id="jamming-toggle", value=False,
                       style={"display": "inline-block"}),
        ], style={"marginRight": "20px", "display": "flex",
                  "alignItems": "center", "gap": "6px"}),

        # Scan counter + status
        html.Div(id="header-status", style={
            "color": PALETTE["success"], "fontSize": "10px",
            "fontFamily": "monospace", "textAlign": "right",
        }),
    ], style={"display": "flex", "alignItems": "center"}),

], style={
    "display": "flex", "justifyContent": "space-between", "alignItems": "center",
    "background": PALETTE["bg_panel"],
    "borderBottom": f"1px solid {PALETTE['border']}",
    "padding": "12px 20px",
})


app.layout = html.Div([
    HEADER,

    # ── Main content ──────────────────────────────────────────────────────
    html.Div([
        # Left column — 3-D scope + bottom row
        html.Div([
            dcc.Graph(id="radar-scope-3d",
                      config={"displayModeBar": True, "scrollZoom": True},
                      style={"height": "480px"}),

            # Bottom row
            html.Div([
                html.Div([
                    dcc.Graph(id="rd-map", config={"displayModeBar": False},
                              style={"height": "260px"}),
                ], style={"flex": "1.2", "marginRight": "8px"}),

                html.Div([
                    dcc.Graph(id="timeline-chart", config={"displayModeBar": False},
                              style={"height": "260px"}),
                ], style={"flex": "1", "marginRight": "8px"}),

                html.Div([
                    html.Div("TRACK DATA", style={
                        "color": PALETTE["accent"], "fontSize": "10px",
                        "fontFamily": "monospace", "padding": "8px 4px 4px 4px",
                    }),
                    html.Div(id="track-table",
                             style={"overflowX": "auto", "overflowY": "auto",
                                    "maxHeight": "220px"}),
                ], style={
                    "flex": "1.1", "background": PALETTE["bg_panel"],
                    "border": f"1px solid {PALETTE['border']}",
                    "borderRadius": "4px", "padding": "4px",
                }),

            ], style={"display": "flex", "marginTop": "8px"}),

        ], style={"flex": "1", "marginRight": "10px"}),

        # Right column — threat cards
        html.Div([
            html.Div("THREAT ASSESSMENT", style={
                "color": PALETTE["accent"], "fontSize": "10px",
                "fontFamily": "monospace", "letterSpacing": "2px",
                "marginBottom": "10px", "paddingBottom": "8px",
                "borderBottom": f"1px solid {PALETTE['border']}",
            }),
            html.Div(id="threat-cards",
                     style={"overflowY": "auto", "maxHeight": "750px"}),
        ], style={
            "width": "260px", "flexShrink": "0",
            "background": PALETTE["bg_panel"],
            "border": f"1px solid {PALETTE['border']}",
            "borderRadius": "4px", "padding": "12px",
        }),

    ], style={"display": "flex", "padding": "10px",
              "background": PALETTE["bg_dark"], "minHeight": "calc(100vh - 60px)"}),

    # Auto-refresh timer
    dcc.Interval(id="scan-timer", interval=SCAN_INTERVAL_MS, n_intervals=0),

    # Hidden store for scan state
    dcc.Store(id="scan-store", data={"scan_count": 0}),

], style={"background": PALETTE["bg_dark"], "minHeight": "100vh",
          "fontFamily": "monospace"})




@app.callback(
    Output("radar-scope-3d",  "figure"),
    Output("rd-map",           "figure"),
    Output("timeline-chart",   "figure"),
    Output("threat-cards",     "children"),
    Output("track-table",      "children"),
    Output("header-status",    "children"),
    Output("scan-store",       "data"),
    Input("scan-timer",        "n_intervals"),
    Input("scenario-select",   "value"),
    Input("jamming-toggle",    "value"),
    State("scan-store",        "data"),
    prevent_initial_call=False,
)
def update_dashboard(n_intervals, scenario, jamming_on, store_data):
    """Master callback: run one simulation tick and update all panels."""

    # Handle control inputs
    trigger_id = ctx.triggered_id

    if trigger_id == "scenario-select" and scenario != ENGINE.scenario:
        ENGINE.set_scenario(scenario)

    if trigger_id == "jamming-toggle":
        ENGINE.toggle_jamming(bool(jamming_on))

    # Run simulation tick
    tracks = ENGINE.tick()

    # Build all figures
    scope_fig     = build_radar_scope_3d(
        tracks,
        ENGINE.latest_predictions,
        ENGINE.latest_detections,
    )
    rd_fig         = build_rd_map_figure(ENGINE.latest_rd_map)
    timeline_fig   = build_timeline_figure(tracks)
    threat_cards   = build_threat_cards(tracks)
    track_table    = build_track_table(tracks)

    n_confirmed  = len(ENGINE.tracker.confirmed_tracks)
    n_critical   = sum(1 for t in tracks
                       if t.alert_level in ("critical", "high"))
    jam_str      = " ⚡ ECM ACTIVE" if ENGINE.noise_cfg.enable_jamming else ""

    status_html = html.Div([
        html.Span(f"SCAN #{ENGINE.scan_count:04d}  ",
                  style={"color": PALETTE["accent"]}),
        html.Span(f"TRACKS: {n_confirmed}  ",
                  style={"color": PALETTE["success"]}),
        html.Span(f"THREATS: {n_critical}",
                  style={"color": PALETTE["danger"] if n_critical > 0
                         else PALETTE["text_dim"]}),
        html.Span(jam_str,
                  style={"color": PALETTE["warning"]}),
    ])

    new_store = {"scan_count": ENGINE.scan_count}
    return (scope_fig, rd_fig, timeline_fig, threat_cards,
            track_table, status_html, new_store)



if __name__ == "__main__":
    print("=" * 62)
    print("  SkySentinel — AI Air Defense Threat Prediction System")
    print("=" * 62)
    print("  Dashboard URL : http://127.0.0.1:8050")
    print("  Press Ctrl+C  to stop")
    print("=" * 62)
    app.run(debug=False, host="127.0.0.1", port=8050)