from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.signal import butter, lfilter




def db_to_linear(db: float) -> float:
    return 10.0 ** (db / 10.0)

def linear_to_db(linear: float) -> float:
    return 10.0 * math.log10(max(linear, 1e-30))




@dataclass
class ThermalNoiseConfig:

    noise_power_dbw: float = -120.0     # Flat override  [dBW/Hz · Hz = dBW]
    use_radar_params: bool = False       # If True, compute from radar params


def add_thermal_noise(
    data_cube: np.ndarray,
    config: ThermalNoiseConfig,
    rng: np.random.Generator,
    radar_params=None,          # RadarParameters instance (optional)
) -> np.ndarray:

    cube = data_cube.copy()
    n_pulses, n_gates = cube.shape

    # Determine noise power
    if config.use_radar_params and radar_params is not None:
        noise_power = radar_params.thermal_noise_power()
    else:
        noise_power = db_to_linear(config.noise_power_dbw)

    sigma = math.sqrt(noise_power / 2.0)   # per quadrature channel

    noise = sigma * (
        rng.standard_normal((n_pulses, n_gates))
        + 1j * rng.standard_normal((n_pulses, n_gates))
    )
    cube += noise
    return cube




@dataclass
class ClutterConfig:

    amplitude_model: str = "weibull"
    shape_param: float = 1.5               # Weibull shape
    clutter_to_noise_ratio_db: float = 30.0
    range_ref_m: float = 50_000.0          # 50 km reference range
    max_range_m: float = 80_000.0          # Clutter-free beyond this
    pulse_correlation: float = 0.95        # Ground clutter is highly correlated
    elevation_mask_deg: float = 3.0        # Ignore clutter above this elevation


def _weibull_complex(
    shape: float, scale: float, n_pulses: int, n_gates: int,
    rng: np.random.Generator,
) -> np.ndarray:

    amplitude = scale * rng.weibull(shape, size=(n_pulses, n_gates))
    phase = rng.uniform(0, 2 * math.pi, size=(n_pulses, n_gates))
    return amplitude * np.exp(1j * phase)


def add_clutter(
    data_cube: np.ndarray,
    config: ClutterConfig,
    rng: np.random.Generator,
    range_gate_length_m: float = 75.0,
    noise_power: float = 1e-12,
) -> np.ndarray:

    cube = data_cube.copy()
    n_pulses, n_gates = cube.shape

    # Reference clutter amplitude from CNR
    cnr_linear = db_to_linear(config.clutter_to_noise_ratio_db)
    ref_amplitude = math.sqrt(cnr_linear * noise_power)

    # Range axis [m]
    ranges = np.arange(n_gates) * range_gate_length_m + 1.0  # avoid zero

    # Clutter gate mask: only within max_range
    clutter_gates = int(min(config.max_range_m, n_gates * range_gate_length_m)
                        / range_gate_length_m)

    # Range-dependent amplitude scaling  A(R) ∝ 1/R  (distributed clutter)
    range_weight = np.zeros(n_gates)
    range_weight[:clutter_gates] = (
        config.range_ref_m / ranges[:clutter_gates]
    )

    # Generate base clutter (n_pulses × n_gates)
    scale = ref_amplitude
    raw_clutter = _weibull_complex(
        config.shape_param, scale, n_pulses, n_gates, rng
    )

    # Apply range weighting
    raw_clutter *= range_weight[np.newaxis, :]

    # Inter-pulse correlation via AR(1) filter along slow-time axis
    α = config.pulse_correlation
    if α > 0.0 and n_pulses > 1:
        corr_clutter = np.zeros_like(raw_clutter)
        corr_clutter[0] = raw_clutter[0]
        for p in range(1, n_pulses):
            corr_clutter[p] = α * corr_clutter[p - 1] + math.sqrt(1 - α**2) * raw_clutter[p]
    else:
        corr_clutter = raw_clutter

    cube += corr_clutter
    return cube




@dataclass
class JammingConfig:

    jam_type: str = "barrage"
    jammer_to_noise_ratio_db: float = 20.0
    affected_range_fraction: float = 1.0        # barrage / spot
    sweep_rate_gates_per_pulse: float = 10.0    # swept jammer
    sweep_bandwidth_fraction: float = 0.1       # swept: fractional BW per step
    drfm_delay_gates: int = 50                  # DRFM false-target offset
    drfm_doppler_shift_hz: float = 500.0        # DRFM velocity spoofing
    prf_hz: float = 1000.0                      # Needed for DRFM phase calc


def add_jamming(
    data_cube: np.ndarray,
    config: JammingConfig,
    rng: np.random.Generator,
    noise_power: float = 1e-12,
    reference_pulse: Optional[np.ndarray] = None,
) -> np.ndarray:

    cube = data_cube.copy()
    n_pulses, n_gates = cube.shape

    jnr_linear = db_to_linear(config.jammer_to_noise_ratio_db)
    jam_amplitude = math.sqrt(jnr_linear * noise_power)

    # ── Barrage Jamming ─────────────────────────────────────────────────────
    if config.jam_type == "barrage":
        n_jammed = max(1, int(config.affected_range_fraction * n_gates))
        jam_mask = np.zeros(n_gates)
        jam_mask[:n_jammed] = 1.0
        # Randomise which gates are jammed (not always front-loaded)
        rng.shuffle(jam_mask)

        jam_noise = jam_amplitude * (
            rng.standard_normal((n_pulses, n_gates))
            + 1j * rng.standard_normal((n_pulses, n_gates))
        )
        cube += jam_noise * jam_mask[np.newaxis, :]

    # ── Spot Jamming ────────────────────────────────────────────────────────
    elif config.jam_type == "spot":
        # Spot jammer: concentrated power in a small range window
        # representing a narrowband emission that saturates nearby gates
        spot_width = max(1, int(0.05 * n_gates))  # 5 % of range axis
        spot_centre = rng.integers(spot_width, n_gates - spot_width)

        jam_noise = jam_amplitude * 3.0 * (   # 3× more powerful than barrage
            rng.standard_normal((n_pulses, spot_width))
            + 1j * rng.standard_normal((n_pulses, spot_width))
        )
        cube[:, spot_centre - spot_width//2: spot_centre + spot_width//2] += jam_noise

    # ── Swept Jamming ────────────────────────────────────────────────────────
    elif config.jam_type == "swept":
        bw_gates = max(1, int(config.sweep_bandwidth_fraction * n_gates))
        for p in range(n_pulses):
            # Sweep start gate advances each pulse
            sweep_start = int((p * config.sweep_rate_gates_per_pulse) % n_gates)
            sweep_end = min(sweep_start + bw_gates, n_gates)
            width = sweep_end - sweep_start

            jam_slice = jam_amplitude * 2.0 * (
                rng.standard_normal(width) + 1j * rng.standard_normal(width)
            )
            cube[p, sweep_start:sweep_end] += jam_slice

   
    elif config.jam_type == "drfm":
        if reference_pulse is None:
            raise ValueError("DRFM jamming requires a reference_pulse array.")

        pulse_len = len(reference_pulse)
        pri_sec = 1.0 / config.prf_hz

        for p in range(n_pulses):
            # Re-transmit a delayed copy of the reference pulse
            delay_g = config.drfm_delay_gates
            end_g = delay_g + pulse_len

            if end_g > n_gates:
                end_g = n_gates

            available = end_g - delay_g
            if available <= 0:
                continue

            # Apply Doppler modulation (false velocity injection)
            doppler_phase = (
                2.0 * math.pi
                * config.drfm_doppler_shift_hz
                * (p * pri_sec)
            )
            false_echo = (
                jam_amplitude
                * reference_pulse[:available]
                * np.exp(1j * doppler_phase)
            )
            cube[p, delay_g: delay_g + available] += false_echo

    else:
        raise ValueError(
            f"Unknown jam_type '{config.jam_type}'. "
            "Choose: barrage | spot | swept | drfm"
        )

    return cube




@dataclass
class SpoofingConfig:

    ghost_range_m: float = 120_000.0
    ghost_radial_vel_mps: float = -250.0    # Closing target (negative)
    ghost_snr_db: float = 15.0
    n_ghosts: int = 1
    ghost_spread_m: float = 5_000.0
    prf_hz: float = 1000.0
    wavelength_m: float = 0.1              # Carrier wavelength for Doppler


def add_spoofing(
    data_cube: np.ndarray,
    config: SpoofingConfig,
    rng: np.random.Generator,
    reference_pulse: np.ndarray,
    range_gate_length_m: float = 75.0,
    noise_power: float = 1e-12,
) -> tuple[np.ndarray, list[dict]]:

    cube = data_cube.copy()
    n_pulses, n_gates = cube.shape
    pulse_len = len(reference_pulse)
    pri_sec = 1.0 / config.prf_hz
    ghost_truths: list[dict] = []

    snr_linear = db_to_linear(config.ghost_snr_db)
    ghost_amplitude = math.sqrt(snr_linear * noise_power)

    for g_idx in range(config.n_ghosts):
        # Scatter ghost positions around the centre
        if config.n_ghosts > 1:
            offset_m = rng.uniform(-config.ghost_spread_m / 2,
                                   config.ghost_spread_m / 2)
        else:
            offset_m = 0.0

        ghost_range = config.ghost_range_m + offset_m
        ghost_vel   = config.ghost_radial_vel_mps + rng.uniform(-5, 5)

        gate_idx = int(2.0 * ghost_range / (3.0e8 / (n_gates / (2.0 * ghost_range / 3e8 * n_gates)))
                       * n_gates / (2.0 * ghost_range / 3e8))
        # Simpler: compute directly from range and gate spacing
        gate_idx = int(
            (2.0 * ghost_range / 3.0e8)    # two-way delay [s]
            * (3.0e8 / (2.0 * range_gate_length_m))   # gates per second
        )

        f_doppler = -2.0 * ghost_vel / config.wavelength_m

        ghost_truths.append({
            "ghost_id": g_idx,
            "range_m": ghost_range,
            "radial_vel_mps": ghost_vel,
            "gate_idx": gate_idx,
            "f_doppler_hz": f_doppler,
            "snr_db": config.ghost_snr_db,
        })

        for p in range(n_pulses):
            if gate_idx >= n_gates or gate_idx < 0:
                continue

            doppler_phase = 2.0 * math.pi * f_doppler * (p * pri_sec)
            end_g = min(gate_idx + pulse_len, n_gates)
            available = end_g - gate_idx

            ghost_echo = (
                ghost_amplitude
                * reference_pulse[:available]
                * np.exp(1j * doppler_phase)
            )
            cube[p, gate_idx: gate_idx + available] += ghost_echo

    return cube, ghost_truths


# ──────────────────────────────────────────────────────────────────────────────
# 5. Composite Noise Pipeline
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class NoisePipelineConfig:

    # ── Thermal noise ────────────────────────────────────
    enable_thermal: bool = True
    thermal: ThermalNoiseConfig = field(default_factory=ThermalNoiseConfig)

    # ── Clutter ──────────────────────────────────────────
    enable_clutter: bool = True
    clutter: ClutterConfig = field(default_factory=ClutterConfig)

    # ── Jamming ──────────────────────────────────────────
    enable_jamming: bool = False
    jamming: JammingConfig = field(default_factory=JammingConfig)

    # ── Spoofing ─────────────────────────────────────────
    enable_spoofing: bool = False
    spoofing: SpoofingConfig = field(default_factory=SpoofingConfig)


def apply_noise_pipeline(
    data_cube: np.ndarray,
    config: NoisePipelineConfig,
    rng: np.random.Generator,
    range_gate_length_m: float = 75.0,
    reference_pulse: Optional[np.ndarray] = None,
    radar_params=None,
    noise_power: float = 1e-12,
) -> tuple[np.ndarray, dict]:

    cube = data_cube.copy()
    report: dict = {"stages_applied": []}

    # ── Stage 1: Thermal Noise ───────────────────────────────────────────────
    if config.enable_thermal:
        cube = add_thermal_noise(
            cube, config.thermal, rng, radar_params=radar_params
        )
        report["stages_applied"].append("thermal_noise")
        report["thermal_noise_power_dbw"] = config.thermal.noise_power_dbw

    # ── Stage 2: Clutter ─────────────────────────────────────────────────────
    if config.enable_clutter:
        cube = add_clutter(
            cube, config.clutter, rng,
            range_gate_length_m=range_gate_length_m,
            noise_power=noise_power,
        )
        report["stages_applied"].append("clutter")
        report["clutter_cnr_db"] = config.clutter.clutter_to_noise_ratio_db

    # ── Stage 3: Jamming ─────────────────────────────────────────────────────
    if config.enable_jamming:
        cube = add_jamming(
            cube, config.jamming, rng,
            noise_power=noise_power,
            reference_pulse=reference_pulse,
        )
        report["stages_applied"].append(f"jamming_{config.jamming.jam_type}")
        report["jamming_jnr_db"] = config.jamming.jammer_to_noise_ratio_db

    # ── Stage 4: Spoofing ────────────────────────────────────────────────────
    ghost_truths: list[dict] = []
    if config.enable_spoofing:
        if reference_pulse is None:
            raise ValueError("Spoofing requires reference_pulse to be provided.")
        cube, ghost_truths = add_spoofing(
            cube, config.spoofing, rng,
            reference_pulse=reference_pulse,
            range_gate_length_m=range_gate_length_m,
            noise_power=noise_power,
        )
        report["stages_applied"].append("spoofing")
        report["n_ghosts"] = config.spoofing.n_ghosts

    report["ghost_truths"] = ghost_truths
    return cube, report




def estimate_noise_floor(
    data_cube: np.ndarray,
    percentile: float = 20.0,
) -> float:

    gate_power = np.mean(np.abs(data_cube) ** 2, axis=0)  # average across pulses
    return float(np.percentile(gate_power, percentile))


def compute_sinr(
    clean_cube: np.ndarray,
    degraded_cube: np.ndarray,
) -> np.ndarray:

    signal_power = np.mean(np.abs(clean_cube) ** 2, axis=0)
    noise_power  = np.mean(np.abs(degraded_cube - clean_cube) ** 2, axis=0)
    sinr_db = 10.0 * np.log10(
        (signal_power + 1e-30) / (noise_power + 1e-30)
    )
    return sinr_db




if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    from radar_simulation.generate_signals import (
        RadarParameters, generate_cpi, generate_lfm_pulse,
    )

    print("=" * 60)
    print("  SkySentinel — Noise Model Self-Test")
    print("=" * 60)

    # ── 1. Generate a clean data cube ───────────────────────────────────────
    rng_main = np.random.default_rng(seed=7)
    radar = RadarParameters(n_pulses_per_cpi=64, max_range_m=400_000)
    clean_cube, meta, targets = generate_cpi(
        scenario="mixed_threat", radar=radar, seed=7, verbose=False
    )
    ref_pulse = generate_lfm_pulse(radar, rng_main)

    # ── 2. Apply full noise pipeline (all stages enabled) ───────────────────
    pipeline_cfg = NoisePipelineConfig(
        enable_thermal=True,
        thermal=ThermalNoiseConfig(noise_power_dbw=-110.0),
        enable_clutter=True,
        clutter=ClutterConfig(
            clutter_to_noise_ratio_db=35.0,
            max_range_m=100_000.0,
            pulse_correlation=0.97,
        ),
        enable_jamming=True,
        jamming=JammingConfig(
            jam_type="barrage",
            jammer_to_noise_ratio_db=15.0,
            affected_range_fraction=0.6,
        ),
        enable_spoofing=True,
        spoofing=SpoofingConfig(
            ghost_range_m=200_000.0,
            ghost_radial_vel_mps=-300.0,
            ghost_snr_db=18.0,
            n_ghosts=2,
        ),
    )

    degraded_cube, report = apply_noise_pipeline(
        clean_cube, pipeline_cfg,
        rng=rng_main,
        range_gate_length_m=radar.range_gate_length_m,
        reference_pulse=ref_pulse,
        noise_power=radar.thermal_noise_power(),
    )

    print(f"  Pipeline stages : {report['stages_applied']}")
    print(f"  Ghost truths    : {report.get('ghost_truths', [])}")
    noise_floor = estimate_noise_floor(degraded_cube)
    print(f"  Noise floor est : {linear_to_db(noise_floor):.1f} dB")

    sinr = compute_sinr(clean_cube, degraded_cube)
    print(f"  SINR range      : {sinr.min():.1f} → {sinr.max():.1f} dB")
    print("=" * 60)

    # ── 3. Visualisation ────────────────────────────────────────────────────
    BG = "#0a0f1e"
    GRID_C = "#1e2a3a"
    TEXT_C = "white"

    fig = plt.figure(figsize=(16, 10), facecolor=BG)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)
    fig.suptitle("SkySentinel — Noise & Interference Model",
                 color=TEXT_C, fontsize=14, fontweight="bold", y=0.97)

    def styled_ax(ax, title):
        ax.set_facecolor(BG)
        ax.tick_params(colors=TEXT_C, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(GRID_C)
        ax.set_title(title, color=TEXT_C, fontsize=9)
        ax.grid(color=GRID_C, linewidth=0.5)

    # Clean range profile
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(np.abs(clean_cube[0]), color="#00d4ff", lw=0.8)
    ax0.set_xlabel("Range Gate", color=TEXT_C, fontsize=8)
    ax0.set_ylabel("|Amplitude|", color=TEXT_C, fontsize=8)
    styled_ax(ax0, "Clean — Range Profile (Pulse 0)")

    # Degraded range profile
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.plot(np.abs(degraded_cube[0]), color="#ff6b35", lw=0.8)
    ax1.set_xlabel("Range Gate", color=TEXT_C, fontsize=8)
    styled_ax(ax1, "Degraded — Range Profile (Pulse 0)")

    # SINR profile
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.plot(sinr, color="#a8ff78", lw=0.8)
    ax2.axhline(0, color="#ff4444", lw=0.8, linestyle="--", label="0 dB")
    ax2.set_xlabel("Range Gate", color=TEXT_C, fontsize=8)
    ax2.set_ylabel("SINR (dB)", color=TEXT_C, fontsize=8)
    ax2.legend(fontsize=7, facecolor=BG, labelcolor=TEXT_C)
    styled_ax(ax2, "SINR per Range Gate")

    def rd_map(cube, ax, title):
        rd = np.abs(np.fft.fftshift(np.fft.fft(cube, axis=0), axes=0))
        rd_db = 20 * np.log10(rd + 1e-12)
        im = ax.imshow(rd_db, aspect="auto", origin="lower",
                       cmap="inferno",
                       extent=[0, cube.shape[1], -500, 500])
        ax.set_xlabel("Range Gate", color=TEXT_C, fontsize=8)
        ax.set_ylabel("Doppler (Hz)", color=TEXT_C, fontsize=8)
        styled_ax(ax, title)
        return im

    ax3 = fig.add_subplot(gs[1, 0])
    rd_map(clean_cube, ax3, "Clean — Range-Doppler Map")

    ax4 = fig.add_subplot(gs[1, 1])
    rd_map(degraded_cube, ax4, "Degraded — Range-Doppler Map")

    # Clutter-only difference
    ax5 = fig.add_subplot(gs[1, 2])
    diff = np.abs(degraded_cube) - np.abs(clean_cube)
    diff_db = 20 * np.log10(np.abs(diff) + 1e-12)
    ax5.imshow(diff_db, aspect="auto", origin="lower", cmap="plasma",
               extent=[0, clean_cube.shape[1], 0, clean_cube.shape[0]])
    ax5.set_xlabel("Range Gate", color=TEXT_C, fontsize=8)
    ax5.set_ylabel("Pulse Index", color=TEXT_C, fontsize=8)
    styled_ax(ax5, "Interference Layer (Degraded − Clean)")

    out_path = "noise_model_preview.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"\n  Preview saved → {out_path}")