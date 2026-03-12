from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.signal import windows, convolve




def _next_pow2(n: int) -> int:
    """Return the smallest power of 2 that is >= n."""
    return 1 << (n - 1).bit_length()


def db(x: np.ndarray) -> np.ndarray:
    """Convert linear amplitude/power array to dB (20·log10 for amplitudes)."""
    return 20.0 * np.log10(np.abs(x) + 1e-30)



@dataclass
class MatchedFilterConfig:

    window_type: str = "taylor"
    nfft_override: Optional[int] = None


def matched_filter(
    data_cube: np.ndarray,
    reference_pulse: np.ndarray,
    config: MatchedFilterConfig = MatchedFilterConfig(),
) -> np.ndarray:

    n_pulses, n_gates = data_cube.shape
    pulse_len = len(reference_pulse)

    # FFT length: large enough for linear (non-circular) convolution
    nfft = config.nfft_override or _next_pow2(n_gates + pulse_len - 1)
    nfft = max(nfft, n_gates)

    # ── Sidelobe window applied to the reference spectrum ───────────────────
    wtype = config.window_type.lower()
    if wtype == "none":
        win = np.ones(pulse_len)
    elif wtype == "hamming":
        win = windows.hamming(pulse_len)
    elif wtype == "hann":
        win = windows.hann(pulse_len)
    elif wtype == "blackman":
        win = windows.blackman(pulse_len)
    elif wtype == "taylor":
        win = windows.taylor(pulse_len, nbar=4, sll=40)
    else:
        raise ValueError(f"Unknown window type: {wtype!r}")

    # Matched filter transfer function (complex conjugate in freq domain)
    ref_windowed = reference_pulse * win
    H_mf = np.conj(np.fft.fft(ref_windowed, n=nfft))

    # Apply filter to each pulse (row) via FFT convolution
    compressed = np.zeros((n_pulses, n_gates), dtype=np.complex128)
    for p in range(n_pulses):
        row_fft = np.fft.fft(data_cube[p], n=nfft)
        conv    = np.fft.ifft(row_fft * H_mf)
        # Trim to original gate count (discard convolution tail)
        compressed[p] = conv[:n_gates]

    return compressed




@dataclass
class MTIConfig:

    order: int = 2
    blind_velocity_fraction: float = 0.05   # ~5 % near v=0 suppressed


def mti_filter(
    data_cube: np.ndarray,
    config: MTIConfig = MTIConfig(),
) -> np.ndarray:

    n_pulses, n_gates = data_cube.shape
    order = config.order

    if n_pulses <= order:
        raise ValueError(
            f"MTI order {order} requires at least {order+1} pulses; "
            f"got {n_pulses}."
        )

    # Binomial difference coefficients with alternating sign
    from math import comb
    coeffs = np.array(
        [(-1) ** k * comb(order, k) for k in range(order + 1)],
        dtype=np.complex128,
    )

    n_out = n_pulses - order
    output = np.zeros((n_out, n_gates), dtype=np.complex128)

    for i in range(n_out):
        for k, c in enumerate(coeffs):
            output[i] += c * data_cube[i + order - k]

    return output




@dataclass
class DopplerFFTConfig:

    window_type: str = "hann"
    nfft_doppler: Optional[int] = None
    output_db: bool = False


def doppler_fft(
    data_cube: np.ndarray,
    config: DopplerFFTConfig = DopplerFFTConfig(),
) -> np.ndarray:

    n_pulses, n_gates = data_cube.shape
    nfft = config.nfft_doppler or n_pulses

    # Apply slow-time window to reduce Doppler sidelobes
    wtype = config.window_type.lower()
    if wtype == "none":
        win = np.ones(n_pulses)
    elif wtype == "hann":
        win = windows.hann(n_pulses)
    elif wtype == "hamming":
        win = windows.hamming(n_pulses)
    elif wtype == "blackman":
        win = windows.blackman(n_pulses)
    elif wtype == "taylor":
        win = windows.taylor(n_pulses, nbar=4, sll=40)
    else:
        raise ValueError(f"Unknown window: {wtype!r}")

    windowed = data_cube * win[:, np.newaxis]

    # FFT along slow-time axis (axis 0), then fftshift to centre zero-Doppler
    rd_map = np.fft.fftshift(
        np.fft.fft(windowed, n=nfft, axis=0),
        axes=0,
    )

    if config.output_db:
        return db(rd_map).astype(np.float64)
    return rd_map


def rd_map_axes(
    n_doppler_bins: int,
    n_range_gates: int,
    prf_hz: float,
    range_gate_length_m: float,
    wavelength_m: float,
) -> tuple[np.ndarray, np.ndarray]:

    doppler_freq_axis = np.fft.fftshift(
        np.fft.fftfreq(n_doppler_bins, d=1.0 / prf_hz)
    )
    velocity_axis = -doppler_freq_axis * wavelength_m / 2.0   # v = -f_D·λ/2
    range_axis    = np.arange(n_range_gates) * range_gate_length_m / 1000.0

    return velocity_axis, range_axis



@dataclass
class CFARConfig:

    cfar_type: str = "CA"
    guard_cells_range: int = 2
    guard_cells_doppler: int = 2
    training_cells_range: int = 16
    training_cells_doppler: int = 8
    pfa: float = 1e-4
    os_rank: int = 18         # relevant for OS-CFAR
    apply_2d: bool = True
    min_snr_db: float = 8.0   # Hard SNR gate (rejects weak false alarms)


def _cfar_threshold_factor(n_train: int, pfa: float, cfar_type: str) -> float:
    """
    Analytical CFAR threshold multiplier α such that P_FA = pfa.

    For CA-CFAR: α = n_train · (pfa^(-1/n_train) − 1)
    """
    if cfar_type in ("CA", "GOCA", "SOCA"):
        return n_train * (pfa ** (-1.0 / n_train) - 1.0)
    else:
        # OS-CFAR: approximation
        return n_train * (pfa ** (-1.0 / n_train) - 1.0) * 1.2


@dataclass
class Detection:

    range_gate: int         # Range gate index (integer)
    doppler_bin: int        # Doppler bin index (integer, centred)
    range_m: float          # Estimated range [m]
    velocity_mps: float     # Estimated radial velocity [m/s]
    power_db: float         # Detection cell power [dB]
    threshold_db: float     # Local CFAR threshold [dB]
    snr_db: float           # SNR estimate = power − threshold  [dB]
    azimuth_deg: float = 0.0   # Filled by upstream angle-of-arrival module
    elevation_deg: float = 0.0

    def to_dict(self) -> dict:
        return {
            "range_gate": self.range_gate,
            "doppler_bin": self.doppler_bin,
            "range_m": self.range_m,
            "velocity_mps": self.velocity_mps,
            "power_db": self.power_db,
            "threshold_db": self.threshold_db,
            "snr_db": self.snr_db,
            "azimuth_deg": self.azimuth_deg,
            "elevation_deg": self.elevation_deg,
        }


def cfar_detector(
    rd_map: np.ndarray,
    config: CFARConfig,
    prf_hz: float,
    range_gate_length_m: float,
    wavelength_m: float,
) -> list[Detection]:

    # Work with power (real-valued)
    power_map = np.abs(rd_map) ** 2
    n_doppler, n_gates = power_map.shape

    # Axes for physical unit conversion
    velocity_axis, range_axis_km = rd_map_axes(
        n_doppler, n_gates, prf_hz, range_gate_length_m, wavelength_m
    )
    range_axis_m = range_axis_km * 1000.0

    # Guard and training margins
    gr = config.guard_cells_range
    tr = config.training_cells_range
    gd = config.guard_cells_doppler if config.apply_2d else 0
    td = config.training_cells_doppler if config.apply_2d else 0

    half_r = gr + tr
    half_d = gd + td

    n_train_r = 2 * tr
    n_train_d = 2 * td if config.apply_2d else 0
    n_train_total = (n_train_r + n_train_d) * (1 if not config.apply_2d else 1)
    # 2-D: training cells on 4 sides minus corners
    if config.apply_2d:
        n_train_total = (
            (2 * tr + 2 * gr + 1) * (2 * td + 2 * gd + 1)
            - (2 * gr + 1) * (2 * gd + 1)
        )

    alpha = _cfar_threshold_factor(max(n_train_total, 1), config.pfa, config.cfar_type)

    detections: list[Detection] = []

    d_range = range(half_d, n_doppler - half_d) if config.apply_2d else range(n_doppler)
    r_range = range(half_r, n_gates - half_r)

    for d_idx in d_range:
        for r_idx in r_range:
            cut = power_map[d_idx, r_idx]

            # Extract training window (exclude guard zone + CUT)
            if config.apply_2d:
                window = power_map[
                    d_idx - half_d: d_idx + half_d + 1,
                    r_idx - half_r: r_idx + half_r + 1,
                ]
                guard = power_map[
                    d_idx - gd: d_idx + gd + 1,
                    r_idx - gr: r_idx + gr + 1,
                ]
                # Training cells = window minus guard zone
                training_flat = window.ravel()
                guard_flat    = guard.ravel()
                # Build mask: cells in window but not in guard
                training_cells = []
                for dw in range(-half_d, half_d + 1):
                    for rw in range(-half_r, half_r + 1):
                        if abs(dw) > gd or abs(rw) > gr:
                            training_cells.append(power_map[d_idx + dw, r_idx + rw])
                training = np.array(training_cells)
            else:
                # 1-D along range only
                left  = power_map[d_idx, r_idx - half_r: r_idx - gr]
                right = power_map[d_idx, r_idx + gr + 1: r_idx + half_r + 1]
                training = np.concatenate([left, right])

            if len(training) == 0:
                continue

            # Noise estimate based on CFAR type
            if config.cfar_type == "CA":
                noise_est = np.mean(training)
            elif config.cfar_type == "GOCA":
                left_m  = np.mean(training[: len(training) // 2])
                right_m = np.mean(training[len(training) // 2 :])
                noise_est = max(left_m, right_m)
            elif config.cfar_type == "SOCA":
                left_m  = np.mean(training[: len(training) // 2])
                right_m = np.mean(training[len(training) // 2 :])
                noise_est = min(left_m, right_m)
            elif config.cfar_type == "OS":
                k = min(config.os_rank, len(training) - 1)
                noise_est = float(np.sort(training)[k])
            else:
                noise_est = np.mean(training)

            threshold = alpha * noise_est

            if cut > threshold:
                power_db     = float(db(np.sqrt(cut)))
                threshold_db = float(db(np.sqrt(threshold)))
                snr_db       = power_db - threshold_db

                if snr_db < config.min_snr_db:
                    continue

                det = Detection(
                    range_gate   = r_idx,
                    doppler_bin  = d_idx,
                    range_m      = float(range_axis_m[r_idx]),
                    velocity_mps = float(velocity_axis[d_idx]),
                    power_db     = power_db,
                    threshold_db = threshold_db,
                    snr_db       = snr_db,
                )
                detections.append(det)

    # Sort by ascending range
    detections.sort(key=lambda d: d.range_m)
    return detections




@dataclass
class DSPPipelineConfig:
    """Master config that chains all four DSP stages."""
    matched_filter: MatchedFilterConfig = field(default_factory=MatchedFilterConfig)
    mti: MTIConfig = field(default_factory=MTIConfig)
    doppler: DopplerFFTConfig = field(default_factory=DopplerFFTConfig)
    cfar: CFARConfig = field(default_factory=CFARConfig)

    enable_mti: bool = True     # Disable if studying jamming without MTI
    enable_cfar: bool = True


def run_dsp_pipeline(
    data_cube: np.ndarray,
    reference_pulse: np.ndarray,
    prf_hz: float,
    range_gate_length_m: float,
    wavelength_m: float,
    config: DSPPipelineConfig = DSPPipelineConfig(),
) -> tuple[np.ndarray, np.ndarray, list[Detection]]:
    """
    Run the full DSP chain on a (noisy) radar data cube.

    Parameters
    ----------
    data_cube           : np.ndarray  (n_pulses × n_range_gates), complex128
    reference_pulse     : np.ndarray  normalised LFM pulse
    prf_hz              : float
    range_gate_length_m : float
    wavelength_m        : float
    config              : DSPPipelineConfig

    Returns
    -------
    compressed_cube : np.ndarray  — after matched filter
    rd_map          : np.ndarray  — Range-Doppler map (complex)
    detections      : list[Detection]
    """
    # Stage 1 — Matched filter (pulse compression)
    compressed = matched_filter(data_cube, reference_pulse, config.matched_filter)

    # Stage 2 — MTI clutter cancellation
    if config.enable_mti:
        mti_out = mti_filter(compressed, config.mti)
    else:
        mti_out = compressed

    # Stage 3 — Doppler FFT
    rd_map = doppler_fft(mti_out, config.doppler)

    # Stage 4 — CFAR detection
    detections: list[Detection] = []
    if config.enable_cfar:
        detections = cfar_detector(
            rd_map, config.cfar,
            prf_hz=prf_hz,
            range_gate_length_m=range_gate_length_m,
            wavelength_m=wavelength_m,
        )

    return compressed, rd_map, detections




if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    from radar_simulation.generate_signals import (
        RadarParameters, generate_cpi, generate_lfm_pulse,
    )
    from radar_simulation.noise_model import (
        NoisePipelineConfig, ThermalNoiseConfig, ClutterConfig,
        apply_noise_pipeline,
    )

    print("=" * 62)
    print("  SkySentinel — DSP Filters Self-Test")
    print("=" * 62)

    # ── Build clean + noisy data cube ───────────────────────────────────────
    rng = np.random.default_rng(seed=99)
    radar = RadarParameters(n_pulses_per_cpi=128, max_range_m=400_000, prf_hz=2000)

    clean_cube, meta, targets = generate_cpi(
        scenario="mixed_threat", radar=radar, seed=99, verbose=False
    )
    ref_pulse = generate_lfm_pulse(radar, rng)

    noise_cfg = NoisePipelineConfig(
        enable_thermal=True,
        thermal=ThermalNoiseConfig(noise_power_dbw=-108.0),
        enable_clutter=True,
        clutter=ClutterConfig(clutter_to_noise_ratio_db=30.0, max_range_m=80_000.0),
        enable_jamming=False,
        enable_spoofing=False,
    )
    noisy_cube, _ = apply_noise_pipeline(
        clean_cube, noise_cfg, rng=rng,
        range_gate_length_m=radar.range_gate_length_m,
        reference_pulse=ref_pulse,
        noise_power=radar.thermal_noise_power(),
    )

    # ── Run DSP pipeline ────────────────────────────────────────────────────
    dsp_cfg = DSPPipelineConfig(
        matched_filter=MatchedFilterConfig(window_type="taylor"),
        mti=MTIConfig(order=2),
        doppler=DopplerFFTConfig(window_type="hann"),
        cfar=CFARConfig(
            cfar_type="CA",
            guard_cells_range=2, training_cells_range=12,
            guard_cells_doppler=2, training_cells_doppler=6,
            pfa=1e-4, apply_2d=True, min_snr_db=6.0,
        ),
        enable_mti=True,
        enable_cfar=True,
    )

    compressed, rd_map, detections = run_dsp_pipeline(
        noisy_cube, ref_pulse,
        prf_hz=radar.prf_hz,
        range_gate_length_m=radar.range_gate_length_m,
        wavelength_m=radar.wavelength_m,
        config=dsp_cfg,
    )

    print(f"  Pulse-compressed cube : {compressed.shape}")
    print(f"  Range-Doppler map     : {rd_map.shape}")
    print(f"  CFAR detections       : {len(detections)}")
    print()
    for i, d in enumerate(detections[:10]):
        print(
            f"  Det {i+1:02d}  R={d.range_m/1e3:7.1f} km  "
            f"v={d.velocity_mps:+7.1f} m/s  "
            f"SNR={d.snr_db:+5.1f} dB"
        )
    if len(detections) > 10:
        print(f"  ... (+{len(detections)-10} more)")
    print("=" * 62)

    # ── Velocity / range axes ───────────────────────────────────────────────
    n_dopp, n_gates = rd_map.shape
    vel_ax, rng_ax_km = rd_map_axes(
        n_dopp, n_gates,
        radar.prf_hz, radar.range_gate_length_m, radar.wavelength_m
    )

    # ── Visualisation ────────────────────────────────────────────────────────
    BG = "#0a0f1e"
    TC = "white"
    GC = "#1e2a3a"

    fig = plt.figure(figsize=(16, 10), facecolor=BG)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)
    fig.suptitle("SkySentinel — DSP Signal Processing Pipeline",
                 color=TC, fontsize=14, fontweight="bold")

    def sa(ax, title, xlabel="", ylabel=""):
        ax.set_facecolor(BG)
        ax.tick_params(colors=TC, labelsize=8)
        for sp in ax.spines.values():
            sp.set_color(GC)
        ax.set_title(title, color=TC, fontsize=9)
        ax.set_xlabel(xlabel, color=TC, fontsize=8)
        ax.set_ylabel(ylabel, color=TC, fontsize=8)
        ax.grid(color=GC, linewidth=0.5)

    # Raw noisy range profile
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(np.abs(noisy_cube[0]), color="#ff6b35", lw=0.7)
    sa(ax0, "Raw Noisy Range Profile (Pulse 0)", "Range Gate", "|Amplitude|")

    # After matched filter
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.plot(np.abs(compressed[0]), color="#00d4ff", lw=0.7)
    sa(ax1, "After Matched Filter (Pulse Compressed)", "Range Gate", "|Amplitude|")

    # After MTI
    mti_out_local = mti_filter(compressed, dsp_cfg.mti)
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.plot(np.abs(mti_out_local[0]), color="#a8ff78", lw=0.7)
    sa(ax2, "After MTI Filter (Clutter Cancelled)", "Range Gate", "|Amplitude|")

    # Range-Doppler map
    rd_db = db(rd_map)
    vmin, vmax = float(np.percentile(rd_db, 5)), float(np.percentile(rd_db, 99.5))
    ax3 = fig.add_subplot(gs[1, 0:2])
    extent = [rng_ax_km[0], rng_ax_km[-1], vel_ax[0], vel_ax[-1]]
    im = ax3.imshow(
        rd_db, aspect="auto", origin="lower",
        cmap="inferno", vmin=vmin, vmax=vmax, extent=extent,
    )
    # Overlay CFAR detections
    for d in detections:
        r_km = d.range_m / 1000.0
        ax3.plot(r_km, d.velocity_mps, "c+", markersize=8, markeredgewidth=1.2)
    sa(ax3, "Range-Doppler Map with CFAR Detections  (+)",
       "Range (km)", "Radial Velocity (m/s)")
    cb = fig.colorbar(im, ax=ax3, pad=0.01)
    cb.set_label("Power (dB)", color=TC, fontsize=8)
    cb.ax.yaxis.set_tick_params(color=TC, labelsize=7)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=TC)

    # Detection SNR histogram
    ax4 = fig.add_subplot(gs[1, 2])
    if detections:
        snrs = [d.snr_db for d in detections]
        ax4.hist(snrs, bins=20, color="#00d4ff", edgecolor="#0a0f1e", linewidth=0.5)
        ax4.axvline(np.median(snrs), color="#ff6b35", lw=1.2,
                    linestyle="--", label=f"Median {np.median(snrs):.1f} dB")
        ax4.legend(fontsize=7, facecolor=BG, labelcolor=TC)
    sa(ax4, "Detection SNR Distribution", "SNR (dB)", "Count")

    out_path = "filters_preview.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"\n  Preview saved → {out_path}")