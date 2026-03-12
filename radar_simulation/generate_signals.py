
from __future__ import annotations

import math
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy.signal import chirp



SPEED_OF_LIGHT: float = 3.0e8          # metres per second
BOLTZMANN: float = 1.38e-23            # J / K  (thermal noise floor)
REFERENCE_TEMP_K: float = 290.0        # Standard reference temperature (K)



@dataclass
class RadarParameters:


    # --- Transmitter ---
    peak_power_watts: float = 250_000.0      # Tx peak power  (250 kW)
    carrier_frequency_hz: float = 3.0e9     # S-band carrier (~3 GHz)
    pulse_width_sec: float = 5.0e-6          # Uncompressed pulse width  (5 µs)
    bandwidth_hz: float = 2.0e6             # Waveform bandwidth → range res.
    prf_hz: float = 1000.0                  # Pulse Repetition Frequency (1 kHz)
    n_pulses_per_cpi: int = 64              # Pulses per Coherent Processing Interval

    # --- Antenna ---
    antenna_gain_db: float = 34.0           # Main-lobe gain (dB)
    beamwidth_az_deg: float = 1.5           # Azimuth 3-dB beamwidth
    beamwidth_el_deg: float = 5.0           # Elevation 3-dB beamwidth

    # --- Receiver ---
    receiver_noise_figure_db: float = 4.0   # Noise figure (dB)
    losses_db: float = 3.0                  # System losses (feed, propagation)
    sample_rate_hz: float = 5.0e6           # ADC sample rate

    # --- Detection geometry ---
    max_range_m: float = 500_000.0          # Instrumented range  (500 km)
    range_gate_length_m: float = 75.0       # Range cell size (≈ c / 2B)

    # ── Derived quantities (computed post-init) ──
    wavelength_m: float = field(init=False)
    pri_sec: float = field(init=False)
    range_resolution_m: float = field(init=False)
    velocity_resolution_mps: float = field(init=False)
    unambiguous_range_m: float = field(init=False)
    unambiguous_velocity_mps: float = field(init=False)

    def __post_init__(self) -> None:
        self.wavelength_m = SPEED_OF_LIGHT / self.carrier_frequency_hz
        self.pri_sec = 1.0 / self.prf_hz
        self.range_resolution_m = SPEED_OF_LIGHT / (2.0 * self.bandwidth_hz)
        self.velocity_resolution_mps = (
            self.wavelength_m * self.prf_hz / (2.0 * self.n_pulses_per_cpi)
        )
        self.unambiguous_range_m = SPEED_OF_LIGHT * self.pri_sec / 2.0
        self.unambiguous_velocity_mps = self.wavelength_m * self.prf_hz / 4.0

    @property
    def antenna_gain_linear(self) -> float:
        return 10.0 ** (self.antenna_gain_db / 10.0)

    @property
    def noise_figure_linear(self) -> float:
        return 10.0 ** (self.receiver_noise_figure_db / 10.0)

    @property
    def losses_linear(self) -> float:
        return 10.0 ** (self.losses_db / 10.0)

    def thermal_noise_power(self) -> float:
        """Thermal noise power in the receiver bandwidth [W]."""
        return (
            BOLTZMANN
            * REFERENCE_TEMP_K
            * self.bandwidth_hz
            * self.noise_figure_linear
        )

    def __repr__(self) -> str:
        return (
            f"RadarParameters("
            f"f0={self.carrier_frequency_hz/1e9:.2f} GHz, "
            f"PRF={self.prf_hz:.0f} Hz, "
            f"λ={self.wavelength_m*100:.1f} cm, "
            f"R_max={self.max_range_m/1e3:.0f} km)"
        )




@dataclass
class TargetState:


    target_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    threat_type: str = "unknown"     # 'missile' | 'drone' | 'aircraft' | 'unknown'

    # Cartesian position  [m]
    x: float = 0.0
    y: float = 100_000.0
    z: float = 10_000.0

    # Cartesian velocity  [m/s]
    vx: float = -200.0
    vy: float = 0.0
    vz: float = 0.0

    # Signature
    rcs_m2: float = 1.0              # Radar cross-section  [m²]
    rcs_fluctuation_model: str = "swerling1"   # 'constant' | 'swerling1' | 'swerling2'

    def range_m(self) -> float:
        """Radial distance from the radar origin [m]."""
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def azimuth_deg(self) -> float:
        """Azimuth angle measured clockwise from North [deg]."""
        return math.degrees(math.atan2(self.x, self.y)) % 360.0

    def elevation_deg(self) -> float:
        """Elevation angle above the horizontal plane [deg]."""
        return math.degrees(math.atan2(self.z, math.sqrt(self.x**2 + self.y**2)))

    def radial_velocity_mps(self) -> float:
        """
        Radial (range-rate) velocity — positive = receding, negative = closing.
        """
        r = self.range_m()
        if r < 1.0:
            return 0.0
        return (self.x * self.vx + self.y * self.vy + self.z * self.vz) / r

    def speed_mps(self) -> float:
        return math.sqrt(self.vx**2 + self.vy**2 + self.vz**2)

    def update_position(self, dt_sec: float) -> None:
        """Advance kinematic state by one time step (simple linear motion)."""
        self.x += self.vx * dt_sec
        self.y += self.vy * dt_sec
        self.z += self.vz * dt_sec

    def to_dict(self) -> dict:
        return {
            "target_id": self.target_id,
            "threat_type": self.threat_type,
            "x_m": self.x,
            "y_m": self.y,
            "z_m": self.z,
            "vx_mps": self.vx,
            "vy_mps": self.vy,
            "vz_mps": self.vz,
            "range_m": self.range_m(),
            "azimuth_deg": self.azimuth_deg(),
            "elevation_deg": self.elevation_deg(),
            "radial_vel_mps": self.radial_velocity_mps(),
            "speed_mps": self.speed_mps(),
            "rcs_m2": self.rcs_m2,
        }

    def __repr__(self) -> str:
        return (
            f"Target(id={self.target_id}, type={self.threat_type}, "
            f"R={self.range_m()/1e3:.1f} km, "
            f"Az={self.azimuth_deg():.1f}°, "
            f"v_r={self.radial_velocity_mps():.1f} m/s)"
        )




# Realistic RCS values (m²) for common airborne threat types
RCS_PRESETS: dict[str, float] = {
    "ballistic_missile": 0.05,   # stealthy re-entry body
    "cruise_missile":    0.10,
    "drone_small":       0.01,   # micro-UAV
    "drone_medium":      0.10,
    "fighter_jet":       5.00,
    "commercial_aircraft": 30.0,
    "helicopter":        3.00,
}

# Speed ranges [m/s] for scenario generation
SPEED_RANGES: dict[str, tuple[float, float]] = {
    "ballistic_missile": (1500.0, 4000.0),
    "cruise_missile":    (200.0, 350.0),
    "drone_small":       (10.0,  40.0),
    "drone_medium":      (40.0,  120.0),
    "fighter_jet":       (200.0, 600.0),
    "commercial_aircraft": (220.0, 280.0),
    "helicopter":        (30.0,  80.0),
}

# Altitude bands [m AGL]
ALTITUDE_RANGES: dict[str, tuple[float, float]] = {
    "ballistic_missile": (20_000.0, 120_000.0),
    "cruise_missile":    (50.0,    500.0),
    "drone_small":       (20.0,    300.0),
    "drone_medium":      (100.0,   2_000.0),
    "fighter_jet":       (1_000.0, 15_000.0),
    "commercial_aircraft": (8_000.0, 12_000.0),
    "helicopter":        (50.0,   3_000.0),
}


# ──────────────────────────────────────────────────────────────────────────────
# RCS Fluctuation Models (Swerling Cases)
# ──────────────────────────────────────────────────────────────────────────────

def sample_rcs(mean_rcs: float, model: str, rng: np.random.Generator) -> float:

    if model == "constant":
        return mean_rcs
    elif model in ("swerling1", "swerling2"):
        # Exponential distribution ≡ Chi-squared with 2 DoF
        return float(rng.exponential(scale=mean_rcs))
    elif model in ("swerling3", "swerling4"):
        # Chi-squared with 4 DoF → Erlang(k=2)
        return float(rng.gamma(shape=2.0, scale=mean_rcs / 2.0))
    else:
        return mean_rcs




def compute_received_power(
    radar: RadarParameters,
    range_m: float,
    rcs_m2: float,
) -> float:

    if range_m < 1.0:
        range_m = 1.0   # guard against division by zero at very close range

    numerator = (
        radar.peak_power_watts
        * radar.antenna_gain_linear ** 2
        * radar.wavelength_m ** 2
        * rcs_m2
    )
    denominator = (
        (4.0 * math.pi) ** 3
        * range_m ** 4
        * radar.losses_linear
    )
    return numerator / denominator




def generate_lfm_pulse(
    radar: RadarParameters,
    rng: np.random.Generator,
) -> np.ndarray:

    n_samples = int(radar.pulse_width_sec * radar.sample_rate_hz)
    t = np.linspace(0.0, radar.pulse_width_sec, n_samples, endpoint=False)

    f_start = -radar.bandwidth_hz / 2.0
    f_stop  =  radar.bandwidth_hz / 2.0

    # Real chirp waveform
    real_part = chirp(t, f0=f_start, f1=f_stop, t1=radar.pulse_width_sec,
                      method="linear", phi=0)

    # Analytic (I/Q) representation via Hilbert transform approximation
    # For a chirp, the imaginary part is a 90° phase-shifted version
    imag_part = chirp(t, f0=f_start, f1=f_stop, t1=radar.pulse_width_sec,
                      method="linear", phi=-90)

    pulse = (real_part + 1j * imag_part).astype(np.complex128)

    # Normalise to unit energy
    pulse /= np.sqrt(np.mean(np.abs(pulse) ** 2))
    return pulse



def generate_echo_pulse(
    radar: RadarParameters,
    target: TargetState,
    reference_pulse: np.ndarray,
    rng: np.random.Generator,
    pulse_index: int = 0,
) -> tuple[np.ndarray, dict]:

    # ── Geometry ────────────────────────────────────────────────────────────
    range_m = target.range_m()
    radial_vel = target.radial_velocity_mps()

    # ── RCS sample ──────────────────────────────────────────────────────────
    # Swerling1: RCS constant within CPI, new draw each call = each CPI
    inst_rcs = sample_rcs(target.rcs_m2, target.rcs_fluctuation_model, rng)

    # ── Signal power ────────────────────────────────────────────────────────
    p_rx = compute_received_power(radar, range_m, inst_rcs)
    p_noise = radar.thermal_noise_power()
    snr_linear = p_rx / p_noise
    snr_db = 10.0 * math.log10(max(snr_linear, 1e-30))

    # ── Range gate index ────────────────────────────────────────────────────
    two_way_delay_sec = 2.0 * range_m / SPEED_OF_LIGHT
    range_gate_idx = int(two_way_delay_sec * radar.sample_rate_hz)

    # ── Doppler phase ────────────────────────────────────────────────────────
    # Doppler frequency shift [Hz]
    f_doppler = -2.0 * radial_vel / radar.wavelength_m  # negative = closing target
    # Accumulated phase across pulses (slow-time modulation)
    doppler_phase = 2.0 * math.pi * f_doppler * (pulse_index * radar.pri_sec)

    # ── Build range-gate buffer ──────────────────────────────────────────────
    pulse_len = len(reference_pulse)
    n_range_gates = int(radar.max_range_m / radar.range_gate_length_m)
    echo_buffer = np.zeros(n_range_gates, dtype=np.complex128)

    # Clip to valid buffer window
    end_gate = range_gate_idx + pulse_len
    if range_gate_idx >= n_range_gates or range_gate_idx < 0:
        # Target outside instrumented range
        metadata = {
            "target_id": target.target_id,
            "range_m": range_m,
            "rcs_m2": inst_rcs,
            "snr_db": -np.inf,
            "f_doppler_hz": f_doppler,
            "in_range": False,
        }
        return echo_buffer, metadata

    # Window pulse to fit buffer
    available = min(pulse_len, n_range_gates - range_gate_idx)
    scaled_pulse = (
        math.sqrt(snr_linear)
        * reference_pulse[:available]
        * np.exp(1j * doppler_phase)
    )
    echo_buffer[range_gate_idx: range_gate_idx + available] += scaled_pulse

    metadata = {
        "target_id": target.target_id,
        "range_m": range_m,
        "rcs_m2": inst_rcs,
        "snr_db": snr_db,
        "f_doppler_hz": f_doppler,
        "range_gate_idx": range_gate_idx,
        "in_range": True,
    }
    return echo_buffer, metadata



def build_radar_data_cube(
    radar: RadarParameters,
    targets: list[TargetState],
    rng: np.random.Generator,
    dt_between_pulses_sec: Optional[float] = None,
) -> tuple[np.ndarray, pd.DataFrame]:

    dt = dt_between_pulses_sec if dt_between_pulses_sec else radar.pri_sec
    n_range_gates = int(radar.max_range_m / radar.range_gate_length_m)

    data_cube = np.zeros(
        (radar.n_pulses_per_cpi, n_range_gates), dtype=np.complex128
    )

    reference_pulse = generate_lfm_pulse(radar, rng)
    rows: list[dict] = []

    for pulse_idx in range(radar.n_pulses_per_cpi):
        for target in targets:
            echo, meta = generate_echo_pulse(
                radar, target, reference_pulse, rng, pulse_index=pulse_idx
            )
            meta["pulse_index"] = pulse_idx
            rows.append(meta)
            data_cube[pulse_idx] += echo

        # Advance all target positions (linear motion model)
        for target in targets:
            target.update_position(dt)

    echo_metadata = pd.DataFrame(rows)
    return data_cube, echo_metadata




def create_scenario_targets(
    scenario: str = "mixed_threat",
    rng: Optional[np.random.Generator] = None,
) -> list[TargetState]:

    if rng is None:
        rng = np.random.default_rng(seed=42)

    def make_target(
        threat_type: str,
        range_km: float,
        bearing_deg: float,
        elevation_deg: float = 5.0,
        seed_offset: int = 0,
    ) -> TargetState:
        """Helper: place a target at given polar coordinates."""
        r = range_km * 1e3
        az_rad = math.radians(bearing_deg)
        el_rad = math.radians(elevation_deg)

        # Convert spherical → Cartesian (North-East-Up)
        x = r * math.cos(el_rad) * math.sin(az_rad)   # East
        y = r * math.cos(el_rad) * math.cos(az_rad)   # North
        z = r * math.sin(el_rad)                       # Up

        # Velocity — heading towards radar
        spd_lo, spd_hi = SPEED_RANGES.get(threat_type, (100.0, 300.0))
        speed = rng.uniform(spd_lo, spd_hi)
        # Unit vector pointing from target towards origin
        vx = -x / r * speed
        vy = -y / r * speed
        vz = 0.0  # level flight simplification (can be changed)

        alt_lo, alt_hi = ALTITUDE_RANGES.get(threat_type, (1000.0, 10000.0))
        altitude = rng.uniform(alt_lo, alt_hi)

        return TargetState(
            threat_type=threat_type,
            x=x, y=y, z=altitude,
            vx=vx, vy=vy, vz=vz,
            rcs_m2=RCS_PRESETS.get(threat_type, 1.0),
            rcs_fluctuation_model="swerling1",
        )

    if scenario == "single_missile":
        return [make_target("ballistic_missile", range_km=300, bearing_deg=45)]

    elif scenario == "drone_swarm":
        bearings = [10, 35, 60, 110, 160]
        return [
            make_target("drone_small", range_km=rng.uniform(30, 80),
                        bearing_deg=b, elevation_deg=1.5)
            for b in bearings
        ]

    elif scenario == "mixed_threat":
        return [
            make_target("cruise_missile",   range_km=250, bearing_deg=30,  elevation_deg=1.0),
            make_target("drone_medium",     range_km=80,  bearing_deg=120, elevation_deg=2.0),
            make_target("fighter_jet",      range_km=180, bearing_deg=270, elevation_deg=8.0),
            make_target("commercial_aircraft", range_km=350, bearing_deg=190, elevation_deg=12.0),
        ]

    elif scenario == "saturation_attack":
        configs = [
            ("cruise_missile",  180, 10),
            ("cruise_missile",  210, 30),
            ("ballistic_missile", 320, 60),
            ("drone_medium",     90,  90),
            ("drone_small",      60, 130),
            ("drone_small",      75, 150),
            ("fighter_jet",     220, 200),
            ("cruise_missile",  195, 300),
        ]
        return [
            make_target(tt, range_km=r, bearing_deg=b)
            for i, (tt, r, b) in enumerate(configs)
        ]

    else:
        raise ValueError(
            f"Unknown scenario '{scenario}'. "
            "Choose: single_missile | drone_swarm | mixed_threat | saturation_attack"
        )




def generate_cpi(
    scenario: str = "mixed_threat",
    radar: Optional[RadarParameters] = None,
    seed: int = 0,
    verbose: bool = True,
) -> tuple[np.ndarray, pd.DataFrame, list[TargetState]]:
    """
    High-level entry point: create targets → build data cube → return results.

    Parameters
    ----------
    scenario : str          – Scenario name (see create_scenario_targets)
    radar    : RadarParameters | None  – Uses default params if None
    seed     : int          – Random seed for full reproducibility
    verbose  : bool         – Print summary to stdout

    Returns
    -------
    data_cube     : np.ndarray  (n_pulses × n_range_gates), complex128
    echo_metadata : pd.DataFrame
    targets       : list[TargetState]  (positions at END of CPI)
    """
    if radar is None:
        radar = RadarParameters()

    rng = np.random.default_rng(seed=seed)
    targets = create_scenario_targets(scenario, rng=rng)

    t0 = time.perf_counter()
    data_cube, meta = build_radar_data_cube(radar, targets, rng)
    elapsed = time.perf_counter() - t0

    if verbose:
        print("=" * 60)
        print("  SkySentinel — Radar Signal Generator")
        print("=" * 60)
        print(f"  Scenario        : {scenario}")
        print(f"  Radar           : {radar}")
        print(f"  Targets         : {len(targets)}")
        print(f"  Data cube shape : {data_cube.shape}  (pulses × range gates)")
        print(f"  Cube dtype      : {data_cube.dtype}")
        print(f"  Generation time : {elapsed*1000:.1f} ms")
        print()
        # Per-target SNR summary (averaged over pulses)
        for tid in meta["target_id"].unique():
            sub = meta[(meta["target_id"] == tid) & (meta["in_range"])]
            if len(sub):
                mean_snr = sub["snr_db"].mean()
                t_type = sub.iloc[0].get("threat_type", "?")
                rng_km  = sub.iloc[0]["range_m"] / 1e3
                print(
                    f"  [{tid}] {t_type:20s}  "
                    f"R={rng_km:6.1f} km  "
                    f"SNR={mean_snr:+6.1f} dB"
                )
        print("=" * 60)

    return data_cube, meta, targets



if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # ── 1.  Build radar data cube for the mixed-threat scenario ─────────────
    radar_cfg = RadarParameters(
        peak_power_watts=500_000,
        carrier_frequency_hz=3.0e9,
        prf_hz=2000,
        n_pulses_per_cpi=32,
        max_range_m=400_000,
    )

    cube, metadata, final_targets = generate_cpi(
        scenario="mixed_threat",
        radar=radar_cfg,
        seed=42,
        verbose=True,
    )

    # ── 2.  Quick visualisation — Range-Doppler map (abs magnitude) ─────────
    range_doppler = np.abs(np.fft.fftshift(np.fft.fft(cube, axis=0), axes=0))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("SkySentinel — Radar Data Cube (Mixed Threat Scenario)",
                 fontsize=13, fontweight="bold")

    # Fast-time (range) profile of pulse 0
    axes[0].plot(np.abs(cube[0]), color="#00d4ff", linewidth=0.8)
    axes[0].set_title("Fast-Time Range Profile — Pulse 0")
    axes[0].set_xlabel("Range Gate Index")
    axes[0].set_ylabel("|Echo Amplitude|")
    axes[0].set_facecolor("#0a0f1e")
    axes[0].tick_params(colors="white")
    axes[0].yaxis.label.set_color("white")
    axes[0].xaxis.label.set_color("white")
    axes[0].title.set_color("white")
    fig.patch.set_facecolor("#0a0f1e")
    axes[0].spines["bottom"].set_color("#444")
    axes[0].spines["left"].set_color("#444")

    # Range-Doppler map (log scale)
    rd_log = 20 * np.log10(range_doppler + 1e-12)
    img = axes[1].imshow(
        rd_log,
        aspect="auto",
        origin="lower",
        cmap="inferno",
        extent=[0, cube.shape[1], -radar_cfg.prf_hz / 2, radar_cfg.prf_hz / 2],
    )
    axes[1].set_title("Range-Doppler Map (dB)")
    axes[1].set_xlabel("Range Gate")
    axes[1].set_ylabel("Doppler Frequency (Hz)")
    axes[1].title.set_color("white")
    axes[1].xaxis.label.set_color("white")
    axes[1].yaxis.label.set_color("white")
    axes[1].tick_params(colors="white")
    axes[1].set_facecolor("#0a0f1e")
    axes[1].spines["bottom"].set_color("#444")
    axes[1].spines["left"].set_color("#444")
    cbar = fig.colorbar(img, ax=axes[1], pad=0.02)
    cbar.set_label("Power (dB)", color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    plt.tight_layout()
    plt.savefig("radar_data_cube_preview.png", dpi=150, bbox_inches="tight")
    print("\n  Preview saved → radar_data_cube_preview.png")
    plt.show()