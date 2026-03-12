from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

import numpy as np



class TrackStatus(Enum):
    TENTATIVE  = auto()   # Just initiated — not yet confirmed
    CONFIRMED  = auto()   # M-of-N threshold passed → valid track
    COASTING   = auto()   # No association last N scans — extrapolating
    DELETED    = auto()   # Track terminated



@dataclass
class Track:

    track_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    status: TrackStatus = TrackStatus.TENTATIVE

    # ── Kalman state ──────────────────────────────────────────────────────
    x: np.ndarray = field(default_factory=lambda: np.zeros(6))   # state vector
    P: np.ndarray = field(default_factory=lambda: np.eye(6) * 1e6)  # covariance

    # ── Track history (for visualisation / ML) ────────────────────────────
    positions: list[tuple[float, float, float]] = field(default_factory=list)
    velocities: list[tuple[float, float, float]] = field(default_factory=list)
    timestamps: list[float] = field(default_factory=list)
    measurements_history: list[dict] = field(default_factory=list)

    # ── M-of-N track management counters ─────────────────────────────────
    hit_count: int = 0       # Number of scans with successful association
    miss_count: int = 0      # Consecutive scans without association
    total_scans: int = 0     # Total scans since initiation

    # ── Metadata (filled by threat classifier later) ──────────────────────
    threat_type: str = "unknown"
    threat_confidence: float = 0.0
    alert_level: str = "none"   # 'none' | 'low' | 'medium' | 'high' | 'critical'

    # ── Particle filter state (optional) ─────────────────────────────────
    particles: Optional[np.ndarray] = None    # shape (N_particles, 6)
    weights:   Optional[np.ndarray] = None    # shape (N_particles,)

    def position_3d(self) -> tuple[float, float, float]:
        """Current best-estimate position (x, y, z) in metres."""
        return (float(self.x[0]), float(self.x[1]), float(self.x[2]))

    def velocity_3d(self) -> tuple[float, float, float]:
        """Current best-estimate velocity (vx, vy, vz) in m/s."""
        return (float(self.x[3]), float(self.x[4]), float(self.x[5]))

    def speed_mps(self) -> float:
        return float(np.linalg.norm(self.x[3:6]))

    def range_m(self) -> float:
        return float(np.linalg.norm(self.x[0:3]))

    def azimuth_deg(self) -> float:
        return math.degrees(math.atan2(self.x[0], self.x[1])) % 360.0

    def elevation_deg(self) -> float:
        horiz = math.sqrt(self.x[0]**2 + self.x[1]**2)
        return math.degrees(math.atan2(self.x[2], horiz))

    def radial_velocity_mps(self) -> float:
        r = self.range_m()
        if r < 1.0:
            return 0.0
        return float(np.dot(self.x[0:3], self.x[3:6]) / r)

    def position_uncertainty_m(self) -> float:
        """1-σ position uncertainty (trace of position block of P)."""
        return float(math.sqrt(self.P[0, 0] + self.P[1, 1] + self.P[2, 2]))

    def is_active(self) -> bool:
        return self.status in (TrackStatus.TENTATIVE,
                               TrackStatus.CONFIRMED,
                               TrackStatus.COASTING)

    def snapshot(self, timestamp: float) -> None:
        """Append current state to history lists."""
        self.positions.append(self.position_3d())
        self.velocities.append(self.velocity_3d())
        self.timestamps.append(timestamp)

    def to_dict(self) -> dict:
        return {
            "track_id":        self.track_id,
            "status":          self.status.name,
            "x_m":             float(self.x[0]),
            "y_m":             float(self.x[1]),
            "z_m":             float(self.x[2]),
            "vx_mps":          float(self.x[3]),
            "vy_mps":          float(self.x[4]),
            "vz_mps":          float(self.x[5]),
            "range_m":         self.range_m(),
            "azimuth_deg":     self.azimuth_deg(),
            "elevation_deg":   self.elevation_deg(),
            "speed_mps":       self.speed_mps(),
            "radial_vel_mps":  self.radial_velocity_mps(),
            "pos_sigma_m":     self.position_uncertainty_m(),
            "hit_count":       self.hit_count,
            "miss_count":      self.miss_count,
            "threat_type":     self.threat_type,
            "alert_level":     self.alert_level,
        }

    def __repr__(self) -> str:
        return (
            f"Track(id={self.track_id}, {self.status.name}, "
            f"R={self.range_m()/1e3:.1f} km, "
            f"v={self.speed_mps():.1f} m/s, "
            f"σ={self.position_uncertainty_m():.0f} m)"
        )


# ──────────────────────────────────────────────────────────────────────────────
# EKF Process & Measurement Models
# ──────────────────────────────────────────────────────────────────────────────

def build_transition_matrix(dt: float) -> np.ndarray:

    F = np.eye(6)
    F[0, 3] = dt   # x  += vx · dt
    F[1, 4] = dt   # y  += vy · dt
    F[2, 5] = dt   # z  += vz · dt
    return F


def build_process_noise(dt: float, sigma_a: float = 10.0) -> np.ndarray:

    q = sigma_a ** 2
    dt2 = dt ** 2
    dt3 = dt ** 3
    dt4 = dt ** 4

    # Continuous white-noise acceleration model discretised to dt
    Q = np.zeros((6, 6))
    for i, j in [(0, 3), (1, 4), (2, 5)]:   # (pos, vel) pairs per axis
        Q[i - 3 if i >= 3 else i,
          i - 3 if i >= 3 else i] += q * dt4 / 4   # pos-pos
    # Build block-diagonal structure
    block = np.array([
        [dt4 / 4, dt3 / 2],
        [dt3 / 2, dt2    ],
    ]) * q
    Q[0:2, 0:2] = block  # x-vx block
    Q[1:3, 1:3] = block  # y-vy (will overwrite Q[1,1] — fix below)
    # Rebuild cleanly axis by axis
    Q = np.zeros((6, 6))
    for pos_idx, vel_idx in [(0, 3), (1, 4), (2, 5)]:
        Q[pos_idx, pos_idx] = q * dt4 / 4
        Q[pos_idx, vel_idx] = q * dt3 / 2
        Q[vel_idx, pos_idx] = q * dt3 / 2
        Q[vel_idx, vel_idx] = q * dt2
    return Q


def measurement_function(x: np.ndarray) -> np.ndarray:

    r = math.sqrt(x[0]**2 + x[1]**2 + x[2]**2)
    r = max(r, 1.0)  # guard
    v_r = (x[0]*x[3] + x[1]*x[4] + x[2]*x[5]) / r
    return np.array([r, v_r])


def measurement_jacobian(x: np.ndarray) -> np.ndarray:

    px, py, pz = x[0], x[1], x[2]
    vx, vy, vz = x[3], x[4], x[5]
    r = math.sqrt(px**2 + py**2 + pz**2)
    r = max(r, 1.0)
    r3 = r ** 3

    # ∂h₁/∂x  (range gradient w.r.t. state)
    dh1 = np.array([px/r, py/r, pz/r, 0.0, 0.0, 0.0])

    # ∂h₂/∂x  (radial velocity gradient w.r.t. state)
    vdotp = px*vx + py*vy + pz*vz
    dh2 = np.array([
        vx/r - px * vdotp / r3,
        vy/r - py * vdotp / r3,
        vz/r - pz * vdotp / r3,
        px/r,
        py/r,
        pz/r,
    ])
    return np.vstack([dh1, dh2])


def build_measurement_noise(
    range_sigma_m: float = 300.0,
    velocity_sigma_mps: float = 3.0,
) -> np.ndarray:

    return np.diag([range_sigma_m**2, velocity_sigma_mps**2])


# ──────────────────────────────────────────────────────────────────────────────
# Extended Kalman Filter — single target, single update step
# ──────────────────────────────────────────────────────────────────────────────

class EKF:

    def __init__(
        self,
        initial_state: np.ndarray,
        initial_P: np.ndarray,
        R: np.ndarray,
        sigma_a: float = 10.0,
    ) -> None:
        self.x = initial_state.copy()
        self.P = initial_P.copy()
        self.R = R
        self.sigma_a = sigma_a

    def predict(self, dt: float) -> tuple[np.ndarray, np.ndarray]:

        F = build_transition_matrix(dt)
        Q = build_process_noise(dt, self.sigma_a)

        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        return self.x.copy(), self.P.copy()

    def update(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

        H   = measurement_jacobian(self.x)
        z_hat = measurement_function(self.x)
        nu  = z - z_hat           # innovation

        S   = H @ self.P @ H.T + self.R    # innovation covariance
        K   = self.P @ H.T @ np.linalg.inv(S)   # Kalman gain

        self.x = self.x + K @ nu

        # Joseph form: P = (I-KH)P(I-KH)ᵀ + KRKᵀ  (always positive definite)
        I_KH = np.eye(6) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T

        return self.x.copy(), self.P.copy()

    def innovation_distance(self, z: np.ndarray) -> float:

        H     = measurement_jacobian(self.x)
        z_hat = measurement_function(self.x)
        nu    = z - z_hat
        S     = H @ self.P @ H.T + self.R
        return float(nu.T @ np.linalg.inv(S) @ nu)


# ──────────────────────────────────────────────────────────────────────────────
# Particle Filter — alternative tracker for high-manoeuvre targets
# ──────────────────────────────────────────────────────────────────────────────

class ParticleFilter:


    def __init__(
        self,
        initial_state: np.ndarray,
        n_particles: int = 500,
        sigma_a: float = 20.0,
        R: Optional[np.ndarray] = None,
    ) -> None:
        self.n = n_particles
        self.sigma_a = sigma_a
        self.R = R if R is not None else build_measurement_noise()

        # Initialise particles around the initial state with some spread
        spread = np.array([500.0, 500.0, 200.0, 15.0, 15.0, 5.0])
        rng = np.random.default_rng()
        self.particles = initial_state + rng.standard_normal((n_particles, 6)) * spread
        self.weights = np.ones(n_particles) / n_particles

    def predict(self, dt: float, rng: Optional[np.random.Generator] = None) -> None:

        if rng is None:
            rng = np.random.default_rng()

        # Constant-velocity propagation
        self.particles[:, 0] += self.particles[:, 3] * dt
        self.particles[:, 1] += self.particles[:, 4] * dt
        self.particles[:, 2] += self.particles[:, 5] * dt

        # Process noise (acceleration perturbation)
        noise_std = self.sigma_a * dt
        self.particles[:, 3] += rng.normal(0, noise_std, self.n)
        self.particles[:, 4] += rng.normal(0, noise_std, self.n)
        self.particles[:, 5] += rng.normal(0, noise_std * 0.3, self.n)  # less vertical

    def update(self, z: np.ndarray) -> None:

        log_weights = np.zeros(self.n)
        R_diag = np.diag(self.R)

        for i, p in enumerate(self.particles):
            z_hat = measurement_function(p)
            residual = z - z_hat
            # Log-likelihood (diagonal R approximation for speed)
            log_weights[i] = -0.5 * np.sum(residual**2 / R_diag)

        # Stable weight normalisation in log-space
        log_weights -= np.max(log_weights)
        self.weights = np.exp(log_weights)
        self.weights /= self.weights.sum() + 1e-300

    def resample(self, rng: Optional[np.random.Generator] = None) -> None:
        """
        Systematic resampling — O(N) and low variance compared to
        multinomial resampling.
        """
        if rng is None:
            rng = np.random.default_rng()

        cumsum = np.cumsum(self.weights)
        u0 = rng.uniform(0, 1.0 / self.n)
        positions = u0 + np.arange(self.n) / self.n

        indices = np.searchsorted(cumsum, positions)
        self.particles = self.particles[indices]
        self.weights = np.ones(self.n) / self.n

    def effective_sample_size(self) -> float:
        """N_eff = 1 / Σ wᵢ².  Resample when N_eff < N/2."""
        return float(1.0 / (np.sum(self.weights ** 2) + 1e-300))

    def state_estimate(self) -> np.ndarray:
        """Weighted mean of particles — best state estimate."""
        return np.average(self.particles, weights=self.weights, axis=0)

    def covariance_estimate(self) -> np.ndarray:
        """Weighted covariance of particles."""
        mean = self.state_estimate()
        diff = self.particles - mean
        return np.einsum("i,ij,ik->jk", self.weights, diff, diff)



@dataclass
class AssociationConfig:

    gate_threshold: float = 12.0
    max_unassociated_range_m: float = 450_000.0


def associate_measurements(
    tracks: list[Track],
    measurements: list[dict],
    config: AssociationConfig,
) -> tuple[dict[str, dict], list[dict]]:

    unmatched = list(measurements)
    associations: dict[str, dict] = {}

    # Score matrix: rows = tracks, cols = measurements
    active_tracks = [t for t in tracks if t.is_active()]

    if not active_tracks or not measurements:
        return associations, unmatched

    score_matrix = np.full((len(active_tracks), len(measurements)), np.inf)

    for ti, track in enumerate(active_tracks):
        ekf = track._ekf  # type: ignore[attr-defined]
        for mi, meas in enumerate(measurements):
            z = np.array([meas["range_m"], meas["velocity_mps"]])
            d2 = ekf.innovation_distance(z)
            if d2 < config.gate_threshold:
                score_matrix[ti, mi] = d2

    used_measurements: set[int] = set()

    # Greedy: assign lowest-cost track-measurement pairs first
    while True:
        min_val = np.min(score_matrix)
        if np.isinf(min_val):
            break
        ti, mi = np.unravel_index(np.argmin(score_matrix), score_matrix.shape)
        track = active_tracks[ti]
        associations[track.track_id] = measurements[mi]
        used_measurements.add(mi)

        # Prevent re-use of this row (track) and column (measurement)
        score_matrix[ti, :] = np.inf
        score_matrix[:, mi] = np.inf

    unmatched = [m for i, m in enumerate(measurements)
                 if i not in used_measurements]
    return associations, unmatched




@dataclass
class TrackManagerConfig:

    confirm_m_hits: int = 3
    confirm_n_scans: int = 5
    max_coast_scans: int = 4
    use_particle_filter: bool = False
    sigma_a: float = 15.0
    range_sigma_m: float = 300.0
    vel_sigma_mps: float = 3.0
    dt_sec: float = 1.0
    association: AssociationConfig = field(default_factory=AssociationConfig)
    n_particles: int = 300   # PF particle count (if use_particle_filter=True)


class TrackManager:

    def __init__(self, config: TrackManagerConfig) -> None:
        self.config = config
        self._tracks: dict[str, Track] = {}
        self._scan_count: int = 0
        self._rng = np.random.default_rng()
        self._R = build_measurement_noise(
            config.range_sigma_m, config.vel_sigma_mps
        )

    # ── Public entry point ───────────────────────────────────────────────────

    def update(
        self,
        measurements: list[dict],
        timestamp: float,
    ) -> list[Track]:

        self._scan_count += 1
        dt = self.config.dt_sec

        # ── Step 1: Predict all active tracks forward ────────────────────
        for track in self._tracks.values():
            if not track.is_active():
                continue
            if self.config.use_particle_filter and track.particles is not None:
                track._pf.predict(dt, self._rng)  # type: ignore[attr-defined]
                xs = track._pf.state_estimate()   # type: ignore[attr-defined]
                track.x = xs
                track.P = track._pf.covariance_estimate()  # type: ignore
            else:
                track._ekf.predict(dt)            # type: ignore[attr-defined]
                track.x = track._ekf.x
                track.P = track._ekf.P

        # ── Step 2: Data association ─────────────────────────────────────
        active_tracks = [t for t in self._tracks.values() if t.is_active()]
        associations, unmatched = associate_measurements(
            active_tracks, measurements, self.config.association
        )

        # ── Step 3: Update matched tracks ────────────────────────────────
        for track_id, meas in associations.items():
            track = self._tracks[track_id]
            z = np.array([meas["range_m"], meas["velocity_mps"]])

            if self.config.use_particle_filter and track.particles is not None:
                track._pf.update(z)              # type: ignore[attr-defined]
                if track._pf.effective_sample_size() < self.config.n_particles / 2:  # type: ignore
                    track._pf.resample(self._rng) # type: ignore[attr-defined]
                track.x = track._pf.state_estimate()  # type: ignore[attr-defined]
                track.P = track._pf.covariance_estimate()  # type: ignore[attr-defined]
            else:
                track._ekf.update(z)             # type: ignore[attr-defined]
                track.x = track._ekf.x
                track.P = track._ekf.P

            track.hit_count   += 1
            track.miss_count   = 0
            track.total_scans += 1
            track.measurements_history.append({**meas, "timestamp": timestamp})
            track.snapshot(timestamp)

        # ── Step 4: Coast unmatched tracks ───────────────────────────────
        for track in active_tracks:
            if track.track_id not in associations:
                track.miss_count  += 1
                track.total_scans += 1
                track.snapshot(timestamp)

                if (track.status == TrackStatus.CONFIRMED
                        and track.miss_count <= self.config.max_coast_scans):
                    track.status = TrackStatus.COASTING
                elif track.miss_count > self.config.max_coast_scans:
                    track.status = TrackStatus.DELETED

        # ── Step 5: Initiate new tentative tracks ─────────────────────────
        for meas in unmatched:
            if meas["range_m"] > self.config.association.max_unassociated_range_m:
                continue
            new_track = self._init_track(meas, timestamp)
            self._tracks[new_track.track_id] = new_track

        # ── Step 6: M-of-N confirmation logic ────────────────────────────
        for track in self._tracks.values():
            if track.status == TrackStatus.TENTATIVE:
                if (track.hit_count >= self.config.confirm_m_hits
                        and track.total_scans <= self.config.confirm_n_scans):
                    track.status = TrackStatus.CONFIRMED
                elif track.total_scans > self.config.confirm_n_scans:
                    track.status = TrackStatus.DELETED

            elif track.status == TrackStatus.COASTING:
                if track.miss_count == 0:
                    track.status = TrackStatus.CONFIRMED   # re-confirm

        return self.all_tracks

    # ── Private helpers ──────────────────────────────────────────────────────

    def _init_track(self, meas: dict, timestamp: float) -> Track:
       
        az_rad = math.radians(meas.get("azimuth_deg", 0.0))
        el_rad = math.radians(meas.get("elevation_deg", 2.0))
        r      = meas["range_m"]
        v_r    = meas["velocity_mps"]

        # Convert spherical → Cartesian
        cos_el = math.cos(el_rad)
        px = r * cos_el * math.sin(az_rad)
        py = r * cos_el * math.cos(az_rad)
        pz = r * math.sin(el_rad)

        # Velocity: radial component along unit range vector
        ux = px / r;  uy = py / r;  uz = pz / r
        vx = v_r * ux;  vy = v_r * uy;  vz = v_r * uz

        x0 = np.array([px, py, pz, vx, vy, vz])

        # Large initial covariance (high uncertainty at track birth)
        P0 = np.diag([
            (0.1 * r) ** 2,    # x position uncertainty
            (0.1 * r) ** 2,    # y position uncertainty
            (0.05 * r) ** 2,   # z (elevation less certain)
            (50.0) ** 2,       # vx uncertainty
            (50.0) ** 2,       # vy uncertainty
            (20.0) ** 2,       # vz uncertainty
        ])

        track = Track(x=x0, P=P0)

        # Attach EKF instance to track (stored as private attr)
        ekf = EKF(x0, P0, self._R, sigma_a=self.config.sigma_a)
        object.__setattr__(track, "_ekf", ekf)

        if self.config.use_particle_filter:
            pf = ParticleFilter(x0, self.config.n_particles,
                                self.config.sigma_a, self._R)
            object.__setattr__(track, "_pf", pf)
            track.particles = pf.particles
            track.weights   = pf.weights

        track.hit_count   = 1
        track.total_scans = 1
        track.snapshot(timestamp)
        track.measurements_history.append({**meas, "timestamp": timestamp})
        return track

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def all_tracks(self) -> list[Track]:
        return [t for t in self._tracks.values() if t.is_active()]

    @property
    def confirmed_tracks(self) -> list[Track]:
        return [t for t in self._tracks.values()
                if t.status == TrackStatus.CONFIRMED]

    @property
    def track_count(self) -> dict[str, int]:
        counts: dict[str, int] = {s.name: 0 for s in TrackStatus}
        for t in self._tracks.values():
            counts[t.status.name] += 1
        return counts

    def summary(self) -> str:
        counts = self.track_count
        return (
            f"Scan #{self._scan_count:04d} | "
            f"Active: {len(self.all_tracks)} | "
            f"Confirmed: {counts['CONFIRMED']} | "
            f"Tentative: {counts['TENTATIVE']} | "
            f"Coasting: {counts['COASTING']}"
        )




def detections_to_measurements(detections) -> list[dict]:
    """
    Convert a list of filters.Detection objects to the dict format expected
    by TrackManager.update().
    """
    return [
        {
            "range_m":       d.range_m,
            "velocity_mps":  d.velocity_mps,
            "azimuth_deg":   getattr(d, "azimuth_deg", 0.0),
            "elevation_deg": getattr(d, "elevation_deg", 2.0),
            "snr_db":        d.snr_db,
        }
        for d in detections
    ]




if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    from radar_simulation.generate_signals import (
        RadarParameters, generate_cpi, generate_lfm_pulse,
    )
    from radar_simulation.noise_model import (
        NoisePipelineConfig, ThermalNoiseConfig, ClutterConfig,
        apply_noise_pipeline,
    )
    from radar_simulation.filters import (
        run_dsp_pipeline, DSPPipelineConfig,
    )

    print("=" * 62)
    print("  SkySentinel — Multi-Target Kalman Tracker Self-Test")
    print("=" * 62)

    # ── Simulation parameters ────────────────────────────────────────────
    N_SCANS   = 25          # number of radar scans to simulate
    DT_SEC    = 1.0         # scan interval

    radar = RadarParameters(n_pulses_per_cpi=64, max_range_m=400_000,
                            prf_hz=2000)
    rng_g = np.random.default_rng(seed=42)

    tracker = TrackManager(TrackManagerConfig(
        sigma_a=12.0,
        dt_sec=DT_SEC,
        confirm_m_hits=3,
        confirm_n_scans=5,
        max_coast_scans=3,
    ))

    # ── Run N_SCANS iterations ───────────────────────────────────────────
    track_paths: dict[str, list[tuple[float, float]]] = {}

    for scan_idx in range(N_SCANS):
        timestamp = scan_idx * DT_SEC

        # Fresh data cube per scan (targets move between scans)
        cube, meta, _ = generate_cpi(
            scenario="mixed_threat", radar=radar,
            seed=scan_idx, verbose=False
        )
        ref_pulse = generate_lfm_pulse(radar, rng_g)

        noise_cfg = NoisePipelineConfig(
            enable_thermal=True,
            thermal=ThermalNoiseConfig(noise_power_dbw=-108.0),
            enable_clutter=True,
            clutter=ClutterConfig(clutter_to_noise_ratio_db=28.0,
                                  max_range_m=70_000.0),
        )
        noisy_cube, _ = apply_noise_pipeline(
            cube, noise_cfg, rng=rng_g,
            range_gate_length_m=radar.range_gate_length_m,
            reference_pulse=ref_pulse,
            noise_power=radar.thermal_noise_power(),
        )

        dsp_cfg = DSPPipelineConfig(enable_mti=True, enable_cfar=True)
        _, _, detections = run_dsp_pipeline(
            noisy_cube, ref_pulse,
            prf_hz=radar.prf_hz,
            range_gate_length_m=radar.range_gate_length_m,
            wavelength_m=radar.wavelength_m,
            config=dsp_cfg,
        )

        meas = detections_to_measurements(detections)
        active = tracker.update(meas, timestamp)
        print(f"  {tracker.summary()}")

        for t in tracker.confirmed_tracks:
            tid = t.track_id
            if tid not in track_paths:
                track_paths[tid] = []
            track_paths[tid].append((t.x[0] / 1e3, t.x[1] / 1e3))  # km

    print()
    print(f"  Final confirmed tracks : {len(tracker.confirmed_tracks)}")
    for t in tracker.confirmed_tracks:
        print(f"    {t}")
    print("=" * 62)

    # ── Visualisation — 2-D track plot ───────────────────────────────────
    BG = "#0a0f1e";  TC = "white";  GC = "#1e2a3a"

    fig, ax = plt.subplots(figsize=(10, 10), facecolor=BG)
    ax.set_facecolor(BG)
    ax.tick_params(colors=TC)
    for sp in ax.spines.values():
        sp.set_color(GC)
    ax.set_title("SkySentinel — Multi-Target Kalman Tracker",
                 color=TC, fontsize=13, fontweight="bold")
    ax.set_xlabel("East (km)", color=TC)
    ax.set_ylabel("North (km)", color=TC)
    ax.grid(color=GC, linewidth=0.6, linestyle="--")

    # Radar site
    ax.plot(0, 0, "w*", markersize=14, zorder=10, label="Radar Site")

    colours = ["#00d4ff", "#ff6b35", "#a8ff78", "#f7c59f",
               "#e84393", "#b8d8d8", "#ffd700", "#c77dff"]

    for i, (tid, path) in enumerate(track_paths.items()):
        if len(path) < 2:
            continue
        xs = [p[0] for p in path]
        ys = [p[1] for p in path]
        col = colours[i % len(colours)]
        ax.plot(xs, ys, "-o", color=col, markersize=3, linewidth=1.5,
                label=f"Track {tid}")
        ax.plot(xs[-1], ys[-1], "^", color=col, markersize=9)  # current pos

    # Detection radius rings
    for r_km in [100, 200, 300, 400]:
        theta = np.linspace(0, 2 * np.pi, 200)
        ax.plot(r_km * np.cos(theta), r_km * np.sin(theta),
                color=GC, linewidth=0.7, linestyle=":")

    ax.legend(loc="upper right", fontsize=8, facecolor=BG, labelcolor=TC,
              framealpha=0.7)
    ax.set_xlim(-420, 420);  ax.set_ylim(-420, 420)

    out = "tracker_preview.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"\n  Preview saved → {out}")