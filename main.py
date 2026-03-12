from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np

# ── Project root ─────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Radar simulation ─────────────────────────────────────────────────
from radar_simulation.generate_signals import (
    RadarParameters,
    generate_cpi,
    generate_lfm_pulse,
)

from radar_simulation.noise_model import (
    NoisePipelineConfig,
    ThermalNoiseConfig,
    ClutterConfig,
    apply_noise_pipeline,
)

from radar_simulation.filters import (
    DSPPipelineConfig,
    run_dsp_pipeline,
)

# ── Tracking ─────────────────────────────────────────────────────────
from multi_target_tracking.kalman_tracker import (
    TrackManager,
    TrackManagerConfig,
    detections_to_measurements,
    TrackStatus,
)

# ── ML Models ────────────────────────────────────────────────────────
from ml_models.threat_classifier import (
    ThreatClassifier,
    ClassifierConfig,
)

from ml_models.trajectory_predictor import (
    TrajectoryPredictorInference,
    ModelConfig,
    TrainConfig,
)

# ── Utils ────────────────────────────────────────────────────────────
from utils.metrics import (
    LatencyProfiler,
    print_classification_report,
)

from utils.data_loader import (
    SimulationLogger,
    list_scenarios,
)

# ── Terminal colors ───────────────────────────────────────────────────
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"


# ─────────────────────────────────────────────────────────────────────
# Banner
# ─────────────────────────────────────────────────────────────────────
def banner() -> None:
    print(
        f"""
{CYAN}{BOLD}
  ╔═══════════════════════════════════════════════════════════════╗
  ║   SKY SENTINEL – AI AIR DEFENSE THREAT PREDICTION SYSTEM     ║
  ║                         v1.0.0                                ║
  ╚═══════════════════════════════════════════════════════════════╝
{RESET}
"""
    )


# ─────────────────────────────────────────────────────────────────────
# Build / Load ML Models
# ─────────────────────────────────────────────────────────────────────
def build_models(no_train: bool = False) -> tuple[
    ThreatClassifier, TrajectoryPredictorInference
]:

    clf_path = PROJECT_ROOT / "ml_models/saved_models/threat_classifier.pkl"
    pred_path = PROJECT_ROOT / "ml_models/saved_models/trajectory_predictor.pt"

    if no_train and clf_path.exists():
        print(f"{DIM}Loading threat classifier...{RESET}")
        classifier = ThreatClassifier.load(clf_path)
    else:
        print(f"{CYAN}Training threat classifier...{RESET}")
        classifier = ThreatClassifier(
            ClassifierConfig(n_estimators=150, n_per_class=800)
        ).train(verbose=True, evaluate=True)

        clf_path.parent.mkdir(parents=True, exist_ok=True)
        classifier.save(clf_path)

    if no_train and pred_path.exists():
        print(f"{DIM}Loading trajectory predictor...{RESET}")
        predictor = TrajectoryPredictorInference.from_saved(
            pred_path,
            PROJECT_ROOT / "ml_models/saved_models/trajectory_scaler.npz",
        )
    else:
        print(f"{CYAN}Training trajectory predictor...{RESET}")

        model_cfg = ModelConfig(
            hidden_dim=64,
            seq_len=12,
            pred_horizon=10,
        )

        train_cfg = TrainConfig(
            epochs=30,
            n_train_seqs=2000,
            batch_size=64,
            patience=8,
            device="auto",
        )

        predictor = TrajectoryPredictorInference.train_and_build(
            model_cfg,
            train_cfg,
            verbose=True,
        )

    return classifier, predictor


# ─────────────────────────────────────────────────────────────────────
# Core Simulation
# ─────────────────────────────────────────────────────────────────────
def run_simulation(
    scenario: str,
    n_scans: int,
    seed: int,
    clf: ThreatClassifier,
    predictor: TrajectoryPredictorInference,
    log_dir: Optional[Path] = None,
) -> tuple[list, dict, dict]:

    rng = np.random.default_rng(seed)

    radar = RadarParameters(
        n_pulses_per_cpi=64,
        max_range_m=400000,
        prf_hz=2000,
    )

    tracker = TrackManager(
        TrackManagerConfig(
            sigma_a=15.0,
            dt_sec=1.0,
            confirm_m_hits=3,
            confirm_n_scans=5,
            max_coast_scans=4,
        )
    )

    noise_cfg = NoisePipelineConfig(
        enable_thermal=True,
        thermal=ThermalNoiseConfig(noise_power_dbw=-108),
        enable_clutter=True,
        clutter=ClutterConfig(
            clutter_to_noise_ratio_db=28,
            max_range_m=80000,
        ),
    )

    dsp_cfg = DSPPipelineConfig(
        enable_mti=True,
        enable_cfar=True,
    )

    profiler = LatencyProfiler()

    logger = None
    if log_dir:
        logger = SimulationLogger(
            log_dir,
            scenario_name=scenario,
            run_id=f"{scenario}_{seed}",
        )

    classifications: dict = {}
    predictions: dict = {}

    print(f"\n{CYAN}Running simulation...{RESET}\n")

    for scan in range(1, n_scans + 1):

        with profiler.measure("signal"):
            cube, _, _ = generate_cpi(
                scenario=scenario,
                radar=radar,
                seed=seed + scan,
            )

            ref = generate_lfm_pulse(radar, rng)

        with profiler.measure("noise"):
            noisy, _ = apply_noise_pipeline(
                cube,
                noise_cfg,
                rng=rng,
                range_gate_length_m=radar.range_gate_length_m,
                reference_pulse=ref,
                noise_power=radar.thermal_noise_power(),
            )

        with profiler.measure("dsp"):
            _, _, detections = run_dsp_pipeline(
                noisy,
                ref,
                prf_hz=radar.prf_hz,
                range_gate_length_m=radar.range_gate_length_m,
                wavelength_m=radar.wavelength_m,
                config=dsp_cfg,
            )

        with profiler.measure("tracking"):
            measurements = detections_to_measurements(detections)
            tracks = tracker.update(measurements, scan)

        confirmed = tracker.confirmed_tracks

        with profiler.measure("classification"):
            if confirmed:
                results = clf.classify_batch(confirmed)

                for t, r in zip(confirmed, results):
                    classifications[t.track_id] = r
                    t.threat_type = r.predicted_class
                    t.alert_level = r.alert_level

        with profiler.measure("prediction"):
            preds = predictor.predict_batch(confirmed, scan)

            for p in preds:
                predictions[p.track_id] = p

        if logger:
            logger.log_scan(
                scan,
                tracks,
                detections,
                classifications,
                preds,
                profiler.to_dict(),
            )

        print(
            f"Scan {scan:>3} | "
            f"Tracks {len(tracks):>3} | "
            f"Confirmed {len(confirmed):>3} | "
            f"Detections {len(detections):>3}"
        )

    if logger:
        logger.save()

    profiler.report()

    return tracks, classifications, predictions


# ─────────────────────────────────────────────────────────────────────
# Dashboard Mode
# ─────────────────────────────────────────────────────────────────────
def mode_dashboard(args: argparse.Namespace) -> None:

    print(f"{GREEN}Launching dashboard...{RESET}")

    try:
        from visualization.dashboard import app
    except Exception:
        print(f"{RED}Dash is not installed.{RESET}")
        sys.exit(1)

    print(f"{CYAN}Open http://127.0.0.1:8050{RESET}")
    app.run(debug=False)


# ─────────────────────────────────────────────────────────────────────
# Simulation Mode
# ─────────────────────────────────────────────────────────────────────
def mode_simulate(args: argparse.Namespace) -> None:

    clf, predictor = build_models(args.no_train)

    tracks, classifications, _ = run_simulation(
        args.scenario,
        args.scans,
        args.seed,
        clf,
        predictor,
        Path(args.log_dir),
    )

    confirmed = [
        t for t in tracks if t.status == TrackStatus.CONFIRMED
    ]

    print(f"\n{BOLD}Final Tracks{RESET}")
    print("-" * 40)

    for t in confirmed:
        cl = classifications.get(t.track_id)

        conf = f"{cl.confidence:.0%}" if cl else "-"

        print(
            f"{CYAN}{t.track_id}{RESET} | "
            f"{t.threat_type:<20} | "
            f"Alert {t.alert_level:<8} | "
            f"Conf {conf}"
        )


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        prog="SkySentinel",
        description="AI Air Defense Threat Prediction System",
    )

    parser.add_argument(
        "--mode",
        choices=["dashboard", "simulate"],
        default="dashboard",
    )

    parser.add_argument(
        "--scenario",
        default="mixed_threat",
    )

    parser.add_argument(
        "--scans",
        type=int,
        default=25,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--no-train",
        action="store_true",
    )

    parser.add_argument(
        "--log-dir",
        default=str(PROJECT_ROOT / "outputs/logs"),
    )

    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────
# Entry
# ─────────────────────────────────────────────────────────────────────
def main() -> None:

    banner()

    args = parse_args()

    scenarios = list_scenarios()

    if args.scenario not in scenarios:
        print(f"{RED}Unknown scenario{RESET}")
        print(scenarios)
        sys.exit(1)

    modes = {
        "dashboard": mode_dashboard,
        "simulate": mode_simulate,
    }

    modes[args.mode](args)


if __name__ == "__main__":
    main()