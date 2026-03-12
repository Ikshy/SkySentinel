from __future__ import annotations

import csv
import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


SCENARIOS_PATH = Path(__file__).parent.parent / "scenarios" / "sample_scenarios.json"



def load_scenario(name: str, scenarios_path: Path = SCENARIOS_PATH) -> dict:

    with open(scenarios_path) as f:
        data = json.load(f)

    scenarios = data.get("scenarios", {})
    if name not in scenarios:
        available = list(scenarios.keys())
        raise KeyError(
            f"Scenario '{name}' not found. Available: {available}"
        )
    return scenarios[name]


def list_scenarios(scenarios_path: Path = SCENARIOS_PATH) -> list[str]:
    """Return a list of all available scenario names."""
    with open(scenarios_path) as f:
        data = json.load(f)
    return list(data.get("scenarios", {}).keys())


def load_radar_preset(name: str, scenarios_path: Path = SCENARIOS_PATH) -> dict:
    """Load a radar hardware preset by name."""
    with open(scenarios_path) as f:
        data = json.load(f)
    presets = data.get("radar_presets", {})
    if name not in presets:
        raise KeyError(f"Radar preset '{name}' not found. "
                       f"Available: {list(presets.keys())}")
    return presets[name]




class SimulationLogger:


    def __init__(
        self,
        output_dir: Path,
        scenario_name: str = "unknown",
        run_id: Optional[str] = None,
    ) -> None:
        self.output_dir    = Path(output_dir)
        self.scenario_name = scenario_name
        self.run_id        = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self._track_rows:  list[dict] = []
        self._scan_logs:   list[dict] = []
        self._start_time   = time.time()

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def log_scan(
        self,
        scan_idx:        int,
        tracks:          list,
        detections:      list,
        classifications: dict,
        predictions:     list,
        latency_ms:      dict,
    ) -> None:
        """Record one complete radar scan worth of data."""
        timestamp = time.time() - self._start_time

        scan_entry = {
            "scan_idx":        scan_idx,
            "timestamp_s":     round(timestamp, 3),
            "n_tracks":        len(tracks),
            "n_detections":    len(detections),
            "n_confirmed":     sum(1 for t in tracks
                                   if hasattr(t, "status") and
                                   t.status.name == "CONFIRMED"),
            "latency_ms":      latency_ms,
            "detections": [
                {"range_m": d.range_m, "velocity_mps": d.velocity_mps,
                 "snr_db": d.snr_db}
                for d in detections
            ],
        }
        self._scan_logs.append(scan_entry)

        # Per-track row (flat, for CSV export)
        for track in tracks:
            cl = classifications.get(
                track.track_id if hasattr(track, "track_id") else "", None
            )
            row = {
                "scan_idx":       scan_idx,
                "timestamp_s":    round(timestamp, 3),
                "track_id":       getattr(track, "track_id", "?"),
                "status":         getattr(track, "status", "?"),
                "x_m":            round(float(track.x[0]), 1),
                "y_m":            round(float(track.x[1]), 1),
                "z_m":            round(float(track.x[2]), 1),
                "vx_mps":         round(float(track.x[3]), 2),
                "vy_mps":         round(float(track.x[4]), 2),
                "vz_mps":         round(float(track.x[5]), 2),
                "range_km":       round(track.range_m() / 1000.0, 2),
                "speed_mps":      round(track.speed_mps(), 1),
                "altitude_m":     round(float(track.x[2]), 1),
                "azimuth_deg":    round(track.azimuth_deg(), 1),
                "threat_type":    getattr(track, "threat_type", "unknown"),
                "alert_level":    getattr(track, "alert_level", "none"),
                "confidence":     round(getattr(track, "threat_confidence", 0.0), 3),
                "pos_sigma_m":    round(track.position_uncertainty_m(), 1),
                "hit_count":      getattr(track, "hit_count", 0),
                "miss_count":     getattr(track, "miss_count", 0),
            }
            if cl:
                row["hostile_prob"] = round(cl.hostile_probability, 3)
                for cls_name, prob in cl.class_probabilities.items():
                    row[f"p_{cls_name[:5]}"] = round(prob, 3)

            self._track_rows.append(row)

    def save(self) -> tuple[Path, Path]:

        csv_path  = self.output_dir / f"tracks_{self.run_id}.csv"
        json_path = self.output_dir / f"simulation_{self.run_id}.json"

        # ── CSV — track states ───────────────────────────────────────────
        if self._track_rows:
            df = pd.DataFrame(self._track_rows)
            df.to_csv(csv_path, index=False)

        # ── JSON — full simulation log ───────────────────────────────────
        full_log = {
            "run_id":    self.run_id,
            "scenario":  self.scenario_name,
            "n_scans":   len(self._scan_logs),
            "scans":     self._scan_logs,
        }
        with open(json_path, "w") as f:
            json.dump(full_log, f, indent=2)

        print(f"  ✓ CSV  log → {csv_path}")
        print(f"  ✓ JSON log → {json_path}")
        return csv_path, json_path

    def to_dataframe(self) -> pd.DataFrame:
        """Return recorded track rows as a Pandas DataFrame."""
        return pd.DataFrame(self._track_rows)

    def summary(self) -> dict:
        """High-level summary of the completed simulation run."""
        df = self.to_dataframe()
        if df.empty:
            return {"run_id": self.run_id, "n_scans": 0}

        return {
            "run_id":           self.run_id,
            "scenario":         self.scenario_name,
            "n_scans":          len(self._scan_logs),
            "unique_tracks":    df["track_id"].nunique(),
            "total_detections": sum(s["n_detections"] for s in self._scan_logs),
            "mean_range_km":    round(df["range_km"].mean(), 1),
            "threat_types":     df["threat_type"].value_counts().to_dict(),
            "alert_levels":     df["alert_level"].value_counts().to_dict(),
        }



def get_run_seed(scenario: str, run_number: int = 0) -> int:
    """
    Compute a deterministic seed from a scenario name and run number,
    enabling reproducible batch evaluation.

    >>> get_run_seed("mixed_threat", 0)   # always returns the same int
    """
    import hashlib
    h = hashlib.md5(f"{scenario}_{run_number}".encode()).hexdigest()
    return int(h[:8], 16) % (2**31)


def batch_seeds(scenario: str, n_runs: int = 10) -> list[int]:
    """Return a list of n_runs reproducible seeds for a given scenario."""
    return [get_run_seed(scenario, i) for i in range(n_runs)]