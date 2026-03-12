

from __future__ import annotations

import math
import pickle
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore", category=UserWarning)



THREAT_CLASSES = [
    "ballistic_missile",
    "cruise_missile",
    "drone",
    "fighter_jet",
    "commercial_aircraft",
]

# Alert level thresholds — probability of being a hostile threat class
HOSTILE_CLASSES = {"ballistic_missile", "cruise_missile", "drone", "fighter_jet"}

ALERT_THRESHOLDS = {
    "critical": 0.85,   # Confirmed hostile, high confidence
    "high":     0.65,
    "medium":   0.40,
    "low":      0.20,
    "none":     0.00,
}

# Alert level colors for dashboard integration
ALERT_COLORS = {
    "critical": "#ff0000",
    "high":     "#ff6600",
    "medium":   "#ffcc00",
    "low":      "#00ccff",
    "none":     "#00ff88",
}

MODEL_DIR  = Path(__file__).parent / "saved_models"
MODEL_PATH = MODEL_DIR / "threat_classifier.pkl"




@dataclass
class TrackFeatures:

    # Kinematic
    speed_mps:          float = 0.0
    radial_vel_mps:     float = 0.0
    altitude_m:         float = 0.0
    range_m:            float = 0.0
    elevation_deg:      float = 0.0
    vertical_speed_mps: float = 0.0
    accel_est_mps2:     float = 0.0
    speed_variance:     float = 0.0
    heading_change_rate: float = 0.0
    range_rate_change:  float = 0.0
    altitude_variance:  float = 0.0
    aspect_ratio:       float = 0.0
    kinetic_energy_proxy: float = 0.0
    azimuth_rate_dps:   float = 0.0

    # Signature
    snr_mean_db:        float = 0.0
    snr_variance_db:    float = 0.0
    track_age_scans:    float = 0.0
    hit_rate:           float = 1.0

    # Trajectory shape
    path_curvature:     float = 0.0
    straightness_index: float = 1.0
    altitude_trend:     float = 0.0
    speed_trend:        float = 0.0

    FEATURE_NAMES: list[str] = field(default_factory=lambda: [
        "speed_mps", "radial_vel_mps", "altitude_m", "range_m",
        "elevation_deg", "vertical_speed_mps", "accel_est_mps2",
        "speed_variance", "heading_change_rate", "range_rate_change",
        "altitude_variance", "aspect_ratio", "kinetic_energy_proxy",
        "azimuth_rate_dps", "snr_mean_db", "snr_variance_db",
        "track_age_scans", "hit_rate", "path_curvature",
        "straightness_index", "altitude_trend", "speed_trend",
    ])

    def to_array(self) -> np.ndarray:
        return np.array([
            self.speed_mps, self.radial_vel_mps, self.altitude_m,
            self.range_m, self.elevation_deg, self.vertical_speed_mps,
            self.accel_est_mps2, self.speed_variance, self.heading_change_rate,
            self.range_rate_change, self.altitude_variance, self.aspect_ratio,
            self.kinetic_energy_proxy, self.azimuth_rate_dps,
            self.snr_mean_db, self.snr_variance_db,
            self.track_age_scans, self.hit_rate,
            self.path_curvature, self.straightness_index,
            self.altitude_trend, self.speed_trend,
        ], dtype=np.float32)


def extract_features(track) -> TrackFeatures:

    f = TrackFeatures()

    # ── Handle both Track objects and dict-based synthetic tracks ────────
    if isinstance(track, dict):
        pos_hist = np.array(track.get("positions", [[0, 0, 0]]))
        vel_hist = np.array(track.get("velocities", [[0, 0, 0]]))
        meas_hist = track.get("measurements_history", [])
        x_state = np.array(track.get("x", [0]*6))
        hit  = track.get("hit_count", 1)
        total = track.get("total_scans", 1)
    else:
        pos_hist  = np.array(track.positions) if track.positions else np.zeros((1, 3))
        vel_hist  = np.array(track.velocities) if track.velocities else np.zeros((1, 3))
        meas_hist = track.measurements_history
        x_state   = track.x
        hit   = track.hit_count
        total = max(track.total_scans, 1)

    # ── Current state ─────────────────────────────────────────────────────
    vx, vy, vz = float(x_state[3]), float(x_state[4]), float(x_state[5])
    px, py, pz = float(x_state[0]), float(x_state[1]), float(x_state[2])

    speed = math.sqrt(vx**2 + vy**2 + vz**2)
    r     = math.sqrt(px**2 + py**2 + pz**2)
    r     = max(r, 1.0)
    v_r   = (px*vx + py*vy + pz*vz) / r
    el_deg = math.degrees(math.atan2(pz, math.sqrt(px**2 + py**2)))

    f.speed_mps           = speed
    f.radial_vel_mps      = v_r
    f.altitude_m          = pz
    f.range_m             = r
    f.elevation_deg       = el_deg
    f.vertical_speed_mps  = vz
    f.kinetic_energy_proxy = 0.5 * speed**2
    f.aspect_ratio        = speed / (abs(vz) + 1.0)
    f.hit_rate            = hit / total
    f.track_age_scans     = float(total)

    # ── History-based features (need at least 3 points) ─────────────────
    n = len(pos_hist)
    if n >= 3:
        speeds = np.linalg.norm(vel_hist, axis=1)
        f.speed_variance   = float(np.var(speeds))
        f.altitude_variance = float(np.var(pos_hist[:, 2]))

        # Acceleration estimate from velocity differences
        if n >= 4:
            vel_diffs = np.diff(vel_hist[-4:], axis=0)
            f.accel_est_mps2 = float(np.mean(np.linalg.norm(vel_diffs, axis=1)))

        # Range-rate change (d(v_r)/dt)
        if n >= 2:
            r_prev = float(np.linalg.norm(pos_hist[-2]))
            r_prev = max(r_prev, 1.0)
            v_r_prev = float(np.dot(pos_hist[-2], vel_hist[-2]) / r_prev)
            f.range_rate_change = v_r - v_r_prev

        # Heading change rate (azimuth rate)
        if n >= 2:
            az_now  = math.atan2(px, py)
            az_prev = math.atan2(float(pos_hist[-2][0]),
                                  float(pos_hist[-2][1]))
            daz = az_now - az_prev
            # Wrap to [-π, π]
            daz = (daz + math.pi) % (2 * math.pi) - math.pi
            f.azimuth_rate_dps    = math.degrees(abs(daz))
            f.heading_change_rate = abs(daz)

        # Path curvature — average deviation of direction vectors
        if n >= 4:
            dirs = np.diff(pos_hist[-5:], axis=0)
            norms = np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-8
            unit_dirs = dirs / norms
            dots = np.sum(unit_dirs[:-1] * unit_dirs[1:], axis=1)
            dots = np.clip(dots, -1, 1)
            angles = np.arccos(dots)
            f.path_curvature = float(np.mean(angles))

        # Straightness index
        net_disp = float(np.linalg.norm(pos_hist[-1] - pos_hist[0]))
        total_path = float(np.sum(np.linalg.norm(np.diff(pos_hist, axis=0), axis=1)))
        f.straightness_index = net_disp / (total_path + 1e-8)

        # Altitude trend (linear slope via least squares)
        if n >= 4:
            t_ax = np.arange(n, dtype=float)
            f.altitude_trend = float(np.polyfit(t_ax, pos_hist[:, 2], 1)[0])
            f.speed_trend    = float(np.polyfit(t_ax[:len(speeds)], speeds, 1)[0])

    # ── SNR features from measurement history ────────────────────────────
    snr_vals = [m.get("snr_db", 0.0) for m in meas_hist if "snr_db" in m]
    if snr_vals:
        f.snr_mean_db    = float(np.mean(snr_vals))
        f.snr_variance_db = float(np.var(snr_vals))

    return f




# Realistic kinematic distributions per threat type
THREAT_DISTRIBUTIONS = {
    "ballistic_missile": dict(
        speed=(1500, 4000),     alt=(10000, 80000),
        sigma_a=5.0,            rcs_mean=0.05,
        snr_mean=(5, 20),       altitude_trend=(-100, 0),
    ),
    "cruise_missile": dict(
        speed=(200, 350),       alt=(30, 500),
        sigma_a=15.0,           rcs_mean=0.1,
        snr_mean=(8, 22),       altitude_trend=(-2, 2),
    ),
    "drone": dict(
        speed=(5, 80),          alt=(20, 1500),
        sigma_a=10.0,           rcs_mean=0.02,
        snr_mean=(2, 15),       altitude_trend=(-5, 5),
    ),
    "fighter_jet": dict(
        speed=(200, 600),       alt=(500, 15000),
        sigma_a=50.0,           rcs_mean=5.0,
        snr_mean=(15, 35),      altitude_trend=(-30, 30),
    ),
    "commercial_aircraft": dict(
        speed=(220, 280),       alt=(7000, 12000),
        sigma_a=1.5,            rcs_mean=30.0,
        snr_mean=(20, 40),      altitude_trend=(-1, 1),
    ),
}


def generate_synthetic_tracks(
    n_per_class: int = 800,
    history_len: int = 10,
    rng: Optional[np.random.Generator] = None,
) -> tuple[list[dict], list[str]]:

    if rng is None:
        rng = np.random.default_rng(seed=2024)

    tracks: list[dict] = []
    labels: list[str]  = []

    for class_name, dist in THREAT_DISTRIBUTIONS.items():
        for _ in range(n_per_class):
            speed = rng.uniform(*dist["speed"])
            alt   = rng.uniform(*dist["alt"])
            sigma_a = dist["sigma_a"]

            # Random start position (100–400 km range)
            r_start = rng.uniform(100e3, 400e3)
            az = rng.uniform(0, 2*math.pi)
            el = math.radians(rng.uniform(0.5, 15))

            px = r_start * math.cos(el) * math.sin(az)
            py = r_start * math.cos(el) * math.cos(az)
            pz = alt

            # Velocity heading toward radar
            vx = -px/r_start * speed + rng.normal(0, speed * 0.05)
            vy = -py/r_start * speed + rng.normal(0, speed * 0.05)
            vz = rng.uniform(*dist["altitude_trend"])

            positions  = []
            velocities = []
            state = np.array([px, py, pz, vx, vy, vz])

            for t in range(history_len):
                positions.append(tuple(state[:3]))
                velocities.append(tuple(state[3:]))
                # Propagate with Singer acceleration noise
                ax = rng.normal(0, sigma_a)
                ay = rng.normal(0, sigma_a)
                az_a = rng.normal(0, sigma_a * 0.15)
                state[3] += ax;  state[4] += ay;  state[5] += az_a
                state[0] += state[3];  state[1] += state[4];  state[2] += state[5]
                state[2] = max(state[2], 10.0)   # ground clamp

            # SNR based on RCS and range
            snr_lo, snr_hi = dist["snr_mean"]
            meas_hist = [
                {"snr_db": rng.uniform(snr_lo, snr_hi)}
                for _ in range(history_len)
            ]

            track_dict = {
                "x":                    np.array([*positions[-1], *velocities[-1]]),
                "positions":            positions,
                "velocities":           velocities,
                "measurements_history": meas_hist,
                "hit_count":            history_len,
                "total_scans":          history_len,
            }
            tracks.append(track_dict)
            labels.append(class_name)

    return tracks, labels


# ──────────────────────────────────────────────────────────────────────────────
# Classifier Configuration
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ClassifierConfig:

    use_random_forest:  bool = True
    use_gradient_boost: bool = True
    use_svm:            bool = True
    n_estimators:       int  = 200
    calibrate:          bool = True
    n_per_class:        int  = 1000
    cross_val_folds:    int  = 5
    random_state:       int  = 42



@dataclass
class ClassificationResult:
    """
    Output of a single threat classification inference call.
    """
    track_id:           str
    predicted_class:    str
    class_probabilities: dict[str, float]    # class_name → probability
    confidence:         float                 # max probability
    alert_level:        str                   # none/low/medium/high/critical
    alert_color:        str                   # hex color for dashboard
    hostile_probability: float               # sum of hostile class probs
    features_used:      Optional[np.ndarray] = None  # for explainability

    def top_classes(self, n: int = 3) -> list[tuple[str, float]]:
        """Return top-n class predictions sorted by probability."""
        return sorted(
            self.class_probabilities.items(),
            key=lambda kv: kv[1],
            reverse=True,
        )[:n]

    def to_dict(self) -> dict:
        return {
            "track_id":          self.track_id,
            "predicted_class":   self.predicted_class,
            "probabilities":     self.class_probabilities,
            "confidence":        self.confidence,
            "alert_level":       self.alert_level,
            "alert_color":       self.alert_color,
            "hostile_prob":      self.hostile_probability,
        }

    def __repr__(self) -> str:
        return (
            f"Classification(id={self.track_id}, "
            f"class={self.predicted_class}, "
            f"conf={self.confidence:.2f}, "
            f"alert={self.alert_level})"
        )


def _map_alert_level(hostile_prob: float) -> tuple[str, str]:
    """Map hostile probability to alert level and dashboard color."""
    for level, threshold in ALERT_THRESHOLDS.items():
        if hostile_prob >= threshold:
            return level, ALERT_COLORS[level]
    return "none", ALERT_COLORS["none"]


# ──────────────────────────────────────────────────────────────────────────────
# Ensemble Classifier
# ──────────────────────────────────────────────────────────────────────────────

class ThreatClassifier:


    def __init__(self, config: ClassifierConfig = ClassifierConfig()) -> None:
        self.config   = config
        self.pipeline: Optional[Pipeline] = None
        self.label_encoder = LabelEncoder()
        self.label_encoder.classes_ = np.array(THREAT_CLASSES)
        self._trained = False

    # ── Training ─────────────────────────────────────────────────────────────

    def train(
        self,
        verbose: bool = True,
        evaluate: bool = True,
    ) -> "ThreatClassifier":

        cfg = self.config

        # ── Generate features ────────────────────────────────────────────
        if verbose:
            print(f"  Generating {cfg.n_per_class * len(THREAT_CLASSES)} "
                  f"synthetic tracks …")

        rng    = np.random.default_rng(cfg.random_state)
        tracks, labels = generate_synthetic_tracks(
            n_per_class=cfg.n_per_class, rng=rng
        )

        # Extract feature vectors
        feats = np.array([extract_features(t).to_array() for t in tracks])
        y_enc = self.label_encoder.transform(labels)

        if verbose:
            print(f"  Feature matrix shape: {feats.shape}")

        # ── Build ensemble ───────────────────────────────────────────────
        estimators = []

        if cfg.use_random_forest:
            rf = RandomForestClassifier(
                n_estimators=cfg.n_estimators,
                max_depth=None,
                min_samples_leaf=2,
                max_features="sqrt",
                class_weight="balanced",
                random_state=cfg.random_state,
                n_jobs=-1,
            )
            estimators.append(("rf", rf))

        if cfg.use_gradient_boost:
            gb = GradientBoostingClassifier(
                n_estimators=min(cfg.n_estimators, 150),
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                random_state=cfg.random_state,
            )
            estimators.append(("gb", gb))

        if cfg.use_svm:
            svm = SVC(
                kernel="rbf",
                C=10.0,
                gamma="scale",
                probability=True,
                class_weight="balanced",
                random_state=cfg.random_state,
            )
            estimators.append(("svm", svm))

        if not estimators:
            raise ValueError("At least one classifier must be enabled.")

        # Soft voting ensemble
        voting_clf = VotingClassifier(
            estimators=estimators,
            voting="soft",
            n_jobs=-1,
        )

        # Calibration wrapper (Platt scaling with 5-fold isotonic)
        if cfg.calibrate and len(estimators) > 0:
            base_clf = CalibratedClassifierCV(
                voting_clf, cv=3, method="sigmoid"
            )
        else:
            base_clf = voting_clf

        # Wrap in a pipeline with standard scaler
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    base_clf),
        ])

        # ── Fit ──────────────────────────────────────────────────────────
        if verbose:
            print(f"  Training ensemble ({', '.join(e[0] for e in estimators)}) …")

        self.pipeline.fit(feats, y_enc)
        self._trained = True

        # ── Cross-validation evaluation ──────────────────────────────────
        if evaluate:
            cv = StratifiedKFold(n_splits=cfg.cross_val_folds,
                                 shuffle=True, random_state=cfg.random_state)
            scores = cross_val_score(
                self.pipeline, feats, y_enc,
                cv=cv, scoring="accuracy", n_jobs=-1,
            )
            if verbose:
                print(f"\n  Cross-validation accuracy: "
                      f"{scores.mean()*100:.1f}% ± {scores.std()*100:.1f}%")

            # Full classification report on training set
            y_pred = self.pipeline.predict(feats)
            if verbose:
                print("\n  Classification Report (training set):")
                print(classification_report(
                    y_enc, y_pred,
                    target_names=self.label_encoder.classes_,
                    digits=3,
                ))

            # Feature importances (RF only)
            if cfg.use_random_forest and verbose:
                self._print_feature_importances(feats)

        return self

    def _print_feature_importances(self, feats: np.ndarray) -> None:
        """Print top-10 feature importances from the Random Forest."""
        try:
            # Navigate pipeline → calibrated clf → voting clf → RF
            clf_step = self.pipeline.named_steps["clf"]
            if hasattr(clf_step, "calibrated_classifiers_"):
                # CalibratedClassifierCV: take first fold's base estimator
                base = clf_step.calibrated_classifiers_[0].estimator
            else:
                base = clf_step

            # VotingClassifier
            if hasattr(base, "estimators_"):
                for name, est in zip(
                    [e[0] for e in base.estimators],
                    base.estimators_,
                ):
                    if name == "rf" and hasattr(est, "feature_importances_"):
                        fi = est.feature_importances_
                        feat_names = TrackFeatures().FEATURE_NAMES
                        ranked = sorted(
                            zip(feat_names, fi), key=lambda x: x[1], reverse=True
                        )
                        print("\n  Top-10 Feature Importances (Random Forest):")
                        for fname, imp in ranked[:10]:
                            bar = "█" * int(imp * 200)
                            print(f"    {fname:<28} {imp:.4f}  {bar}")
                        break
        except Exception:
            pass  # Best-effort; don't break the pipeline

    # ── Inference ────────────────────────────────────────────────────────────

    def classify(self, track) -> ClassificationResult:

        if not self._trained or self.pipeline is None:
            raise RuntimeError("Classifier not trained. Call .train() first.")

        feat   = extract_features(track)
        fvec   = feat.to_array().reshape(1, -1)
        track_id = track.get("track_id", "???") if isinstance(track, dict) \
                   else track.track_id

        # Predict probabilities
        probs_raw = self.pipeline.predict_proba(fvec)[0]

        # Map to class names (label encoder order)
        class_probs = {
            cls: float(probs_raw[i])
            for i, cls in enumerate(self.label_encoder.classes_)
        }

        predicted_class = self.label_encoder.classes_[int(np.argmax(probs_raw))]
        confidence      = float(np.max(probs_raw))

        # Hostile probability
        hostile_prob = sum(
            class_probs.get(c, 0.0)
            for c in HOSTILE_CLASSES
        )

        alert_level, alert_color = _map_alert_level(hostile_prob)

        return ClassificationResult(
            track_id            = track_id,
            predicted_class     = predicted_class,
            class_probabilities = class_probs,
            confidence          = confidence,
            alert_level         = alert_level,
            alert_color         = alert_color,
            hostile_probability = hostile_prob,
            features_used       = fvec[0],
        )

    def classify_batch(self, tracks: list) -> list[ClassificationResult]:
        """Classify a list of tracks efficiently in one batch."""
        if not self._trained or self.pipeline is None:
            raise RuntimeError("Classifier not trained.")

        results = []
        feats   = np.array([extract_features(t).to_array() for t in tracks])
        probs_all = self.pipeline.predict_proba(feats)

        for i, (track, probs_raw) in enumerate(zip(tracks, probs_all)):
            track_id = track.get("track_id", str(i)) if isinstance(track, dict) \
                       else track.track_id
            class_probs = {
                cls: float(probs_raw[j])
                for j, cls in enumerate(self.label_encoder.classes_)
            }
            predicted_class = self.label_encoder.classes_[int(np.argmax(probs_raw))]
            confidence      = float(np.max(probs_raw))
            hostile_prob    = sum(
                class_probs.get(c, 0.0) for c in HOSTILE_CLASSES
            )
            alert_level, alert_color = _map_alert_level(hostile_prob)

            results.append(ClassificationResult(
                track_id            = track_id,
                predicted_class     = predicted_class,
                class_probabilities = class_probs,
                confidence          = confidence,
                alert_level         = alert_level,
                alert_color         = alert_color,
                hostile_probability = hostile_prob,
                features_used       = feats[i],
            ))
        return results



    def save(self, path: Path = MODEL_PATH) -> None:
        """Pickle the fitted pipeline to disk."""
        if not self._trained:
            raise RuntimeError("Cannot save untrained classifier.")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump({
                "pipeline":      self.pipeline,
                "label_encoder": self.label_encoder,
                "config":        self.config,
            }, fh)
        print(f"  ✓ Classifier saved → {path}")

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> "ThreatClassifier":
        """Load a previously saved classifier."""
        if not path.exists():
            raise FileNotFoundError(
                f"Classifier not found at {path}. "
                "Run ThreatClassifier().train().save() first."
            )
        with open(path, "rb") as fh:
            data = pickle.load(fh)
        obj = cls(config=data["config"])
        obj.pipeline      = data["pipeline"]
        obj.label_encoder = data["label_encoder"]
        obj._trained      = True
        return obj




if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    print("=" * 64)
    print("  SkySentinel — Threat Classifier Self-Test")
    print("=" * 64)

    # ── Train classifier ─────────────────────────────────────────────────
    clf = ThreatClassifier(ClassifierConfig(n_per_class=600, n_estimators=150))
    clf.train(verbose=True, evaluate=True)
    clf.save()

    # ── Classify a batch of test tracks ──────────────────────────────────
    print("\n  Testing on fresh synthetic tracks …")
    rng = np.random.default_rng(seed=9999)
    test_tracks, test_labels = generate_synthetic_tracks(n_per_class=20, rng=rng)
    results = clf.classify_batch(test_tracks)

    print(f"\n  Sample predictions ({len(results)} tracks):")
    for r, true_lbl in zip(results[:12], test_labels[:12]):
        match = "✓" if r.predicted_class == true_lbl else "✗"
        print(
            f"  {match} True: {true_lbl:<22}  "
            f"Pred: {r.predicted_class:<22}  "
            f"Conf: {r.confidence:.2f}  Alert: {r.alert_level}"
        )

    correct = sum(r.predicted_class == lbl
                  for r, lbl in zip(results, test_labels))
    print(f"\n  Test accuracy (batch): {correct}/{len(results)} = "
          f"{correct/len(results)*100:.1f}%")

    # ── Visualisation — probability heatmap + confusion ──────────────────
    BG = "#0a0f1e";  TC = "white";  GC = "#1e2a3a"

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), facecolor=BG)
    fig.suptitle("SkySentinel — Threat Classifier Analysis",
                 color=TC, fontsize=13, fontweight="bold")

    # 1. Class probability heatmap (first 20 test tracks)
    ax0 = axes[0]
    probs_matrix = np.array([
        [r.class_probabilities.get(c, 0.0) for c in THREAT_CLASSES]
        for r in results[:20]
    ])
    im = ax0.imshow(probs_matrix.T, aspect="auto", cmap="plasma",
                    vmin=0, vmax=1)
    ax0.set_xticks(range(20))
    ax0.set_xticklabels([f"T{i}" for i in range(20)],
                        color=TC, fontsize=7, rotation=45)
    ax0.set_yticks(range(len(THREAT_CLASSES)))
    ax0.set_yticklabels(
        [c.replace("_", "\n") for c in THREAT_CLASSES],
        color=TC, fontsize=8
    )
    ax0.set_facecolor(BG)
    ax0.set_title("Class Probability Heatmap (20 test tracks)", color=TC)
    ax0.set_xlabel("Track Index", color=TC, fontsize=9)
    cb = fig.colorbar(im, ax=ax0, pad=0.02)
    cb.set_label("Probability", color=TC, fontsize=8)
    cb.ax.yaxis.set_tick_params(color=TC)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=TC)

    # 2. Alert level pie chart
    ax1 = axes[1]
    alert_counts: dict[str, int] = {lvl: 0 for lvl in ALERT_COLORS}
    for r in results:
        alert_counts[r.alert_level] = alert_counts.get(r.alert_level, 0) + 1

    non_zero = {k: v for k, v in alert_counts.items() if v > 0}
    wedge_colors = [ALERT_COLORS[k] for k in non_zero]
    wedges, texts, autotexts = ax1.pie(
        non_zero.values(),
        labels=non_zero.keys(),
        autopct="%1.0f%%",
        colors=wedge_colors,
        textprops={"color": TC, "fontsize": 10},
        wedgeprops={"edgecolor": BG, "linewidth": 2},
        startangle=140,
    )
    for at in autotexts:
        at.set_color(BG)
        at.set_fontsize(9)
    ax1.set_facecolor(BG)
    ax1.set_title("Alert Level Distribution", color=TC, fontsize=11)

    for ax in axes:
        ax.set_facecolor(BG)
        ax.tick_params(colors=TC)
        for sp in ax.spines.values():
            sp.set_color(GC)

    plt.tight_layout()
    out = "threat_classifier_preview.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"\n  Preview saved → {out}")
    print("=" * 64)