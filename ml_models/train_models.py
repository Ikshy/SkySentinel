import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml_models.threat_classifier import ThreatClassifier, ClassifierConfig
from ml_models.trajectory_predictor import (
    TrajectoryPredictorInference, ModelConfig, TrainConfig,
)

SAVE_DIR = Path(__file__).parent / "saved_models"

if __name__ == "__main__":
    print("=" * 58)
    print("  SkySentinel — Model Training Script")
    print("=" * 58)

    # ── Train Threat Classifier ───────────────────────────────────────────
    print("\n  [1/2] Training Threat Classifier …")
    clf = ThreatClassifier(
        ClassifierConfig(n_per_class=1000, n_estimators=200)
    ).train(verbose=True, evaluate=True)
    clf.save(SAVE_DIR / "threat_classifier.pkl")

    # ── Train Trajectory Predictor ────────────────────────────────────────
    print("\n  [2/2] Training Trajectory Predictor …")
    m_cfg = ModelConfig(hidden_dim=128, seq_len=15, pred_horizon=10)
    t_cfg = TrainConfig(epochs=60, n_train_seqs=5000, patience=12,
                        batch_size=64, device="auto")
    TrajectoryPredictorInference.train_and_build(m_cfg, t_cfg, verbose=True)

    print("\n   All models saved to:", SAVE_DIR)
    print("  Run with --no-train to skip training on next launch.")
    print("=" * 58)