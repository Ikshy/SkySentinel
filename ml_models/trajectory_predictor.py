from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split




MODEL_DIR   = Path(__file__).parent / "saved_models"
MODEL_PATH  = MODEL_DIR / "trajectory_predictor.pt"
SCALER_PATH = MODEL_DIR / "trajectory_scaler.npz"

INPUT_FEATURES  = 6   # [x, y, z, vx, vy, vz]
OUTPUT_FEATURES = 3   # [x, y, z]  — predict position only




@dataclass
class ModelConfig:
    """
    Hyper-parameters for the encoder-decoder LSTM predictor.
    """
    input_dim:      int   = INPUT_FEATURES   # features per timestep
    hidden_dim:     int   = 128              # LSTM hidden units
    encoder_layers: int   = 2               # encoder LSTM depth
    decoder_layers: int   = 1               # decoder LSTM depth
    dropout:        float = 0.20            # dropout probability
    seq_len:        int   = 15              # input history length (timesteps)
    pred_horizon:   int   = 10             # timesteps to predict ahead
    dt_sec:         float = 1.0            # seconds per timestep


@dataclass
class TrainConfig:
    """
    Training hyper-parameters.
    """
    epochs:        int   = 60
    batch_size:    int   = 64
    lr:            float = 3e-4
    weight_decay:  float = 1e-5
    val_fraction:  float = 0.15
    patience:      int   = 12             # early-stopping patience (epochs)
    clip_grad:     float = 1.0            # gradient clipping norm
    n_train_seqs:  int   = 5_000          # synthetic training sequences
    device:        str   = "auto"         # 'auto' | 'cpu' | 'cuda' | 'mps'

    def resolve_device(self) -> torch.device:
        if self.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(self.device)




class FeatureScaler:


    def __init__(self) -> None:
        self.mean_: Optional[np.ndarray] = None
        self.std_:  Optional[np.ndarray] = None
        self._fitted = False

    def fit(self, sequences: np.ndarray) -> "FeatureScaler":
        """
        Parameters
        ----------
        sequences : np.ndarray  shape (N, T, 6)
        """
        flat = sequences.reshape(-1, sequences.shape[-1])
        self.mean_ = flat.mean(axis=0)
        self.std_  = flat.std(axis=0) + 1e-8
        self._fitted = True
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        assert self._fitted, "Scaler not fitted."
        return (x - self.mean_) / self.std_

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        assert self._fitted, "Scaler not fitted."
        return x * self.std_[:OUTPUT_FEATURES] + self.mean_[:OUTPUT_FEATURES]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, mean=self.mean_, std=self.std_)

    @classmethod
    def load(cls, path: Path) -> "FeatureScaler":
        data = np.load(path)
        scaler = cls()
        scaler.mean_ = data["mean"]
        scaler.std_  = data["std"]
        scaler._fitted = True
        return scaler




THREAT_PROFILES = {
    "ballistic_missile": dict(
        speed=(1500, 4000), alt=(5000, 80000),  sigma_a=5.0,  rcs=0.05
    ),
    "cruise_missile":    dict(
        speed=(200,  350),  alt=(50,   500),    sigma_a=15.0, rcs=0.1
    ),
    "drone_medium":      dict(
        speed=(30,   120),  alt=(100,  2000),   sigma_a=8.0,  rcs=0.1
    ),
    "fighter_jet":       dict(
        speed=(200,  600),  alt=(1000, 15000),  sigma_a=50.0, rcs=5.0
    ),
    "commercial":        dict(
        speed=(220,  280),  alt=(8000, 12000),  sigma_a=2.0,  rcs=30.0
    ),
}


def _simulate_trajectory(
    profile: dict,
    total_steps: int,
    dt: float,
    rng: np.random.Generator,
) -> np.ndarray:

    speed_lo, speed_hi = profile["speed"]
    alt_lo,   alt_hi   = profile["alt"]
    sigma_a = profile["sigma_a"]

    # Random initial position (200–400 km range, random bearing)
    r0  = rng.uniform(200e3, 400e3)
    az  = rng.uniform(0, 2 * math.pi)
    el  = math.radians(rng.uniform(1, 15))

    px = r0 * math.cos(el) * math.sin(az)
    py = r0 * math.cos(el) * math.cos(az)
    pz = rng.uniform(alt_lo, alt_hi)

    # Initial velocity heading toward radar
    speed = rng.uniform(speed_lo, speed_hi)
    vx = -px / r0 * speed + rng.normal(0, speed * 0.05)
    vy = -py / r0 * speed + rng.normal(0, speed * 0.05)
    vz = rng.normal(0, speed * 0.02)

    states = np.zeros((total_steps, 6))
    states[0] = [px, py, pz, vx, vy, vz]

    for t in range(1, total_steps):
        # Random acceleration (Singer model)
        ax = rng.normal(0, sigma_a)
        ay = rng.normal(0, sigma_a)
        az_acc = rng.normal(0, sigma_a * 0.2)  # less vertical manoeuvre

        vx += ax * dt;  vy += ay * dt;  vz += az_acc * dt
        px += vx * dt;  py += vy * dt;  pz += vz * dt

        # Keep altitude within profile bounds (soft clamp)
        pz = float(np.clip(pz, alt_lo * 0.5, alt_hi * 1.5))

        states[t] = [px, py, pz, vx, vy, vz]

    return states


def generate_training_data(
    n_sequences: int,
    model_cfg: ModelConfig,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:

    rng = np.random.default_rng(seed)
    seq_total = model_cfg.seq_len + model_cfg.pred_horizon

    profile_names = list(THREAT_PROFILES.keys())
    X_list, Y_list = [], []

    for _ in range(n_sequences):
        pname   = rng.choice(profile_names)
        profile = THREAT_PROFILES[pname]
        traj    = _simulate_trajectory(profile, seq_total, model_cfg.dt_sec, rng)

        x_seq = traj[:model_cfg.seq_len]                     # (seq_len, 6)
        y_seq = traj[model_cfg.seq_len:, :3]                 # (pred_horizon, 3)

        X_list.append(x_seq)
        Y_list.append(y_seq)

    return np.array(X_list, dtype=np.float32), np.array(Y_list, dtype=np.float32)




class TrajectoryDataset(Dataset):
    """
    Wraps normalised (X, Y) arrays as a PyTorch Dataset.
    """

    def __init__(self, X: np.ndarray, Y: np.ndarray) -> None:
        self.X = torch.from_numpy(X.astype(np.float32))
        self.Y = torch.from_numpy(Y.astype(np.float32))

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.Y[idx]



class TrajectoryPredictor(nn.Module):

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # ── Encoder ───────────────────────────────────────────────────────
        self.encoder = nn.LSTM(
            input_size  = cfg.input_dim,
            hidden_size = cfg.hidden_dim,
            num_layers  = cfg.encoder_layers,
            batch_first = True,
            dropout     = cfg.dropout if cfg.encoder_layers > 1 else 0.0,
        )

        # ── Decoder ───────────────────────────────────────────────────────
        # Input to decoder: previous predicted position (3) + step embedding (hidden_dim)
        self.step_embed = nn.Embedding(cfg.pred_horizon + 1, cfg.hidden_dim)
        self.decoder_input_proj = nn.Linear(OUTPUT_FEATURES + cfg.hidden_dim,
                                            cfg.hidden_dim)
        self.decoder = nn.LSTM(
            input_size  = cfg.hidden_dim,
            hidden_size = cfg.hidden_dim,
            num_layers  = cfg.decoder_layers,
            batch_first = True,
            dropout     = 0.0,
        )

        # ── Output heads ──────────────────────────────────────────────────
        self.fc_mean    = nn.Linear(cfg.hidden_dim, OUTPUT_FEATURES)
        self.fc_log_std = nn.Linear(cfg.hidden_dim, OUTPUT_FEATURES)

        # ── Layer normalisation for training stability ────────────────────
        self.enc_ln = nn.LayerNorm(cfg.hidden_dim)
        self.dec_ln = nn.LayerNorm(cfg.hidden_dim)

        # ── Dropout ───────────────────────────────────────────────────────
        self.drop = nn.Dropout(cfg.dropout)

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier uniform initialisation for all linear & LSTM weight matrices."""
        for name, p in self.named_parameters():
            if "weight_ih" in name or "weight_hh" in name:
                nn.init.orthogonal_(p)
            elif "bias" in name:
                nn.init.zeros_(p)
            elif "weight" in name and p.dim() >= 2:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        x: torch.Tensor,                       # (B, seq_len, input_dim)
        target: Optional[torch.Tensor] = None, # (B, pred_horizon, 3) — teacher forcing
        teacher_forcing_ratio: float = 0.5,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        B = x.size(0)
        device = x.device

        # ── Encode ────────────────────────────────────────────────────────
        enc_out, (h_enc, c_enc) = self.encoder(x)

        # Apply layer norm to last encoder hidden state
        h_enc_normed = self.enc_ln(h_enc[-1])  # (B, hidden_dim)

        # Project encoder states to decoder depth (if layers differ)
        # Simple approach: repeat encoder's last layer state for decoder layers
        h_dec = h_enc[-1:].expand(self.cfg.decoder_layers, -1, -1).contiguous()
        c_dec = c_enc[-1:].expand(self.cfg.decoder_layers, -1, -1).contiguous()

        # ── Decode auto-regressively ──────────────────────────────────────
        means_list:    list[torch.Tensor] = []
        log_stds_list: list[torch.Tensor] = []

        # Seed: last known position from input sequence
        prev_pos = x[:, -1, :OUTPUT_FEATURES]   # (B, 3)

        for step in range(self.cfg.pred_horizon):
            step_idx = torch.full((B,), step, dtype=torch.long, device=device)
            step_emb = self.step_embed(step_idx)   # (B, hidden_dim)

            dec_in_raw = torch.cat([prev_pos, step_emb], dim=-1)  # (B, 3+hidden)
            dec_in = self.drop(F.relu(self.decoder_input_proj(dec_in_raw)))
            dec_in = dec_in.unsqueeze(1)   # (B, 1, hidden_dim)

            dec_out, (h_dec, c_dec) = self.decoder(dec_in, (h_dec, c_dec))
            dec_feat = self.dec_ln(dec_out.squeeze(1))  # (B, hidden_dim)

            mu      = self.fc_mean(dec_feat)          # (B, 3)
            log_std = self.fc_log_std(dec_feat)       # (B, 3)
            log_std = torch.clamp(log_std, -6.0, 3.0) # prevent extreme values

            means_list.append(mu)
            log_stds_list.append(log_std)

            # Teacher forcing: use ground-truth or own prediction
            use_teacher = (
                target is not None
                and torch.rand(1).item() < teacher_forcing_ratio
            )
            prev_pos = target[:, step, :] if use_teacher else mu.detach()

        means    = torch.stack(means_list,    dim=1)   # (B, H, 3)
        log_stds = torch.stack(log_stds_list, dim=1)   # (B, H, 3)
        return means, log_stds




def gaussian_nll_loss(
    means:    torch.Tensor,   # (B, H, 3)
    log_stds: torch.Tensor,   # (B, H, 3)
    targets:  torch.Tensor,   # (B, H, 3)
) -> torch.Tensor:

    var   = torch.exp(2.0 * log_stds) + 1e-8
    nll   = log_stds + 0.5 * ((targets - means) ** 2) / var
    return nll.mean()


# ──────────────────────────────────────────────────────────────────────────────
# Training Loop
# ──────────────────────────────────────────────────────────────────────────────

def train(
    model_cfg:  ModelConfig  = ModelConfig(),
    train_cfg:  TrainConfig  = TrainConfig(),
    save_dir:   Path         = MODEL_DIR,
    verbose:    bool         = True,
) -> TrajectoryPredictor:

    device = train_cfg.resolve_device()
    if verbose:
        print(f"  Training device : {device}")

    # ── Generate synthetic data ──────────────────────────────────────────
    if verbose:
        print(f"  Generating {train_cfg.n_train_seqs} synthetic trajectories …")

    X_raw, Y_raw = generate_training_data(train_cfg.n_train_seqs, model_cfg)

    # ── Fit scaler on X and normalise ────────────────────────────────────
    scaler = FeatureScaler().fit(X_raw)
    X_norm = scaler.transform(X_raw).astype(np.float32)

    # Normalise Y positions using the same position scale
    pos_mean = scaler.mean_[:3]
    pos_std  = scaler.std_[:3]
    Y_norm   = ((Y_raw - pos_mean) / pos_std).astype(np.float32)

    # ── Dataset / DataLoader ─────────────────────────────────────────────
    dataset  = TrajectoryDataset(X_norm, Y_norm)
    n_val    = max(1, int(len(dataset) * train_cfg.val_fraction))
    n_train  = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(0))

    train_loader = DataLoader(train_ds, batch_size=train_cfg.batch_size,
                              shuffle=True, num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=train_cfg.batch_size * 2,
                              shuffle=False, num_workers=0)

    # ── Model, optimiser, scheduler ─────────────────────────────────────
    model = TrajectoryPredictor(model_cfg).to(device)
    optim = torch.optim.AdamW(model.parameters(),
                               lr=train_cfg.lr,
                               weight_decay=train_cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=train_cfg.epochs, eta_min=train_cfg.lr * 0.05
    )

    best_val_loss  = float("inf")
    patience_count = 0
    best_state     = None

    if verbose:
        print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"  Training {n_train} / validation {n_val} sequences")
        print(f"  {'Epoch':>6} {'Train NLL':>12} {'Val NLL':>12} {'LR':>10}")
        print("  " + "-" * 46)

    for epoch in range(1, train_cfg.epochs + 1):
        # ── Training ────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optim.zero_grad()

            # Decay teacher forcing from 0.8 → 0.2 over training
            tf_ratio = 0.8 - 0.6 * (epoch / train_cfg.epochs)
            means, log_stds = model(xb, target=yb, teacher_forcing_ratio=tf_ratio)
            loss = gaussian_nll_loss(means, log_stds, yb)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), train_cfg.clip_grad)
            optim.step()
            train_loss += loss.item() * len(xb)

        train_loss /= n_train

        # ── Validation ──────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                means, log_stds = model(xb, teacher_forcing_ratio=0.0)
                val_loss += gaussian_nll_loss(means, log_stds, yb).item() * len(xb)
        val_loss /= n_val

        scheduler.step()
        lr_now = scheduler.get_last_lr()[0]

        if verbose and (epoch % 5 == 0 or epoch == 1):
            print(f"  {epoch:>6d} {train_loss:>12.4f} {val_loss:>12.4f} {lr_now:>10.2e}")

        # ── Early stopping ──────────────────────────────────────────────
        if val_loss < best_val_loss - 1e-4:
            best_val_loss  = val_loss
            patience_count = 0
            best_state     = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_count += 1
            if patience_count >= train_cfg.patience:
                if verbose:
                    print(f"  Early stop at epoch {epoch}  (best val={best_val_loss:.4f})")
                break

    # ── Save ────────────────────────────────────────────────────────────
    if best_state:
        model.load_state_dict(best_state)

    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "model_cfg":   model_cfg,
        "best_val_nll": best_val_loss,
    }, save_dir / "trajectory_predictor.pt")

    scaler.save(save_dir / "trajectory_scaler.npz")

    if verbose:
        print(f"\n  ✓ Model saved  → {save_dir / 'trajectory_predictor.pt'}")
        print(f"  ✓ Scaler saved → {save_dir / 'trajectory_scaler.npz'}")

    model.eval()
    return model



@dataclass
class PredictedTrajectory:

    track_id:        str
    pred_positions:  np.ndarray   # (pred_horizon, 3)  — mean (x, y, z)
    pred_std:        np.ndarray   # (pred_horizon, 3)  — 1-σ uncertainty
    pred_timestamps: np.ndarray   # (pred_horizon,)    — absolute time [s]
    last_known_pos:  np.ndarray   # (3,)               — last tracked position
    impact_point_m:  Optional[tuple[float, float, float]]  # extrapolated CPA

    def range_at_step(self, step: int) -> float:
        """Predicted range from radar at a future step."""
        p = self.pred_positions[step]
        return float(np.linalg.norm(p))

    def confidence_interval(
        self,
        step: int,
        sigma: float = 2.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (lower, upper) CI bounds at `sigma`-σ for given step."""
        mu  = self.pred_positions[step]
        std = self.pred_std[step]
        return mu - sigma * std, mu + sigma * std

    def to_dict(self) -> dict:
        return {
            "track_id":       self.track_id,
            "pred_positions": self.pred_positions.tolist(),
            "pred_std":       self.pred_std.tolist(),
            "pred_timestamps": self.pred_timestamps.tolist(),
            "last_known_pos": self.last_known_pos.tolist(),
        }


class TrajectoryPredictorInference:


    def __init__(
        self,
        model:   TrajectoryPredictor,
        scaler:  FeatureScaler,
        cfg:     ModelConfig,
        device:  Optional[torch.device] = None,
    ) -> None:
        self.model  = model.eval()
        self.scaler = scaler
        self.cfg    = cfg
        self.device = device or torch.device("cpu")

    @classmethod
    def from_saved(
        cls,
        model_path:  Path = MODEL_PATH,
        scaler_path: Path = SCALER_PATH,
        device_str:  str  = "auto",
    ) -> "TrajectoryPredictorInference":
        """Load a previously saved model + scaler from disk."""
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                "Run train() first or call TrajectoryPredictorInference.train_and_build()."
            )
        data   = torch.load(model_path, map_location="cpu")
        cfg    = data["model_cfg"]
        model  = TrajectoryPredictor(cfg)
        model.load_state_dict(data["model_state"])

        scaler = FeatureScaler.load(scaler_path)

        # Device
        if device_str == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device_str)

        return cls(model.to(device), scaler, cfg, device)

    @classmethod
    def train_and_build(
        cls,
        model_cfg: ModelConfig = ModelConfig(),
        train_cfg: TrainConfig = TrainConfig(),
        verbose:   bool        = True,
    ) -> "TrajectoryPredictorInference":
        """Convenience: train from scratch and return inference wrapper."""
        model  = train(model_cfg, train_cfg, verbose=verbose)
        scaler = FeatureScaler.load(MODEL_DIR / "trajectory_scaler.npz")
        device = train_cfg.resolve_device()
        return cls(model.to(device), scaler, model_cfg, device)

    def predict(
        self,
        track,                          # multi_target_tracking.kalman_tracker.Track
        current_timestamp: float = 0.0,
    ) -> Optional[PredictedTrajectory]:

        min_hist = self.cfg.seq_len

        if len(track.positions) < min_hist:
            return None   # not enough history yet

        # Build input state sequence from track history
        recent_pos = np.array(track.positions[-min_hist:])   # (seq_len, 3)
        recent_vel = np.array(track.velocities[-min_hist:])  # (seq_len, 3)
        state_seq  = np.concatenate([recent_pos, recent_vel], axis=1)  # (seq_len, 6)

        # Normalise
        norm_seq = self.scaler.transform(state_seq).astype(np.float32)
        x_tensor = torch.from_numpy(norm_seq).unsqueeze(0).to(self.device)

        # Inference
        with torch.no_grad():
            means, log_stds = self.model(x_tensor, teacher_forcing_ratio=0.0)

        means_np    = means.squeeze(0).cpu().numpy()       # (H, 3)
        log_stds_np = log_stds.squeeze(0).cpu().numpy()   # (H, 3)

        # Inverse-transform positions back to metres
        pred_pos = self.scaler.inverse_transform(means_np)
        pred_std = np.exp(log_stds_np) * self.scaler.std_[:3]

        # Future timestamps
        timestamps = (
            current_timestamp
            + np.arange(1, self.cfg.pred_horizon + 1) * self.cfg.dt_sec
        )

        # Rough closest-point-of-approach (CPA) extrapolation
        impact = self._extrapolate_impact(pred_pos, pred_std)

        return PredictedTrajectory(
            track_id        = track.track_id,
            pred_positions  = pred_pos,
            pred_std        = pred_std,
            pred_timestamps = timestamps,
            last_known_pos  = np.array(track.position_3d()),
            impact_point_m  = impact,
        )

    def predict_batch(
        self,
        tracks:            list,
        current_timestamp: float = 0.0,
    ) -> list[PredictedTrajectory]:
        """Run predictions for a list of tracks. Returns only non-None results."""
        results = []
        for t in tracks:
            p = self.predict(t, current_timestamp)
            if p is not None:
                results.append(p)
        return results

    @staticmethod
    def _extrapolate_impact(
        pred_positions: np.ndarray,
        pred_std:       np.ndarray,
        ground_z:       float = 0.0,
    ) -> Optional[tuple[float, float, float]]:

        zs = pred_positions[:, 2]
        if zs[-1] > ground_z:
            return None   # trajectory not heading to ground

        for i in range(len(zs) - 1):
            if zs[i] > ground_z >= zs[i + 1]:
                # Linear interpolation
                t_frac = (ground_z - zs[i]) / (zs[i + 1] - zs[i] + 1e-8)
                x = pred_positions[i, 0] + t_frac * (
                    pred_positions[i + 1, 0] - pred_positions[i, 0]
                )
                y = pred_positions[i, 1] + t_frac * (
                    pred_positions[i + 1, 1] - pred_positions[i, 1]
                )
                return (float(x), float(y), float(ground_z))
        return None



if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    print("=" * 64)
    print("  SkySentinel — Trajectory Predictor Self-Test")
    print("=" * 64)

    # Use a quick config for fast self-test
    m_cfg = ModelConfig(
        hidden_dim=64, encoder_layers=2, seq_len=12, pred_horizon=8, dt_sec=1.0
    )
    t_cfg = TrainConfig(
        epochs=30, batch_size=64, n_train_seqs=2000,
        patience=8, device="auto"
    )

    print("\n  [1/3] Training …")
    predictor = TrajectoryPredictorInference.train_and_build(
        model_cfg=m_cfg, train_cfg=t_cfg, verbose=True
    )


    print("\n  [2/3] Building synthetic track …")
    rng = np.random.default_rng(seed=77)
    profile = THREAT_PROFILES["cruise_missile"]
    traj    = _simulate_trajectory(profile, m_cfg.seq_len + m_cfg.pred_horizon + 5,
                                   m_cfg.dt_sec, rng)

    class FakeTrack:
        track_id  = "TEST01"
        positions  = [tuple(s[:3]) for s in traj[:m_cfg.seq_len]]
        velocities = [tuple(s[3:]) for s in traj[:m_cfg.seq_len]]
        def position_3d(self):
            return self.positions[-1]

    fake_track = FakeTrack()
    result     = predictor.predict(fake_track, current_timestamp=m_cfg.seq_len)
    gt_future  = traj[m_cfg.seq_len: m_cfg.seq_len + m_cfg.pred_horizon, :3]

    # ADE: Average Displacement Error
    ade = float(np.mean(np.linalg.norm(result.pred_positions - gt_future, axis=1)))
    print(f"\n  [3/3] Results:")
    print(f"    Predicted steps  : {m_cfg.pred_horizon}")
    print(f"    ADE              : {ade/1e3:.2f} km")
    if result.impact_point_m:
        print(f"    Extrapolated CPA : ({result.impact_point_m[0]/1e3:.1f}, "
              f"{result.impact_point_m[1]/1e3:.1f}) km")


    BG = "#0a0f1e";  TC = "white"
    fig = plt.figure(figsize=(12, 7), facecolor=BG)
    ax  = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(BG)
    fig.patch.set_facecolor(BG)

    hist_pos = np.array(fake_track.positions)
    ax.plot(hist_pos[:, 0]/1e3, hist_pos[:, 1]/1e3, hist_pos[:, 2]/1e3,
            "o-", color="#00d4ff", lw=1.5, ms=3, label="Track History")

    ax.plot(result.pred_positions[:, 0]/1e3,
            result.pred_positions[:, 1]/1e3,
            result.pred_positions[:, 2]/1e3,
            "s--", color="#ff6b35", lw=2, ms=5, label="LSTM Prediction (mean)")

    ax.plot(gt_future[:, 0]/1e3, gt_future[:, 1]/1e3, gt_future[:, 2]/1e3,
            "^-", color="#a8ff78", lw=1.5, ms=4, label="Ground Truth")

    # Uncertainty tubes (1-σ)
    for step in range(len(result.pred_positions)):
        mu  = result.pred_positions[step] / 1e3
        std = result.pred_std[step] / 1e3
        for dim, col in [(0, "#ff4444"), (1, "#ffaa00")]:
            lo = mu.copy();  hi = mu.copy()
            lo[dim] -= std[dim];  hi[dim] += std[dim]
            ax.plot([lo[0], hi[0]], [lo[1], hi[1]], [lo[2], hi[2]],
                    color=col, alpha=0.35, lw=3)

    ax.set_xlabel("East (km)", color=TC, fontsize=9)
    ax.set_ylabel("North (km)", color=TC, fontsize=9)
    ax.set_zlabel("Altitude (km)", color=TC, fontsize=9)
    ax.set_title("SkySentinel — LSTM Trajectory Prediction", color=TC,
                 fontsize=12, fontweight="bold")
    ax.tick_params(colors=TC, labelsize=7)
    ax.xaxis.pane.fill = False;  ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.legend(loc="upper left", fontsize=8, facecolor=BG, labelcolor=TC)

    out = "trajectory_predictor_preview.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"\n  Preview saved → {out}")
    print("=" * 64)