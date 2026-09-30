"""
TrafficNet: one small CNN-LSTM with three heads, fed by scale-free scene features.

Input : window of the last WINDOW seconds of features sampled at 1 Hz (N_FEATURES per step)
Output: (1) density score H seconds ahead   (0-1, multiply by 100)
        (2) clearance time in seconds       (0-1, multiply by MAX_CLEAR)
        (3) incident logit                  (sigmoid -> probability)

Every feature is a ratio (relative to the scene's own adaptive baseline), so nothing
depends on pixel size or absolute camera geometry. That is what makes the model
transferable across camera angles.

Train:
    python trafficnet.py --csv logs/*.csv --out trafficnet.pt
(the CSVs are produced by:  python live_traffic.py --source video.mp4 --log logs/a.csv --no-show)
"""
import argparse
import glob

import numpy as np

WINDOW = 30          # seconds of history
HORIZON = 30         # seconds ahead for density forecast
MAX_CLEAR = 120.0    # clearance time cap (s)
FEATURES = ["score", "crowd", "occ_rel", "stopped_ratio", "speed_rel",
            "depart_rel", "arrive_rel", "incident_rule"]
N_FEATURES = len(FEATURES)


def build_model():
    import torch.nn as nn

    class TrafficNet(nn.Module):
        def __init__(self, n_features=N_FEATURES, hidden=64):
            super().__init__()
            self.cnn = nn.Sequential(
                nn.Conv1d(n_features, 32, 3, padding=1), nn.ReLU(),
                nn.Conv1d(32, 64, 3, padding=1), nn.ReLU(),
            )
            self.lstm = nn.LSTM(64, hidden, num_layers=2, batch_first=True, dropout=0.2)
            self.trunk = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU())
            self.h_score = nn.Linear(32, 1)
            self.h_clear = nn.Linear(32, 1)
            self.h_incident = nn.Linear(32, 1)

        def forward(self, x):                      # x: (B, T, F)
            z = self.cnn(x.transpose(1, 2)).transpose(1, 2)
            out, _ = self.lstm(z)
            h = self.trunk(out[:, -1])
            return (self.h_score(h).squeeze(-1),
                    self.h_clear(h).squeeze(-1),
                    self.h_incident(h).squeeze(-1))

    return TrafficNet()


# --------------------------------------------------------------------------- labels
def make_dataset(df, clear_thr=30.0, congested_thr=40.0):
    """Self-supervised labels from the recorded feature stream (no manual annotation)."""
    X = df[FEATURES].to_numpy(dtype=np.float32).copy()
    score = df["score"].to_numpy(dtype=np.float32)   # 0-100
    X[:, 0] /= 100.0                                 # live pipeline feeds score/100 too
    inc = df["incident_rule"].to_numpy(dtype=np.float32)
    n = len(df)
    xs, y_score, y_clear, y_inc = [], [], [], []
    for t in range(WINDOW, n - HORIZON):
        xs.append(X[t - WINDOW:t])
        y_score.append(score[t + HORIZON] / 100.0)
        if score[t] >= congested_thr:
            nxt = np.where(score[t:] < clear_thr)[0]
            clear = float(nxt[0]) if len(nxt) else MAX_CLEAR
        else:
            clear = 0.0
        y_clear.append(min(clear, MAX_CLEAR) / MAX_CLEAR)
        y_inc.append(float(inc[t:t + 10].max()))
    return (np.stack(xs), np.array(y_score, np.float32),
            np.array(y_clear, np.float32), np.array(y_inc, np.float32))


def train(csv_paths, out, epochs=60, lr=1e-3, device=None):
    import pandas as pd
    import torch
    import torch.nn as nn

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    per_video = []
    for p in csv_paths:
        df = pd.read_csv(p)
        if len(df) > WINDOW + HORIZON + 10:
            per_video.append(make_dataset(df))
        else:
            print(f"skip {p}: too short ({len(df)} rows)")
    if not per_video:
        raise SystemExit("no usable CSVs")

    # hold out the last video (different scene/angle) if >1 video: honest cross-angle check
    if len(per_video) > 1:
        train_sets, val_set = per_video[:-1], per_video[-1]
        print(f"validating on held-out video: {csv_paths[-1]}")
    else:
        d = per_video[0]
        cut = int(len(d[0]) * 0.8)
        train_sets = [tuple(a[:cut] for a in d)]
        val_set = tuple(a[cut:] for a in d)
        print("only one video: chronological 80/20 split (does NOT test angle independence)")

    cat = lambda i: np.concatenate([s[i] for s in train_sets])
    Xtr, ys, yc, yi = (torch.tensor(cat(i)) for i in range(4))
    Xva, vys, vyc, vyi = (torch.tensor(a) for a in val_set)

    model = build_model().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, 20, 0.5)
    huber, bce = nn.HuberLoss(), nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([max(1.0, float((1 - yi.mean()) / max(yi.mean(), 1e-3)))]).to(device))

    best = 1e9
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(perm), 64):
            idx = perm[i:i + 64]
            xb = Xtr[idx].to(device)
            # small noise augmentation -> robustness to scene-to-scene differences
            xb = xb + 0.02 * torch.randn_like(xb)
            ps, pc, pi = model(xb)
            loss = huber(ps, ys[idx].to(device)) + huber(pc, yc[idx].to(device)) \
                + 0.5 * bce(pi, yi[idx].to(device))
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            ps, pc, _ = model(Xva.to(device))
            mae_s = (ps.cpu() - vys).abs().mean().item() * 100
            mae_c = (pc.cpu() - vyc).abs().mean().item() * MAX_CLEAR
        if mae_s + mae_c < best:
            best = mae_s + mae_c
            torch.save({"state": model.state_dict(), "window": WINDOW, "features": FEATURES}, out)
        if ep % 10 == 0 or ep == epochs - 1:
            print(f"epoch {ep:3d}  val MAE: density {mae_s:5.2f} pts | clearance {mae_c:6.2f} s")
    print(f"saved best model -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", nargs="+", required=True, help="feature logs (globs ok)")
    ap.add_argument("--out", default="trafficnet.pt")
    ap.add_argument("--epochs", type=int, default=60)
    a = ap.parse_args()
    paths = sorted(sum((glob.glob(p) for p in a.csv), []))
    train(paths, a.out, a.epochs)
