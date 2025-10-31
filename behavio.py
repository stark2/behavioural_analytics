# behavio.py
# Dependencies: pip install numpy pandas joblib
"""
Purpose
-------
Behavioral biometrics profiling per (username, contract).
Train: build a per-user profile from historical sessions.
Eval : score new sessions against the corresponding user profile in [0, 1].

Model
-----
1-NN in a per-user standardized feature space (mean/std computed from that user's sessions).
Score = exp( - d2_min / (k * d2_ref) ), where
  d2_min = min squared Euclidean distance to any training vector of this user
  d2_ref = robust scale = median pairwise squared distance among this user's training vectors
  k      = slope control (>1 => slower decay, more forgiving)
Property: exact byte-identical training row -> distance 0 -> score 1.0.

Inputs
------
CSV with at least:
  username, contract, mouseTrace, keystrokes
Optional numeric aggregates (kept for output compatibility):
  mouseAverageVelocity, mouseAverageAcceleration, mouseTotalMovement,
  averageDwellTime, averageTypingSpeed
Other columns are ignored for modeling. windowSize is used only to normalize coordinates.

Outputs
-------
Training saves, per user:
  <key>.scaler.joblib  : {"mean": mu, "std": sd, "d2_ref": ..., "feature_order": [...]}
  <key>.train.npz      : Xs (training matrix in standardized space)
  <key>.json           : {"key": ..., "n_sessions": ...}
  _index.json          : global index

Evaluation:
  --out-csv            : compact CSV with ID, similarity_score, base aggregates
  --out-features-csv   : full engineered features per row + internals (d2_min, d2_ref, k)
"""

import argparse
import json
import math
import os
import re
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from joblib import dump, load

# --------------------- Identifiers and base numeric columns ---------------------

ID_COLS = ["username", "contract"]

# Base aggregates that we keep for output compatibility. They are NOT required for modeling.
BASIC_NUM_COLS = [
    "mouseAverageVelocity",
    "mouseAverageAcceleration",
    "mouseTotalMovement",
    "averageDwellTime",
    "averageTypingSpeed",
]

# Fixed feature order for modeling: base aggregates + engineered features.
FEATURE_ORDER = BASIC_NUM_COLS + [
    # Mouse features from mouseTrace
    "m_speed_mean", "m_speed_std", "m_speed_p95",
    "m_acc_mean", "m_acc_std", "m_acc_p95",
    "m_jerk_p95", "m_pause_frac", "m_straightness",
    "m_dir_change_rate", "m_ballistic_ratio", "m_sample_count",
    # Keystroke features from keystrokes
    "k_dwell_mean", "k_dwell_std", "k_dwell_p95",
    "k_flight_mean", "k_flight_std", "k_flight_p95",
    "k_overlap_ratio", "k_shift_ratio", "k_symbol_ratio",
    "k_speed_cps", "k_events",
]

# --------------------- Parsing utilities ---------------------

def _to_float(x) -> float:
    """Safe float conversion with NaN fallback."""
    try:
        return float(x)
    except Exception:
        return np.nan

def _json_load_relaxed(cell: Any) -> list:
    """
    Robust JSON loader for CSV cells that may be:
    - NaN / None / empty
    - double-escaped JSON (e.g., ""[ ... ]"")
    Returns a Python list or an empty list. Non-list objects are ignored.
    """
    if cell is None:
        return []
    if isinstance(cell, float) and math.isnan(cell):
        return []
    if isinstance(cell, list):
        return cell
    if isinstance(cell, dict):
        # Unexpected for these columns; ignore structurally different input.
        return []

    s = str(cell).strip()
    if not s:
        return []
    if s.lower() in ("nan", "none", "null"):
        return []

    # Try direct JSON
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, list) else []
    except Exception:
        pass

    # Try un-escaping CSV style ""..."" -> "..."
    try:
        s2 = s.replace('""', '"')
        obj = json.loads(s2)
        return obj if isinstance(obj, list) else []
    except Exception:
        return []

def _user_key(row: pd.Series) -> str:
    """Stable per-user key. Casting to str preserves leading zeros in contract."""
    return f"{str(row['username'])}||{str(row['contract'])}"

def _safe_div(a: float, b: float, eps: float = 1e-9) -> float:
    """Numerically safe division that avoids division by zero."""
    b = b if abs(b) > eps else (eps if b >= 0 else -eps)
    return a / b

# --------------------- Mouse features from mouseTrace ---------------------

def extract_mouse_features(row: pd.Series) -> Dict[str, float]:
    """
    Engineer robust mouse features from raw (x,y,t) trace:
      - normalize x,y by window size to reduce device dependence
      - compute speed/acc/jerk percentiles
      - pause and ballistic ratios by speed thresholds
      - straightness and direction-change rate
      - sample count
    """
    # Parse "WxH" window size; default to 1x1 if invalid or missing
    w_str = str(row.get("windowSize", "")).lower().replace(" ", "")
    try:
        wx, wy = map(float, w_str.split("x"))
        if wx <= 0 or wy <= 0:
            wx, wy = 1.0, 1.0
    except Exception:
        wx, wy = 1.0, 1.0

    trace = _json_load_relaxed(row.get("mouseTrace", "[]"))
    if not trace:
        return {
            "m_speed_mean": 0.0, "m_speed_std": 0.0, "m_speed_p95": 0.0,
            "m_acc_mean": 0.0, "m_acc_std": 0.0, "m_acc_p95": 0.0,
            "m_jerk_p95": 0.0, "m_pause_frac": 1.0,
            "m_straightness": 0.0, "m_dir_change_rate": 0.0,
            "m_ballistic_ratio": 0.0, "m_sample_count": 0.0,
        }

    xs, ys, ts = [], [], []
    last_t = None
    for p in trace:
        try:
            x = float(p.get("x", 0.0)) / wx
            y = float(p.get("y", 0.0)) / wy
            t = float(p.get("timestamp", 0.0)) / 1000.0  # seconds
        except Exception:
            continue
        # Accept strictly increasing timestamps only
        if last_t is None or t > last_t:
            xs.append(x); ys.append(y); ts.append(t)
            last_t = t

    n = len(ts)
    if n < 3:
        # Not enough dynamics; emit zeros but keep sample_count
        return {
            "m_speed_mean": 0.0, "m_speed_std": 0.0, "m_speed_p95": 0.0,
            "m_acc_mean": 0.0, "m_acc_std": 0.0, "m_acc_p95": 0.0,
            "m_jerk_p95": 0.0, "m_pause_frac": 1.0,
            "m_straightness": 0.0, "m_dir_change_rate": 0.0,
            "m_ballistic_ratio": 0.0, "m_sample_count": float(n),
        }

    xs = np.asarray(xs); ys = np.asarray(ys); ts = np.asarray(ts)
    dx = np.diff(xs); dy = np.diff(ys); dt = np.diff(ts)
    dt[dt <= 0] = 1e-6

    # Speed in screen-fractions per second
    speed = np.hypot(dx, dy) / dt
    # Acceleration and jerk
    acc = np.diff(speed) / dt[1:] if len(speed) > 1 else np.array([], dtype=float)
    jerk = np.diff(acc) / dt[2:] if len(acc) > 1 else np.array([], dtype=float)

    # Direction-change rate (unwrapped angle differences per second)
    ang = np.arctan2(dy, dx)
    ang_u = np.unwrap(ang)
    d_ang = np.diff(ang_u)
    dir_change_rate = float(np.mean(np.abs(d_ang) / dt[1:])) if len(dt) > 1 else 0.0

    # Pause and ballistic time fractions by speed thresholds
    S_pause = 0.05  # near-still movement
    S_fast  = 0.5   # fast ballistic movement
    time_total = ts[-1] - ts[0]
    time_pause = float(np.sum(dt[speed < S_pause])) if len(speed) else 0.0
    time_fast  = float(np.sum(dt[speed > S_fast])) if len(speed) else 0.0

    # Path length and straightness
    path_len = float(np.sum(np.hypot(dx, dy)))
    disp = float(np.hypot(xs[-1] - xs[0], ys[-1] - ys[0]))
    straightness = _safe_div(disp, path_len) if path_len > 0 else 0.0

    def pct(a: np.ndarray, q: float) -> float:
        return float(np.percentile(a, q)) if a.size else 0.0

    return {
        "m_speed_mean": float(np.mean(speed)),
        "m_speed_std":  float(np.std(speed)),
        "m_speed_p95":  pct(speed, 95),
        "m_acc_mean":   float(np.mean(acc)) if acc.size else 0.0,
        "m_acc_std":    float(np.std(acc))  if acc.size else 0.0,
        "m_acc_p95":    pct(acc, 95),
        "m_jerk_p95":   pct(jerk, 95),
        "m_pause_frac": _safe_div(time_pause, time_total) if time_total > 0 else 1.0,
        "m_straightness": straightness,
        "m_dir_change_rate": dir_change_rate,
        "m_ballistic_ratio": _safe_div(time_fast, time_total) if time_total > 0 else 0.0,
        "m_sample_count": float(n),
    }

# --------------------- Keystroke features from keystrokes ---------------------

def extract_keystroke_features(row: pd.Series) -> Dict[str, float]:
    """
    Engineer robust keyboard features:
      - dwell/flight mean/std/p95
      - overlap ratio (flight<0), shift/symbol ratios
      - typing speed (chars/sec) and event count
    """
    ks = _json_load_relaxed(row.get("keystrokes", "[]"))
    if not ks:
        return {
            "k_dwell_mean": 0.0, "k_dwell_std": 0.0, "k_dwell_p95": 0.0,
            "k_flight_mean": 0.0, "k_flight_std": 0.0, "k_flight_p95": 0.0,
            "k_overlap_ratio": 0.0, "k_shift_ratio": 0.0, "k_symbol_ratio": 0.0,
            "k_speed_cps": 0.0, "k_events": 0.0,
        }

    dwell, flight, keys, times = [], [], [], []
    for e in ks:
        try:
            dwell.append(float(e.get("dwellTime", 0.0)))
            flight.append(float(e.get("flightTime", 0.0)))
            keys.append(str(e.get("key", "")))
            times.append(float(e.get("timestamp", 0.0)) / 1000.0)
        except Exception:
            continue

    dwell = np.asarray(dwell) if len(dwell) else np.array([0.0])
    flight = np.asarray(flight) if len(flight) else np.array([0.0])
    times = np.asarray(times) if len(times) else np.array([0.0, 0.0])
    T = float(times[-1] - times[0]) if times.size > 1 else 0.0

    shift_mask = np.array([k == "Shift" for k in keys], dtype=bool)

    def is_symbol(k: str) -> bool:
        # Treat alphabetic single chars as non-symbols; others as symbols (except Shift and space).
        if k in ("Shift", " "):
            return False
        return not (len(k) == 1 and k.isalpha())

    symbol_mask = np.array([is_symbol(k) for k in keys], dtype=bool)

    def pct(a: np.ndarray, q: float) -> float:
        return float(np.percentile(a, q)) if a.size else 0.0

    return {
        "k_dwell_mean": float(np.mean(dwell)),
        "k_dwell_std":  float(np.std(dwell)),
        "k_dwell_p95":  pct(dwell, 95),
        "k_flight_mean": float(np.mean(flight)),
        "k_flight_std":  float(np.std(flight)),
        "k_flight_p95":  pct(flight, 95),
        "k_overlap_ratio": float(np.mean(flight < 0.0)),
        "k_shift_ratio":   float(np.mean(shift_mask)) if len(keys) else 0.0,
        "k_symbol_ratio":  float(np.mean(symbol_mask)) if len(keys) else 0.0,
        "k_speed_cps":     _safe_div(len(keys), T) if T > 0 else 0.0,
        "k_events":        float(len(keys)),
    }

# --------------------- Feature assembly ---------------------

def row_to_feature_dict(row: pd.Series) -> Dict[str, float]:
    """Assemble full feature dict from a single CSV row."""
    # Base aggregates (if present in CSV).
    base = {c: _to_float(row.get(c, 0.0)) for c in BASIC_NUM_COLS}
    # Engineered features
    m = extract_mouse_features(row)
    k = extract_keystroke_features(row)
    base.update(m)
    base.update(k)
    # Sanitize NaN/Inf
    for kf, vf in list(base.items()):
        if vf is None or (isinstance(vf, float) and (math.isnan(vf) or math.isinf(vf))):
            base[kf] = 0.0
    return base

def build_feature_vector(row: pd.Series, feature_order: List[str]) -> np.ndarray:
    """Build a numpy vector following the fixed feature order."""
    feats = row_to_feature_dict(row)
    return np.array([feats.get(name, 0.0) for name in feature_order], dtype=np.float64)

def build_feature_matrix_and_df(df_sub: pd.DataFrame, feature_order: List[str]) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Build:
      - X: feature matrix with rows in the given order
      - feats_df: DataFrame with all engineered feature columns (for transparency export)
    """
    dicts: List[Dict[str, float]] = []
    rows: List[np.ndarray] = []
    for _, row in df_sub.iterrows():
        d = row_to_feature_dict(row)
        dicts.append(d)
        rows.append(np.array([d.get(name, 0.0) for name in feature_order], dtype=np.float64))
    X = np.vstack(rows) if rows else np.zeros((0, len(feature_order)), dtype=np.float64)
    feats_df = pd.DataFrame(dicts)
    return X, feats_df

# --------------------- Per-user standardization ---------------------

class Scaler:
    """Simple per-user standardizer (mean/std)."""
    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = mean.astype(np.float64)
        self.std = std.astype(np.float64)
        self.std[self.std == 0] = 1e-6

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean[None, :]) / self.std[None, :]

def fit_scaler(X: np.ndarray) -> Scaler:
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1e-6
    return Scaler(mu, sd)

# --------------------- Distances and scoring ---------------------

def pairwise_min_sqdist(x: np.ndarray, Xs: np.ndarray) -> float:
    """Return min ||x - Xi||^2 over all user's training vectors Xi (in standardized space)."""
    diffs = Xs - x[None, :]
    d2 = np.sum(diffs * diffs, axis=1)
    return float(np.min(d2))

def median_pairwise_d2(Xs: np.ndarray) -> float:
    """Median pairwise squared distance among training vectors (robust scale)."""
    n = Xs.shape[0]
    if n <= 1:
        return 1.0
    acc = []
    for i in range(n):
        for j in range(i + 1, n):
            diff = Xs[i] - Xs[j]
            acc.append(float(np.dot(diff, diff)))
    m = np.median(acc) if acc else 1.0
    return float(max(m, 1e-6))

def score_from_distance(d2: float, d2_ref: float, k: float) -> float:
    """Map distance to [0,1] exponentially; d2=0 -> 1.0. k tunes the slope."""
    denom = max(1e-9, k * max(1e-9, d2_ref))
    return float(np.exp(- d2 / denom))

# --------------------- CSV IO ---------------------

def load_sessions_csv(csv_path: str) -> pd.DataFrame:
    """
    Load CSV and normalize essential columns.
    - Keep ID columns as strings to preserve leading zeros.
    - Provide safe defaults for JSON columns (empty string -> []).
    - Cast base numeric columns to float (fill NaN with 0).
    """
    df = pd.read_csv(csv_path)

    # IDs must exist and be strings
    for c in ID_COLS:
        if c not in df.columns:
            raise KeyError(f"Missing identifier column '{c}' in {csv_path}")
        df[c] = df[c].astype(str)

    # Ensure raw JSON columns exist and have empty strings where missing
    if "mouseTrace" not in df.columns:
        df["mouseTrace"] = ""
    else:
        df["mouseTrace"] = df["mouseTrace"].where(df["mouseTrace"].notna(), "")

    if "keystrokes" not in df.columns:
        df["keystrokes"] = ""
    else:
        df["keystrokes"] = df["keystrokes"].where(df["keystrokes"].notna(), "")

    # Base numeric columns: convert if present
    for c in BASIC_NUM_COLS:
        if c in df.columns:
            df[c] = df[c].apply(_to_float).fillna(0.0)

    # Composite user key
    df["_user_key"] = df.apply(_user_key, axis=1)
    return df

# --------------------- Persistence ---------------------

def _safe_name(key: str) -> str:
    """Filesystem-safe stem for a given user key."""
    return re.sub(r'[\\/:"<>|*?]+', "_", key)

def save_user_artifacts(models_dir: str, key: str, scaler: Scaler, Xs: np.ndarray,
                        d2_ref: float, feature_order: List[str]) -> None:
    """Save per-user scaler, training matrix in standardized space, and metadata."""
    os.makedirs(models_dir, exist_ok=True)
    safe = _safe_name(key)

    dump({"mean": scaler.mean, "std": scaler.std, "d2_ref": d2_ref, "feature_order": feature_order},
         os.path.join(models_dir, f"{safe}.scaler.joblib"))

    np.savez_compressed(os.path.join(models_dir, f"{safe}.train.npz"), Xs=Xs)

    with open(os.path.join(models_dir, f"{safe}.json"), "w", encoding="utf-8") as f:
        json.dump({"key": key, "n_sessions": int(Xs.shape[0])}, f, ensure_ascii=False)

def load_user_artifacts(models_dir: str, key: str) -> Tuple["Scaler", np.ndarray, Dict[str, Any], float, List[str]]:
    """Load per-user scaler, training matrix, metadata, and feature order."""
    safe = _safe_name(key)
    scaler_path = os.path.join(models_dir, f"{safe}.scaler.joblib")
    train_path = os.path.join(models_dir, f"{safe}.train.npz")
    meta_path = os.path.join(models_dir, f"{safe}.json")
    if not (os.path.exists(scaler_path) and os.path.exists(train_path) and os.path.exists(meta_path)):
        raise FileNotFoundError(f"artifacts missing for '{key}' in {models_dir}")

    sc = load(scaler_path)
    Xs = np.load(train_path)["Xs"]
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    scaler = Scaler(np.array(sc["mean"]), np.array(sc["std"]))
    d2_ref = float(sc["d2_ref"])
    feature_order = list(sc.get("feature_order", FEATURE_ORDER))
    return scaler, Xs, meta, d2_ref, feature_order

# --------------------- Train and evaluate ---------------------

def train(models_dir: str, csv_path: str) -> None:
    """Train a profile per user and save artifacts."""
    df = load_sessions_csv(csv_path)
    keys = []
    for key, g in df.groupby("_user_key"):
        # Build feature matrix (rows = sessions)
        X_list = [build_feature_vector(row, FEATURE_ORDER) for _, row in g.iterrows()]
        if not X_list:
            continue
        X = np.vstack(X_list)

        # Per-user standardizer and standardized matrix
        scaler = fit_scaler(X)
        Xs = scaler.transform(X)

        # Robust scale
        d2_ref = median_pairwise_d2(Xs)

        # Persist
        save_user_artifacts(models_dir, key, scaler, Xs, d2_ref, FEATURE_ORDER)
        keys.append(key)

    with open(os.path.join(models_dir, "_index.json"), "w", encoding="utf-8") as f:
        json.dump({"models": sorted(keys), "feature_order": FEATURE_ORDER}, f, ensure_ascii=False)

    print(f"trained {len(keys)} NN profiles → {models_dir}")

def evaluate(models_dir: str, csv_path: str, username: str, contract: str,
             out_csv: str = None, k: float = 1.0, out_features_csv: str = None) -> None:
    """
    Score new sessions of (username, contract) against the saved profile.
    Writes:
      - out_csv: compact output, backward compatible
      - out_features_csv: full engineered features + d2_min, d2_ref, k (for transparency)
    """
    df = load_sessions_csv(csv_path)
    key = f"{str(username)}||{str(contract)}"

    scaler, Xs_train, meta, d2_ref, feature_order = load_user_artifacts(models_dir, key)

    sub = df[df["_user_key"] == key].copy()
    if sub.empty:
        raise SystemExit(f"no rows for ({username}, {contract}) in {csv_path}")

    # Full feature matrix for eval + a DataFrame with all engineered features
    X_raw, features_df = build_feature_matrix_and_df(sub, feature_order)
    Xs_eval = scaler.transform(X_raw)

    d2_list, scores = [], []
    for x in Xs_eval:
        d2 = pairwise_min_sqdist(x, Xs_train)
        s = score_from_distance(d2, d2_ref, k)
        d2_list.append(d2)
        scores.append(s)

    sub["similarity_score"] = scores

    # Compact output (ID + score + base aggregates if present)
    out_cols = ID_COLS + ["similarity_score"]
    for c in BASIC_NUM_COLS:
        if c in sub.columns:
            out_cols.append(c)
    out_compact = sub[out_cols]

    if out_csv:
        out_compact.to_csv(out_csv, index=False)
        print(f"written: {out_csv}")
    else:
        # If no file is requested, print to stdout
        print(out_compact.to_csv(index=False), end="")

    # Full features dump with internals for verification that all data are used
    if out_features_csv:
        full = sub[ID_COLS].reset_index(drop=True).join(features_df)
        full["similarity_score"] = scores
        full["d2_min"] = d2_list
        full["d2_ref"] = d2_ref
        full["k"] = k
        full.to_csv(out_features_csv, index=False)
        print(f"written: {out_features_csv}")

# --------------------- CLI ---------------------

def main():
    ap = argparse.ArgumentParser(prog="behavio.py")
    sub = ap.add_subparsers(dest="mode", required=True)

    ap_tr = sub.add_parser("train", help="build and save NN (1-NN) profiles")
    ap_tr.add_argument("--models-dir", required=True)
    ap_tr.add_argument("--csv", required=True)

    ap_ev = sub.add_parser("eval", help="score new sessions for a given user/contract")
    ap_ev.add_argument("--models-dir", required=True)
    ap_ev.add_argument("--csv", required=True)
    ap_ev.add_argument("--username", required=True)
    ap_ev.add_argument("--contract", required=True)
    ap_ev.add_argument("--out-csv")
    ap_ev.add_argument("--k", type=float, default=1.0, help="slope; higher -> slower decay (more forgiving)")
    ap_ev.add_argument("--out-features-csv", help="dump all engineered features and internals for transparency")

    args = ap.parse_args()
    if args.mode == "train":
        train(args.models_dir, args.csv)
    else:
        evaluate(args.models_dir, args.csv, args.username, args.contract, args.out_csv, args.k, args.out_features_csv)

if __name__ == "__main__":
    main()