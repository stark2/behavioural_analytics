# behavio_server.py
# pip install fastapi uvicorn numpy pandas joblib pydantic

from typing import Any, Dict, List, Optional, Union
import json, math, os, re, logging, glob

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from joblib import dump, load

# ---------- config/logging ----------
DEFAULT_MODELS_DIR = os.getenv("MODELS_DIR", "./models")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ---------- identifiers / features ----------
ID_COLS = ["username", "contract"]

BASIC_NUM_COLS = [
    "mouseAverageVelocity",
    "mouseAverageAcceleration",
    "mouseTotalMovement",
    "averageDwellTime",
    "averageTypingSpeed",
]

FEATURE_ORDER = BASIC_NUM_COLS + [
    "m_speed_mean", "m_speed_std", "m_speed_p95",
    "m_acc_mean", "m_acc_std", "m_acc_p95",
    "m_jerk_p95", "m_pause_frac", "m_straightness",
    "m_dir_change_rate", "m_ballistic_ratio", "m_sample_count",
    "k_dwell_mean", "k_dwell_std", "k_dwell_p95",
    "k_flight_mean", "k_flight_std", "k_flight_p95",
    "k_overlap_ratio", "k_shift_ratio", "k_symbol_ratio",
    "k_speed_cps", "k_events",
]

# Accept messy CSV numerics as float|int|string
Numeric = Union[float, int, str]

# ---------- pydantic models ----------
class Session(BaseModel):
    username: str
    contract: str

    password: Optional[str] = None

    # Relax numeric inputs to avoid 422 on messy CSV
    mouseAverageVelocity: Optional[Numeric] = None
    mouseAverageAcceleration: Optional[Numeric] = None
    mouseTotalMovement: Optional[Numeric] = None
    averageDwellTime: Optional[Numeric] = None
    averageTypingSpeed: Optional[Numeric] = None

    mouseTrace: Optional[Any] = None
    keystrokes: Optional[Any] = None
    device: Optional[str] = None
    browser: Optional[str] = None
    screenResolution: Optional[str] = None
    windowSize: Optional[str] = None
    timezone: Optional[str] = None

    @field_validator("username", "contract", mode="before")
    @classmethod
    def _to_str(cls, v):
        if v is None:
            raise ValueError("username/contract required")
        return str(v).strip()

class TrainRequest(BaseModel):
    models_dir: str = Field(default=DEFAULT_MODELS_DIR)
    sessions: List[Session]

class TrainResponse(BaseModel):
    models_dir: str
    trained_profiles: int
    keys: List[str]

class EvalRequest(BaseModel):
    models_dir: str = Field(default=DEFAULT_MODELS_DIR)
    username: str
    contract: str
    sessions: List[Session]
    k: float = 1.0
    explain: bool = False
    topN: int = 8

class TopFeature(BaseModel):
    feature: str
    contrib: float
    share: float

class Explanation(BaseModel):
    nn_index: int
    top_features: List[TopFeature]
    delta_raw: Optional[Dict[str, float]] = None

class EvalResponse(BaseModel):
    models_dir: str
    key: str
    compact: List[Dict[str, Any]]
    features: List[Dict[str, Any]]
    explanations: Optional[List[Explanation]] = None

class ModelsListResponse(BaseModel):
    models_dir: str
    keys: List[str]

# ---------- utils ----------
def _to_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return np.nan

def _json_load_relaxed(cell: Any) -> list:
    """Parse list from various encodings, return [] on failure."""
    if cell is None:
        return []
    if isinstance(cell, float) and math.isnan(cell):
        return []
    if isinstance(cell, list):
        return cell
    if isinstance(cell, dict):
        return []
    s = str(cell).strip()
    if not s or s.lower() in ("nan", "none", "null"):
        return []
    # try plain
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, list) else []
    except Exception:
        pass
    # try de-escaped CSV JSON
    try:
        s2 = s.replace('""', '"')
        obj = json.loads(s2)
        return obj if isinstance(obj, list) else []
    except Exception:
        return []

def _user_key_from_row(row: pd.Series) -> str:
    return f"{str(row['username'])}||{str(row['contract'])}"

def _user_key_from_strings(username: str, contract: str) -> str:
    return f"{str(username)}||{str(contract)}"

def _safe_div(a: float, b: float, eps: float = 1e-9) -> float:
    b = b if abs(b) > eps else (eps if b >= 0 else -eps)
    return a / b

def _safe_name(key: str) -> str:
    return re.sub(r'[\\/:"<>|*?]+', "_", key)

# ---------- feature engineering ----------
def extract_mouse_features(row: pd.Series) -> Dict[str, float]:
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
            t = float(p.get("timestamp", 0.0)) / 1000.0
        except Exception:
            continue
        if last_t is None or t > last_t:
            xs.append(x); ys.append(y); ts.append(t)
            last_t = t

    n = len(ts)
    if n < 3:
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

    speed = np.hypot(dx, dy) / dt
    acc = np.diff(speed) / dt[1:] if len(speed) > 1 else np.array([], dtype=float)
    jerk = np.diff(acc) / dt[2:] if len(acc) > 1 else np.array([], dtype=float)

    ang_u = np.unwrap(np.arctan2(dy, dx))
    d_ang = np.diff(ang_u)
    dir_change_rate = float(np.mean(np.abs(d_ang) / dt[1:])) if len(dt) > 1 else 0.0

    S_pause, S_fast = 0.05, 0.5
    time_total = ts[-1] - ts[0]
    time_pause = float(np.sum(dt[speed < S_pause])) if len(speed) else 0.0
    time_fast  = float(np.sum(dt[speed > S_fast])) if len(speed) else 0.0

    path_len = float(np.sum(np.hypot(dx, dy)))
    disp = float(np.hypot(xs[-1] - xs[0], ys[-1] - ys[0]))
    straightness = _safe_div(disp, path_len) if path_len > 0 else 0.0

    def pct(a, q): return float(np.percentile(a, q)) if a.size else 0.0

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

def extract_keystroke_features(row: pd.Series) -> Dict[str, float]:
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
        if k in ("Shift", " "): return False
        return not (len(k) == 1 and k.isalpha())
    symbol_mask = np.array([is_symbol(k) for k in keys], dtype=bool)

    def pct(a, q): return float(np.percentile(a, q)) if a.size else 0.0

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

def row_to_feature_dict(row: pd.Series) -> Dict[str, float]:
    base = {c: _to_float(row.get(c, 0.0)) for c in BASIC_NUM_COLS}
    base.update(extract_mouse_features(row))
    base.update(extract_keystroke_features(row))
    for k, v in list(base.items()):
        if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
            base[k] = 0.0
    return base

def build_feature_vector(row: pd.Series, feature_order: List[str]) -> np.ndarray:
    d = row_to_feature_dict(row)
    return np.array([d.get(name, 0.0) for name in feature_order], dtype=np.float64)

def build_feature_matrix_and_df(df_sub: pd.DataFrame, feature_order: List[str]):
    dicts, rows = [], []
    for _, row in df_sub.iterrows():
        d = row_to_feature_dict(row)
        dicts.append(d)
        rows.append(np.array([d.get(name, 0.0) for name in feature_order], dtype=np.float64))
    X = np.vstack(rows) if rows else np.zeros((0, len(feature_order)), dtype=np.float64)
    feats_df = pd.DataFrame(dicts)
    return X, feats_df

# ---------- standardization / scoring ----------
class Scaler:
    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = mean.astype(np.float64)
        self.std = std.astype(np.float64)
        self.std[self.std == 0] = 1e-6
    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean[None, :]) / self.std[None, :]

def fit_scaler(X: np.ndarray) -> Scaler:
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd[sd == 0] = 1e-6
    return Scaler(mu, sd)

def pairwise_min_sqdist(x: np.ndarray, Xs: np.ndarray) -> float:
    diffs = Xs - x[None, :]
    d2 = np.sum(diffs * diffs, axis=1)
    return float(np.min(d2))

def median_pairwise_d2(Xs: np.ndarray) -> float:
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
    denom = max(1e-9, k * max(1e-9, d2_ref))
    return float(np.exp(- d2 / denom))

# ---------- JSON -> DataFrame ----------
def sessions_to_dataframe(sessions: List[Session]) -> pd.DataFrame:
    dicts = [s.dict() for s in sessions]
    df = pd.DataFrame(dicts)
    for c in ID_COLS:
        if c not in df.columns:
            raise HTTPException(status_code=400, detail=f"Missing identifier column '{c}'")
        df[c] = df[c].astype(str)
    if "mouseTrace" not in df.columns: df["mouseTrace"] = ""
    else: df["mouseTrace"] = df["mouseTrace"].where(df["mouseTrace"].notna(), "")
    if "keystrokes" not in df.columns: df["keystrokes"] = ""
    else: df["keystrokes"] = df["keystrokes"].where(df["keystrokes"].notna(), "")
    # Coerce numeric columns; bad values -> NaN -> 0.0
    for c in BASIC_NUM_COLS:
        if c in df.columns: df[c] = df[c].apply(_to_float).fillna(0.0)
        else: df[c] = 0.0
    df["_user_key"] = df.apply(_user_key_from_row, axis=1)
    return df

# ---------- persistence ----------
def save_user_artifacts(models_dir: str, key: str, scaler: Scaler, Xs: np.ndarray,
                        d2_ref: float, feature_order: List[str]) -> None:
    os.makedirs(models_dir, exist_ok=True)
    safe = _safe_name(key)
    dump({"mean": scaler.mean, "std": scaler.std, "d2_ref": d2_ref, "feature_order": feature_order},
         os.path.join(models_dir, f"{safe}.scaler.joblib"))
    np.savez_compressed(os.path.join(models_dir, f"{safe}.train.npz"), Xs=Xs)
    with open(os.path.join(models_dir, f"{safe}.json"), "w", encoding="utf-8") as f:
        json.dump({"key": key, "n_sessions": int(Xs.shape[0])}, f, ensure_ascii=False)

def load_user_artifacts(models_dir: str, key: str):
    safe = _safe_name(key)
    scaler_path = os.path.join(models_dir, f"{safe}.scaler.joblib")
    train_path  = os.path.join(models_dir, f"{safe}.train.npz")
    meta_path   = os.path.join(models_dir, f"{safe}.json")
    if not (os.path.exists(scaler_path) and os.path.exists(train_path) and os.path.exists(meta_path)):
        raise HTTPException(status_code=404, detail=f"Artifacts missing for '{key}' in {models_dir}")
    sc = load(scaler_path)
    Xs = np.load(train_path)["Xs"]
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    scaler = Scaler(np.array(sc["mean"]), np.array(sc["std"]))
    d2_ref = float(sc["d2_ref"])
    feature_order = list(sc.get("feature_order", FEATURE_ORDER))
    return scaler, Xs, meta, d2_ref, feature_order

def update_index(models_dir: str, keys: List[str]) -> None:
    path = os.path.join(models_dir, "_index.json")
    existing = {"models": []}
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except Exception:
            existing = {"models": []}
    merged = sorted(set(existing.get("models", [])) | set(keys))
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"models": merged, "feature_order": FEATURE_ORDER}, f, ensure_ascii=False)

def list_model_keys(models_dir: str) -> List[str]:
    if not os.path.isdir(models_dir):
        return []
    keys: List[str] = []
    for meta_file in glob.glob(os.path.join(models_dir, "*.json")):
        if os.path.basename(meta_file) == "_index.json":
            continue
        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                meta = json.load(f)
            key = meta.get("key")
            if key:
                keys.append(key)
        except Exception:
            continue
    return sorted(set(keys))

# ---------- core train/eval ----------
def core_train(models_dir: str, df: pd.DataFrame) -> List[str]:
    keys = []
    for key, g in df.groupby("_user_key"):
        X_list = [build_feature_vector(row, FEATURE_ORDER) for _, row in g.iterrows()]
        if not X_list:
            continue
        X = np.vstack(X_list)
        scaler = fit_scaler(X)
        Xs = scaler.transform(X)
        d2_ref = median_pairwise_d2(Xs)
        save_user_artifacts(models_dir, key, scaler, Xs, d2_ref, FEATURE_ORDER)
        keys.append(key)
    if keys:
        update_index(models_dir, keys)
    return keys

def core_eval(models_dir: str, key: str, df_user_sessions: pd.DataFrame, k: float,
              explain: bool, topN: int):
    scaler, Xs_train, meta, d2_ref, feature_order = load_user_artifacts(models_dir, key)
    sub = df_user_sessions[df_user_sessions["_user_key"] == key].copy()
    if sub.empty:
        raise HTTPException(status_code=400, detail=f"No sessions for '{key}' in request")
    X_raw, features_df = build_feature_matrix_and_df(sub, feature_order)
    Xs_eval = scaler.transform(X_raw)

    d2_list, scores, nn_idx = [], [], []
    for xz in Xs_eval:
        diffs = Xs_train - xz[None, :]
        d2_vec = np.sum(diffs * diffs, axis=1)
        j_star = int(np.argmin(d2_vec))
        d2 = float(d2_vec[j_star])
        s = score_from_distance(d2, d2_ref, k)
        d2_list.append(d2); scores.append(s); nn_idx.append(j_star)

    compact = []
    for i in range(len(sub)):
        rec = {
            "username": sub.iloc[i]["username"],
            "contract": sub.iloc[i]["contract"],
            "similarity_score": float(scores[i]),
        }
        for c in BASIC_NUM_COLS:
            rec[c] = float(sub.iloc[i][c]) if c in sub.columns else 0.0
        compact.append(rec)

    features = []
    for i in range(len(sub)):
        row_feat = {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                    for k, v in features_df.iloc[i].to_dict().items()}
        row_feat.update({
            "username": sub.iloc[i]["username"],
            "contract": sub.iloc[i]["contract"],
            "similarity_score": float(scores[i]),
            "d2_min": float(d2_list[i]),
            "d2_ref": float(d2_ref),
            "k": float(k),
            "nn_index": int(nn_idx[i]),
        })
        features.append(row_feat)

    explanations: Optional[List[Explanation]] = None
    if explain:
        explanations = []
        for i, xz in enumerate(Xs_eval):
            j_star = nn_idx[i]
            comp = (xz - Xs_train[j_star]) ** 2
            d2 = float(np.sum(comp))
            order = np.argsort(comp)[::-1]
            top = []
            for idx in order[:max(1, int(topN))]:
                name = feature_order[idx]
                val = float(comp[idx])
                share = float(val / d2) if d2 > 0 else 0.0
                top.append(TopFeature(feature=name, contrib=val, share=share))
            x_train_raw = (Xs_train[j_star] * scaler.std) + scaler.mean
            delta_raw = {feature_order[idx]: float(X_raw[i, idx] - x_train_raw[idx])
                         for idx in range(len(feature_order))}
            explanations.append(Explanation(nn_index=j_star, top_features=top, delta_raw=delta_raw))

    return compact, features, explanations

# ---------- FastAPI ----------
app = FastAPI(title="Behavio Profiles API", version="1.3.0")

@app.on_event("startup")
def _startup_load_models():
    keys = list_model_keys(DEFAULT_MODELS_DIR)
    if not keys:
        logging.warning(f"No models found in '{DEFAULT_MODELS_DIR}'. Train first.")
    else:
        logging.info(f"Loaded {len(keys)} model(s) from '{DEFAULT_MODELS_DIR}'.")

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

@app.get("/models", response_model=ModelsListResponse)
def list_models(models_dir: str = DEFAULT_MODELS_DIR):
    keys = list_model_keys(models_dir)
    return ModelsListResponse(models_dir=models_dir, keys=keys)

@app.post("/train", response_model=TrainResponse)
def train_endpoint(req: TrainRequest):
    df = sessions_to_dataframe(req.sessions)
    if df.empty:
        raise HTTPException(status_code=400, detail="Empty sessions payload")
    keys = core_train(req.models_dir, df)
    if not keys:
        logging.warning("Training completed but no profiles were created.")
    return TrainResponse(models_dir=req.models_dir, trained_profiles=len(keys), keys=keys)

@app.post("/eval", response_model=EvalResponse)
def eval_endpoint(req: EvalRequest):
    df = sessions_to_dataframe(req.sessions)
    if df.empty:
        raise HTTPException(status_code=400, detail="Empty sessions payload")
    key = _user_key_from_strings(req.username, req.contract)
    compact, features, explanations = core_eval(
        req.models_dir, key, df, req.k, req.explain, req.topN
    )
    return EvalResponse(models_dir=req.models_dir, key=key,
                        compact=compact, features=features,
                        explanations=explanations)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("behavio_server2:app", host="0.0.0.0", port=8000, reload=False)