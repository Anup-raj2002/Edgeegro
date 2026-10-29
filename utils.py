import numpy as np
import os
from datetime import datetime
import pandas as pd
from config import SMOOTH_ALPHA, CSV_LOG_FILE, TRAIN_CSV

_smooth_state = {}

def dist(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))

def nearest(cands, ref, max_dist):
    if ref is None: return None
    best, best_d = None, max_dist
    for c in cands:
        d = dist(c, ref)
        if d < best_d:
            best, best_d = c, d
    return best

def calc_angle(a, b, c):
    a, b, c = (np.array(p[:2], dtype=np.float32) for p in (a, b, c))
    ba, bc = a - b, c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    return float(np.degrees(np.arccos(np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)))) if denom > 1e-8 else None

def angle_from_vertical(p_from, p_to, down=True):
    v = np.array([p_to[0] - p_from[0], p_to[1] - p_from[1]], dtype=np.float32)
    n = np.linalg.norm(v)
    if n < 1e-6: return None
    ref = np.array([0.0, 1.0 if down else -1.0], dtype=np.float32)
    return float(np.degrees(np.arccos(np.clip(np.dot(v, ref) / n, -1.0, 1.0))))

def shoulder_tilt(rs, ls):
    a, b = (rs, ls) if rs[0] < ls[0] else (ls, rs)
    return abs(float(np.degrees(np.arctan2(b[1] - a[1], b[0] - a[0]))))

def smooth(name, value):
    if value is None:
        _smooth_state.pop(name, None)
        return None
    prev = _smooth_state.get(name)
    new = value if prev is None else SMOOTH_ALPHA * value + (1 - SMOOTH_ALPHA) * prev
    _smooth_state[name] = new
    return new

def log_session(r, selected_device, infer_ms):
    row = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Device": selected_device, "Infer_ms": round(infer_ms, 2),
        "Verdict": r["label"], "RULA_Score": round(r["grand"], 2),
        "Score_A": round(r["A"], 2), "Score_B": round(r["B"], 2),
        "Neck": r["neck"], "Trunk": r["trunk"], "Upper_Arm": r["ua"], "Lower_Arm": r["la"],
        "Neck_Flex_Deg": round(r["neck_deg"], 1),
        "Pct_Low": round(r["pct_low"], 1), "Pct_Medium": round(r["pct_med"], 1), "Pct_High": round(r["pct_high"], 1),
        "Valid_Frames": r["valid_frames"], "Total_Frames": r["total_frames"],
    }
    pd.DataFrame([row]).to_csv(CSV_LOG_FILE, mode="a", header=not os.path.exists(CSV_LOG_FILE), index=False)

def save_train_rows(train_rows):
    if not train_rows: return
    pd.DataFrame(train_rows).to_csv(TRAIN_CSV, mode="a", header=not os.path.exists(TRAIN_CSV), index=False)
    print(f"Saved {len(train_rows)} labelled frames to {TRAIN_CSV}")
    train_rows.clear()