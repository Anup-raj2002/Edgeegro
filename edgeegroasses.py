import os
import sys
import time
from datetime import datetime

import cv2
import numpy as np
import openvino as ov
import openvino.properties as props
import pandas as pd


MODEL_PATH = "models/human-pose-estimation-0001/FP16/human-pose-estimation-0001.xml"
CAMERA_INDEX = 0


TARGET_DEVICE = sys.argv[1].upper() if len(sys.argv) > 1 else "CPU"
infer_ms = 0.0

RECORD_SECONDS = 10         
COUNTDOWN_SECONDS = 3         
CALIB_SECONDS = 3             
MIN_VALID_FRAC = 0.30        
HIGH_FRAC_ESCALATE = 0.40     

CONF_THRESH = 0.20
ROI_X_FRAC = (0.02, 0.98)     
MIN_SHOULDER_FRAC = 0.08
PANEL_W = 450
SMOOTH_ALPHA = 0.4            
CSV_LOG_FILE = "edgeergo_session_log.csv"


STATIC_POSTURE_BONUS = 1      
UPPER_ARM_RATIO = 0.55        
NECK_TOL = 0.93             


TRAIN_CSV = "edgeergo_train.csv"
MODEL_FILE = "risk_model.joblib"
FEATURES = ["neck_deg", "s", "drop", "ratio", "ang", "lat", "tilt"]
LABEL_COLORS = {"Low": (0, 255, 0), "Medium": (0, 255, 255), "High": (0, 0, 255)}
clf = None
if os.path.exists(MODEL_FILE):
    try:
        import joblib
        clf = joblib.load(MODEL_FILE)
        print(f"Loaded trained risk model: {MODEL_FILE}")
    except Exception as e:
        print("Could not load risk model:", e)

LOCK_MAX_JUMP = 2.0          
LOCK_W_RATIO = (0.6, 1.6)    
LOCK_LOST_FRAMES = 15    


core = ov.Core()
available_devices = core.available_devices
print(f"Detected OpenVINO devices: {available_devices}")

tokens = [t for t in TARGET_DEVICE.replace(":", ",").split(",")
          if t not in ("AUTO", "MULTI", "HETERO")]
missing = [t for t in tokens if not any(d.startswith(t) for d in available_devices)]
if missing:
    print(f"ERROR: device(s) {missing} not available on this machine. "
          f"Available: {available_devices}")
    sys.exit(1)

model = core.read_model(MODEL_PATH)
selected_device = TARGET_DEVICE
config = {props.hint.performance_mode(): props.hint.PerformanceMode.LATENCY}
print(f"Compiling model on {selected_device}...")
compiled_model = core.compile_model(model, selected_device, config)
print(f"Pose model successfully loaded on {selected_device}!")

input_layer = compiled_model.input(0)
_, _, IN_H, IN_W = input_layer.shape

heatmap_output = None
for out in compiled_model.outputs:
    if out.shape[1] == 19:
        heatmap_output = out
if heatmap_output is None:
    raise RuntimeError("Heatmap output not found")

KERNEL = np.ones((25, 25), np.uint8)



def dist(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def nearest(cands, ref, max_dist):
    if ref is None:
        return None
    best, best_d = None, max_dist
    for c in cands:
        d = dist(c, ref)
        if d < best_d:
            best, best_d = c, d
    return best


def calc_angle(a, b, c):
    """Angle at point b between a-b and c-b, in degrees."""
    a, b, c = (np.array(p[:2], dtype=np.float32) for p in (a, b, c))
    ba, bc = a - b, c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom < 1e-8:
        return None
    return float(np.degrees(np.arccos(np.clip(np.dot(ba, bc) / denom, -1.0, 1.0))))


def angle_from_vertical(p_from, p_to, down=True):
    """Angle (0-180) between the vector p_from->p_to and the vertical axis."""
    v = np.array([p_to[0] - p_from[0], p_to[1] - p_from[1]], dtype=np.float32)
    n = np.linalg.norm(v)
    if n < 1e-6:
        return None
    ref = np.array([0.0, 1.0 if down else -1.0], dtype=np.float32)
    return float(np.degrees(np.arccos(np.clip(np.dot(v, ref) / n, -1.0, 1.0))))


def shoulder_tilt(rs, ls):
    """Absolute tilt of the shoulder line from horizontal, in degrees."""
    a, b = (rs, ls) if rs[0] < ls[0] else (ls, rs)
    return abs(float(np.degrees(np.arctan2(b[1] - a[1], b[0] - a[0]))))


_smooth_state = {}


def smooth(name, value):
    if value is None:
        _smooth_state.pop(name, None)
        return None
    prev = _smooth_state.get(name)
    new = value if prev is None else SMOOTH_ALPHA * value + (1 - SMOOTH_ALPHA) * prev
    _smooth_state[name] = new
    return new



TABLE_A = [
    [1, 2, 2],
    [2, 3, 3],
    [3, 3, 4],
    [4, 4, 4],
    [5, 5, 6],
    [7, 8, 9],
]
# Table B (legs score 1 = seated, feet supported): rows = neck 1..6, columns = trunk 1..6
TABLE_B = [
    [1, 2, 3, 5, 6, 7],
    [2, 2, 4, 5, 6, 7],
    [3, 3, 4, 5, 6, 7],
    [5, 5, 6, 7, 7, 8],
    [7, 7, 7, 8, 8, 8],
    [8, 8, 8, 8, 9, 9],
]
# Table C: rows = score A (1..8), columns = score B (1..7)
TABLE_C = [
    [1, 2, 3, 3, 4, 5, 5],
    [2, 2, 3, 4, 4, 5, 5],
    [3, 3, 3, 4, 4, 5, 6],
    [3, 3, 3, 4, 5, 6, 6],
    [4, 4, 4, 5, 6, 7, 7],
    [4, 4, 5, 6, 6, 7, 7],
    [5, 5, 6, 6, 7, 7, 7],
    [5, 5, 6, 7, 7, 7, 7],
]


def rula_grand_score(ua, la, neck, trunk):
    ua = int(np.clip(ua, 1, 6))
    la = int(np.clip(la, 1, 3))
    neck = int(np.clip(neck, 1, 6))
    trunk = int(np.clip(trunk, 1, 6))
    score_a = TABLE_A[ua - 1][la - 1] + STATIC_POSTURE_BONUS      # + muscle use (desk work is static)
    score_b = TABLE_B[neck - 1][trunk - 1] + STATIC_POSTURE_BONUS
    grand = TABLE_C[min(score_a, 8) - 1][min(score_b, 7) - 1]
    return grand, score_a, score_b


def frame_rula(P, base):
    """RULA scores for one frame, using front-view geometry + the calibrated upright baseline."""
    w = dist(P["r_sho"], P["l_sho"])
    if w < 1e-6 or P["nose"] is None:
        return None

    # --- Neck: forward flexion shows up as the nose-neck distance shrinking vs the upright baseline
    ratio = dist(P["nose"], P["neck"]) / w
    # NECK_TOL gives a small dead-band so keypoint jitter around the upright baseline reads as 0 deg
    sag = float(np.degrees(np.arccos(np.clip(ratio / (base["h0"] * NECK_TOL), 0.0, 1.0))))
    lat = angle_from_vertical(P["neck"], P["nose"], down=False) or 0.0
    neck = 1 if sag <= 12 else (2 if sag <= 22 else 3)
    if lat > 18:                       # side-bent / twisted
        neck += 1

    # --- Trunk: leaning/slouching toward the camera -> shoulders get wider and the neck drops
    s = w / base["w0"]
    drop = (P["neck"][1] - base["ny0"]) / base["w0"]
    if s > 1.50 or drop > 0.50:
        trunk = 4
    elif s > 1.25 or drop > 0.25:
        trunk = 3
    elif s > 1.12 or drop > 0.12:
        trunk = 2
    else:
        trunk = 1
    if shoulder_tilt(P["r_sho"], P["l_sho"]) > 10 or s < 0.82:  # side-bend / twist
        trunk += 1
    trunk = min(trunk, 6)


    ua_best, la_best, have_arm, ang_best = 1, 1, False, 0.0
    for sho, elb, wri in (("r_sho", "r_elb", "r_wri"), ("l_sho", "l_elb", "l_wri")):
        if P[elb] is None:
            continue
        have_arm = True
        alpha = angle_from_vertical(P[sho], P[elb], down=True) or 0.0          
        q = dist(P[sho], P[elb]) / (UPPER_ARM_RATIO * w)
        beta = float(np.degrees(np.arccos(np.clip(q, 0.0, 1.0))))              
        ang = max(alpha, beta)
        ang_best = max(ang_best, ang)
        ua = 1 if ang <= 25 else (2 if ang <= 45 else (3 if ang <= 90 else 4))
        if alpha > 45:                                                         
            ua += 1
     
        la = 1
        ua_best, la_best = max(ua_best, min(ua, 6)), max(la_best, la)
    if not have_arm:
        ua_best, la_best = 2, 1       

    grand, sa, sb = rula_grand_score(ua_best, la_best, neck, trunk)
    return {"neck_deg": sag, "neck": neck, "trunk": trunk, "ua": ua_best, "la": la_best,
            "A": sa, "B": sb, "grand": grand,
            # raw features for the learned classifier
            "s": s, "drop": drop, "ratio": ratio / base["h0"], "ang": ang_best, "lat": lat,
            "tilt": shoulder_tilt(P["r_sho"], P["l_sho"])}


def grand_to_label(g):
    if g <= 2:
        return "Low Risk", (0, 255, 0)
    if g <= 4:
        return "Medium Risk", (0, 255, 255)
    return "High Risk", (0, 0, 255)


# ============================================================
# POSE DETECTION + SINGLE-PERSON LOCK
# ============================================================
def preprocess(roi):
    h, w = roi.shape[:2]
    scale = min(IN_W / w, IN_H / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(roi, (nw, nh))
    padded = np.zeros((IN_H, IN_W, 3), np.uint8)
    padded[:nh, :nw] = resized
    blob = padded.transpose(2, 0, 1)[None].astype(np.float32)
    return blob, scale


def get_peaks(frame, x0, x1):
    global infer_ms
    roi = frame[:, x0:x1]
    blob, scale = preprocess(roi)

    t0 = time.perf_counter()
    results = compiled_model([blob])
    ms = (time.perf_counter() - t0) * 1000
    infer_ms = ms if infer_ms == 0.0 else 0.9 * infer_ms + 0.1 * ms

    heatmaps = results[heatmap_output][0]
    all_peaks = []
    for i in range(18):
        hm = cv2.resize(heatmaps[i], (IN_W, IN_H), interpolation=cv2.INTER_CUBIC)
        dil = cv2.dilate(hm, KERNEL)
        ys, xs = np.nonzero((hm == dil) & (hm >= CONF_THRESH))
        pts = [(float(x) / scale + x0, float(y) / scale, float(hm[y, x])) for x, y in zip(xs, ys)]
        pts.sort(key=lambda p: -p[2])
        all_peaks.append(pts[:4])
    return all_peaks


class PersonLock:
    """Remembers the target person so anyone else entering the frame is ignored."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.neck = None
        self.w = None
        self.missed = 0

    @property
    def locked(self):
        return self.neck is not None

    def update(self, neck, w):
        self.neck, self.w, self.missed = (neck[0], neck[1]), w, 0

    def miss(self):
        self.missed += 1
        if self.missed > LOCK_LOST_FRAMES:
            self.reset()


def select_person(peaks, min_shoulder_px, roi_cx, roi_half_w, lock):
    best, best_score = None, 0.0
    for neck in peaks[1]:
        for rs in peaks[2]:
            for ls in peaks[5]:
                w = dist(rs, ls)
                if w < min_shoulder_px:
                    continue
                mid = ((rs[0] + ls[0]) / 2, (rs[1] + ls[1]) / 2)
                if dist(neck, mid) > 0.6 * w:
                    continue

                score = w * (neck[2] + rs[2] + ls[2])
                centre_pen = 1.0 - 0.5 * min(abs(neck[0] - roi_cx) / roi_half_w, 1.0)
                score *= centre_pen

                if lock.locked:
                    d = dist(neck, lock.neck)
                    ratio = w / lock.w
                    if d > LOCK_MAX_JUMP * lock.w:
                        continue
                    if not (LOCK_W_RATIO[0] <= ratio <= LOCK_W_RATIO[1]):
                        continue
                    score *= 1.0 / (1.0 + d / lock.w)

                if score > best_score:
                    best, best_score = (neck, rs, ls, w), score

    if best is None:
        if lock.locked:
            lock.miss()
        return None, 0.0

    neck, rs, ls, w = best
    lock.update(neck, w)
    p = {"neck": neck, "r_sho": rs, "l_sho": ls}
    p["nose"] = nearest(peaks[0], neck, 1.5 * w)
    p["r_elb"] = nearest(peaks[3], rs, 1.6 * w)
    p["r_wri"] = nearest(peaks[4], p["r_elb"], 1.6 * w)
    p["l_elb"] = nearest(peaks[6], ls, 1.6 * w)
    p["l_wri"] = nearest(peaks[7], p["l_elb"], 1.6 * w)
    p["r_ear"] = nearest(peaks[16], p["nose"], 1.0 * w)
    p["l_ear"] = nearest(peaks[17], p["nose"], 1.0 * w)
    return p, w


# ============================================================
# SESSION VERDICT
# ============================================================
def summarise(samples, total_frames, min_valid=None):
    """samples: list of frame_rula() dicts for frames where the person was found."""
    need = min_valid if min_valid is not None else max(5, MIN_VALID_FRAC * total_frames)
    if total_frames == 0 or len(samples) < need:
        return None

    grand = np.array([s["grand"] for s in samples])

    def med(key):
        return float(np.median([s[key] for s in samples]))

    pct_low = 100.0 * np.mean(grand <= 2)
    pct_med = 100.0 * np.mean((grand > 2) & (grand <= 4))
    pct_high = 100.0 * np.mean(grand >= 5)

    median_grand = float(np.median(grand))
    label, color = grand_to_label(median_grand)
    if pct_high >= 100 * HIGH_FRAC_ESCALATE:
        label, color = "High Risk", (0, 0, 255)

    ml_label = None
    if clf is not None:                       # learned classifier decides; RULA scores stay on screen
        X = np.array([[s[f] for f in FEATURES] for s in samples])
        vals, counts = np.unique(clf.predict(X), return_counts=True)
        ml_label = str(vals[np.argmax(counts)])
        label, color = f"{ml_label} Risk", LABEL_COLORS.get(ml_label, (200, 200, 200))

    return {
        "ml_label": ml_label,
        "label": label, "color": color,
        "grand": median_grand,
        "neck": med("neck"), "trunk": med("trunk"), "ua": med("ua"), "la": med("la"),
        "A": med("A"), "B": med("B"), "neck_deg": med("neck_deg"),
        "pct_low": pct_low, "pct_med": pct_med, "pct_high": pct_high,
        "valid_frames": len(samples), "total_frames": total_frames,
    }


def log_session(r):
    row = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Device": selected_device,
        "Infer_ms": round(infer_ms, 2),
        "Verdict": r["label"],
        "RULA_Score": round(r["grand"], 2),
        "Score_A": round(r["A"], 2), "Score_B": round(r["B"], 2),
        "Neck": r["neck"], "Trunk": r["trunk"], "Upper_Arm": r["ua"], "Lower_Arm": r["la"],
        "Neck_Flex_Deg": round(r["neck_deg"], 1),
        "Pct_Low": round(r["pct_low"], 1),
        "Pct_Medium": round(r["pct_med"], 1),
        "Pct_High": round(r["pct_high"], 1),
        "Valid_Frames": r["valid_frames"],
        "Total_Frames": r["total_frames"],
    }
    pd.DataFrame([row]).to_csv(
        CSV_LOG_FILE, mode="a", header=not os.path.exists(CSV_LOG_FILE), index=False
    )
    save_train_rows()


def save_train_rows():
    """Append the labelled per-frame features of this clip to the training CSV."""
    if not train_rows:
        return
    pd.DataFrame(train_rows).to_csv(
        TRAIN_CSV, mode="a", header=not os.path.exists(TRAIN_CSV), index=False
    )
    print(f"Saved {len(train_rows)} labelled frames to {TRAIN_CSV}")
    train_rows.clear()


# ============================================================
# DRAWING
# ============================================================
LINKS = [
    ("nose", "neck"),
    ("neck", "r_sho"), ("r_sho", "r_elb"), ("r_elb", "r_wri"),
    ("neck", "l_sho"), ("l_sho", "l_elb"), ("l_elb", "l_wri"),
    ("r_ear", "nose"), ("l_ear", "nose"),
    ("r_sho", "l_sho"),
]


def draw_skeleton(view, P):
    for a, b in LINKS:
        if P.get(a) is not None and P.get(b) is not None:
            cv2.line(view, (int(P[a][0]), int(P[a][1])), (int(P[b][0]), int(P[b][1])), (255, 0, 0), 3)
    for pt in P.values():
        if pt is not None:
            cv2.circle(view, (int(pt[0]), int(pt[1])), 6, (0, 255, 0), -1)


def put(panel, text, y, scale=0.6, color=(255, 255, 255), thick=2):
    cv2.putText(panel, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)


def build_panel(VH, state, info, active_device, calibrated):
    panel = np.full((VH, PANEL_W, 3), 35, np.uint8)
    put(panel, "EdgeErgo - RULA Assessment", 35, 0.7)
    put(panel, f"Device: {active_device} | Inference: {infer_ms:.1f} ms", 55, 0.45, (0, 255, 128), 1)
    cv2.line(panel, (20, 65), (PANEL_W - 20, 65), (120, 120, 120), 1)

    if state == "IDLE":
        cal_txt = "Calibrated" if calibrated else "NOT calibrated"
        put(panel, f"Baseline: {cal_txt}", 95, 0.55, (0, 255, 0) if calibrated else (0, 165, 255), 1)
        put(panel, "[C] calibrate (sit UPRIGHT first)", 135, 0.55, (0, 255, 255), 1)
        put(panel, f"[SPACE] record {RECORD_SECONDS}s assessment", 165, 0.55, (0, 255, 255), 1)
        put(panel, "[P] quick photo assessment", 195, 0.55, (0, 255, 255), 1)
        put(panel, "[Q] quit", 225, 0.55, (0, 255, 255), 1)
        if info.get("msg"):
            put(panel, info["msg"], 265, 0.5, (0, 165, 255), 1)
        put(panel, "Live body-part scores (preview):", 315, 0.5, (180, 180, 180), 1)
        nd = info.get("neck_deg")
        put(panel, f"Neck flexion : {'N/A' if nd is None else f'{nd:.0f} deg'}", 345, 0.5, (180, 180, 180), 1)
        put(panel, f"Neck score   : {info.get('neck', 'N/A')}", 370, 0.5, (180, 180, 180), 1)
        put(panel, f"Trunk score  : {info.get('trunk', 'N/A')}", 395, 0.5, (180, 180, 180), 1)
        put(panel, f"Upper arm    : {info.get('ua', 'N/A')}", 420, 0.5, (180, 180, 180), 1)
        put(panel, "Person detected" if info.get("locked") else "No person detected",
            460, 0.5, (0, 255, 0) if info.get("locked") else (150, 150, 150), 1)

    elif state == "CALIBRATE":
        put(panel, "CALIBRATING", 105, 0.9, (0, 255, 255), 3)
        put(panel, "Sit UPRIGHT, relaxed,", 150, 0.6)
        put(panel, "looking at the screen.", 180, 0.6)
        put(panel, f"{info.get('remaining', CALIB_SECONDS):.0f}", 300, 3.0, (0, 255, 255), 6)

    elif state == "COUNTDOWN":
        put(panel, "Get ready - hold your posture", 105)
        put(panel, f"{info.get('remaining', COUNTDOWN_SECONDS):.0f}", 225, 3.0, (0, 255, 255), 6)
        put(panel, "Locked onto you. Others ignored.", 285, 0.55, (0, 255, 0), 1)

    elif state == "RECORDING":
        put(panel, "RECORDING", 105, 0.9, (0, 0, 255), 3)
        put(panel, f"{info.get('remaining', RECORD_SECONDS):.1f}s left", 155, 0.8)
        put(panel, f"Frames with pose: {info.get('valid', 0)}/{info.get('total', 0)}", 195, 0.55, (180, 180, 180), 1)
        put(panel, "Hold your normal posture.", 235, 0.55, (180, 180, 180), 1)
        bar_w = int((PANEL_W - 40) * info.get("progress", 0.0))
        cv2.rectangle(panel, (20, 265), (PANEL_W - 20, 285), (80, 80, 80), 1)
        cv2.rectangle(panel, (20, 265), (20 + bar_w, 285), (0, 0, 255), -1)

    elif state == "RESULT":
        r = info.get("result")
        if r is None:
            put(panel, "Insufficient data", 115, 0.9, (0, 165, 255), 3)
            put(panel, "Person not detected clearly.", 155, 0.55, (200, 200, 200), 1)
            put(panel, "Check lighting and that your", 180, 0.55, (200, 200, 200), 1)
            put(panel, "head and shoulders are visible.", 205, 0.55, (200, 200, 200), 1)
        else:
            put(panel, "FINAL VERDICT", 95, 0.7)
            put(panel, r["label"], 145, 1.2, r["color"], 3)
            put(panel, f"RULA grand score: {r['grand']:.0f} / 7", 185, 0.6, (230, 230, 230), 1)
            put(panel, f"Score A (arm/wrist): {r['A']:.0f}   Score B (neck/trunk): {r['B']:.0f}",
                210, 0.45, (180, 180, 180), 1)
            if r.get("ml_label"):
                put(panel, f"Verdict from trained classifier: {r['ml_label']}", 228, 0.45, (0, 255, 128), 1)
            put(panel, "Median body-part scores:", 245, 0.5, (180, 180, 180), 1)
            put(panel, f"Neck {r['neck']:.0f} ({r['neck_deg']:.0f} deg)   Trunk {r['trunk']:.0f}", 272, 0.55)
            put(panel, f"Upper arm {r['ua']:.0f}   Lower arm {r['la']:.0f}", 300, 0.55)
            put(panel, "Time spent in each level:", 345, 0.5, (180, 180, 180), 1)
            put(panel, f"Low    {r['pct_low']:.0f}%", 375, 0.6, (0, 255, 0))
            put(panel, f"Medium {r['pct_med']:.0f}%", 405, 0.6, (0, 255, 255))
            put(panel, f"High   {r['pct_high']:.0f}%", 435, 0.6, (0, 0, 255))
            put(panel, f"({r['valid_frames']}/{r['total_frames']} frames used)", 470, 0.5, (150, 150, 150), 1)
        put(panel, "[SPACE] again  [P] photo  [C] recalibrate", VH - 55, 0.45, (0, 255, 255), 1)
        put(panel, "[Q] quit", VH - 30, 0.5, (0, 255, 255), 1)
    return panel


# ============================================================
# CAMERA + STATE MACHINE
# ============================================================
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    raise RuntimeError("Could not open camera")
ret, frame = cap.read()
if not ret:
    raise RuntimeError("Could not read from camera")

VH, VW = frame.shape[:2]
ROI_X0 = int(ROI_X_FRAC[0] * VW)
ROI_X1 = int(ROI_X_FRAC[1] * VW)
ROI_CX = (ROI_X0 + ROI_X1) / 2
ROI_HALF_W = (ROI_X1 - ROI_X0) / 2
MIN_SHOULDER_PX = MIN_SHOULDER_FRAC * (ROI_X1 - ROI_X0)

window_name = "EdgeErgo - Posture Assessment"
cv2.namedWindow(window_name)

lock = PersonLock()
state = "IDLE"
t_start = 0.0
samples, total_frames = [], 0
result = None
photo_mode = False
frozen_view = None
base = None                       # calibrated upright baseline
cal_w, cal_h, cal_y = [], [], []
status_msg = ""
current_label = None              # set with keys 1/2/3 to record labelled training data
train_rows = []

print("EdgeErgo started. C = calibrate (sit upright), SPACE = record, P = photo, Q = quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera read failed - stopping.")
        break

    now = time.time()

    # In IDLE the lock is released so the preview just follows the best person.
    if state == "IDLE":
        lock.reset()

    if state == "RESULT":
        view = frozen_view.copy()
        person = None
    else:
        peaks = get_peaks(frame, ROI_X0, ROI_X1)
        person, _ = select_person(peaks, MIN_SHOULDER_PX, ROI_CX, ROI_HALF_W, lock)
        view = frame.copy()
        cv2.rectangle(view, (ROI_X0, 0), (ROI_X1 - 1, VH - 1), (0, 255, 255), 2)
        if person is not None:
            draw_skeleton(view, person)

    info = {}
    fr = None
    if person is not None and base is not None:
        fr = frame_rula(person, base)

    if state == "IDLE":
        info = {"locked": person is not None, "msg": status_msg}
        if fr is not None:
            info["neck_deg"] = smooth("neck_deg", fr["neck_deg"])
            info["neck"], info["trunk"], info["ua"] = fr["neck"], fr["trunk"], fr["ua"]

    elif state == "CALIBRATE":
        remaining = CALIB_SECONDS - (now - t_start)
        if person is not None and person["nose"] is not None:
            w = dist(person["r_sho"], person["l_sho"])
            cal_w.append(w)
            cal_h.append(dist(person["nose"], person["neck"]) / w)
            cal_y.append(person["neck"][1])
        info = {"remaining": max(remaining, 0.0) + 0.5}
        if remaining <= 0:
            if len(cal_w) >= 10:
                base = {"w0": float(np.median(cal_w)),
                        "h0": float(np.median(cal_h)),
                        "ny0": float(np.median(cal_y))}
                status_msg = "Calibration OK."
                print(f"Calibrated: {base}")
            else:
                status_msg = "Calibration failed - person not seen."
            state = "IDLE"
            info = {"msg": status_msg}

    elif state == "COUNTDOWN":
        remaining = COUNTDOWN_SECONDS - (now - t_start)
        info = {"remaining": max(remaining, 0.0) + 0.5}
        if remaining <= 0:
            state = "RECORDING"
            samples, total_frames = [], 0
            train_rows.clear()
            clip_id = datetime.now().strftime("%H%M%S%f")
            t_start = now

    elif state == "RECORDING":
        total_frames += 1
        if fr is not None:
            samples.append(fr)
            if current_label:
                train_rows.append({"clip_id": clip_id, "label": current_label,
                                   **{f: fr[f] for f in FEATURES}})
        elapsed = now - t_start
        info = {
            "remaining": max(RECORD_SECONDS - elapsed, 0.0),
            "progress": min(elapsed / RECORD_SECONDS, 1.0),
            "valid": len(samples), "total": total_frames,
        }
        if photo_mode:
            if len(samples) >= 8 or total_frames >= 45:
                result = summarise(samples, total_frames, min_valid=3)
                if result is not None:
                    log_session(result)
                frozen_view = view.copy()
                state = "RESULT"
                photo_mode = False
        elif elapsed >= RECORD_SECONDS:
            result = summarise(samples, total_frames)
            if result is not None:
                log_session(result)
            frozen_view = view.copy()
            state = "RESULT"

    if state == "RESULT":
        info = {"result": result}

    panel = build_panel(VH, state, info, selected_device, base is not None)
    cv2.imshow(window_name, np.hstack([view, panel]))

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord("c") and state in ("IDLE", "RESULT"):
        lock.reset()
        cal_w, cal_h, cal_y = [], [], []
        status_msg = ""
        state = "CALIBRATE"
        t_start = time.time()
    if key in (ord("1"), ord("2"), ord("3")) and state in ("IDLE", "RESULT"):
        current_label = {ord("1"): "Low", ord("2"): "Medium", ord("3"): "High"}[key]
        status_msg = f"TRAIN label = {current_label}. Press SPACE to record."
        state = "IDLE"
    if key == ord("0") and state in ("IDLE", "RESULT"):
        current_label = None
        status_msg = "Training label off."
        state = "IDLE"
    if key in (ord(" "), ord("p")) and state in ("IDLE", "RESULT"):
        if base is None:
            status_msg = "Press C first (sit upright)."
            state = "IDLE"
        else:
            lock.reset()
            photo_mode = key == ord("p")
            samples, total_frames = [], 0
            status_msg = ""
            state = "COUNTDOWN"
            t_start = time.time()

cap.release()
cv2.destroyAllWindows()