import cv2
import numpy as np
import openvino as ov
import pandas as pd
from datetime import datetime
import os


MODEL_PATH = "models/human-pose-estimation-0001/FP16/human-pose-estimation-0001.xml"
CAMERA_INDEX = 0

CONF_THRESH = 0.20            
ROI_X_FRAC = (0.20, 0.80)     
MIN_SHOULDER_FRAC = 0.12      
PANEL_W = 450                 
SMOOTH_ALPHA = 0.4            
CSV_LOG_FILE = "edgeergo_session_log.csv"


core = ov.Core()
model = core.read_model(MODEL_PATH)
compiled_model = core.compile_model(model, "AUTO")

print("Pose model loaded successfully!")
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
    if ref is None: return None
    best, best_d = None, max_dist
    for c in cands:
        d = dist(c, ref)
        if d < best_d: best, best_d = c, d
    return best

def calc_angle(a, b, c):
    a, b, c = np.array(a[:2], dtype=np.float32), np.array(b[:2], dtype=np.float32), np.array(c[:2], dtype=np.float32)
    ba, bc = a - b, c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom < 1e-8: return None
    cosine = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))

def tilt_deg(a, b):
    if a[0] > b[0]: a, b = b, a
    return float(np.degrees(np.arctan2(b[1] - a[1], b[0] - a[0])))

_smooth_state = {}
def smooth(name, value):
    if value is None:
        _smooth_state.pop(name, None)
        return None
    prev = _smooth_state.get(name)
    new = value if prev is None else SMOOTH_ALPHA * value + (1 - SMOOTH_ALPHA) * prev
    _smooth_state[name] = new
    return new

def evaluate_rula(neck_angle, r_upper, r_elbow):
    """Basic RULA logic mapping joint angles to risk levels."""
    score = 1
    if neck_angle and neck_angle > 20: score += 2
    elif neck_angle and neck_angle > 10: score += 1
    
    if r_upper and r_upper > 45: score += 2
    elif r_upper and r_upper > 20: score += 1
    
    if score <= 2: return "Low Risk", (0, 255, 0)
    elif score <= 4: return "Medium Risk", (0, 255, 255)
    else: return "High Risk", (0, 0, 255)

def log_data(risk_label, neck, r_upper):
    """Save to local encrypted/CSV storage for privacy compliance."""
    df = pd.DataFrame([{
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Risk_Level": risk_label,
        "Neck_Angle": round(neck, 2) if neck else None,
        "R_Upper_Arm": round(r_upper, 2) if r_upper else None
    }])
    df.to_csv(CSV_LOG_FILE, mode='a', header=not os.path.exists(CSV_LOG_FILE), index=False)

# ============================================================
# POSE DETECTION
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
    roi = frame[:, x0:x1]
    blob, scale = preprocess(roi)
    results = compiled_model([blob])
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

def select_person(peaks, min_shoulder_px):
    best, best_score = None, 0.0
    for neck in peaks[1]:
        for rs in peaks[2]:
            for ls in peaks[5]:
                w = dist(rs, ls)
                if w < min_shoulder_px: continue
                mid = ((rs[0] + ls[0]) / 2, (rs[1] + ls[1]) / 2)
                if dist(neck, mid) > 0.6 * w: continue
                score = w * (neck[2] + rs[2] + ls[2])
                if score > best_score: best, best_score = (neck, rs, ls, w), score
    
    if best is None: return None, 0.0
    
    neck, rs, ls, w = best
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
# CAMERA & MAIN LOOP
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
MIN_SHOULDER_PX = MIN_SHOULDER_FRAC * (ROI_X1 - ROI_X0)

window_name = "EdgeErgo - Predictive Risk Assessment"
cv2.namedWindow(window_name)

LINKS = [
    ("nose", "neck"),
    ("neck", "r_sho"), ("r_sho", "r_elb"), ("r_elb", "r_wri"),
    ("neck", "l_sho"), ("l_sho", "l_elb"), ("l_elb", "l_wri"),
    ("r_ear", "nose"), ("l_ear", "nose"),
    ("r_sho", "l_sho"),
]

print("==========================================")
print("      EdgeErgo Application Started")
print("==========================================")
print("Press 'Q' in the video window to stop.")

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    peaks = get_peaks(frame, ROI_X0, ROI_X1)
    person, shoulder_w = select_person(peaks, MIN_SHOULDER_PX)

    view = frame.copy()
    cv2.rectangle(view, (ROI_X0, 0), (ROI_X1 - 1, VH - 1), (0, 255, 255), 2)

    neck_angle = r_upper = r_elbow = None

    if person is not None:
        P = person

        # Draw skeleton
        for a, b in LINKS:
            if P.get(a) is not None and P.get(b) is not None:
                cv2.line(view, (int(P[a][0]), int(P[a][1])),
                         (int(P[b][0]), int(P[b][1])), (255, 0, 0), 3)
        for pt in P.values():
            if pt is not None:
                cv2.circle(view, (int(pt[0]), int(pt[1])), 6, (0, 255, 0), -1)

        # Measurements
        shoulder_center = ((P["r_sho"][0] + P["l_sho"][0]) / 2,
                           (P["r_sho"][1] + P["l_sho"][1]) / 2)

        if P["nose"] is not None:
            neck_angle = calc_angle(P["nose"], P["neck"], shoulder_center)
        if P["r_elb"] is not None:
            r_upper = calc_angle(P["neck"], P["r_sho"], P["r_elb"])
            if P["r_wri"] is not None:
                r_elbow = calc_angle(P["r_sho"], P["r_elb"], P["r_wri"])

    # Smoothing
    neck_angle = smooth("neck", neck_angle)
    r_upper = smooth("r_upper", r_upper)
    r_elbow = smooth("r_elbow", r_elbow)
    
    risk_label = "Scanning..."
    risk_color = (200, 200, 200)

    if neck_angle is not None or r_upper is not None:
        risk_label, risk_color = evaluate_rula(neck_angle, r_upper, r_elbow)
        
        # Log data every 30 frames to avoid spamming the local CSV storage
        if frame_count % 30 == 0:
            log_data(risk_label, neck_angle, r_upper)

    # UI Panel
    panel = np.full((VH, PANEL_W, 3), 35, np.uint8)

    cv2.putText(panel, "EdgeErgo Dashboard", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.line(panel, (20, 55), (PANEL_W - 20, 55), (120, 120, 120), 1)

    y = 90
    rows = [
        ("Neck angle", neck_angle),
        ("Right upper arm", r_upper),
        ("Right elbow", r_elbow),
    ]
    
    for label, value in rows:
        if value is None:
            text, color = f"{label}: N/A", (150, 150, 150)
        else:
            text, color = f"{label}: {value:.1f} deg", (255, 255, 255)
        cv2.putText(panel, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        y += 40

    cv2.putText(panel, "Current Risk Level:", (20, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(panel, risk_label, (20, y + 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, risk_color, 3)

    cv2.imshow(window_name, np.hstack([view, panel]))
    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()