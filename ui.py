import cv2
import numpy as np
from config import PANEL_W, RECORD_SECONDS, CALIB_SECONDS, COUNTDOWN_SECONDS

LINKS = [
    ("nose", "neck"), ("neck", "r_sho"), ("r_sho", "r_elb"), ("r_elb", "r_wri"),
    ("neck", "l_sho"), ("l_sho", "l_elb"), ("l_elb", "l_wri"),
    ("r_ear", "nose"), ("l_ear", "nose"), ("r_sho", "l_sho")
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

def build_panel(VH, state, info, active_device, calibrated, infer_ms):
    panel = np.full((VH, PANEL_W, 3), 35, np.uint8)
    put(panel, "EdgeErgo - RULA Assessment", 35, 0.7)
    put(panel, f"Device: {active_device} | Inference: {infer_ms:.1f} ms", 55, 0.45, (0, 255, 128), 1)
    cv2.line(panel, (20, 65), (PANEL_W - 20, 65), (120, 120, 120), 1)

    # --- LIVE SCORE READOUT BLOCK ---
    def draw_live_scores(y_start):
        put(panel, "Live body-part scores:", y_start, 0.5, (180, 180, 180), 1)
        nd = info.get("neck_deg")
        put(panel, f"Neck flexion : {'N/A' if nd is None else f'{nd:.0f} deg'}", y_start + 30, 0.5, (180, 180, 180), 1)
        put(panel, f"Neck score   : {info.get('neck', 'N/A')}", y_start + 55, 0.5, (180, 180, 180), 1)
        put(panel, f"Trunk score  : {info.get('trunk', 'N/A')}", y_start + 80, 0.5, (180, 180, 180), 1)
        put(panel, f"Upper arm    : {info.get('ua', 'N/A')}", y_start + 105, 0.5, (180, 180, 180), 1)

    if state == "IDLE":
        cal_txt = "Calibrated" if calibrated else "NOT calibrated"
        put(panel, f"Baseline: {cal_txt}", 95, 0.55, (0, 255, 0) if calibrated else (0, 165, 255), 1)
        put(panel, "[C] calibrate (sit UPRIGHT first)", 135, 0.55, (0, 255, 255), 1)
        put(panel, f"[SPACE] record {RECORD_SECONDS}s assessment", 165, 0.55, (0, 255, 255), 1)
        if info.get("msg"): put(panel, info["msg"], 235, 0.5, (0, 165, 255), 1)
        draw_live_scores(285)

    elif state == "CALIBRATE":
        put(panel, "CALIBRATING", 105, 0.9, (0, 255, 255), 3)
        put(panel, f"{info.get('remaining', CALIB_SECONDS):.0f}", 300, 3.0, (0, 255, 255), 6)

    elif state == "COUNTDOWN":
        put(panel, "Get ready - hold your posture", 105)
        put(panel, f"{info.get('remaining', COUNTDOWN_SECONDS):.0f}", 225, 3.0, (0, 255, 255), 6)

    elif state == "RECORDING":
        put(panel, "RECORDING VIDEO", 105, 0.9, (0, 0, 255), 3)
        put(panel, f"{info.get('remaining', RECORD_SECONDS):.1f}s left", 155, 0.8)
        bar_w = int((PANEL_W - 40) * info.get("progress", 0.0))
        cv2.rectangle(panel, (20, 265), (20 + bar_w, 285), (0, 0, 255), -1)

    elif state == "ANALYZING":
        put(panel, "ANALYZING FRAMES", 105, 0.9, (0, 165, 255), 3)
        if info.get("msg"): put(panel, info["msg"], 155, 0.6, (0, 255, 255), 1)
        draw_live_scores(225)

    elif state == "RESULT":
        r = info.get("result")
        if r is None:
            put(panel, "Insufficient data", 115, 0.9, (0, 165, 255), 3)
        else:
            put(panel, "FINAL VERDICT", 95, 0.7)
            put(panel, r["label"], 145, 1.2, r["color"], 3)
            put(panel, f"RULA grand score: {r['grand']:.0f} / 7", 185, 0.6, (230, 230, 230), 1)
            put(panel, f"Score A (arm/wrist): {r['A']:.0f}   Score B (neck/trunk): {r['B']:.0f}", 210, 0.45, (180, 180, 180), 1)
            
            put(panel, "Median body-part scores:", 245, 0.5, (180, 180, 180), 1)
            put(panel, f"Neck {r['neck']:.0f} ({r['neck_deg']:.0f} deg)   Trunk {r['trunk']:.0f}", 272, 0.55)
            put(panel, f"Upper arm {r['ua']:.0f}   Lower arm {r['la']:.0f}", 300, 0.55)
            
            put(panel, "Time spent in each level:", 345, 0.5, (180, 180, 180), 1)
            put(panel, f"Low    {r['pct_low']:.0f}%", 375, 0.6, (0, 255, 0))
            put(panel, f"Medium {r['pct_med']:.0f}%", 405, 0.6, (0, 255, 255))
            put(panel, f"High   {r['pct_high']:.0f}%", 435, 0.6, (0, 0, 255))
            
    put(panel, "[SPACE] again  [C] recalibrate  [Q] quit", VH - 30, 0.45, (0, 255, 255), 1)
    return panel