import numpy as np
import os
from config import *
from utils import dist, angle_from_vertical, shoulder_tilt, nearest

clf = None
if os.path.exists(MODEL_FILE):
    try:
        import joblib
        clf = joblib.load(MODEL_FILE)
    except Exception:
        pass

class PersonLock:
    def __init__(self):
        self.reset()
    def reset(self):
        self.neck, self.w, self.missed = None, None, 0
    @property
    def locked(self): return self.neck is not None
    def update(self, neck, w):
        self.neck, self.w, self.missed = (neck[0], neck[1]), w, 0
    def miss(self):
        self.missed += 1
        if self.missed > LOCK_LOST_FRAMES: self.reset()

def select_person(peaks, min_shoulder_px, roi_cx, roi_half_w, lock):
    best, best_score = None, 0.0
    for neck in peaks[1]:
        for rs in peaks[2]:
            for ls in peaks[5]:
                w = dist(rs, ls)
                if w < min_shoulder_px: continue
                mid = ((rs[0] + ls[0]) / 2, (rs[1] + ls[1]) / 2)
                if dist(neck, mid) > 0.6 * w: continue

                score = w * (neck[2] + rs[2] + ls[2]) * (1.0 - 0.5 * min(abs(neck[0] - roi_cx) / roi_half_w, 1.0))
                
                if lock.locked:
                    d = dist(neck, lock.neck)
                    ratio = w / lock.w
                    if d > LOCK_MAX_JUMP * lock.w or not (LOCK_W_RATIO[0] <= ratio <= LOCK_W_RATIO[1]):
                        continue
                    score *= 1.0 / (1.0 + d / lock.w)

                if score > best_score:
                    best, best_score = (neck, rs, ls, w), score

    if best is None:
        if lock.locked: lock.miss()
        return None, 0.0

    neck, rs, ls, w = best
    lock.update(neck, w)
    
    p = {"neck": neck, "r_sho": rs, "l_sho": ls,
         "nose": nearest(peaks[0], neck, 1.5 * w),
         "r_elb": nearest(peaks[3], rs, 1.6 * w), "l_elb": nearest(peaks[6], ls, 1.6 * w)}
    p["r_wri"] = nearest(peaks[4], p.get("r_elb"), 1.6 * w) if p.get("r_elb") else None
    p["l_wri"] = nearest(peaks[7], p.get("l_elb"), 1.6 * w) if p.get("l_elb") else None
    p["r_ear"] = nearest(peaks[16], p.get("nose"), 1.0 * w) if p.get("nose") else None
    p["l_ear"] = nearest(peaks[17], p.get("nose"), 1.0 * w) if p.get("nose") else None
    return p, w

def rula_grand_score(ua, la, neck, trunk):
    ua, la = int(np.clip(ua, 1, 6)), int(np.clip(la, 1, 3))
    neck, trunk = int(np.clip(neck, 1, 6)), int(np.clip(trunk, 1, 6))
    score_a = TABLE_A[ua - 1][la - 1] + STATIC_POSTURE_BONUS
    score_b = TABLE_B[neck - 1][trunk - 1] + STATIC_POSTURE_BONUS
    return TABLE_C[min(score_a, 8) - 1][min(score_b, 7) - 1], score_a, score_b

def frame_rula(P, base):
    w = dist(P["r_sho"], P["l_sho"])
    if w < 1e-6: return None

    if P.get("nose") is not None:
        ratio = dist(P["nose"], P["neck"]) / w
        sag = float(np.degrees(np.arccos(np.clip(ratio / (base["h0"] * NECK_TOL), 0.0, 1.0))))
        lat = angle_from_vertical(P["neck"], P["nose"], down=False) or 0.0
        
        neck = 1 if sag <= 8 else (2 if sag <= 15 else 3)
        if lat > 8: 
            neck += 1
    else:
        sag, lat, neck, ratio = 0.0, 0.0, 1, base["h0"]

    s = w / base["w0"]
    drop = (P["neck"][1] - base["ny0"]) / base["w0"]
    trunk = 4 if s > 1.50 or drop > 0.50 else (3 if s > 1.25 or drop > 0.25 else (2 if s > 1.12 or drop > 0.12 else 1))
    if shoulder_tilt(P["r_sho"], P["l_sho"]) > 10 or s < 0.82: trunk += 1

    ua_best, la_best, have_arm, ang_best = 1, 1, False, 0.0
    for sho, elb, wri in (("r_sho", "r_elb", "r_wri"), ("l_sho", "l_elb", "l_wri")):
        if P.get(elb) is None: continue
        have_arm = True
        alpha = angle_from_vertical(P[sho], P[elb], down=True) or 0.0
        beta = float(np.degrees(np.arccos(np.clip(dist(P[sho], P[elb]) / (UPPER_ARM_RATIO * w), 0.0, 1.0))))
        ang = max(alpha, beta)
        ang_best = max(ang_best, ang)
        ua = 1 if ang <= 25 else (2 if ang <= 45 else (3 if ang <= 90 else 4))
        if alpha > 45: ua += 1
        ua_best = max(ua_best, min(ua, 6))

    if not have_arm: ua_best = 2

    grand, sa, sb = rula_grand_score(ua_best, la_best, neck, min(trunk, 6))
    return {"neck_deg": sag, "neck": min(neck, 6), "trunk": min(trunk, 6), "ua": ua_best, "la": la_best,
            "A": sa, "B": sb, "grand": grand,
            "s": s, "drop": drop, "ratio": ratio / base["h0"], "ang": ang_best, "lat": lat,
            "tilt": shoulder_tilt(P["r_sho"], P["l_sho"])}

def summarise(samples, total_frames, min_valid=None):
    need = min_valid if min_valid is not None else max(5, MIN_VALID_FRAC * total_frames)
    if total_frames == 0 or len(samples) < need: return None

    grand = np.array([s["grand"] for s in samples])
    med = lambda key: float(np.median([s[key] for s in samples]))
    
    median_grand = float(np.median(grand))
    label = "High Risk" if median_grand > 4 else ("Medium Risk" if median_grand > 2 else "Low Risk")
    color = LABEL_COLORS.get(label.split()[0], (0,255,0))

    ml_label = None
    if clf is not None:
        X = np.array([[s[f] for f in FEATURES] for s in samples])
        vals, counts = np.unique(clf.predict(X), return_counts=True)
        ml_label = str(vals[np.argmax(counts)])
        label, color = f"{ml_label} Risk", LABEL_COLORS.get(ml_label, (200, 200, 200))

    return {
        "ml_label": ml_label, "label": label, "color": color, "grand": median_grand,
        "neck": med("neck"), "trunk": med("trunk"), "ua": med("ua"), "la": med("la"),
        "A": med("A"), "B": med("B"), "neck_deg": med("neck_deg"),
        "pct_low": 100.0 * np.mean(grand <= 2), "pct_med": 100.0 * np.mean((grand > 2) & (grand <= 4)), 
        "pct_high": 100.0 * np.mean(grand >= 5),
        "valid_frames": len(samples), "total_frames": total_frames,
    }