import cv2
import time
import numpy as np
from datetime import datetime
from config import *
from utils import dist, smooth, log_session, save_train_rows
from inference import PoseEstimator
from rula_engine import PersonLock, select_person, frame_rula, summarise
from ui import draw_skeleton, build_panel

def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera")
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Could not read from camera")

    VH, VW = frame.shape[:2]
    ROI_X0, ROI_X1 = int(ROI_X_FRAC[0] * VW), int(ROI_X_FRAC[1] * VW)
    ROI_CX, ROI_HALF_W = (ROI_X0 + ROI_X1) / 2, (ROI_X1 - ROI_X0) / 2
    MIN_SHOULDER_PX = MIN_SHOULDER_FRAC * (ROI_X1 - ROI_X0)

    estimator = PoseEstimator()
    lock = PersonLock()

    window_name = "EdgeErgo - Posture Assessment"
    cv2.namedWindow(window_name)

    state, t_start = "IDLE", 0.0
    recorded_frames = []
    samples, train_rows = [], []
    result, frozen_view, base = None, None, None
    cal_w, cal_h, cal_y = [], [], []
    status_msg, current_label = "", None
    
    analysis_indices = []
    analysis_current = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        now = time.time()

        if state == "IDLE":
            lock.reset()

        info, fr = {}, None

        if state in ("IDLE", "CALIBRATE", "COUNTDOWN"):
            peaks, infer_ms = estimator.get_peaks(frame, ROI_X0, ROI_X1)
            person, _ = select_person(peaks, MIN_SHOULDER_PX, ROI_CX, ROI_HALF_W, lock)
            view = frame.copy()
            cv2.rectangle(view, (ROI_X0, 0), (ROI_X1 - 1, VH - 1), (0, 255, 255), 2)
            
            if person is not None:
                draw_skeleton(view, person)
                if base is not None:
                    fr = frame_rula(person, base)

        elif state == "RECORDING":
            view = frame.copy()
            cv2.rectangle(view, (ROI_X0, 0), (ROI_X1 - 1, VH - 1), (0, 0, 255), 2)

        elif state == "RESULT":
            view = frozen_view.copy()

        if state == "IDLE":
            info = {"locked": person is not None, "msg": status_msg}
            if fr:
                info.update({"neck_deg": smooth("neck_deg", fr["neck_deg"]), "neck": fr["neck"], "trunk": fr["trunk"], "ua": fr["ua"]})

        elif state == "CALIBRATE":
            remaining = CALIB_SECONDS - (now - t_start)
            if person is not None and person.get("nose"):
                w = dist(person["r_sho"], person["l_sho"])
                cal_w.append(w)
                cal_h.append(dist(person["nose"], person["neck"]) / w)
                cal_y.append(person["neck"][1])
            info = {"remaining": max(remaining, 0.0) + 0.5}

            if remaining <= 0:
                if len(cal_w) >= 10:
                    base = {"w0": float(np.median(cal_w)), "h0": float(np.median(cal_h)), "ny0": float(np.median(cal_y))}
                    status_msg = "Calibration OK."
                else:
                    status_msg = "Calibration failed."
                state, info = "IDLE", {"msg": status_msg}

        elif state == "COUNTDOWN":
            remaining = COUNTDOWN_SECONDS - (now - t_start)
            info = {"remaining": max(remaining, 0.0) + 0.5}
            if remaining <= 0:
                state = "RECORDING"
                recorded_frames = []
                t_start = now

        elif state == "RECORDING":
            recorded_frames.append(frame.copy())
            elapsed = now - t_start
            info = {"remaining": max(RECORD_SECONDS - elapsed, 0.0), "progress": min(elapsed / RECORD_SECONDS, 1.0)}

            if elapsed >= RECORD_SECONDS:
                state = "ANALYZING"
                samples, train_rows = [], []
                analysis_current = 0
                lock.reset()
                
                if len(recorded_frames) > 0:
                    analysis_indices = np.linspace(0, len(recorded_frames) - 1, 10, dtype=int)
                else:
                    analysis_indices = []

        elif state == "ANALYZING":
            if analysis_current < len(analysis_indices):
                idx = analysis_indices[analysis_current]
                img = recorded_frames[idx]
                
                peaks, _ = estimator.get_peaks(img, ROI_X0, ROI_X1)
                p, _ = select_person(peaks, MIN_SHOULDER_PX, ROI_CX, ROI_HALF_W, lock)
                
                view = img.copy()
                cv2.rectangle(view, (ROI_X0, 0), (ROI_X1 - 1, VH - 1), (0, 255, 255), 2)
                info = {"msg": f"Image {analysis_current + 1} of 10"}
                
                if p is not None:
                    draw_skeleton(view, p)
                    if base is not None:
                        frame_score = frame_rula(p, base)
                        if frame_score is not None:
                            samples.append(frame_score)
                            info.update({"neck_deg": frame_score["neck_deg"], "neck": frame_score["neck"], "trunk": frame_score["trunk"], "ua": frame_score["ua"]})
                
                frozen_view = view.copy()
                analysis_current += 1
                
                panel = build_panel(VH, state, info, estimator.device, base is not None, estimator.infer_ms)
                cv2.imshow(window_name, np.hstack([view, panel]))
                cv2.waitKey(350)
                continue
                
            else:
                result = summarise(samples, total_frames=10, min_valid=3)
                if result:
                    log_session(result, estimator.device, estimator.infer_ms)
                state = "RESULT"

        if state == "RESULT":
            info = {"result": result}

        panel = build_panel(VH, state, info, estimator.device, base is not None, estimator.infer_ms)
        cv2.imshow(window_name, np.hstack([view, panel]))

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"): break
        if key == ord("c") and state in ("IDLE", "RESULT"):
            lock.reset(); cal_w, cal_h, cal_y = [], [], []; state, t_start = "CALIBRATE", time.time()
        if key in (ord(" "), ord("p")) and state in ("IDLE", "RESULT"):
            if base is None:
                status_msg, state = "Press C first.", "IDLE"
            else:
                lock.reset(); state, t_start = "COUNTDOWN", time.time()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()