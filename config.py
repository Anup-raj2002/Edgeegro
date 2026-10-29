import sys

MODEL_PATH = "models/human-pose-estimation-0001/FP16/human-pose-estimation-0001.xml"
CAMERA_INDEX = 0
TARGET_DEVICE = sys.argv[1].upper() if len(sys.argv) > 1 else "CPU"

RECORD_SECONDS = 10 
COUNTDOWN_SECONDS = 3 
CALIB_SECONDS = 3 
MIN_VALID_FRAC = 0.30 
HIGH_FRAC_ESCALATE = 0.40 

CONF_THRESH = 0.15 
LOCK_MAX_JUMP = 3.0 
LOCK_W_RATIO = (0.4, 2.0) 
LOCK_LOST_FRAMES = 30 

ROI_X_FRAC = (0.02, 0.98) 
MIN_SHOULDER_FRAC = 0.08
PANEL_W = 450
SMOOTH_ALPHA = 0.4 
CSV_LOG_FILE = "edgeergo_session_log.csv"
TRAIN_CSV = "edgeergo_train.csv"
MODEL_FILE = "risk_model.joblib"
FEATURES = ["neck_deg", "s", "drop", "ratio", "ang", "lat", "tilt"]
LABEL_COLORS = {"Low": (0, 255, 0), "Medium": (0, 255, 255), "High": (0, 0, 255)}

STATIC_POSTURE_BONUS = 1 
UPPER_ARM_RATIO = 0.55 
NECK_TOL = 0.98 

TABLE_A = [
    [1, 2, 2], [2, 3, 3], [3, 3, 4], [4, 4, 4], [5, 5, 6], [7, 8, 9],
]
TABLE_B = [
    [1, 2, 3, 5, 6, 7], [2, 2, 4, 5, 6, 7], [3, 3, 4, 5, 6, 7],
    [5, 5, 6, 7, 7, 8], [7, 7, 7, 8, 8, 8], [8, 8, 8, 8, 9, 9],
]
TABLE_C = [
    [1, 2, 3, 3, 4, 5, 5], [2, 2, 3, 4, 4, 5, 5], [3, 3, 3, 4, 4, 5, 6],
    [3, 3, 3, 4, 5, 6, 6], [4, 4, 4, 5, 6, 7, 7], [4, 4, 5, 6, 6, 7, 7],
    [5, 5, 6, 6, 7, 7, 7], [5, 5, 6, 7, 7, 7, 7],
]