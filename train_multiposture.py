"""Train a sitting-posture risk classifier on the public MultiPosture dataset.

Dataset (CC BY 4.0): https://zenodo.org/records/14230872   -> download data.csv
Usage:   python train_multiposture.py data.csv

It evaluates with a LEAVE-SUBJECTS-OUT split, so the accuracy it prints is for people
the model has never seen (much more honest than a random frame split).
"""
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupKFold

CSV = sys.argv[1] if len(sys.argv) > 1 else "data.csv"
OUT = "multiposture_model.joblib"

LABEL_MAP = {
    "TUP": "Low",
    "TLF": "High",
    "TLB": "Medium",
    "TLR": "Medium",
    "TLL": "Medium",
}
LANDMARKS = ["nose", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
             "left_elbow", "right_elbow", "left_hip", "right_hip"]

df = pd.read_csv(CSV)
print(f"Loaded {len(df)} frames, {df['subject'].nunique()} subjects")
print("Upper-body label codes found:\n", df["upperbody_label"].value_counts().to_string())

unknown = sorted(set(df["upperbody_label"]) - set(LABEL_MAP))
if unknown:
    print(f"\nWARNING: codes not in LABEL_MAP are ignored: {unknown}")
df = df[df["upperbody_label"].isin(LABEL_MAP)].copy()
df["risk"] = df["upperbody_label"].map(LABEL_MAP)
print("\nRisk classes after mapping:\n", df["risk"].value_counts().to_string())


def xyz(name):
    return df[[f"{name}_x", f"{name}_y", f"{name}_z"]].to_numpy(dtype=float)


ls, rs = xyz("left_shoulder"), xyz("right_shoulder")
lh, rh = xyz("left_hip"), xyz("right_hip")
sw = np.linalg.norm(ls - rs, axis=1)
sw[sw < 1e-6] = 1e-6                                   # scale-free: divide by shoulder width

feats = {}
for name in LANDMARKS:
    p = xyz(name)
    for i, axis in enumerate("xyz"):
        feats[f"{name}_{axis}"] = p[:, i] / sw

sho_mid, hip_mid = (ls + rs) / 2, (lh + rh) / 2
trunk = sho_mid - hip_mid                              
nose = xyz("nose")
feats["trunk_fwd_deg"] = np.degrees(np.arctan2(trunk[:, 2], -trunk[:, 1]))     # 
feats["shoulder_tilt"] = np.degrees(np.arctan2(ls[:, 1] - rs[:, 1], ls[:, 0] - rs[:, 0]))

X = pd.DataFrame(feats).to_numpy()
y = df["risk"].to_numpy()
groups = df["subject"].to_numpy()
labels = [l for l in ["Low", "Medium", "High"] if l in set(y)]

n_splits = min(5, len(set(groups)))
all_true, all_pred, accs = [], [], []
for fold, (tr, te) in enumerate(GroupKFold(n_splits=n_splits).split(X, y, groups), 1):
    clf = RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_leaf=3,
                                 class_weight="balanced", random_state=42, n_jobs=-1)
    clf.fit(X[tr], y[tr])
    pred = clf.predict(X[te])
    accs.append(accuracy_score(y[te], pred))
    all_true += list(y[te])
    all_pred += list(pred)
    print(f"Fold {fold}: accuracy on unseen subjects = {accs[-1] * 100:.1f}%")

print(f"\nMean accuracy over {n_splits} subject-wise folds: {np.mean(accs) * 100:.1f}%")
print("\nConfusion matrix (rows = true, cols = predicted):", labels)
print(confusion_matrix(all_true, all_pred, labels=labels))
print("\n", classification_report(all_true, all_pred, labels=labels, zero_division=0))

final = RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_leaf=3,
                               class_weight="balanced", random_state=42, n_jobs=-1)
final.fit(X, y)
joblib.dump({"model": final, "feature_names": list(feats)}, OUT)
print(f"Saved model to {OUT}")