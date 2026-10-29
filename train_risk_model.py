"""Train the EdgeErgo risk classifier from your own labelled clips.

1. In edgeegroasses.py press 1 (Low), 2 (Medium) or 3 (High), then SPACE to record a clip.
   Every recorded frame is saved with that label to edgeergo_train.csv.
2. Run:  python train_risk_model.py
3. It writes risk_model.joblib. edgeegroasses.py loads it automatically next time.
"""
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit, train_test_split

CSV = "edgeergo_train.csv"
OUT = "risk_model.joblib"
FEATURES = ["neck_deg", "s", "drop", "ratio", "ang", "lat", "tilt"]
LABELS = ["Low", "Medium", "High"]

df = pd.read_csv(CSV).dropna(subset=FEATURES + ["label"])
print(f"Loaded {len(df)} frames from {df['clip_id'].nunique()} clips")
print(df.groupby("label")["clip_id"].nunique().rename("clips per class").to_string())
print(df["label"].value_counts().rename("frames per class").to_string())

missing = [l for l in LABELS if l not in set(df["label"])]
if missing:
    sys.exit(f"Need recordings for: {missing}. Record clips for every class first.")

clips_per_class = df.groupby("label")["clip_id"].nunique().min()
X, y, groups = df[FEATURES].values, df["label"].values, df["clip_id"].values

if clips_per_class >= 4:
    # Split by CLIP so frames of the same recording never appear in both train and test.
    # Hold out about 25% of the clips of EACH class so every class is tested.
    rng = np.random.default_rng(42)
    test_clips = set()
    for lab in LABELS:
        clips = np.array(df.loc[df["label"] == lab, "clip_id"].astype(str).unique())
        rng.shuffle(clips)
        test_clips.update(clips[: max(1, round(0.25 * len(clips)))])
    is_test = df["clip_id"].isin(test_clips).values
    tr, te = np.where(~is_test)[0], np.where(is_test)[0]
    split_note = "clip-wise split (honest estimate)"
else:
    print("\nWARNING: fewer than 4 clips in some class - using a frame-wise split. "
          "Accuracy will be OPTIMISTIC because neighbouring frames are near-identical. "
          "Record more clips for a trustworthy number.")
    tr, te = train_test_split(np.arange(len(df)), test_size=0.25, random_state=42, stratify=y)
    split_note = "frame-wise split (optimistic)"

clf = RandomForestClassifier(
    n_estimators=200, max_depth=8, min_samples_leaf=5,
    class_weight="balanced", random_state=42, n_jobs=-1,
)
clf.fit(X[tr], y[tr])
pred = clf.predict(X[te])

print(f"\nTest accuracy ({split_note}): {accuracy_score(y[te], pred) * 100:.1f}%")
print("\nConfusion matrix (rows = true, cols = predicted):", LABELS)
print(confusion_matrix(y[te], pred, labels=LABELS))
print("\n", classification_report(y[te], pred, labels=LABELS, zero_division=0))
print("Feature importance:")
for f, imp in sorted(zip(FEATURES, clf.feature_importances_), key=lambda t: -t[1]):
    print(f"  {f:9s} {imp:.3f}")

# Refit on ALL data for the deployed model
clf.fit(X, y)
joblib.dump(clf, OUT)
print(f"\nSaved model to {OUT}")