import pandas as pd, numpy as np
from pathlib import Path
from joblib import load, dump
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.metrics import mean_absolute_error, make_scorer

'''
Semi-supervised model → adds pseudo-labels from isotonic calibration on unlabeled data.
Trains weighted Random Forest and saves banana_from_images.joblib
'''

LABELED = "../data/processed/banana_images.csv"
UNLAB   = "../data/processed/unlabeled_features.csv"
CALIB   = "../data/models/days_from_dark_iso.joblib"

Path("../data/models").mkdir(parents=True, exist_ok=True)
cv_mae = None

#1 - Load

labeled = pd.read_csv(LABELED)
unlab   = pd.read_csv(UNLAB)

#clean labeled (be safe)

labeled["yellow_pct"] = pd.to_numeric(labeled["yellow_pct"], errors="coerce")
labeled["dark_pct"]   = pd.to_numeric(labeled["dark_pct"],   errors="coerce")
labeled["days_to_death"] = pd.to_numeric(labeled["days_to_death"], errors="coerce")
labeled = labeled.dropna(subset=["yellow_pct","dark_pct","days_to_death"])

#2 - Pseudo-label unlabeled using isotonic (only where dark_pct is valid)

iso = load(CALIB)
unlab["yellow_pct"] = pd.to_numeric(unlab["yellow_pct"], errors="coerce")
unlab["dark_pct"]   = pd.to_numeric(unlab["dark_pct"],   errors="coerce")
unlab = unlab.dropna(subset=["dark_pct"])  #only predict where dark_pct is real
unlab["days_to_death"] = iso.predict(unlab["dark_pct"].values)

#3 - Clipped to a sane range

y_max = labeled["days_to_death"].max()
unlab["days_to_death"] = unlab["days_to_death"].clip(lower=0, upper=y_max)

#4 - Combined with weights

use_cols_L = ["banana_id","day","yellow_pct","dark_pct","days_to_death"]  #labeled has day
use_cols_U = ["yellow_pct","dark_pct","days_to_death"]                    #unlabeled has no day

L = labeled[use_cols_L].copy()
U = unlab[use_cols_U].copy()
U["banana_id"] = np.nan
U["day"] = np.nan

L["weight"] = 3.0
U["weight"] = 1.0

combo = pd.concat([L, U], ignore_index=True)
combo = combo.dropna(subset=["yellow_pct","dark_pct","days_to_death"])


X = combo[["day","yellow_pct","dark_pct"]]
y = combo["days_to_death"].astype(float)
w = combo["weight"].values

#5 - Labeled-only CV

lab_groups = labeled["banana_id"]
n_splits = min(5, max(2, lab_groups.nunique()))

gkf = GroupKFold(n_splits=n_splits)

pre_cv = ColumnTransformer([
    ("num", Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale",  StandardScaler())
    ]), ["day","yellow_pct","dark_pct"])
])

pipe_cv = Pipeline([
    ("pre", pre_cv),
    ("rf", RandomForestRegressor(n_estimators=500, random_state=42))
])

X_lab = L[["day","yellow_pct","dark_pct"]]
y_lab = L["days_to_death"].astype(float)

mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)
cv_scores = -cross_val_score(pipe_cv, X_lab, y_lab, cv=gkf,
                             groups=lab_groups, scoring=mae_scorer)

cv_mae = cv_scores.mean()

#6 - Final fit
pipe = Pipeline([("pre", ColumnTransformer([
                        ("num", Pipeline([
                            ("impute", SimpleImputer(strategy="median")),
                            ("scale",  StandardScaler())
                        ]), ["day","yellow_pct","dark_pct"])
                    ])),
                 ("rf", RandomForestRegressor(n_estimators=700, random_state=42))])

pipe.fit(X, y, rf__sample_weight=w)

#7 - MAE on the combined training data (labeled + pseudo-labeled(1000))
y_pred = pipe.predict(X)
mae_train = mean_absolute_error(y, y_pred)
print("Train-set MAE (days):", round(mae_train, 3))

dump(pipe, "../data/models/banana_from_images.joblib")
print("Saved final model to ../data/models/banana_from_images.joblib")

if cv_mae is not None:
    print("Reference (labeled-only) CV MAE (days):", round(cv_mae, 3))