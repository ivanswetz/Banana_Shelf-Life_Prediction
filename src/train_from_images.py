import os, numpy as np, pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.metrics import mean_absolute_error, make_scorer
from joblib import dump

'''
Same as #pseudo_label_and_train but only for labeled data(banana sets 1,2,3)
'''

OUT = Path(__file__).resolve().parent.parent / "data" / "models"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv("../data/processed/banana_images.csv")

X = df[["day", "yellow_pct", "dark_pct"]]
y = df["days_to_death"].astype(float)
groups = df["banana_id"]

pre = ColumnTransformer([
    ("num", Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale",  StandardScaler())
    ]), ["day", "yellow_pct", "dark_pct"])
])

pipe = Pipeline([
    ("pre", pre),
    ("rf", RandomForestRegressor(n_estimators=600, random_state=42))
])

gkf = GroupKFold(n_splits=min(5, max(2, groups.nunique())))
mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)
mae_scores = -cross_val_score(pipe, X, y, cv=gkf, groups=groups, scoring=mae_scorer)

print("Per-fold MAE (days):", np.round(mae_scores, 3).tolist())
print("Mean MAE (days):", round(mae_scores.mean(), 3))

pipe.fit(X, y)
dump(pipe, OUT / "banana_from_images.joblib")
print("Saved model to", OUT / "banana_from_images.joblib")
