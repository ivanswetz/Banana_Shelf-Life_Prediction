import pandas as pd
from sklearn.isotonic import IsotonicRegression
from joblib import dump
import os

'''
Fits an isotonic regression mapping from dark% → days_to_death (monotonic decreasing). Saves model as days_from_dark_iso.joblib.
'''

LABELED = "../data/processed/banana_images.csv"
os.makedirs("../data/models", exist_ok=True)  #ensure exists

df = pd.read_csv(LABELED)
#Photo-only features
X_dark = df["dark_pct"].values
y_days = df["days_to_death"].astype(float).values

#higher dark: smaller days
iso = IsotonicRegression(increasing=False, y_min=0.0)  #decreasing function
iso.fit(X_dark, y_days)

dump(iso, "../data/models/days_from_dark_iso.joblib")
print("Saved ../data/models/days_from_dark_iso.joblib")
print(f"Dark range seen: {X_dark.min():.1f} .. {X_dark.max():.1f}")
