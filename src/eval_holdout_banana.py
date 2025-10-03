import pandas as pd
from joblib import load
from pathlib import Path
from sklearn.metrics import mean_absolute_error

'''
Evaluates MAE on specific banana sets(1,2 or 3)
'''

DATA = Path(__file__).resolve().parent.parent / "data" / "processed" / "banana_images.csv"
MODEL = Path(__file__).resolve().parent.parent / "data" / "models" / "banana_from_images.joblib"

#Banana ID for test
BANANA_ID = 3

df = pd.read_csv(DATA)
hold = df[df["banana_id"] == BANANA_ID].copy()

if hold.empty:
    raise SystemExit(f"No rows for banana_id={BANANA_ID}. Check your labels.csv and processed CSV.")

X = hold[["day","yellow_pct", "dark_pct"]]
y_true = hold["days_to_death"].astype(float)

model = load(MODEL)
y_pred = model.predict(X)

mae = mean_absolute_error(y_true, y_pred)
print(f"Hold-out banana {BANANA_ID} → MAE: {mae:.2f} days  (n={len(hold)})")
print("Sample preds:", [round(v,2) for v in y_pred[:5]])