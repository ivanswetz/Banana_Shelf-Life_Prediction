import os, glob, pandas as pd
from img_features import color_percents
os.makedirs("../data/processed", exist_ok=True)

'''
Processes 1000 unlabeled banana photos, extracts yellow/dark%, and saves as unlabeled_features.csv
'''

FOLDER = "../data/raw/banana_set_large"      #1000 images
OUT = "../data/processed/unlabeled_features.csv"

rows = []
paths = sorted(glob.glob(os.path.join(FOLDER, "*.*")))
for i, p in enumerate(paths, 1):
    try:
        y, d = color_percents(p)
        rows.append({"image_path": p, "yellow_pct": y, "dark_pct": d})
        print(f"appended succsessfully {i}")
    except Exception as e:
        print(f"[skip] {p}: {e}")

df = pd.DataFrame(rows)
df.to_csv(OUT, index=False)
print(f"Wrote {OUT} with {len(df)} rows")