import os
import pandas as pd
from img_features import color_percents

'''
Loops through labeled banana sets, extracts features, and builds banana_images.csv
'''

os.makedirs("../data/processed", exist_ok=True) #ensure exists

def main():
    labels = pd.read_csv("../data/labels.csv")
    rows = []

    for i, row in labels.iterrows():
        path = row["image_path"]
        try:
            y, d = color_percents(path)  #yellow %, dark %
        except Exception as e:
            print(f"[WARN] Skipping {path}: {e}")
            continue

        out = {
            "banana_id": row["banana_id"],
            "day": row["day"],
            "death_day": row.get("death_day", None),
            "yellow_pct": y,
            "dark_pct": d,
            "image_path": path,
        }
        rows.append(out)

    df = pd.DataFrame(rows)

    #compute target
    if "death_day" in df.columns:
        df["days_to_death"] = df["death_day"] - df["day"]
        df = df[df["days_to_death"].notna() & (df["days_to_death"] >= 0)].copy()

    out_path = "../data/processed/banana_images.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path} with shape {df.shape} and {df['banana_id'].nunique()} bananas")

if __name__ == "__main__":
    main()
