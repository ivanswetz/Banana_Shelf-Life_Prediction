import pandas as pd
from joblib import load
from img_features import color_percents

'''
Inference script: take a single photo, extract features, run trained model, print days-to-death
'''

def main():
    model = load("../data/models/banana_from_images.joblib")
    img_path = r"C:\Users\Ivan\Desktop\projk\wiw.jpg"

    y_pct, d_pct = color_percents(img_path)

    X = pd.DataFrame([{
        "day": float("nan"),
        "yellow_pct": y_pct,
        "dark_pct": d_pct
    }])

    pred = model.predict(X)[0]
    print(f"From photo: yellow={y_pct:.1f}%, dark={d_pct:.1f}%")
    print(f"Predicted days to death: {pred:.2f}")

if __name__ == "__main__":
    main()
