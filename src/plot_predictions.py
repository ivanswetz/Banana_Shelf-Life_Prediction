import pandas as pd
import matplotlib.pyplot as plt
from joblib import load

'''
Builds plots for each banana set
'''

DATA = "../data/processed/banana_images.csv"
MODEL = "../data/models/banana_from_images.joblib"

#Load data + model
df = pd.read_csv(DATA)
model = load(MODEL)

#Loop over each banana_id
for bid in df["banana_id"].unique():
    banana = df[df["banana_id"] == bid].copy()

    #Features
    X = banana[["day","yellow_pct","dark_pct"]]
    y_true = banana["days_to_death"].values
    y_pred = model.predict(X)

    #Plot
    plt.figure(figsize=(6,4))
    plt.plot(banana["day"], y_true, "o-", label="True days_to_death")
    plt.plot(banana["day"], y_pred, "s--", label="Predicted")
    plt.title(f"Banana {bid} — True vs. Predicted")
    plt.xlabel("Day since start")
    plt.ylabel("Days to death")
    plt.legend()
    plt.grid(True)
    plt.show()
