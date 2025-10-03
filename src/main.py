import pandas as pd
import cv2

'''
This main is used for data manipulation to not enter data manually"
'''

data = [
    {"banana_id": 1, "day": 0,  "death_day": 11},
    {"banana_id": 1, "day": 1,  "death_day": 11},
    {"banana_id": 1, "day": 2,  "death_day": 11},
    {"banana_id": 1, "day": 3,  "death_day": 11},
    {"banana_id": 1, "day": 4,  "death_day": 11},
    {"banana_id": 1, "day": 5,  "death_day": 11},
    {"banana_id": 1, "day": 6,  "death_day": 11},
    {"banana_id": 1, "day": 7,  "death_day": 11},
    {"banana_id": 1, "day": 8,  "death_day": 11},
    {"banana_id": 1, "day": 9,  "death_day": 11},
    {"banana_id": 1, "day": 10, "death_day": 11},
    {"banana_id": 1, "day": 11, "death_day": 11},
    {"banana_id": 1, "day": 12, "death_day": 11},
    {"banana_id": 1, "day": 13, "death_day": 11},
    {"banana_id": 1, "day": 14, "death_day": 11},
    {"banana_id": 1, "day": 15, "death_day": 11},
    {"banana_id": 2, "day": 0, "death_day": 5},
    {"banana_id": 2, "day": 1, "death_day": 5},
    {"banana_id": 2, "day": 2, "death_day": 5},
    {"banana_id": 2, "day": 3, "death_day": 5},
    {"banana_id": 2, "day": 4, "death_day": 5},
    {"banana_id": 2, "day": 5, "death_day": 5},
    {"banana_id": 3, "day": 0, "death_day": 6},
    {"banana_id": 3, "day": 1, "death_day": 6},
    {"banana_id": 3, "day": 2, "death_day": 6},
    {"banana_id": 3, "day": 3, "death_day": 6},
    {"banana_id": 3, "day": 3, "death_day": 6},
    {"banana_id": 3, "day": 4, "death_day": 6},
    {"banana_id": 3, "day": 5, "death_day": 6},
    {"banana_id": 3, "day": 6, "death_day": 6},
    {"banana_id": 3, "day": 7, "death_day": 6}

]

#set 1
data1 = [{"banana_id": 1, "day": d, "death_day": 11} for d in range(16)]
df1 = pd.DataFrame(data1)
df1["image_path"] = df1["day"].apply(lambda x: f"../data/raw/banana_set_1/Screenshot_{x + 1}.jpg")

#set 2
data2 = [{"banana_id": 2, "day": d, "death_day": 5} for d in range(6)]
df2 = pd.DataFrame(data2)
df2["image_path"] = df2["day"].apply(lambda x: f"../data/raw/banana_set_2/Screenshot_{x + 1}.jpg")

#set 3
data3 = [{"banana_id": 3, "day": d, "death_day": 6} for d in range(8)]
df3 = pd.DataFrame(data3)
df3["image_path"] = df3["day"].apply(lambda x: f"../data/raw/banana_set_3/Screenshot_{x + 1}.jpg")

#combination 3 sets
df = pd.concat([df1, df2, df3], ignore_index=True)

df.to_csv("../data/labels.csv", index=False)
print("Saved to data/labels.csv with", len(df), "rows")