import pandas as pd

train = pd.read_csv("data/train_data.csv")

train = train.dropna()

train.to_csv("data/processed_data.csv", index=False)

print("Data preprocessing done")
