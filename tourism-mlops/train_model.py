import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

data = pd.read_csv("data/processed_data.csv")

X = data.drop("ProdTaken", axis=1)
y = data["ProdTaken"]

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X,y)

joblib.dump(model, "best_model.pkl")
print("Model trained")
