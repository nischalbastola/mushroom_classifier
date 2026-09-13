import numpy as np
import pandas as pd

df = pd.read_csv("data.csv")

print(df.shape)
print(df["Mushroom_quality"].value_counts())
print(df.isna().sum())

# Keep unknown categorical values as an explicit category.
df = df.replace("?", "unknown")

# Encode the target separately because it is the value the model must predict.
y = df["Mushroom_quality"].map({"e": 0, "p": 1}).to_numpy(dtype=np.int8)

# Remove the target and the constant feature before one-hot encoding.
features = df.drop(columns=["Mushroom_quality", "veil_type"])

# Create one binary column for every category in every feature.
X = pd.get_dummies(features, dtype=np.int8).to_numpy(dtype=np.int8)

print("Encoded feature matrix:", X.shape)
print("Target vector:", y.shape)