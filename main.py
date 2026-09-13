import numpy as np
import pandas as pd

df = pd.read_csv("data.csv")

print(df.shape)
print(df["Mushroom_quality"].value_counts())
print(df.isna().sum())

# kept unknown as a seperate category instead of dropping it.
df = df.replace("?", "unknown")

# Seperataing Y values 
y = df["Mushroom_quality"].map({"e": 0, "p": 1}).to_numpy(dtype=np.int8)

# removing target and veil_type feature as it has only one category.
features = df.drop(columns=["Mushroom_quality", "veil_type"])

# one hot encoding. (X is the feature matrix)
encoded_features = pd.get_dummies(features, dtype=np.int8)
X = encoded_features.to_numpy(dtype=np.int8)

print("Original feature count:", features.shape[1])
print("Encoded feature count:", encoded_features.shape[1])

print("Encoded feature matrix:", X.shape)
print("Target vector:", y.shape)

# Split the data into training and testing sets.
rng = np.random.default_rng(42)

shuffled = rng.permutation(len(X))
split_index = int(len(X) * 0.8)

train_indices = shuffled[:split_index]
test_indices = shuffled[split_index:]

X_train = X[train_indices]
X_test = X[test_indices]
y_train = y[train_indices]
y_test = y[test_indices]


