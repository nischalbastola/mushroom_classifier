import numpy as np
import pandas as pd


def gini_impurity(labels):
    """Return the Gini impurity of a group of labels."""
    if len(labels) == 0:
        return 0.0

    _, counts = np.unique(labels, return_counts=True)
    probabilities = counts / len(labels)

    return 1.0 - np.sum(probabilities ** 2)


def split_gini_impurity(left_labels, right_labels):
    """Return weighted Gini impurity after a binary split."""
    total_samples = len(left_labels) + len(right_labels)

    if total_samples == 0:
        return 0.0

    left_weight = len(left_labels) / total_samples
    right_weight = len(right_labels) / total_samples

    return (
        left_weight * gini_impurity(left_labels)
        + right_weight * gini_impurity(right_labels)
    )


def find_best_split(X_data, y_data, feature_indices=None):
    """Find the feature with the lowest weighted Gini impurity."""
    best_feature = None
    best_gini = float("inf")

    if feature_indices is None:
        feature_indices = range(X_data.shape[1])

    for feature_index in feature_indices:
        left_mask = X_data[:, feature_index] == 0
        right_mask = X_data[:, feature_index] == 1

        left_labels = y_data[left_mask]
        right_labels = y_data[right_mask]

        if len(left_labels) == 0 or len(right_labels) == 0:
            continue

        current_gini = split_gini_impurity(left_labels, right_labels)

        if current_gini < best_gini:
            best_gini = current_gini
            best_feature = feature_index

    return best_feature, best_gini


def majority_class(labels):
    """Return the most common class in a group of labels."""
    return int(np.bincount(labels).argmax())


def build_tree(
    X_data,
    y_data,
    depth=0,
    max_depth=8,
    min_samples_split=2,
    max_features=None,
    rng=None,
):
    """Recursively build a binary decision tree from one-hot features."""
    node = {
        "prediction": majority_class(y_data),
        "depth": depth,
        "samples": len(y_data),
    }

    if (
        len(y_data) == 0
        or len(np.unique(y_data)) == 1
        or depth >= max_depth
        or len(y_data) < min_samples_split
    ):
        node["leaf"] = True
        return node

    if max_features is None:
        max_features = max(1, int(np.sqrt(X_data.shape[1])))

    if rng is None:
        rng = np.random.default_rng(42)

    feature_indices = rng.choice(
        X_data.shape[1],
        size=min(max_features, X_data.shape[1]),
        replace=False,
    )
    feature_index, split_gini = find_best_split(X_data, y_data, feature_indices)
    if feature_index is None or split_gini >= gini_impurity(y_data):
        node["leaf"] = True
        return node

    left_mask = X_data[:, feature_index] == 0
    right_mask = ~left_mask

    node["leaf"] = False
    node["feature_index"] = feature_index
    node["gini"] = split_gini
    node["left"] = build_tree(
        X_data[left_mask],
        y_data[left_mask],
        depth + 1,
        max_depth,
        min_samples_split,
        max_features,
        rng,
    )
    node["right"] = build_tree(
        X_data[right_mask],
        y_data[right_mask],
        depth + 1,
        max_depth,
        min_samples_split,
        max_features,
        rng,
    )
    return node


def predict_tree(node, row):
    """Predict one label by walking from the root to a leaf."""
    if node["leaf"]:
        return node["prediction"]

    if row[node["feature_index"]] == 0:
        return predict_tree(node["left"], row)
    return predict_tree(node["right"], row)


def predict_tree_batch(tree, X_data):
    """Predict labels for every row in a feature matrix."""
    return np.array([predict_tree(tree, row) for row in X_data], dtype=np.int8)


def train_random_forest(
    X_data,
    y_data,
    n_trees=25,
    max_depth=8,
    min_samples_split=2,
    seed=123,
):
    """Train trees on bootstrap samples with random features at each split."""
    rng = np.random.default_rng(seed)
    trees = []

    for _ in range(n_trees):
        sample_indices = rng.integers(0, len(X_data), size=len(X_data))
        tree = build_tree(
            X_data[sample_indices],
            y_data[sample_indices],
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            rng=rng,
        )
        trees.append(tree)

    return trees


def predict_forest(trees, X_data):
    """Predict by majority vote across all trees."""
    tree_predictions = np.array(
        [predict_tree_batch(tree, X_data) for tree in trees],
        dtype=np.int8,
    )
    return (np.mean(tree_predictions, axis=0) >= 0.5).astype(np.int8)


def evaluate_predictions(actual, predicted):
    """Return accuracy and a confusion matrix in [actual, predicted] order."""
    confusion_matrix = np.zeros((2, 2), dtype=np.int64)
    np.add.at(confusion_matrix, (actual, predicted), 1)

    accuracy = np.mean(actual == predicted)
    poisonous_recall = confusion_matrix[1, 1] / max(
        1, confusion_matrix[1].sum()
    )
    return accuracy, poisonous_recall, confusion_matrix


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


print("\nGini tests:")
print("Pure edible:", gini_impurity(np.array([0, 0, 0])))
print("Mixed:", gini_impurity(np.array([0, 1])))
print("Training set:", gini_impurity(y_train))

best_feature, best_gini = find_best_split(X_train, y_train)
print("\nBest split:")
print("Feature index:", best_feature)
print("Feature name:", encoded_features.columns[best_feature])
print("Weighted Gini:", best_gini)

tree = build_tree(X_train, y_train, max_depth=8)
tree_predictions = predict_tree_batch(tree, X_test)
tree_accuracy, tree_recall, tree_matrix = evaluate_predictions(
    y_test, tree_predictions
)

print("\nDecision tree:")
print("Test accuracy:", tree_accuracy)
print("Poisonous recall:", tree_recall)
print("Confusion matrix:\n", tree_matrix)

forest = train_random_forest(X_train, y_train, n_trees=25, max_depth=8)
forest_predictions = predict_forest(forest, X_test)
forest_accuracy, forest_recall, forest_matrix = evaluate_predictions(
    y_test, forest_predictions
)

print("\nRandom forest:")
print("Trees:", len(forest))
print("Test accuracy:", forest_accuracy)
print("Poisonous recall:", forest_recall)
print("Confusion matrix:\n", forest_matrix)
