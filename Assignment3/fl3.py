import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Load dataset
data = pd.read_csv("Assignment3/Housing.xls")

# Select numeric features only
features = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']
target = 'price'

X = data[features].values
y = data[target].values
scaler = StandardScaler()
X = scaler.fit_transform(X)


num_clients = 5

X_splits = np.array_split(X, num_clients)
y_splits = np.array_split(y, num_clients)

client_datasets = list(zip(X_splits, y_splits))


def train_local_model(X, y, w, b, lr=0.01, epochs=20):
    n = len(y)
    for _ in range(epochs):
        y_pred = X @ w + b
        error = y_pred - y
        dw = (1/n) * X.T @ error
        db = (1/n) * np.sum(error)
        w -= lr * dw
        b -= lr * db
    return w, b
def fedavg(local_weights, local_biases, data_sizes):
    total_samples = sum(data_sizes)
    w_global = np.zeros_like(local_weights[0])
    b_global = 0.0

    for w, b, size in zip(local_weights, local_biases, data_sizes):
        w_global += (size / total_samples) * w
        b_global += (size / total_samples) * b

    return w_global, b_global



num_features = X.shape[1]
w_global = np.zeros(num_features)
b_global = 0.0

rounds = 200

for r in range(rounds):
    local_weights = []
    local_biases = []
    data_sizes = []

    for X_c, y_c in client_datasets:
        w_local, b_local = train_local_model(
            X_c, y_c,
            w_global.copy(),
            b_global
        )

        local_weights.append(w_local)
        local_biases.append(b_local)
        data_sizes.append(len(y_c))

    # Server aggregation
    w_global, b_global = fedavg(
        local_weights, local_biases, data_sizes
    )

    # Evaluate global model
    y_pred_global = X @ w_global + b_global
    mse = mean_squared_error(y, y_pred_global)

    print(f"Round {r+1:02d} | Global MSE: {mse:.2f}")



print("\nFinal Global Model Parameters")
print("Weights:", w_global)
print("Bias:", b_global)
