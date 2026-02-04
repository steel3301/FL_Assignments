import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

train_df = pd.read_csv("Assignment2/train.csv")

y = train_df["SalePrice"]
X = train_df.drop("SalePrice", axis=1)

num_cols = X.select_dtypes(include=["int64", "float64"]).columns
cat_cols = X.select_dtypes(include=["object"]).columns

for col in num_cols:
    X[col] = X[col].fillna(X[col].mean())

for col in cat_cols:
    X[col] = X[col].fillna(X[col].mode()[0])

X = pd.get_dummies(X, drop_first=True)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

data = pd.concat([X_scaled, y.reset_index(drop=True)], axis=1)

num_clients = 5
client_data = np.array_split(data, num_clients)

clients = {}

for i in range(num_clients):
    client_train, client_test = train_test_split(client_data[i], test_size=0.2, random_state=42)
    clients[f"client_{i+1}"] = {
        "train": client_train,
        "test": client_test
    }

for c in clients:
    print(c, clients[c]["train"].shape, clients[c]["test"].shape)
