import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# -----------------------------------
# Step 1: Load dataset
# -----------------------------------
data = pd.read_csv("Assignment1/sales_data.csv")

print("Original Dataset Shape:", data.shape)
print(data.head())

# -----------------------------------
# Step 2: Handle missing values
# -----------------------------------
data = data.fillna(data.mean())

# -----------------------------------
# Step 3: Feature-label separation
# -----------------------------------
X = data[['date','product','category','price','quantity']]
y = data['revenue']


products = ['Smartphone', 'Laptop', 'T-Shirt', 'Headphones' ,'Watch' ,'Tablet', 'Coat',
 'Smartwatch', 'Speaker', 'Backpack', 'Hoodie' ,'Sneakers', 'Wallet', 'Jeans']
categories = ['Electronics', 'Clothing', 'Accessories', 'Bags', 'Shoes', 'Clohting', 'Bgas','Shoeses']


print("\n\n")
print(X['product'].unique())

print("\n\n")
print(X['category'].unique())

# -----------------------------------
# Step 4: Normalize features
# -----------------------------------
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------------
# Step 5: Train-test split
# -----------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.25, random_state=42
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# -----------------------------------
# Step 6: Robust federated partitioning
# -----------------------------------
def create_federated_clients(X, y, num_clients):
    """
    Safely splits data across federated clients.
    Works even for small datasets.
    """
    client_data = {}
    indices = np.arange(len(X))
    np.random.shuffle(indices)

    splits = np.array_split(indices, num_clients)

    for i, idx in enumerate(splits):
        client_data[f'client_{i+1}'] = {
            'X': X[idx],
            'y': y.iloc[idx]
        }

    return client_data

# -----------------------------------
# Step 7: Create federated clients
# -----------------------------------
num_clients = 3   # SAFE for small dataset
clients = create_federated_clients(X_train, y_train, num_clients)

# -----------------------------------
# Step 8: Display client data info
# -----------------------------------
for client, data in clients.items():
    print(f"\n{client}")
    print("Local X shape:", data['X'].shape)
    print("Local y shape:", data['y'].shape)
