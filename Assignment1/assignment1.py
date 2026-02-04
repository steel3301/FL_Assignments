import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

data = pd.read_csv("Assignment1/sales_data.csv").dropna()
X = data[['price','quantity']]
y = data['revenue']

clients = np.array_split(data, 3)

def train(client_df):
    Xc = client_df[['price','quantity']]
    yc = client_df['revenue']
    m = LinearRegression()
    m.fit(Xc, yc)
    return m.coef_, m.intercept_

coefs, intercepts = [], []
for c in clients:
    coef, intercept = train(c)
    coefs.append(coef)
    intercepts.append(intercept)

global_coef = np.mean(coefs, axis=0)
global_intercept = np.mean(intercepts)

print("Global coefficients:", global_coef)
print("Global intercept:", global_intercept)