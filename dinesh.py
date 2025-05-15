# EBPL-DS: Forecasting House Prices Accurately Using Smart Regression Techniques

# ✅ Step 1: Import Libraries
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# ✅ Step 2: Load Sample Housing Dataset (from seaborn)
df = sns.load_dataset("diamonds").dropna().copy()

# ✅ Step 3: Simulate a "House Price" dataset
df = df[["carat", "depth", "table", "x", "y", "z", "price"]]
df.columns = ['Size', 'Depth', 'Table', 'Width', 'Height', 'Length', 'SalePrice']

# ✅ Step 4: Preprocessing
X = df.drop("SalePrice", axis=1)
y = df["SalePrice"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ✅ Step 5: Split Data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ✅ Step 6: Train Models
lr = LinearRegression()
rf = RandomForestRegressor(n_estimators=100, random_state=42)

lr.fit(X_train, y_train)
rf.fit(X_train, y_train)

# ✅ Step 7: Predict
lr_preds = lr.predict(X_test)
rf_preds = rf.predict(X_test)

# ✅ Step 8: Evaluate
print("🔍 Linear Regression:")
print("RMSE:", np.sqrt(mean_squared_error(y_test, lr_preds)))
print("R²:", r2_score(y_test, lr_preds))

print("\n🌲 Random Forest Regressor:")
print("RMSE:", np.sqrt(mean_squared_error(y_test, rf_preds)))
print("R²:", r2_score(y_test, rf_preds))

# ✅ Step 9: Visualization
plt.figure(figsize=(8, 4))
plt.scatter(y_test, rf_preds, alpha=0.3, label="Predicted vs Actual")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Random Forest Predictions")
plt.legend()
plt.grid(True)
plt.show()
