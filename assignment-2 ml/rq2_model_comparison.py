import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Create folders
os.makedirs("tables", exist_ok=True)
os.makedirs("figures", exist_ok=True)

# Load dataset
df = pd.read_csv("btcusd_1-min_data.csv")

# 🔥 SPEED FIX 1: Use only recent data (reduce size)
df = df.tail(50000)   # adjust: 20k–100k depending on your system

# Time handling
df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
df = df.sort_values('Timestamp')

# Feature engineering
df['Close_lag1'] = df['Close'].shift(1)
df['Target'] = df['Close'].shift(-1)
df = df.dropna()

# Train-test split
split = int(len(df) * 0.8)
train = df[:split]
test = df[split:]

X_train = train[['Close_lag1']]
y_train = train['Target']

X_test = test[['Close_lag1']]
y_test = test['Target']

# Scaling (for SVR)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Models
models = {
    "Linear Regression": LinearRegression(),

    # 🔥 SPEED FIX 2: Smaller Random Forest
    "Random Forest": RandomForestRegressor(
        n_estimators=50,   # reduced from 100+
        max_depth=10,
        random_state=42,
        n_jobs=-1
    ),

    # 🔥 SPEED FIX 3: Faster SVR (simplified)
    "SVR": SVR(C=10, gamma='scale')
}

results = []

# Train & evaluate
for name, model in models.items():

    if name == "SVR":
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    results.append([name, mae, rmse, r2])

# Save table
results_df = pd.DataFrame(results, columns=["Model", "MAE", "RMSE", "R2"])
results_df.to_csv("tables/RQ2_model_comparison.csv", index=False)

# Plot
plt.figure()
results_df.set_index("Model")[["MAE", "RMSE", "R2"]].plot(kind="bar")
plt.title("RQ2: Model Comparison")
plt.xticks(rotation=15)

plt.savefig("figures/RQ2_model_comparison.pdf")
plt.close()

print("RQ2 completed! Fast execution.")