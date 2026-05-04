import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Create folders
os.makedirs("tables", exist_ok=True)
os.makedirs("figures", exist_ok=True)

# Load data
df = pd.read_csv("btcusd_1-min_data.csv")

# 🔥 Speed optimization
df = df.tail(50000)

# Time handling
df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
df = df.sort_values('Timestamp')

# Feature engineering (simple)
df['lag1'] = df['Close'].shift(1)
df['Target'] = df['Close'].shift(-1)
df = df.dropna()

# Train-test split
split = int(len(df) * 0.8)

X = df[['lag1']]
y = df['Target']

# Helper function
def evaluate(X_train, X_test, y_train, y_test):
    model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return (
        mean_absolute_error(y_test, y_pred),
        np.sqrt(mean_squared_error(y_test, y_pred)),
        r2_score(y_test, y_pred)
    )

results = []

# =========================
# Scenario 1: Normal
# =========================
mae, rmse, r2 = evaluate(X[:split], X[split:], y[:split], y[split:])
results.append(["Normal Split", mae, rmse, r2])

# =========================
# Scenario 2: Less training data (60%)
# =========================
split_small = int(len(df) * 0.6)
mae, rmse, r2 = evaluate(X[:split_small], X[split_small:], y[:split_small], y[split_small:])
results.append(["Less Training Data", mae, rmse, r2])

# =========================
# Scenario 3: Add noise
# =========================
np.random.seed(42)
noise = np.random.normal(0, 0.01, size=X.shape)
X_noisy = X + noise

mae, rmse, r2 = evaluate(X_noisy[:split], X_noisy[split:], y[:split], y[split:])
results.append(["With Noise", mae, rmse, r2])

# =========================
# Scenario 4: Missing values
# =========================
X_missing = X.copy()
mask = np.random.rand(*X_missing.shape) < 0.1
X_missing[mask] = np.nan
X_missing = X_missing.fillna(X_missing.mean())

mae, rmse, r2 = evaluate(X_missing[:split], X_missing[split:], y[:split], y[split:])
results.append(["With Missing Values", mae, rmse, r2])

# =========================
# Save results
# =========================
results_df = pd.DataFrame(results, columns=["Scenario", "MAE", "RMSE", "R2"])
results_df.to_csv("tables/RQ6_robustness_analysis.csv", index=False)

# =========================
# Plot 1: Bar Chart
# =========================
plt.figure()
results_df.set_index("Scenario")[["MAE", "RMSE", "R2"]].plot(kind="bar")
plt.title("RQ6: Robustness Analysis")

plt.savefig("figures/RQ6_robustness_boxplot.pdf")
plt.close()

# =========================
# Plot 2: Line Chart
# =========================
plt.figure()

plt.plot(results_df["Scenario"], results_df["MAE"], marker='o', label="MAE")
plt.plot(results_df["Scenario"], results_df["RMSE"], marker='o', label="RMSE")
plt.plot(results_df["Scenario"], results_df["R2"], marker='o', label="R2")

plt.xticks(rotation=20)
plt.legend()
plt.title("RQ6: Performance Under Different Conditions")

plt.savefig("figures/RQ6_robustness_line.pdf")
plt.close()

print("RQ6 completed! Files saved.")