import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Create folders
os.makedirs("tables", exist_ok=True)
os.makedirs("figures", exist_ok=True)

# Load data
df = pd.read_csv("btcusd_1-min_data.csv")

# 🔥 Speed fix: reduce dataset size
df = df.tail(50000)

# Time handling
df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
df = df.sort_values('Timestamp')

# =========================
# Preprocessing Functions
# =========================

# 1. Raw (only lag)
def raw_data(df):
    temp = df.copy()
    temp['lag1'] = temp['Close'].shift(1)
    temp['Target'] = temp['Close'].shift(-1)
    temp = temp.dropna()
    return temp[['lag1']], temp['Target']

# 2. With indicator (moving average)
def with_indicator(df):
    temp = df.copy()
    temp['lag1'] = temp['Close'].shift(1)
    temp['SMA'] = temp['Close'].rolling(10).mean().shift(1)
    temp['Target'] = temp['Close'].shift(-1)
    temp = temp.dropna()
    return temp[['lag1', 'SMA']], temp['Target']

# 3. With scaling
def with_scaling(df):
    X, y = with_indicator(df)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    return pd.DataFrame(X_scaled, columns=X.columns), y

# 4. Full (indicator + scaling + outlier removal)
def full_pipeline(df):
    X, y = with_indicator(df)

    # Remove outliers (simple)
    Q1 = X.quantile(0.25)
    Q3 = X.quantile(0.75)
    IQR = Q3 - Q1
    mask = ~((X < (Q1 - 1.5 * IQR)) | (X > (Q3 + 1.5 * IQR))).any(axis=1)

    X = X[mask]
    y = y[mask]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return pd.DataFrame(X_scaled, columns=X.columns), y

# Strategies
strategies = {
    "Raw": raw_data,
    "Indicator": with_indicator,
    "Scaled": with_scaling,
    "Full": full_pipeline
}

results = []

# =========================
# Run Experiments
# =========================
for name, func in strategies.items():

    X, y = func(df)

    # Time-series split
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    # Simple model (same for fairness)
    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    results.append([name, mae, rmse, r2])

# =========================
# Save Table
# =========================
results_df = pd.DataFrame(results, columns=["Strategy", "MAE", "RMSE", "R2"])
results_df.to_csv("tables/RQ3_preprocessing_impact.csv", index=False)

# =========================
# Bar Plot
# =========================
plt.figure()
results_df.set_index("Strategy")[["MAE", "RMSE", "R2"]].plot(kind="bar")
plt.title("RQ3: Preprocessing Impact")

plt.savefig("figures/RQ3_preprocessing_ablation.pdf")
plt.close()

# =========================
# Line Plot
# =========================
plt.figure()

plt.plot(results_df["Strategy"], results_df["MAE"], marker='o', label="MAE")
plt.plot(results_df["Strategy"], results_df["RMSE"], marker='o', label="RMSE")
plt.plot(results_df["Strategy"], results_df["R2"], marker='o', label="R2")

plt.legend()
plt.title("RQ3: Performance Trend")

plt.savefig("figures/RQ3_preprocessing_line.pdf")
plt.close()

print("RQ3 completed! Files saved.")