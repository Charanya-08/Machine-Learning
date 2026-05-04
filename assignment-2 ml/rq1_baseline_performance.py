import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Create folders
os.makedirs("tables", exist_ok=True)
os.makedirs("figures", exist_ok=True)

# Load data
df = pd.read_csv("btcusd_1-min_data.csv")

# Convert time and sort
df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
df = df.sort_values('Timestamp')

# Feature engineering (lag)
df['Close_lag1'] = df['Close'].shift(1)

# Target (next step price)
df['Target'] = df['Close'].shift(-1)

df = df.dropna()

# Train-test split (80/20)
split = int(len(df) * 0.8)
train = df[:split]
test = df[split:]

X_train = train[['Close_lag1']]
y_train = train['Target']

X_test = test[['Close_lag1']]
y_test = test['Target']

# Model (baseline)
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# Save results
results = pd.DataFrame({
    "Model": ["Linear Regression"],
    "MAE": [mae],
    "RMSE": [rmse],
    "R2": [r2]
})

results.to_csv("tables/RQ1_baseline_performance.csv", index=False)

# Simple plot
import matplotlib.pyplot as plt

plt.figure()
plt.plot(y_test.values[:100], label="Actual")
plt.plot(y_pred[:100], label="Predicted")
plt.legend()

plt.savefig("figures/RQ1_baseline_comparison.pdf")
plt.close()

print("Done! Files saved in tables/ and figures/")