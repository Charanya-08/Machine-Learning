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

# 🔥 Speed fix: reduce size
df = df.tail(50000)

# Time handling
df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
df = df.sort_values('Timestamp')

# =========================
# Feature Engineering (Simple)
# =========================
df['lag1'] = df['Close'].shift(1)
df['lag2'] = df['Close'].shift(2)
df['SMA'] = df['Close'].rolling(10).mean().shift(1)
df['Volatility'] = (df['High'] - df['Low']).shift(1)

df['Target'] = df['Close'].shift(-1)

df = df.dropna()

features = ['lag1', 'lag2', 'SMA', 'Volatility']
X = df[features]
y = df['Target']

# Train-test split
split = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

# =========================
# Model (Fast + Good)
# =========================
model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2:", r2_score(y_test, y_pred))

# =========================
# Feature Importance
# =========================
importance = model.feature_importances_

feat_df = pd.DataFrame({
    "Feature": features,
    "Importance": importance
}).sort_values(by="Importance", ascending=False)

# Simple interpretation
interpretation = {
    "lag1": "Previous price strongly influences next price",
    "lag2": "Short-term trend continuation",
    "SMA": "Moving average captures trend",
    "Volatility": "Market fluctuation indicator"
}

feat_df["Interpretation"] = feat_df["Feature"].map(interpretation)

# Save table
feat_df.to_csv("tables/RQ4_top_features.csv", index=False)

# =========================
# Plot
# =========================
plt.figure()
plt.barh(feat_df["Feature"], feat_df["Importance"])
plt.gca().invert_yaxis()
plt.title("RQ4: Feature Importance")

plt.savefig("figures/RQ4_feature_importance.pdf")
plt.close()

print("RQ4 completed! Files saved.")