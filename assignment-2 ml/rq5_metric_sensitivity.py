import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Create folders
os.makedirs("tables", exist_ok=True)
os.makedirs("figures", exist_ok=True)

# Load data
df = pd.read_csv("btcusd_1-min_data.csv")

# 🔥 Speed fix
df = df.tail(50000)

# Time handling
df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
df = df.sort_values('Timestamp')

# Feature engineering
df['lag1'] = df['Close'].shift(1)
df['Target'] = df['Close'].shift(-1)
df = df.dropna()

# Train-test split
split = int(len(df) * 0.8)
train = df[:split]
test = df[split:]

X_train = train[['lag1']]
y_train = train['Target']

X_test = test[['lag1']]
y_test = test['Target']

# Models (simple + fast)
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
}

results = []

# Train & evaluate
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    results.append([name, mae, rmse, r2])

# Create dataframe
df_results = pd.DataFrame(results, columns=["Model", "MAE", "RMSE", "R2"])

# Ranking
df_results["Rank_MAE"] = df_results["MAE"].rank(ascending=True)
df_results["Rank_RMSE"] = df_results["RMSE"].rank(ascending=True)
df_results["Rank_R2"] = df_results["R2"].rank(ascending=False)

df_results["Overall_Rank"] = df_results[["Rank_MAE","Rank_RMSE","Rank_R2"]].mean(axis=1)

# Save table
df_results.to_csv("tables/RQ5_model_rankings.csv", index=False)

# =========================
# Plot 1: Metric Comparison
# =========================
plt.figure()
df_results.set_index("Model")[["MAE","RMSE","R2"]].plot(kind="bar")
plt.title("RQ5: Metric Comparison")

plt.savefig("figures/RQ5_metric_sensitivity.pdf")
plt.close()

# =========================
# Plot 2: Ranking Line Plot
# =========================
plt.figure()

for i in range(len(df_results)):
    plt.plot(["MAE","RMSE","R2"],
             [df_results.loc[i,"Rank_MAE"],
              df_results.loc[i,"Rank_RMSE"],
              df_results.loc[i,"Rank_R2"]],
             marker='o',
             label=df_results.loc[i,"Model"])

plt.gca().invert_yaxis()
plt.legend()
plt.title("RQ5: Ranking Sensitivity")

plt.savefig("figures/RQ5_bump_chart.pdf")
plt.close()

print("RQ5 completed! Files saved.")