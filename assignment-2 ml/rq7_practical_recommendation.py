import pandas as pd
import os
import matplotlib.pyplot as plt

# Create folders
os.makedirs("tables", exist_ok=True)
os.makedirs("figures", exist_ok=True)

# =========================
# Decision Matrix (Simple Scores)
# =========================
data = {
    "Model": ["Linear Regression", "Random Forest"],
    "Performance": [3, 4],
    "Interpretability": [5, 3],
    "Robustness": [3, 4],
    "Speed": [5, 4]
}

df = pd.DataFrame(data)

# Overall score
df["Final Score"] = df[["Performance","Interpretability","Robustness","Speed"]].mean(axis=1)

# Save table
df.to_csv("tables/RQ7_decision_matrix.csv", index=False)

# =========================
# Plot 1: Bar Chart
# =========================
plt.figure()
df.set_index("Model")["Final Score"].plot(kind="bar")
plt.title("RQ7: Final Model Comparison")

plt.savefig("figures/RQ7_radar_chart.pdf")
plt.close()

# =========================
# Plot 2: Scatter Plot
# =========================
plt.figure()

for i in range(len(df)):
    plt.scatter(df.loc[i,"Performance"], df.loc[i,"Interpretability"])
    plt.text(df.loc[i,"Performance"], df.loc[i,"Interpretability"], df.loc[i,"Model"])

plt.xlabel("Performance")
plt.ylabel("Interpretability")
plt.title("RQ7: Model Trade-off")

plt.savefig("figures/RQ7_bubble_chart.pdf")
plt.close()

# =========================
# Final Recommendation
# =========================
best_model = df.sort_values("Final Score", ascending=False).iloc[0]["Model"]

print("Best Model:", best_model)
print("RQ7 completed! Files saved.")