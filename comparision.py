import pandas as pd
import matplotlib.pyplot as plt


# Results from models
results = {"Model": ["Decision Tree", "Random Forest"],"MAE": [44160.48, 31572.66],"MSE": [4848725317.09, 2395652720.04],"R2 Score": [0.63, 0.82]}

df_results = pd.DataFrame(results)

print("Model Comparison")
print(df_results)

# Best model based on R2
best_model = df_results.loc[df_results["R2 Score"].idxmax(), "Model"]
print(f"\nBest Performing Model: {best_model}")

df_results.set_index("Model")[["R2 Score"]].plot(kind="bar")
plt.title("Model Comparison using R2 Score")
plt.ylabel("R2 Score")
plt.ylim(0,1)
plt.show()
