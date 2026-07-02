import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("housing.csv")
print(df)
import matplotlib.pyplot as plt

plt.scatter(x=df["median_income"],y= df["median_house_value"],s=10,alpha=0.3)
plt.xlabel("Median Income")
plt.ylabel("Median House Value")
plt.title("Income vs House Price")
plt.show()