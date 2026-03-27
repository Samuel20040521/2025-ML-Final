import numpy as np
import pandas as pd
from scipy.stats import beta

# Sampling
N = 500_000  # half million for good estimate
a, b = 0.8, 0.7
samples = beta.rvs(a, b, size=N)

# Bins: 0~1 in 0.01 intervals
bins = np.arange(0, 1.01, 0.01)
hist, edges = np.histogram(samples, bins=bins)
probs = hist / N

# Format result as table
df = pd.DataFrame({
    "bin_start": edges[:-1],
    "bin_end": edges[1:],
    "probability": probs
})
##plot the graph
import matplotlib.pyplot as plt
plt.bar(df["bin_start"], df["probability"], width=0.01, align='edge', edgecolor='black')
plt.xlabel("Value")
plt.ylabel("Probability")
plt.title(f"Beta Distribution (a={a}, b={b}) Histogram")
plt.xlim(0, 1)
plt.grid(axis='y', alpha=0.75)
##show the y value
for i, prob in enumerate(df["probability"]):
    if 0.003>prob >=0:  # only label bars with significant height
        plt.text(df["bin_start"][i] + 0.005, prob + 0.001, f"{prob:.3f}", ha='center', va='bottom', fontsize=8)
plt.show()