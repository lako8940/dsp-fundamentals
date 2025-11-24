import numpy as np

m = np.arange(1, 200)
n = .712

t = m * n

# Print whole numbers and their indices
indices = np.where(t == np.floor(t))[0]
for i in indices:
    print(f"Index: {i}, Value: {t[i]}")
