import numpy as np

with np.load("data.npz") as data:
    X_train = data['X_train']

print(X_train.shape)