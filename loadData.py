import numpy as np

with np.load("extraSplitData.npz") as data:
    Y_train = data['Y_train']

print(Y_train)