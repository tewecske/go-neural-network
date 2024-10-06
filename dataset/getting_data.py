import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

x, y = spiral_data(samples=100, classes=3)

print(x)
print("---------------------")
print(y)

