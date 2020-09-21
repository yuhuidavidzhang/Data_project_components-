
"""Calculate entropy"""
n = 1000000
probability = 0.2

import math
entropy = - (math.log2(probability) * probability + math.log2(1-probability) * (1-probability))

print(entropy * n / 8)

"""
The value_counts() function is used to get a Series containing counts of unique values.

The resulting object will be in descending order so that the first element is the most frequently-occurring element"""

value_counts()


import numpy
x = np.array([1, 1, 2, 4])
y = np.log2(2)

print(y)


[0. 0. 1. 2.]


o = np.log2(freq + 1e-6)