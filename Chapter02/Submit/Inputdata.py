# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 17:49:07 2019

@author: Sumail
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv('iris.data', header=None)
df1 = df[0:50]
df2 = df[100:150]
df = pd.concat([df1,df2])
print(df[40:60])

y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)

# extract sepal length and petal length
X = df.iloc[0:100, [1, 3]].values

# plot data
plt.scatter(X[:50, 0], X[:50, 1],
            color='red', marker='o', label='setosa')
plt.scatter(X[50:100, 0], X[50:100, 1],
            color='blue', marker='x', label='virginica')

plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')

plt.savefig('images/02_06.png', dpi=300)
plt.show()