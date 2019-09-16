# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 17:59:04 2019

@author: Sumail
"""
import Perceptron 
import matplotlib.pyplot as plt
import numpy as np
import Inputdata
import AdalineGD

ppn = Perceptron.Perceptron(eta=0.1, n_iter=10)

ppn.fit(Inputdata.X, Inputdata.y)

plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of updates')

# plt.savefig('images/02_07.png', dpi=300)
plt.show()

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

ada1 = AdalineGD.AdalineGD(n_iter=10, eta=0.01).fit(Inputdata.X, Inputdata.y)
ax[0].plot(range(1, len(ada1.cost_) + 1), np.log10(ada1.cost_), marker='o')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(Sum-squared-error)')
ax[0].set_title('Adaline - Learning rate 0.01')

ada2 = AdalineGD.AdalineGD(n_iter=10, eta=0.0001).fit(Inputdata.X, Inputdata.y)
ax[1].plot(range(1, len(ada2.cost_) + 1), ada2.cost_, marker='o')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Sum-squared-error')
ax[1].set_title('Adaline - Learning rate 0.0001')

# plt.savefig('images/02_11.png', dpi=300)
plt.show()