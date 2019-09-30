# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 17:03:12 2019

@author: Sumail
"""

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np
import plot_decision_regions

iris = datasets.load_iris()
dict1 = [[0,1], [0,2], [0,3], [1,2], [1,3], [2,3]]
dict2 = [[0,1,2], [0,1,3], [0,2,3], [1,2,3]]
dict3 = [[0,1,2,3]]
dictc = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
dictp = ['l1', 'l2']
dict11 = []
for j in dictp:
    print(j)
    for i in range(len(dict1)):
        X = iris.data[:, dict1[i]]
        y = iris.target
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=1, stratify=y)
        sc = StandardScaler()
        sc.fit(X_train)
        X_train_std = sc.transform(X_train)
        X_test_std = sc.transform(X_test)
        X_combined_std = np.vstack((X_train_std, X_test_std))
        y_combined = np.hstack((y_train, y_test))
        lr = LogisticRegression(C=100.0, random_state=1, solver = 'liblinear', multi_class = 'auto', penalty = j)
        lr.fit(X_train_std, y_train)
        print('2 features are [%d,%d]. iteration is %d. test set accuracy is %f'%(dict1[i][0], dict1[i][1], lr.n_iter_, lr.score(X_test_std, y_test)))
        dict11.append([dict1[i][0], dict1[i][1], lr.n_iter_, lr.score(X_test_std, y_test)])
    for i in range(len(dict2)):
        X = iris.data[:, dict2[i]]
        y = iris.target
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=1, stratify=y)
        sc = StandardScaler()
        sc.fit(X_train)
        X_train_std = sc.transform(X_train)
        X_test_std = sc.transform(X_test)
        X_combined_std = np.vstack((X_train_std, X_test_std))
        y_combined = np.hstack((y_train, y_test))  
        lr = LogisticRegression(C=100.0, random_state=1, solver = 'liblinear', multi_class = 'auto', penalty = j)
        lr.fit(X_train_std, y_train)
        print('3 features are [%d,%d,%d]. iteration is %d. test set accuracy is %f'%(dict2[i][0], dict2[i][1],dict2[i][2], lr.n_iter_, lr.score(X_test_std, y_test)))
        dict11.append([dict2[i][0], dict2[i][1], dict2[i][2], lr.n_iter_, lr.score(X_test_std, y_test)])
    for i in range(len(dict3)):
        X = iris.data[:, dict3[i]]
        y = iris.target
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=1, stratify=y)
        sc = StandardScaler()
        sc.fit(X_train)
        X_train_std = sc.transform(X_train)
        X_test_std = sc.transform(X_test)
        X_combined_std = np.vstack((X_train_std, X_test_std))
        y_combined = np.hstack((y_train, y_test))
        lr = LogisticRegression(C=100.0, random_state=1, solver = 'liblinear', multi_class = 'auto', penalty = j)
        lr.fit(X_train_std, y_train)
        print('4 features are [%d,%d,%d,%d]. iteration is %d. test set accuracy is %f'%(dict3[i][0], dict3[i][1], dict3[i][2], dict3[i][3], lr.n_iter_, lr.score(X_test_std, y_test)))
'''for dict in dict11:
    print('2 features are [%d,%d,%d]. iteration is %d. test set accuracy is %f'%(dict[0], dict[1], dict[2], dict[3], dict[4]))'''