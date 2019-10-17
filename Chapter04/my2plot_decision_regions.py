# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 17:53:41 2019

@author: Sumail
"""
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.2):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    x3_min, x3_max = X[:, 2].min() - 1, X[:, 2].max() + 1
    x4_min, x4_max = X[:, 3].min() - 1, X[:, 3].max() + 1
    xx1, xx2, xx3, xx4 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution),
                           np.arange(x3_min, x3_max, resolution),
                           np.arange(x4_min, x4_max, resolution))
    
    tempX = np.array([xx1.ravel(), xx2.ravel(),xx3.ravel(), xx4.ravel()]).T
    print(tempX.shape)
    pca = PCA(n_components=2)
    pca.fit(tempX)
    newX=pca.fit_transform(tempX)
    
    tempshape = xx1.shape
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel(),xx3.ravel(), xx4.ravel()]).T)
    newX = newX.reshape(2, len(Z))
    print(newX.shape)
    '''x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    temp1 = np.zeros(len(xx1.ravel()))
    temp2 = np.zeros(len(xx1.ravel()))
    for i in range(len(xx1.ravel())):
        temp1[i] = X[:, 2].max()
        temp2[i] = X[:, 3].max()
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel(),temp1, temp2]).T)'''
    '''print(Z.shape)
    print(xx1.shape)
    
    newX = newX.reshape(2, len(Z))
    print(newX.shape)
    newX1 = newX[0]
    newX2 = newX[1]
    
    newX1 = newX1.reshape(tempshape)
    print(newX1.shape)
    pca = PCA(n_components=2)
    pca.fit(newX1)
    newX11=pca.fit_transform(newX1)
    print(newX11.shape)
    newX2 = newX2.reshape(tempshape)
    Z = Z.reshape((xx1.shape[0]*xx1.shape[1], xx1.shape[2]*xx1.shape[3]))
    print(Z.shape)'''
    
    plt.contourf(newX[0], newX[1], Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.8, 
                    c=colors[idx],
                    marker=markers[idx], 
                    label=cl, 
                    edgecolor='black')

    # highlight test samples
    '''if test_idx:
        # plot all samples
        X_test, y_test = X[test_idx, :], y[test_idx]

        plt.scatter(X_test[:, 0],
                    X_test[:, 1],
                    c='',
                    edgecolor='black',
                    alpha=1.0,
                    linewidth=1,
                    marker='o',
                    s=100, 
                    label='test set')'''