# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 17:43:39 2023

@author: neural.net_
"""

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (8,8)
from scipy.spatial import Voronoi, voronoi_plot_2d

def create_random_data():
    X= np.random.normal(0.7, 0.5, size=(150, 2))
    X1 = -1 + 1.2 * np.random.rand(50,2)
    X2 = np.random.normal(0.5, 0.65, size=(50, 2))
    X2[:,-1] = X2[:,-1] -1.2
    X[50:100, :] = X1
    X[100:, :] = X2
    return X

class Kmeans_Clustering():
    "Implementation of the naïve k-means class (Lloyd-Algorithmus)."
    def __init__(self, K, dims, max_iter=10):
        self.K = K
        self.mu = np.random.rand(K, dims)
        self.max_iter = max_iter
    
    def plot(self, X, C, idx):
        color = plt.cm.rainbow(np.linspace(0, 1, self.K))
        points = self.mu
        vor = Voronoi(points)
        fig = voronoi_plot_2d(vor, show_vertices=False, line_colors='black',
                      line_width=2, line_alpha=0.6, point_size=2)
        for k, clr in zip(range(self.K), color):
            plt.plot(self.mu[k,0 ], self.mu[k,1 ], 's', c=clr, markersize=22)
            mask = np.where(C==k)
            plt.plot(X[mask,0], X[mask,1], '.', c=clr, markersize=22)
            plt.title('Kmeans Clustering')
            plt.text(-.55, 1.25, 'Iteration: '+str(idx),  fontsize=26, ha='left', va='top')
            plt.xlim([-0.6, 1.3])
            plt.ylim([-1.6, 1.3])
            frame1 = plt.gca()
            frame1.axes.xaxis.set_ticklabels([])
            frame1.axes.yaxis.set_ticklabels([])
            plt.show()
        
    def MSE(self, X):
        mse = np.empty((self.K, X.shape[0]))
        mse[:] = np.nan
        for k in range(self.K):
            mse[k, :] = ((X - self.mu[k])**2).mean(axis=1)
        return mse
    
    def update(self, X, C):
        for k in range(self.K):
            mask = np.where(C==k)
            data = X[mask]
            self.mu[k] = data.mean(axis=0)
        return self.mu
            
    def assign(self, mse):
        self.C = np.argmin(mse, axis=0) 
        return self.C
            
    def fit(self, X):
        self.C = np.zeros(X.shape[0])
        for i in range(self.max_iter):
            # cluster assignment
            mse = self.MSE(X)
            prev_C = self.C
            C = self.assign(mse)  
            # update cluster representation
            mu = self.update(X,C)
            # plot clusters
            self.plot(X, C, i)
            # Check convergence criterion
            if (prev_C == C).all():
                break
        return mse, mu, C
    
# Parameters for kmeans-Algorithms
K = 3               # number of clusters
dims = 2            # dimension of data
max_iter = 20       # maximal numer of iterations

X = create_random_data()
kmeans = Kmeans_Clustering(K, dims, max_iter)
mse, mu, C = kmeans.fit(X)
