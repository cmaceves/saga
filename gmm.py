import os
import sys
import copy
import math
import numpy as np
import pandas as pd
from math_util import maximum_likelihood

class GMM:
    def __init__(self, n_components, mean_init, max_iter = 10):
        self.n_componets = n_components
        self.max_iter = max_iter
        self.mean_vector = mean_init
        # pi list contains the fraction of the dataset for every cluster
        self.pi = [1/self.n_componets for comp in range(self.n_componets)]

    def multivariate_normal(self, X, mean_vector, covariance_matrix):
        term1 = (2 * np.pi * covariance_matrix) ** (-0.5)
        constant = -0.5
        if X != mean_vector:
            term2 = constant * ((X - mean_vector)**2) * (1/covariance_matrix)
        else:
            term2 = constant * ((0.00001)**2) * (1/covariance_matrix)
        return(term1 * np.exp(term2))
    
    def _kmeans_init(self, X, mean_vector):
        """
        Initialize the covariance matrices in a k-means like way.
        """
        arr = np.array(mean_vector)
        clusters = []
        
        for i in range(self.n_componets):
            clusters.append([])
        for x in X:
            location = (np.abs(arr - x)).argmin()
            clusters[location].append(x)        
        base_cov = 0.0005
        self.covariance_matrixes = [float(np.cov(x)) if len(x) > 1 else base_cov for x in clusters]
        self.covariance_matrixes = [x if x > 0 else base_cov for x in self.covariance_matrixes]        
        self.covariance_matrixes = [base_cov] * self.n_componets

    def fit(self, X):       
        self._kmeans_init(X, self.mean_vector)
        
        for iteration in range(self.max_iter):
            if iteration % 1 == 0:
                print("training iteration:", iteration)            
            """
            print("pi:", self.pi)
            print("cov top:", self.covariance_matrixes)
            print("all means:", [round(x, 4) for x in self.mean_vector])
            """
            for val in self.pi:
                if str(val) == 'nan':
                    print(self.pi)
                    break
            
            ''' --------------------------   E - STEP   -------------------------- '''
            # Initiating the r matrix, evrey row contains the probabilities
            # for every cluster for this row
            self.r = np.zeros((len(X), self.n_componets))
            probas = []
            all_probas = []
            
            # Calculating the r matrix
            for n in range(len(X)):
                for k in range(self.n_componets):                    
                    self.r[n][k] = self.pi[k] * self.multivariate_normal(X[n], self.mean_vector[k], self.covariance_matrixes[k])
                    all_values = [self.multivariate_normal(X[n], self.mean_vector[j], self.covariance_matrixes[j]) for j in range(self.n_componets)]
                    all_values = [self.pi[j] * x for j,x in enumerate(all_values)]
                    val = self.r[n][k] / sum(all_values)
                    if sum(all_values) == 0:
                        continue
                                   
                    self.r[n][k] /= sum(all_values)

            # Calculating the N which the sum of the normalized likelihood per category
            N = np.sum(self.r, axis=0)
            N = [x if x > 0 else 0.0001 for x in N]
            ''' --------------------------   M - STEP   -------------------------- '''            
            original_mean = self.mean_vector 
            
            """     
            # Initializing the mean vector as a zero vector 
            self.mean_vector = np.zeros((self.n_componets)) 
            # Updating the mean vector
            for k in range(self.n_componets):
                for n in range(len(X)):
                    self.mean_vector[k] += self.r[n][k] * X[n]
            self.mean_vector = [1/N[k]*self.mean_vector[k] for k in range(self.n_componets)]
            """
            # Initiating the list of the covariance matrixes
            self.covariance_matrixes = [float(0.0)] * self.n_componets
            # Updating the covariance matrices
            for k in range(self.n_componets):
                if all(x == 0 for x in list(self.r[:,k])):
                    weight = [1] * len(self.r[:, k])
                else:
                    weight = self.r[:,k]
                self.covariance_matrixes[k] = float(np.cov(X, aweights=(weight), ddof=0))
                            
            self.covariance_matrixes = [1/N[k]*self.covariance_matrixes[k] for k in range(self.n_componets)]
            self.covariance_matrixes = [x if x != 0 else 0.00001 for x in self.covariance_matrixes]            
            self.covariance_matrixes = [x if not math.isinf(x) else 0.00001 for x in self.covariance_matrixes]
            self.covariance_matrixes = [x if str(x) != 'nan' else 0.00001 for x in self.covariance_matrixes]   
            self.pi = [N[k]/len(X) for k in range(self.n_componets)]
            self.pi = [x if x != 0 else 0.0001 for x in self.pi]

    def score(self, X):
        probas = []
        scores = []
        for n in range(len(X)):
            tmp = [self.multivariate_normal(X[n], self.mean_vector[k], self.covariance_matrixes[k]) * self.pi[k] for k in range(self.n_componets)]
            probas.append(tmp)
            scores.append(max(tmp))
        return(scores, probas)

