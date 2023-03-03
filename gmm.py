import sys
import numpy as np
import pandas as pd
from math_util import maximum_likelihood

class GMM:
    '''
        This class is the implementation of the Gaussian Mixture Models 
        inspired by sci-kit learn implementation.
    '''
    def __init__(self, n_components, mean_init, max_iter = 30):
        '''
            This functions initializes the model by seting the following paramenters:
                :param n_components: int
                    The number of clusters in which the algorithm must split
                    the data set
                :param max_iter: int, default = 100
                    The number of iteration that the algorithm will go throw to find the clusters
                :param comp_names: list of strings, default=None
                    In case it is setted as a list of string it will use to
                    name the clusters
        '''
        self.n_componets = n_components
        self.max_iter = max_iter
        self.mean_vector = mean_init
        # pi list contains the fraction of the dataset for every cluster
        self.pi = [1/self.n_componets for comp in range(self.n_componets)]

    def multivariate_normal(self, X, mean_vector, covariance_matrix):
        '''
            This function implements the multivariat normal derivation formula,
            the normal distribution for vectors it requires the following parameters
                :param X: 1-d numpy array
                    The row-vector for which we want to calculate the distribution
                :param mean_vector: 1-d numpy array
                    The row-vector that contains the means for each column
                :param covariance_matrix: 2-d numpy array (matrix)
                    The 2-d matrix that contain the covariances for the features
        '''
        term1 = (2 * np.pi * covariance_matrix) ** (-0.5) 
        term2 = -0.5 * ((X - mean_vector)**2) * (1/covariance_matrix)
        #term1 = (1 * np.pi * covariance_matrix) ** (-0.5) 
       
        #if this term is larger, then the likelihood is larger
        #this is negative, therefore we want abs() to be smaller
        #change constance to place larger burden on linear closeness
        #term2 = -1 * ((X - mean_vector)**2) * (1/covariance_matrix)
        
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
        self.covariance_matrixes = [float(np.cov(x)) if len(x) > 1 else 0.0001 for x in clusters]

    def fit(self, X):       
        self._kmeans_init(X, self.mean_vector)
        for iteration in range(self.max_iter):
            """ 
            print("mean:", self.mean_vector)
            print("pi:", self.pi)
            print("cov:", self.covariance_matrixes)
            """
            ''' --------------------------   E - STEP   -------------------------- '''
            # Initiating the r matrix, evrey row contains the probabilities
            # for every cluster for this row
            self.r = np.zeros((len(X), self.n_componets))
            probas = []
            all_probas = []
            # Calculating the r matrix
            for n in range(len(X)):
                for k in range(self.n_componets):
                    """
                    self.r[n][k] = self.pi[k] * self.multivariate_normal(X[n], self.mean_vector[k], self.covariance_matrixes[k])
                    self.r[n][k] /= sum([self.pi[j]*self.multivariate_normal(X[n], self.mean_vector[j], self.covariance_matrixes[j]) for j in range(self.n_componets)])
                    """
                    self.r[n][k] = self.multivariate_normal(X[n], self.mean_vector[k], self.covariance_matrixes[k])
                    self.r[n][k] /= sum([self.multivariate_normal(X[n], self.mean_vector[j], self.covariance_matrixes[j]) for j in range(self.n_componets)])
                    
                    #print(self.r[n][k], self.covariance_matrixes[k])
                
                #print(self.r[n], X[n])
                probas.append(max(self.r[n][:]))
                all_probas.append(list(self.r[n]))
            #print("min:", min(probas))
            loc = probas.index(min(probas))
            #print("min point:", X[loc])
            #print("min probas:", all_probas[loc])
            # Calculating the N which the sum of the normalized likelihood per category
            N = np.sum(self.r, axis=0)
            #print("iteration:", iteration)
            #print("N:", list(N))
            #print(list(self.covariance_matrixes))
            #print("mean:", self.mean_vector)
            #print("pi:", self.pi)
            ''' --------------------------   M - STEP   -------------------------- '''
            """ 
            # Initializing the mean vector as a zero vector 
            self.mean_vector = np.zeros((self.n_componets, len(X)))
            
            # Updating the mean vector
            for k in range(self.n_componets):
                for n in range(len(X)):
                    self.mean_vector[k] += self.r[n][k] * X[n]
            
            self.mean_vector = [1/N[k]*self.mean_vector[k] for k in range(self.n_componets)]
            self.mean_vector = [list(np.unique(x))[0] for x in self.mean_vector]
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
                        
            # Updating the pi list
            self.pi = [N[k]/len(X) for k in range(self.n_componets)]
           
            #make sure the pi value isn't zero
            self.pi = [x if x != 0 else 0.0001 for x in self.pi]

    def predict(self, X):
        probas = []
        for n in range(len(X)):
            probas.append([self.multivariate_normal(X[n], self.mean_vector[k], self.covariance_matrixes[k])
                           for k in range(self.n_componets)])
        cluster = []
        for x, proba in zip(X, probas):
            loc = proba.index(max(proba))
            cluster.append(self.mean_vector[loc])

        return(cluster)

    def score(self, X):
        probas = []
        scores = []
        for n in range(len(X)):
            tmp = [self.multivariate_normal(X[n], self.mean_vector[k], self.covariance_matrixes[k]) * self.pi[k] for k in range(self.n_componets)]
            scores.append(max(tmp))
        return(scores)

