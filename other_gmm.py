import numpy as np
from scipy import random
import matplotlib.transforms as transforms
from scipy.stats import multivariate_normal

class GMM():
    def __init__(self, k, dim, init_mu=None, init_sigma=None, init_pi=None):
        '''
        Define a model with known number of clusters and dimensions.
        input:
            - k: Number of Gaussian clusters
            - dim: Dimension 
            - init_mu: initial value of mean of clusters (k, dim)
                       (default) random from uniform[-10, 10]
            - init_sigma: initial value of covariance matrix of clusters (k, dim, dim)
                          (default) Identity matrix for each cluster
            - init_pi: initial value of cluster weights (k,)
                       (default) equal value to all cluster i.e. 1/k
        '''
        self.k = k
        self.dim = dim
        if(init_mu is None):
            init_mu = random.rand(k, dim)*20 - 10
        self.mu = init_mu
        if(init_sigma is None):
            init_sigma = np.zeros((k, dim, dim))
            for i in range(k):
                init_sigma[i] = np.eye(dim)
        self.sigma = init_sigma
        if(init_pi is None):
            init_pi = np.ones(self.k)/self.k
        self.pi = init_pi
   
    def init_em(self, X):
        '''
        Initialization for EM algorithm.
        input:
            - X: data (batch_size, dim)
        '''
        self.data = X
        self.num_points = X.shape[0]
        self.z = np.zeros((self.num_points, self.k))
    
    def e_step(self):
        '''
        E-step of EM algorithm.
        '''
        for i in range(self.k):
            self.z[:, i] = self.pi[i] * multivariate_normal.pdf(self.data, mean=self.mu[i], cov=self.sigma[i])
        self.z /= self.z.sum(axis=1, keepdims=True)
    
    def m_step(self):
        '''
        M-step of EM algorithm.
        '''
        sum_z = self.z.sum(axis=0)
        self.pi = sum_z / self.num_points
        #self.mu = np.matmul(self.z.T, self.data)
        #self.mu /= sum_z[:, None]
        for i in range(self.k):
            j = np.expand_dims(self.data, axis=1) - self.mu[i]
            s = np.matmul(j.transpose([0, 2, 1]), j)
            self.sigma[i] = np.matmul(s.transpose(1, 2, 0), self.z[:, i] )
            self.sigma[i] /= sum_z[i]
            
    def log_likelihood(self, X):
        '''
        Compute the log-likelihood of X under current parameters
        input:
            - X: Data (batch_size, dim)
        output:
            - log-likelihood of X: Sum_n Sum_k log(pi_k * N( X_n | mu_k, sigma_k ))
        '''
        ll = []
        for d in X:
            tot = 0
            for i in range(self.k):
                #if it's zero, set it to something negilbly small so the program doesn't break
                if self.sigma[i][0][0] == 0:
                    self.sigma[i][0][0] = 0.0000001
                    print("here", self.sigma[i][0][0], self.pi[i])
                try:
                    x = self.pi[i] * multivariate_normal.pdf(d, mean=self.mu[i], cov=self.sigma[i])
                except:
                    print('this', self.sigma[i][0][0], self.pi[i], self.mu[i])
                    sys.exit(0)
                tot += x
            ll.append(np.log(tot))
        return(np.sum(ll))

    def score(self, X):
        all_like = []
        assignments = []
        scores = []
        for d in X:
            tmp = []
            for i in range(self.k):
                x = self.pi[i] * multivariate_normal.pdf(d, mean=self.mu[i], cov=self.sigma[i])
                tmp.append(x)
            all_like.append(tmp)
            score = max(tmp)
            scores.append(score)
            idx = tmp.index(score)
            assignments.append(self.mu[idx])
        return(assignments, scores, all_like)


