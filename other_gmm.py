import os
import sys
import copy
import numpy as np
from scipy import random
import matplotlib.transforms as transforms
from scipy.stats import multivariate_normal
np.seterr(all="ignore")
class GMM():
    def __init__(self, k, dim, init_mu=None, init_sigma=None, init_pi=None, name=None, solution=None, default_sigma=None, fixed_means=False):
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
        self.fixed_means = fixed_means
        self.init_mu = init_mu
        self.iteration = 0
        if default_sigma is None:
            self.default_sigma = 0.1
        else:
            self.default_sigma = default_sigma
        self.k = k
        self.dim = dim
        if solution is not None:
            self.solution = solution
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
        if name is not None:
            self.name = name   
        self.sigma_reset = {}

    def init_em(self, X, positions, combination_solutions, combination_sums):
        '''
        Initialization for EM algorithm.
        input:
            - X: data (batch_size, dim)
        '''
        self.data = X
        self.positions = np.array(positions)
        self.num_points = X.shape[0]
        self.z = np.zeros((self.num_points, self.k))
        self.mu_combo = []
        for m in self.mu:
            idx = combination_sums.index(m)
            self.mu_combo.append(combination_solutions[idx])
        self._positional_conflicts()

    def _positional_conflicts(self):
        """
        Calculate where we might have conflicts in our position assignments.
        """
        unique, counts = np.unique(self.positions, return_counts=True)
        position_conflicts = [x for x,y in zip(list(unique), list(counts)) if y > 1]
        self.position_conflict_idx = []
        for p in position_conflicts:
            idxs = [i for i,x in enumerate(self.positions) if x == p]
            self.position_conflict_idx.append(idxs)

    def e_step(self):
        '''
        E-step of EM algorithm.
        '''
        for i in range(self.k):
            if self.sigma[i][0][0] == 0 or str(self.sigma[i][0][0]) == 'nan':
                self.sigma[i][0][0] = self.default_sigma
                if self.mu[i][0] not in self.sigma_reset:
                    self.sigma_reset[self.mu[i][0]] = self.iteration
                else:
                    self.sigma_reset[self.mu[i][0]] = self.iteration
            x = self.pi[i] * multivariate_normal.pdf(self.data, mean=self.mu[i], cov=self.sigma[i])
            x = [a if str(a) != 'nan' else 0 for a in x]
            self.z[:, i] = x

        #assignment is max of z in k direction
        max_idxs = np.argmax(self.z, axis=1)
        max_vals = np.amax(self.z, axis=1)
        for pc in self.position_conflict_idx:
            pcids = [max_idxs[i] for i in pc] #get assignments for position conflicts
            pcvals = [max_vals[i] for i in pc] #get assignments for position conflicts
            max_pcval = max(pcvals)
            max_pc_idx = pcvals.index(max_pcval)
            combos = [self.mu_combo[i] for i in pcids]
            no_overlap_combo = combos[max_pc_idx]
            pc_top = pc[max_pc_idx]
            for pos in pc:
                if pos == pc_top:
                    continue
                check = self.z[pos, :]
                index = list(range(0,len(check)))
                zipped = list(zip(check, index))
                zipped.sort(reverse=True)
                check, index = zip(*zipped)
                combo_check = [self.mu_combo[i] for i in index]
                for c, i, cc in zip(check, index, combo_check):
                    if len([x for x in cc if x in no_overlap_combo]) > 0:
                        self.z[pos, i] = 0                       
                    else:
                        break    
        self.z /= self.z.sum(axis=1, keepdims=True)
    
    def m_step(self):
        '''
        M-step of EM algorithm.
        '''
        sum_z = self.z.sum(axis=0)
        self.pi = sum_z / self.num_points
        if self.fixed_means is False:
            self.mu = np.matmul(self.z.T, self.data)
            self.mu /= sum_z[:, None]
       
        d = np.expand_dims(self.data, axis=1)
        for i in range(self.k):
            j = d - self.mu[i]
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
                if self.sigma[i][0][0] == 0 or str(self.sigma[i][0][0]) == 'nan':
                    self.sigma[i][0][0] = self.default_sigma
                x = self.pi[i] * multivariate_normal.pdf(d, mean=self.mu[i], cov=self.sigma[i])                
                tot += x
            ll.append(np.log(tot))
        return(np.sum(ll))

    def score(self, X):
        all_like = []
        assignments = []
        scores = []
        ll = []
        for d in X:
            tmp = []
            tot = 0
            for i in range(self.k):
                if self.sigma[i][0][0] == 0 or str(self.sigma[i][0][0]) == 'nan':
                    self.sigma[i][0][0] = self.default_sigma
                x = self.pi[i] * multivariate_normal.pdf(d, mean=self.mu[i], cov=self.sigma[i])
                tmp.append(x)
                tot += x
            ll.append(np.log(tot))
            all_like.append(tmp)
            score = max(tmp)
            scores.append(score)
            idx = tmp.index(score)
            assignments.append(self.mu[idx])
        return(assignments, scores, all_like, ll)

    def other_score(self, X, positions):
        unique, counts = np.unique(positions, return_counts=True)
        position_conflicts = [x for x,y in zip(list(unique), list(counts)) if y > 1]
        position_conflict_idx = []
        for p in position_conflicts:
            idxs = [i for i,x in enumerate(positions) if x == p]
            position_conflict_idx.append(idxs)

        z = np.zeros((len(X), self.k))
        for i in range(self.k):
            if self.sigma[i][0][0] == 0 or str(self.sigma[i][0][0]) == 'nan':
                self.sigma[i][0][0] = self.default_sigma
                if self.mu[i][0] not in self.sigma_reset:
                    self.sigma_reset[self.mu[i][0]] = self.iteration
                else:
                    self.sigma_reset[self.mu[i][0]] = self.iteration
            x = self.pi[i] * multivariate_normal.pdf(X, mean=self.mu[i], cov=self.sigma[i])
            x = [a if str(a) != 'nan' else 0 for a in x]
            z[:, i] = x

        #assignment is max of z in k direction
        max_idxs = np.argmax(z, axis=1)
        max_vals = np.amax(z, axis=1)
        for pc in position_conflict_idx:
            pcids = [max_idxs[i] for i in pc] #get assignments for position conflicts
            pcvals = [max_vals[i] for i in pc] #get assignments for position conflicts
            max_pcval = max(pcvals)
            max_pc_idx = pcvals.index(max_pcval)
            combos = [self.mu_combo[i] for i in pcids]
            no_overlap_combo = combos[max_pc_idx]
            pc_top = pc[max_pc_idx]
            for pos in pc:
                if pos == pc_top:
                    continue
                if pos == 5165:
                    print(pcids)
                    print(pcvals)
                check = z[pos, :]
                index = list(range(0,len(check)))
                zipped = list(zip(check, index))
                zipped.sort(reverse=True)
                check, index = zip(*zipped)
                combo_check = [self.mu_combo[i] for i in index]
                for c, i, cc in zip(check, index, combo_check):
                    if len([x for x in cc if x in no_overlap_combo]) > 0:
                        z[pos, i] = 0                       
                    else:
                        break    
        all_like = []
        assignments = []
        scores = []
        ll = []
       
        for i in range(len(z)):
            tmp = []
            tot = 0
            for j in range(self.k):
                pass
            like = z[i,:]
            tot += sum(like)
            ll.append(np.log(tot))
            all_like.append(like)
            scores.append(max(like))
            idx = list(like).index(max(like))
            assignments.append(self.mu[idx])
        return(assignments, scores, all_like, ll)

