import os
import sys
import copy
import itertools
import numpy as np
from numpy import random
import matplotlib.transforms as transforms
from scipy.stats import multivariate_normal
np.seterr(all="ignore")

class GMM():
    def __init__(self, k, dim, init_mu=None, init_sigma=None, init_pi=None, name=None, solution=None, default_sigma=None, fixed_means=False, filename=None):
        self.aics = []
        self.fixed_means = fixed_means
        self.init_mu = init_mu
        self.iteration = 0
        if default_sigma is None:
            self.default_sigma = 1
        else:
            self.default_sigma = default_sigma
        if filename is not None:
            self.filename = filename
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
        self.aic = 1000

    def init_em(self, X, positions, combination_solutions, combination_sums, error):
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
            self.mu_combo.append(list(combination_solutions[idx]))
           
        self._positional_conflicts()
        self.possible_assign_pairs(error)       
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

    def possible_assign_pairs(self,error):
        means = np.squeeze(self.mu)
        combos = self.mu_combo
        possible_assign_pairs = []
        two_combos = list(itertools.permutations(combos, 2))
        for combination in two_combos:
            a = list(combination[0])
            b = list(combination[1])
            flat = copy.deepcopy(a)
            flat.extend(b)
            total = sum(flat)
            if total < 1-error or total > 1+error:
                continue
            u, count = np.unique(flat, return_counts=True)
            overlap = [x for x in count if x > 1]
            if len(overlap) == 0:
                idx_a = combos.index(a)
                idx_b = combos.index(b)
                pair = [idx_a, idx_b]
                possible_assign_pairs.append(pair)                          
        three_combos = list(itertools.permutations(combos, 3))
        for combination in three_combos:
            if len(combination) != 3:
                continue
            a = list(combination[0])
            b = list(combination[1])
            c = list(combination[2])
            flat = copy.deepcopy(a)
            flat.extend(b)
            flat.extend(c)
            total = sum(flat)
            u, count = np.unique(flat, return_counts=True)
            overlap = [x for x in count if x > 1]
            if total < 1-error or total > 1+error:
                continue
            if len(overlap) == 0:
                idx_a = combos.index(combination[0])
                idx_b = combos.index(combination[1])
                idx_c = combos.index(combination[2])
                pair = [idx_a, idx_b, idx_c] 
                possible_assign_pairs.append(pair)                          
        four_combos = list(itertools.combinations(combos, 4))
        for combination in four_combos:
            if len(combination) != 4:
                continue
            a = list(combination[0])
            b = list(combination[1])
            c = list(combination[2])
            d = list(combination[3])
            flat = copy.deepcopy(a)
            flat.extend(b)
            flat.extend(c)
            flat.extend(d)
            total = sum(flat)
            u, count = np.unique(flat, return_counts=True)
            overlap = [x for x in count if x > 1]
            if total < 1-error or total > 1+error:
                continue
            if len(overlap) == 0:
                idx_a = combos.index(combination[0])
                idx_b = combos.index(combination[1])
                idx_c = combos.index(combination[2])
                idx_d = combos.index(combination[3])
                pair = [idx_a, idx_b, idx_c, idx_d] 
                possible_assign_pairs.append(pair)                         
        self.possible_assign_pairs = possible_assign_pairs

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

        for pc in position_conflict_idx:
            length = len(pc)
            ml_combos = []
            for pair in self.possible_assign_pairs:
                ml = 0
                tmp = []
                if len(pair) != length:
                    ml_combos.append(0)
                    continue
                for a,b in zip(pc, pair):
                    ml += z[a,b]
                    tmp.append(z[a,b])
                #ml_combos.append(ml)
                ml_combos.append(min(tmp))
            #this could cause issues
            if len([x for x in ml_combos if x == 0]) == len(ml_combos):
                for pos in pc:
                    for i, x in enumerate(z[pos,:]):
                        z[pos, i] = 0
                continue
            
            max_val = max(ml_combos)
            ml_idx = ml_combos.index(max(ml_combos))
            for pos, assign in zip(pc, self.possible_assign_pairs[ml_idx]):
                largest = z[pos, assign]
                for i, x in enumerate(z[pos,:]):
                    if z[pos, i] > largest:
                        z[pos, i] = 0
        assignments = []
        scores = []
        aic_scores = []
        pos_zero = []
        ll = []
        all_diff  = 0 
        all_like = []
        for i in range(len(z)):
            tmp = []
            tot = 0
            for j in range(self.k):
                pass
            like = z[i,:]
            tot += sum(like)
            if tot > 0:
                ll.append(np.log(tot))
            else:
                ll.append(0)
            idx = list(like).index(max(like))
            diff = abs(X[i]-self.mu[idx])
            all_diff += diff[0]
            assignments.append(self.mu[idx])
            scores.append(max(like))
            all_like.append(like)
        sll = [np.log(x) if x > 0 else 0 for x in scores]
        self.aic = (2 * len(self.mu)) - (2 * sum(sll))
        return(assignments, scores, all_like, ll, self.aic, all_diff)

