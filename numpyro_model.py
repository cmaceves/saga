"""
Use numpyro to model beta distributions of frequencies to find key peaks in samples.
"""
import os
import sys
import copy
import pulp
import json
import numpyro
import pickle
import itertools
import argparse
import numpy as np
import pandas as pd
import networkx as nx
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpyro.distributions as dist


from jax import random
from numpyro import handlers
from numpyro.infer import MCMC, NUTS
from numpyro.diagnostics import hpdi
from numpyro.infer import Predictive
from scipy.stats import beta
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
import model_util
import file_util

def solve_for_log_normal_parameters(mean, variance):
        sigma2 = np.log(variance/mean**2 + 1)
        mu = np.log(mean) - sigma2/2
        return (mu, sigma2)

def get_alpha_beta(mu, sigma):
    alpha = mu**2 * ((1 - mu) / sigma**2 - 1 / mu)
    beta = alpha * (1 / mu - 1)
    return (alpha, beta)

def find_opposing_cluster(position, frequency, index, assigned_variants):
    """
    Post processing code to leverage positional "duos" to combine clusters. Given a position and frequency, find the alterative cluster.
    """

    for i, av in enumerate(assigned_variants):
        if len(av) == 0:
            continue
        freqs = [float(a.split("_")[1]) for a in av]
        if np.mean(freqs) < 0.03:
            continue
        found=False
        if i == index:
            continue
        for edge in av:
            pos, freq = edge.split("_")
            if str(pos) == str(position):
                found=True
        if found: 
            return(i)
    return(None)

def preprocess_data():
    """
    Open the variants file and find the frequencies that are relevant for training.
    """
    df = pd.read_csv('file_5_sorted.calmd.tsv', sep='\t')
    alt_freq = df['ALT_FREQ'].tolist()
    ref_freq = [x/y for x,y in zip(df['REF_DP'].tolist(), df['TOTAL_DP'].tolist())]

    total_freq = copy.deepcopy(alt_freq)
    total_freq.extend(ref_freq)
    #values cannot == 0 or == 1
    total_freq = [float(round(x,3)) for x in total_freq if 0 < round(x,3) < 1]

def model2(data, n):
    weights = numpyro.sample('weights', dist.Dirichlet(concentration=jnp.ones(n)))
    
    alphas = []
    betas = []
    for i in range(n):
        alpha = numpyro.sample(f'alpha_{i}', dist.Gamma(i + 1, 5))
        beta = numpyro.sample(f'beta_{i}', dist.Gamma(2, 5))
        alphas.append(alpha)
        betas.append(beta)

    with numpyro.plate('data', len(data)):
        assignment = numpyro.sample('assignment', dist.Categorical(weights))
        alpha_assigned = jnp.array(alphas)[assignment]
        beta_assigned = jnp.array(betas)[assignment]
        numpyro.sample('obs', dist.Beta(alpha_assigned, beta_assigned), obs=data)

def model(data, n):
    # define weights with a dirichlet prior
    weights = numpyro.sample('weights', dist.Dirichlet(concentration=jnp.ones(n)))
    print(data.shape)
    """
    alpha1 = numpyro.sample('alpha', dist.LogNormal(loc=a, scale=b))
    alpha2 = numpyro.sample('alpha2', dist.LogNormal(loc=a, scale=b))
    beta1 = numpyro.sample('beta', dist.LogNormal(loc=a, scale=b))
    beta2 = numpyro.sample('beta2', dist.LogNormal(loc=a, scale=b))

    alphas = np.array([alpha1, alpha2])
    betas = np.array([beta1, beta2])
    
    """
    with numpyro.plate('components', n):
        # define alpha and beta  for n Beta distributions
        alpha = numpyro.sample('alpha', dist.LogNormal(loc=a, scale=b))
        beta = numpyro.sample('beta', dist.LogNormal(loc=c, scale=d))
    
    # define beta distributions
    with numpyro.plate('data', len(data)):
        assignment = numpyro.sample('assignment', dist.Categorical(weights))
        numpyro.sample('obs', dist.Beta(alpha[assignment], beta[assignment]), obs=data)

def run_model(total_freq, n_val):
    data = jnp.array(total_freq)
    # Run mcmc
    kernel = NUTS(model)
    mcmc = MCMC(kernel, num_warmup=50, num_samples=1000)
    mcmc.run(random.PRNGKey(112358), data=data, n=n_val)
    mcmc.print_summary()
    samples = mcmc.get_samples()

    fig, ax = plt.subplots(1, 1)
    cm = plt.get_cmap('gist_rainbow')
    ax.set_prop_cycle(color=[cm(1.*i/n_val) for i in range(n_val)])
    all_distributions = []
    for n in range(n_val):
        alpha = np.mean(samples['alpha'][:,n])
        betas = np.mean(samples['beta'][:,n])
        mean, var, skew, kurt = beta.stats(alpha, betas, moments='mvsk')
        #if n == 3:
        #    continue
        weight = np.mean(samples['weights'][:,n])
        x = np.linspace(beta.ppf(0.01, alpha, betas),
                        beta.ppf(0.99, alpha, betas), 500)
        ax.plot(x, beta.pdf(x, alpha, betas),
               '-', lw=3, alpha=0.6, label='beta pdf 0')
        
        dist = beta.pdf(x, alpha, betas)
        dist = beta.pdf(total_freq, alpha, betas)
        all_distributions.append(dist)
        """
        for x_val, d_val in zip(x, dist):
            print(x_val, d_val)
        """
        print("N", n, "mean", mean, "var", var, "weight", weight)

    #plt.savefig("./analysis/figures/test.png")
    """
    samples = mcmc.get_samples()
    # Try doing more checks here .. 
    predictive = Predictive(model, samples)
    predictions = predictive(random.PRNGKey(112358), data = jnp.array(data), n = 2)["obs"]
    print(samples.keys())
    """
    return(all_distributions)

def run_numpyro_model(sample_id):
    """
    parser = argparse.ArgumentParser(description='Use numpyro to solve peaks in sample.')
    parser.add_argument('-s','--sample_id', help='The name of the sample ID', required=True)
    args = vars(parser.parse_args())
    print(args)
    sys.exit(0)
    """
    output_dir = "/home/chrissy/Desktop/saga_spike_in_results/" + sample_id
    reference_file = "/home/chrissy/Desktop/sequence.fasta"
    bed_file = "/home/chrissy/Desktop/sarscov2_v2_primers.bed"
    variants_json = os.path.join(output_dir, sample_id+"_variants.txt")
    
    if os.path.isfile(variants_json):
        with open(variants_json, "r") as rfile:
            primer_dict = json.load(rfile)
    else:
        print("variants file not found")
        sys.exit(1)

    reference_sequence = file_util.parse_reference_sequence(reference_file)
    primer_positions = file_util.parse_bed_file(bed_file)

    frequency, nucs, positions, depth, low_depth_positions, reference_positions, ambiguity_dict, \
        total_mutated_pos, training_removed = file_util.parse_variants(primer_dict, primer_positions, reference_sequence)
    complexity, complexity_estimate = model_util.create_complexity_estimate(total_mutated_pos)
    print(complexity, complexity_estimate)
    og_frequency = copy.deepcopy(frequency)
    og_positions = copy.deepcopy(positions)
    frequency = [round(x,3) for x in frequency if 0.01 < round(x,3) < 0.98]
    positions = [x for x,y in zip(positions, og_frequency) if 0.01 < round(y,3) < 0.98]

    global a
    global b
    global c
    global d
    print(len(frequency))
    if complexity == "low":
        r_min = 4
        r_max = 6
        n_max = 31
        #alpha 1.75/0.5
        a = 1.75
        b = 0.5
        #beta 2.25/0.5
        c = 2.25
        d = 0.5
    elif complexity == "extremely low":
        n_max = 16
        r_min = 3
        r_max = 4
        a = 1.75
        b = 0.5
        c = 2.25
        d = 0.5
    elif complexity == "singular":
        a = 5
        b = 1
        c = 5
        b = 1
        r_min = 2
        r_max = 3
        n_max = 16
    else:
        sys.exit(1)
    #frequency.sort()
    print(len(frequency), len(positions))
    """
    for freq, pos in zip(frequency, positions):
        print(freq, pos)
    """
    all_distributions = run_model(frequency, n_max)
    assigned_points = []
    assigned_pos = []
    assigned_variants = []
    for i in range(n_max):
        assigned_points.append([])
        assigned_pos.append([])
        assigned_variants.append([])
    permutations = []
    permutations.extend(list(itertools.permutations(list(range(len(all_distributions))), r=3)))
    permutations.extend(list(itertools.permutations(list(range(len(all_distributions))), r=2)))
    permutations.extend(list(itertools.permutations(list(range(len(all_distributions))), r=1)))
    seen_pos = []    
    for i, pos in enumerate(positions):
        if pos in seen_pos:
            continue
        all_idxs = [j for j, p in enumerate(positions) if p == pos]
        all_freqs = [frequency[j] for j in all_idxs]
        all_probs = []
        for j in all_idxs:
            tmp = [x[j] for x in all_distributions]
            all_probs.append(tmp)
        all_combo_likelihood = []
        for permute in permutations:
            tmp = 0
            if len(permute) == len(all_idxs):
                for idx, probs in zip(permute, all_probs):
                    tmp += probs[idx]
            all_combo_likelihood.append(tmp)
        best_i = all_combo_likelihood.index(max(all_combo_likelihood))
        for permute, f in zip(permutations[best_i], all_freqs):
            assigned_pos[permute].append(pos)
            assigned_points[permute].append(f)
            assigned_variants[permute].append(str(pos)+"_"+str(f))
        seen_pos.append(pos)

    
    #sys.exit(0)
    means = []
    for i, (ap, pos) in enumerate(zip(assigned_points, assigned_pos)):
        if len(ap) > 0:
            print("\n", i, ap)
            print(pos)
            means.append(round(np.mean(ap),3))
            print(round(np.mean(ap),3))
        else:
            means.append(0)
    unsorted_means = copy.deepcopy(means)     
    means.sort()
    print(means)
    print(sample_id)   
    unique_pos, unique_count = np.unique(positions, return_counts=True)
    duo_pos = [x for x,y in zip(unique_pos,unique_count) if y == 2]
    variants = [str(p)+"_"+str(f) for p,f in zip(positions, frequency)]
    new_cluster = {}
    clusters = {}
    for i, (mean, edge) in enumerate(zip(unsorted_means, assigned_variants)):
        new_cluster[i] = []
        clusters[i] = []
        if mean < 0.03:
            new_cluster[i].append(i)
            continue
        if len(edge) > 0:
            #look at every variant assigned to this code
            for e in edge:
                position, frequency = e.split("_")
                if int(position) in duo_pos:
                    #lets go find the matching cluster
                    match_index = find_opposing_cluster(position, frequency, i, assigned_variants)
                    if match_index is not None:
                        new_cluster[i].append(match_index)
        if len(new_cluster[i]) == 0:
            new_cluster[i].append(i)
    for key, value in new_cluster.items():
        if len(value) == 0:
            continue
        value = list(np.unique(value))
        for index in value:
            clusters[int(index)].extend(value)
    kept_chains = []
    #find the longest chain containing each index
    for i in range(len(clusters)):
        longest=0
        chain=[]
        for key, value in clusters.items():
            if i in value:
                value = list(np.unique(value))
                if len(value) > longest:
                    longest = len(value)
                    chain = value
        if len(chain) > 0 and chain not in kept_chains:
            kept_chains.append(chain)
    #print(clusters)
    #print(kept_chains)
    new_point_assign = []
    new_means = []
    for chain in kept_chains:
        tmp = []
        for index in chain:
            tmp.extend(assigned_points[index])
        if len(tmp) != 0:
            new_point_assign.append(tmp)
            new_means.append(np.mean(tmp))

    new_means = [round(x,3) for x in new_means]
    print("new means", new_means)
    print(unsorted_means)
    for mean,point in zip(new_means, new_point_assign):
        print(mean, point)
    means = new_means
    allow_resample = 5
    #print(kept_chains)
    #code has been post-processed
    all_means = copy.deepcopy(means)
    for i in range(allow_resample):
        all_means.extend([x for x in means if x > 0.02])

    all_means.sort()
    all_combos = pulp.allcombinations(all_means, r_max)
    error = 0.10
    possible_solutions = []
    max_val = 1 + error
    min_val = 1 - error
    for combo in all_combos:
        #print(list(combo))
        if min_val < sum(combo) < max_val:
            if combo not in possible_solutions:
                if r_min <= len(combo) < r_max:
                    print(combo)
                    possible_solutions.append(combo)

    #set up new model parameters and priors
    length_new_means = len(new_means)
    solution = list(possible_solutions[0])
    n = len(solution)
    combination_solutions, combination_sums = model_util.generate_combinations(n, solution)
    combination_sums = [round(x,2) for x in combination_sums]
    
    #find the solution peaks that most closely match the observed
    arr = np.array(combination_sums)
    used_peaks = []

    for value in new_means:
        i = np.abs(arr-value).argmin()
        used_peaks.append(arr[i])
    
    for mu in used_peaks:
        sigma = 0.05
        alpha, beta = get_alpha_beta(mu, sigma)  
        a, b = solve_for_log_normal_parameters(alpha, 0.1)  
        c, d = solve_for_log_normal_parameters(beta, 0.1)
        a = float(a)
        b = float(b)
        c = float(c)
        d = float(d)
        #alpha = numpyro.sample('alpha', dist.LogNormal(loc=a, scale=b))
        #beta = numpyro.sample('beta', dist.LogNormal(loc=c, scale=d))

    
    sys.exit(0)     
