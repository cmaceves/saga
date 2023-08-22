"""
TODO:
mdoularize
save models
rewrite as command line param style model

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

def create_conflict_permutations(n):
    """
    Create possible permutations of assignment, opperating under the assumption that two variants may be assigned to the same group.
    """
    permute_simple = list(range(n))
    permute_resample = copy.deepcopy(permute_simple)
    for i in range(3):
        permute_resample.extend(permute_simple)
    all_permutations = itertools.permutations(permute_resample, 3)
    keep_permutations = []
    for permute in all_permutations:
        if len(permute) > 1:
            values, counts = np.unique(permute, return_counts=True)
            counts = counts[counts > 1]
            if len(counts) > 0:
                keep_permutations.append(permute)
    return(keep_permutations)

def find_conflicting_frequencies(positions, frequency):
    """
    At every position, find variants within 0.03 of each other and label them as possible conflicting in terms of assignment.
    """
    unique_positions = list(np.unique(positions))
    conflicting_frequencies = {}
    for pos in unique_positions:
        idxs = [i for i,x in enumerate(positions) if x == pos]
        if len(idxs) <= 1:
            continue
        
        #find every possible duo combo
        all_combos = pulp.allcombinations(idxs, 2)
        problem_combos = []
        for combo in all_combos:
            if len(combo) > 1:
                val1 =  frequency[combo[0]]
                val2 = frequency[combo[1]]
                if abs(val1 - val2) < 0.03:
                    problem_combos.append(list(combo))
        if len(problem_combos) == 0:
            continue
        flat = [item for sublist in problem_combos for item in sublist]
        freq = [x for i,x in enumerate(frequency) if i in flat]
        conflicting_frequencies[pos] = freq

    return(conflicting_frequencies)
    
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

def model(data, n):
    weights = numpyro.sample('weights', dist.Dirichlet(concentration=jnp.ones(n)))
    
    alphas = []
    betas = []
    for i in range(n):
        a = a_params[i]
        b = b_params[i]
        c = c_params[i]
        d = d_params[i]
        
        alpha = numpyro.sample(f'alpha_{i}', dist.LogNormal(a, b))
        beta = numpyro.sample(f'beta_{i}', dist.LogNormal(c, d))
        alphas.append(alpha)
        betas.append(beta)
    
    with numpyro.plate('data', len(data)):
        assignment = numpyro.sample('assignment', dist.Categorical(weights))
        alpha_assigned = jnp.array(alphas)[assignment]
        beta_assigned = jnp.array(betas)[assignment]
        numpyro.sample('obs', dist.Beta(alpha_assigned, beta_assigned), obs=data)
    """
    with numpyro.plate('components', n):
        # define alpha and beta  for n Beta distributions
        alpha = numpyro.sample('alpha', dist.LogNormal(a, b))
        beta = numpyro.sample('beta', dist.LogNormal(c, d))
    with numpyro.plate('data', len(data)):
        assignment = numpyro.sample('assignment', dist.Categorical(weights))
        alpha_assigned = alpha[assignment]
        beta_assigned = alpha[assignment]
        numpyro.sample('obs', dist.Beta(alpha_assigned, beta_assigned), obs=data)
    """

def run_model(total_freq, n_val):
    data = jnp.array(total_freq)
    # Run mcmc
    kernel = NUTS(model)
    mcmc = MCMC(kernel, num_warmup=50, num_samples=1000)
    mcmc.run(random.PRNGKey(112358), data=data, n=n_val)
    #mcmc.print_summary()
    samples = mcmc.get_samples()

    fig, ax = plt.subplots(1, 1)
    cm = plt.get_cmap('gist_rainbow')
    ax.set_prop_cycle(color=[cm(1.*i/n_val) for i in range(n_val)])
    all_distributions = []
    for n in range(n_val):
        alpha = np.mean(samples['alpha_%s'%n])
        betas = np.mean(samples['beta_%s'%n])
        #alpha = np.mean(samples['alpha'][:,n])
        #betas = np.mean(samples['beta'][:,n])
        mean, var, skew, kurt = beta.stats(alpha, betas, moments='mvsk')
        weight = np.mean(samples['weights'][:,n])
        x = np.linspace(beta.ppf(0.01, alpha, betas),
                        beta.ppf(0.99, alpha, betas), 500)
        ax.plot(x, beta.pdf(x, alpha, betas),
               '-', lw=3, alpha=0.6, label='beta pdf 0')
        
        dist = beta.pdf(x, alpha, betas)
        dist = beta.pdf(total_freq, alpha, betas)
        all_distributions.append(dist)
    #plt.savefig("./analysis/figures/test.png")
    """
    samples = mcmc.get_samples()
    # Try doing more checks here .. 
    predictive = Predictive(model, samples)
    predictions = predictive(random.PRNGKey(112358), data = jnp.array(data), n = 2)["obs"]
    print(samples.keys())
    """
    return(all_distributions)

def run_numpyro_model(sample_id, gt_mut_dict=None):
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
    global a_params
    global b_params
    global c_params
    global d_params
    a_params = []
    b_params = []
    c_params = []
    d_params = []
    if complexity == "low":
        num = 2
        for mu in np.arange(0.1, 1.0, 0.1):
            if mu <= 0.50:
                num = 2
            if mu >= 0.70:
                num = 1
            mu = round(mu, 2)
            sigma = 0.1
            alpha, beta = get_alpha_beta(mu, sigma)  
            a, b = solve_for_log_normal_parameters(alpha, 1)  
            c, d = solve_for_log_normal_parameters(beta, 1)
            a_params.extend([a]*num)
            b_params.extend([b]*num)
            c_params.extend([c]*num)
            d_params.extend([d]*num)
        r_min = 4
        r_max = 7
        n_max = len(a_params)
    elif complexity == "extremely low":
        num = 2
        for mu in np.arange(0.2, 1.0, 0.2):
            if mu <= 0.50:
                num = 2
            if mu > 0.50:
                num = 2
            mu = round(mu, 2)
            sigma = 0.1
            alpha, beta = get_alpha_beta(mu, sigma)  
            a, b = solve_for_log_normal_parameters(alpha, 1)  
            c, d = solve_for_log_normal_parameters(beta, 1)
            a_params.extend([a]*num)
            b_params.extend([b]*num)
            c_params.extend([c]*num)
            d_params.extend([d]*num)
        n_max = len(a_params)
        r_min = 3
        r_max = 4
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
    """
    for p, f in zip(positions, frequency):
        print(p, f)
    sys.exit(0)
    """ 
    #total_return_length = 25
    #refined_kde_peaks = model_util.define_kde(frequency, complexity, 3000, total_return_length, 2)
    #print(refined_kde_peaks)
    #sys.exit(0)
    #print(len(frequency), len(positions))
    #find positions with indentity problem
    conflicting_frequencies = find_conflicting_frequencies(positions, frequency)
    assigned_points = []
    assigned_pos = []
    assigned_variants = []
    assigned_probs = []
    for i in range(n_max):
        assigned_points.append([])
        assigned_pos.append([])
        assigned_variants.append([])
        assigned_probs.append([])
    
    permutations = []
    permutations.extend(list(itertools.permutations(list(range(n_max)), r=3)))
    permutations.extend(list(itertools.permutations(list(range(n_max)), r=2)))
    permutations.extend(list(itertools.permutations(list(range(n_max)), r=1)))
    seen_pos = []    

    #add in additional permutations in the event we will allow variants to be assigned to the same group
    conflict_permutations = copy.deepcopy(permutations)
    conflict_permutations.extend(create_conflict_permutations(n_max))

    all_distributions = run_model(frequency, n_max)
    #assign out variants
    for i, pos in enumerate(positions):
        if pos in seen_pos:
            continue
        if pos in conflicting_frequencies:
            used_permutations = conflict_permutations
        else:
            used_permutations = permutations
        all_idxs = [j for j, p in enumerate(positions) if p == pos]
        all_freqs = [frequency[j] for j in all_idxs]
        all_probs = []
        for j in all_idxs:
            tmp = [x[j] for x in all_distributions]
            all_probs.append(tmp)
        all_combo_likelihood = []
        for permute in used_permutations:
            tmp = 0
            if len(permute) == len(all_idxs):
                for idx, probs in zip(permute, all_probs):
                    tmp += probs[idx]
            all_combo_likelihood.append(tmp)
        best_i = all_combo_likelihood.index(max(all_combo_likelihood))
        for permute, f in zip(used_permutations[best_i], all_freqs):
            assigned_pos[permute].append(pos)
            assigned_points[permute].append(f)
            assigned_variants[permute].append(str(pos)+"_"+str(f))
        seen_pos.append(pos)
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
                position, freq = e.split("_")
                if int(position) in duo_pos:
                    #lets go find the matching cluster
                    match_index = find_opposing_cluster(position, freq, i, assigned_variants)
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
    print(kept_chains)
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
    print("new means", new_means, len(new_means))
    print(unsorted_means)
    means = [x for x in means if x > 0]
    #means = new_means
    print("means", means, len(means))
    
    allow_resample = 0
    #print(kept_chains)
    #code has been post-processed
    #set up new model parameters and priors
    all_means = copy.deepcopy(means)
    for i in range(allow_resample):
        all_means.extend([x for x in means if x > 0.02])

    all_means.sort()
    all_combos = pulp.allcombinations(all_means, r_max-1)
    error = 0.05
    possible_solutions = []
    max_val = 1 + error
    min_val = 1 - error

    #
    for combo in all_combos:
        if min_val < sum(combo) < max_val:
            if combo not in possible_solutions:
                if r_min <= len(combo):
                    print(len(possible_solutions), combo)
                    possible_solutions.append(combo)

    for solution in possible_solutions:
        n = len(solution)
        combination_solutions, combination_sums = model_util.generate_combinations(n, solution)
        combination_sums = [round(x,2) for x in combination_sums]

        #find the solution peaks that most closely match the observed
        arr = np.array([x for x in combination_sums if 0 < x < 1])
        used_peaks = []    
        for value in means:
            i = np.abs(arr-value).argmin()
            if arr[i] not in used_peaks and 0 < arr[i] < 0.99:
                used_peaks.append(arr[i])
        #make sure we have a "100%" peak and a noise peak     
        largest = max(used_peaks)
        smallest = min(used_peaks)
        if abs(1-largest) > 0.05:        
            used_peaks.append(0.97)
        if abs(0.03-smallest) > 0.05:
            used_peaks.append(0.03)
        print("combos", arr)
        print("used peaks", used_peaks)
        print("solution", solution)
        #get new params
        a_params = []
        b_params = []
        c_params = []
        d_params = []  
        m = len(used_peaks)
        for mu in used_peaks:
            sigma = 0.1
            alpha, beta = get_alpha_beta(mu, sigma)  
            a, b = solve_for_log_normal_parameters(alpha, 0.1)  
            c, d = solve_for_log_normal_parameters(beta, 0.1)
            if np.isnan(a) or np.isnan(b) or np.isnan(c) or np.isnan(d):                
                if abs(1-mu) <= 0.01:
                    alpha, beta = get_alpha_beta(mu, sigma)  
                    a, b = solve_for_log_normal_parameters(alpha, 0.1)  
                    c, d = solve_for_log_normal_parameters(beta, 0.1)
                #continue
            a_params.append(float(a))
            b_params.append(float(b))
            c_params.append(float(c))
            d_params.append(float(d))
            #print(a_params, b_params, c_params, d_params)

        all_distributions = run_model(frequency, m)
        arr = np.array(all_distributions)
        best_probs = np.amax(arr, axis=0)
        print("\n", solution, sum(np.log(best_probs)))

        assigned_points = []
        assigned_pos = []
        assigned_variants = []
        assigned_probs = []
        for i in range(len(used_peaks)):
            assigned_points.append([])
            assigned_pos.append([])
            assigned_variants.append([])
            assigned_probs.append([])

        permutations = []
        permutations.extend(list(itertools.permutations(list(range(len(all_distributions))), r=3)))
        permutations.extend(list(itertools.permutations(list(range(len(all_distributions))), r=2)))
        permutations.extend(list(itertools.permutations(list(range(len(all_distributions))), r=1)))
        conflict_permutations = copy.deepcopy(permutations)
        conflict_permutations.extend(create_conflict_permutations(len(all_distributions)))

        seen_pos = []
        total_probs = 0
        means = []
        for i, pos in enumerate(positions):
            if pos in seen_pos:
                continue
            if pos in conflicting_frequencies:
                used_permutations = conflict_permutations
            else:
                used_permutations = permutations
            all_idxs = [j for j, p in enumerate(positions) if p == pos]
            all_freqs = [frequency[j] for j in all_idxs]
            all_probs = []
            for j in all_idxs:
                tmp = [x[j] for x in all_distributions]
                all_probs.append(tmp)
            all_combo_likelihood = []
            for permute in used_permutations:
                tmp = 0
                if len(permute) == len(all_idxs):
                    for idx, probs in zip(permute, all_probs):
                        tmp += probs[idx]
                all_combo_likelihood.append(tmp)
            best_i = all_combo_likelihood.index(max(all_combo_likelihood))
            for permute, f in zip(used_permutations[best_i], all_freqs):
                assigned_pos[permute].append(pos)
                assigned_points[permute].append(f)
                assigned_variants[permute].append(str(pos)+"_"+str(f))
                assigned_probs[permute].append(all_distributions[permute][i])
                total_probs += all_distributions[permute][i]
            seen_pos.append(pos)

        print(len(used_peaks), "total probs", total_probs)
        for i, (ap, pos, probs) in enumerate(zip(assigned_points, assigned_pos, assigned_probs)):
            if len(ap) > 0:
                #print("\n", i, ap)
                #print(pos)
                means.append(round(np.mean(ap),3))
                print(round(np.mean(ap),3))
                #print(assigned_probs)
            else:
                means.append(0)
        print(means, total_probs)
    sys.exit(0)     
