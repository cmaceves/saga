"""
TODO:
mdoularize
save models
rewrite as command line param style model

Use numpyro to model beta distributions of frequencies to find key peaks in samples.
"""
import os
import sys
import jax
import copy
import pulp
import json
import numpyro
import pickle
import itertools
import argparse
import numpy as np
import pandas as pd
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpyro.distributions as dist

import random as rnd
from jax import random
from numpyro import handlers
from numpyro.infer import MCMC, NUTS
from numpyro.diagnostics import hpdi
from numpyro.infer import Predictive
from scipy.stats import kstest
from scipy.stats import beta
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
import model_util
import file_util

def account_for_peaks(possible_solutions, means, associated_points):
    """
    Attempt to look at the hiearchy of peaks account for by each solution.
    """
    print(len(possible_solutions))
    kept_solutions = []
    
    for solution in possible_solutions:
        n = len(solution)
        combination_solutions, combination_sums = model_util.generate_combinations(n, solution)
        combination_sums = [round(x,3) for x in combination_sums]
        all_possible_dist = []
        for combo in combination_solutions:
            if len(combo) < 2:
                continue
            mixture_points = []
            small = 100
            for item in combo:
                i = means.index(item)
                mixture_points.append(associated_points[i])
                if len(associated_points[i]) < small:
                    small = len(associated_points[i])
            added = [0]*small
            for tmp in mixture_points:
                sample_list = rnd.choices(tmp, k=small)
                added = [x+y for x,y in zip(sample_list, added)]
            added = [round(x,3) for x in added]
            all_possible_dist.append(added)
        valid = True

        #every grouping of points must be accounted for by the solution
        for mean, ap in zip(means, associated_points):
            if mean in solution:
                continue
            found = False
            #loop through all our combo distributions
            for dist in all_possible_dist:
                p = kstest(ap, dist)
                if p.pvalue > 0.05:
                    found = True
                    break
            if not found:
                valid = False
                break
        if valid:
            kept_solutions.append(solution)
    return(kept_solutions)

def calculate_prob_ratio(chosen_prob, probabilities, frequency, threshold=5):
    """
    Given the probabilities and assignment for a variants, calculate something like the odds ratio. If over a certain threshold return true and call the variant. If a probability is larger than the probability of the selected assignment we ignore it because it's not biologically possible to be the point of assignment.
    """
    chosen_prob = np.log(chosen_prob+1)
    print(probabilities)
    probabilities = np.log([x+1 for x in list(probabilities)])
    print(probabilities)
    diffs = list(chosen_prob-probabilities)
    closest = [x for x in diffs if x > 0]
    closest_index = [i for i,x in enumerate(diffs) if x > 0]
    idx = closest.index(min(closest))
    og_idx = closest_index[idx]
    
    nearest_prob = probabilities[og_idx]
    if nearest_prob == 0:
        return(True)
    ratio = chosen_prob/nearest_prob
    #if frequency == 0.5:
    print(frequency, ratio, chosen_prob, nearest_prob)
    if ratio > threshold:
        return(True)
    else:
        return(False)
    
def choose_solution(filename, output_dir, sample_name):
    """
    Given all distributions for every solution, choose the correct return value. Write the variant assignments to a json file along with information on probability of assignment, and which variants cannot be called and why.
    """
    assignment_output = os.path.join(output_dir, sample_name+"_assignments.json")
    with open(filename, "r") as jfile:
        data = json.load(jfile)
        solutions = data['solutions']
        distributions = data['distributions']
        positions = data['positions']
        frequencies = data['frequency']
        nucs = data['nucs']
        used_peaks = data['used_peaks']

    all_aic = []
    all_assignments = []
    for sol, dist, up in zip(solutions, distributions, used_peaks):
        aic, peak_dict = assign_variants(dist, positions, frequencies, nucs, sol, up)
        all_aic.append(aic)
        all_assignments.append(peak_dict)
        sys.exit(0)
    idx = np.array(all_aic).argmin()
    with open(assignment_output, "w") as afile:
        json.dump(all_assignments[idx], afile)    

def assign_variants(all_distributions, positions, frequencies, nucs, solution, used_peaks):
    """
    Given a set of distributions, positions, frequencies and nucs create a dictionary of consensus sequences.
    """
    #print("\nsolution", solution)
    conflicting_frequencies = find_conflicting_frequencies(positions, frequencies)
    permutations = create_permutations(len(all_distributions)) 
    #add in additional permutations in the event we will allow variants to be assigned to the same group
    conflict_permutations = copy.deepcopy(permutations)
    conflict_permutations.extend(create_conflict_permutations(len(all_distributions)))
    
    assigned_pos, assigned_points, assigned_variants, total_prob, not_assigned = assign_points_groups(positions, frequencies, conflict_permutations, conflicting_frequencies, permutations, all_distributions, used_peaks)
    
    #long form, variants per peak
    peak_variants = {}
    for pos, points, var in zip(assigned_pos, assigned_points, assigned_variants):
        variants = []
        for v in var:
            tmp = v.split("_")
            p = tmp[0]
            f = tmp[1]
            nuc = 'X'
            for ps, fq, nu in zip(positions, frequencies, nucs):
                if str(ps) == p and str(fq) == f:
                    nuc = nu
                    break
            variants.append(str(ps)+nuc)
        peak_variants[round(float(np.mean(points)),3)] = variants
     
    combination_solutions, combination_sums = model_util.generate_combinations(len(solution), solution)
    combination_sums = [round(x,2) for x in combination_sums]
    arr = np.array(combination_sums)
    
    aic = (2 * len(used_peaks)) - (2 * total_prob)
    print(aic, not_assigned)
    #place peak variants into categories based on individual population
    individual_peaks = {}
    for population in solution:
        if 0.03 < population < 0.97:
            individual_peaks[population] = [] 

    #individual mapping
    peak_mapping = {}
    rounded_solution = [round(x,2) for x in solution]
    for original, new in zip(used_peaks, list(peak_variants.keys())):
        if original in rounded_solution: 
            peak_mapping[original] = new

    #assign out variants that are shared
    for i, (key, value) in enumerate(peak_variants.items()):
        match_peak = used_peaks[i]
        #all peak
        if match_peak >= 0.97:
            for sol in solution:
                if 0.03 < sol < 0.97:
                    individual_peaks[sol].extend(value)
            continue
        #noise peak
        elif match_peak <= 0.03:
            continue
        idx = combination_sums.index(match_peak)
        for sol in combination_solutions[idx]:
            if 0.03 < sol < 0.97:
                individual_peaks[sol].extend(value)

    #change the frequency values of the individual populations based on what the model spit out
    final_peak_assign = {}
    for key, value in individual_peaks.items():
        key = round(key,2)
        new_key = peak_mapping[key]
        final_peak_assign[new_key] = value

    return(aic, final_peak_assign)

def create_permutations(n_max):
    """
    Find all possible permutations that variants could be assigned to.
    """
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
    return(permutations)

def assign_points_groups(positions, frequency, conflict_permutations, conflicting_frequencies, permutations, all_distributions, used_peaks=None):
    """
    Given a set of distributions, assign each variant to the most probable group given biological constraints.
    """
    seen_pos = []
    assigned_pos = []
    assigned_points = []
    assigned_variants = []
    total_prob = 0
    not_assigned = []
    for i in range(len(all_distributions)):
        assigned_pos.append([])
        assigned_points.append([])
        assigned_variants.append([])

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
        for permute, f, k in zip(used_permutations[best_i], all_freqs, all_idxs):
            assigned_pos[permute].append(pos)
            assigned_points[permute].append(f)
            assigned_variants[permute].append(str(pos)+"_"+str(f))
            #print(all_distributions[permute][k])
            chosen_prob = all_distributions[permute][k]
            all_probs = np.array(all_distributions)[:,k]
            #ratio_success = calculate_prob_ratio(chosen_prob, all_probs, f)
            ratio_success = True
            if ratio_success:
                total_prob += np.log(chosen_prob)
            else:
                not_assigned.append(str(pos)+"_"+str(f))
        seen_pos.append(pos)
 
    return(assigned_pos, assigned_points, assigned_variants, total_prob, not_assigned)

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

def run_model(total_freq, n_val, num_warmup=50, num_samples=1000):
    """
    Run model and return distribtuion objects
    """
    data = jnp.array(total_freq)
    # Run mcmc
    kernel = NUTS(model)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples)
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
        mean, var, skew, kurt = beta.stats(alpha, betas, moments='mvsk')
        weight = np.mean(samples['weights'][:,n])
        x = np.linspace(beta.ppf(0.01, alpha, betas),
                        beta.ppf(0.99, alpha, betas), 500)
        dist = beta.pdf(x, alpha, betas)
        dist = beta.pdf(total_freq, alpha, betas)
        all_distributions.append(dist)
    return(all_distributions)

def run_numpyro_model(sample_id, gt_mut_dict=None):
    """
    parser = argparse.ArgumentParser(description='Use numpyro to solve peaks in sample.')
    parser.add_argument('-s','--sample_id', help='The name of the sample ID', required=True)
    args = vars(parser.parse_args())
    print(args)
    sys.exit(0)
    """
    #print(jax.devices())
    #print(jax.lib.xla_bridge.get_backend().platform)
    #sys.exit(0)
    numpyro.set_platform('cpu')
    
    output_dir = "/home/chrissy/Desktop/saga_spike_in_results/" + sample_id
    output_file = output_dir + "/" + sample_id + "_beta_dist.json"
    reference_file = "/home/chrissy/Desktop/sequence.fasta"
    bed_file = "/home/chrissy/Desktop/sarscov2_v2_primers.bed"
    variants_json = os.path.join(output_dir, sample_id+"_variants.txt")   

    output_dict = {"solutions":[], "distributions":[], "frequency":[], "positions":[], "nucs":[], "complexity":"", "used_peaks":[]}

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
    #print(complexity, complexity_estimate)
    og_frequency = copy.deepcopy(frequency)
    og_positions = copy.deepcopy(positions)
    frequency = [round(x,3) for x in frequency if 0.01 < round(x,3) < 0.98]
    positions = [x for x,y in zip(positions, og_frequency) if 0.01 < round(y,3) < 0.98]
    nucs = [x for x,y in zip(nucs, og_frequency) if 0.01 < round(y,3) < 0.98]
    global a_params
    global b_params
    global c_params
    global d_params
    a_params = []
    b_params = []
    c_params = []
    d_params = []
    output_dict['complexity'] = complexity
    allow_resample = 0
    if complexity == "low":
        num = 2
        allow_resample = 3
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
        r_min = 5
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
        num = 2
        ranges = [0.03, 0.10, 0.90, 0.97]
        for mu in ranges:
            mu = round(mu, 2)
            sigma = 0.1
            alpha, beta = get_alpha_beta(mu, sigma)  
            a, b = solve_for_log_normal_parameters(alpha, 1)  
            c, d = solve_for_log_normal_parameters(beta, 1)
            a_params.extend([a]*num)
            b_params.extend([b]*num)
            c_params.extend([c]*num)
            d_params.extend([d]*num)
        r_min = 2
        r_max = 3
        n_max = len(a_params)
    
    else:
        sys.exit(1)

    #find positions with indentity problem
    conflicting_frequencies = find_conflicting_frequencies(positions, frequency)
    permutations = create_permutations(n_max) 

    #add in additional permutations in the event we will allow variants to be assigned to the same group
    conflict_permutations = copy.deepcopy(permutations)
    conflict_permutations.extend(create_conflict_permutations(n_max))
    all_distributions = run_model(frequency, n_max)
    assigned_pos, assigned_points, assigned_variants, total_prob, not_assigned = assign_points_groups(positions, frequency, conflict_permutations, conflicting_frequencies, permutations, all_distributions)

    means = []
    final_assigned_points = []
    for i, (ap, pos) in enumerate(zip(assigned_points, assigned_pos)):
        if len(ap) > 0:
            print("\n", i, ap)
            #print(pos)
            means.append(round(np.mean(ap),3))
            final_assigned_points.append(ap)
            #print(round(np.mean(ap),3))
        else:
            means.append(0)

    #if p value < 0.05 reject null
    #null is data are distributed from same distribution
    #p = kstest(final_assigned_points[1], final_assigned_points[1])
    #print(p.pvalue)
    means = [x for x in means if x > 0]
    print("means", means, len(means))    
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
    

    sorted_means = copy.deepcopy(means)
    sorted_means.sort()
        
    for combo in all_combos:
        if min_val < sum(combo) < max_val:
            if combo not in possible_solutions:
                if r_min <= len(combo) < r_max and combo not in possible_solutions and sorted_means[0] in combo:
                    print(combo)
                    possible_solutions.append(combo)

    useful_solutions = account_for_peaks(possible_solutions, means, final_assigned_points) 
    output_dict = {"useful_solutions":useful_solutions, "all_solutions":possible_solutions}
    with open(output_file, "w") as ofile:    
        json.dump(output_dict, ofile)
    return(1)
    #print(solution_dictionary[3])
    output_dict['solutions'] = possible_solutions
    output_dict['positions'] = positions
    output_dict['frequency'] = frequency
    output_dict['nucs'] = nucs 
    print("means", means)
    og_means = copy.deepcopy(means)
    for solution in possible_solutions:
        print("\n")
        n = len(solution)
        combination_solutions, combination_sums = model_util.generate_combinations(n, solution)
        combination_sums = [round(x,2) for x in combination_sums]

        #find the solution peaks that most closely match the observed
        arr = np.array([x for x in combination_sums if 0 < x < 1])
        used_peaks = []    
         
        for value in og_means:
            i = np.abs(arr-value).argmin()
            if arr[i] not in used_peaks and 0.03 <= arr[i] <= 0.97 and abs(arr[i]-value) < 0.05:
                used_peaks.append(arr[i])
        
        #make sure we have a "100%" peak and a noise peak     
        if len(used_peaks) > 0:
            largest = max(used_peaks)
            smallest = min(used_peaks)
            if abs(1-largest) > 0.05:        
                used_peaks.append(0.97)
            if abs(0.03-smallest) > 0.05:
                used_peaks.append(0.03)
        else:
            used_peaks = [0.03, 0.97]
        largest = max(used_peaks)
        smallest = min(used_peaks)
       
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
            sigma = 0.01
            alpha, beta = get_alpha_beta(mu, sigma)  
            
            #print("mu", mu, "sigma", sigma, "alpha", alpha, "beta",beta)
            a, b = solve_for_log_normal_parameters(alpha, 0.001)  
            c, d = solve_for_log_normal_parameters(beta, 0.001)
            if np.isnan(a) or np.isnan(b) or np.isnan(c) or np.isnan(d):                
                continue
            #print(a,b,c,d)
            a_params.append(float(a))
            b_params.append(float(b))
            c_params.append(float(c))
            d_params.append(float(d))
        all_distributions = run_model(frequency, m, num_warmup=500, num_samples=3000)
        output_dict['distributions'].append([list(x) for x in list(all_distributions)])
        output_dict['used_peaks'].append(used_peaks)
    #print(output_dict) 
    with open(output_file, "w") as ofile:    
        json.dump(output_dict, ofile)
    #sys.exit(0)
    return(0)

