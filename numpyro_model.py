import os
import sys
import jax
import copy
import pulp
import json
import numpyro
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
from scipy import stats
from scipy.stats import kstest
from scipy.stats import beta

import model_util
import file_util
DEBUG = False

def map_solution_experimental_peaks(experimental, solution, combination_sums, combination_solutions, used_peaks):
    """
    For any solution we have the "initial" solution modes which we use to seed our second model, however when the model learns the distributions it changes the modes slighlty. Here we map the learned cluster modes back to the original solution mode.
    """
    individual_mapping = {}
    all_mapping = {}
    #first lets just map individual lineages to experimental modes
    for e_peak, u_peak in zip(experimental, used_peaks):
        if u_peak not in solution:
            continue
        if u_peak not in individual_mapping:
            individual_mapping[u_peak] = []
        individual_mapping[u_peak].append(e_peak)
        all_mapping[e_peak] = [e_peak]
    experimental_individual = list(all_mapping.keys())

    #next we map combinations of experimental modes to individual experimental modes
    for e_peak, u_peak in zip(experimental, used_peaks):
        if u_peak in solution:
            continue
        if 0.98 == e_peak:
            all_mapping[e_peak] = experimental_individual
        elif 0.02 == u_peak:
            all_mapping[e_peak] = []
        else:
            idx = combination_sums.index(u_peak)
            tmp = [] #store the translated combination of peaks
            conflicts = []
            for c_peak in combination_solutions[idx]:
                if len(individual_mapping[c_peak]) < 2:
                    tmp.extend(individual_mapping[c_peak])
                #we had two peaks at the same initial mode and now we need to resolve it                
                else:
                    conflicts.append(c_peak)
            #how much have we concretely acounted for when trying to sum to this mode?
            total_accounted_for = sum(tmp)
            #how much do we lack?
            diff = abs(e_peak-total_accounted_for)
            unique_conflicts, unique_counts = np.unique(conflicts, return_counts=True)
            
            pick = []
            choices = []
            for ucon, ucount in zip(list(unique_conflicts), list(unique_counts)):
                matches = individual_mapping[ucon]
                #we use all identical modes to form this peak
                if len(matches) == ucount:
                    tmp.extend(matches)
                    diff -= sum(matches)
                else:
                    pick.append(ucount) #the number of peaks we choose from the identical modes
                    choices.append(matches)
            if len(unique_conflicts) > 1:
                print("Not generalizable peak sorting code")
                sys.exit(1)
            #TODO: here we have the ability to choose multiple peaks, this isn't currently generalizable...
            for p, c in zip(pick, choices):
                csol, csum = model_util.generate_combinations(p, c)          
                kept_csol = []
                kept_csum = []
                dist_diff = [] #how much does choosing this set of peaks close the gap for what we still need to account for
                for a, b in zip(csol, csum):
                    if len(a) == p:
                        kept_csol.append(a)
                        kept_csum.append(b) 
                for a, b in zip(kept_csol, kept_csum):
                    dist_diff.append(abs(diff-b))
                
                idx = dist_diff.index(min(dist_diff))
                tmp.extend(list(kept_csol[idx]))
            all_mapping[e_peak] = tmp

    return(all_mapping, experimental_individual)

def calculate_prob_ratio(chosen_prob, probabilities, frequency, used_peaks, combo_solutions, combo_sums, pos=None, threshold=5, noise_peak = 0.02, universal_peak = 0.98):
    """
    Given the probabilities and assignment for a variants, calculate something like the odds ratio. If over a certain threshold return true and call the variant. If a probability is larger than the probability of the selected assignment we ignore it because it's not biologically possible to be the point of assignment.
    """
    diffs = list(chosen_prob-probabilities)
    closest = [x for x in diffs if x > 0]   
    closest_index = [i for i,x in enumerate(diffs) if x > 0]
    #we only have one group where the probability of assignment is zero
    if len(closest) == 0:
        return(True, [], [])
    idx = closest.index(min(closest))
    og_idx = closest_index[idx]
    nearest_prob = probabilities[og_idx]
    if nearest_prob == 0:
        return(True, [], [])
    ratio = chosen_prob/nearest_prob
    if ratio > threshold:
        return(True, [], [])
    else:
        #if we've failed the initial test, we now try and see which groups differ between the two "chosen" solutions in an attempt to resolve small differences
        #print("ratio test failed")
        #print(chosen_prob, nearest_prob, probabilities, frequency)
        l_idx = probabilities.argmax() #index of the chosen prob
        peak_l = used_peaks[l_idx]
        peak_s = used_peaks[og_idx]
        if peak_l == universal_peak:
            combo_l = combo_solutions[-1] 
        elif peak_s == universal_peak:
            combo_s = combo_solutions[-1]
        if peak_l == noise_peak:
            combo_l = set([0.02])
        elif peak_s == noise_peak:
            combo_s = set([0.02])
        try:
            idx_l = combo_sums.index(peak_l)    
            combo_l = combo_solutions[idx_l]
        except:
            pass
        try:
            idx_s = combo_sums.index(peak_s)
            combo_s = combo_solutions[idx_s]
        except:
            pass
       
        overlap = [x for x in combo_l if x in combo_s]
        non_overlap = [x for x in combo_l if x not in combo_s]
        non_overlap.extend([x for x in combo_s if x not in combo_l])
        return(False, overlap, non_overlap)
    
def choose_solution(filename, output_dir, sample_name):
    """
    Given all distributions for every solution, choose the correct return value. Write the variant assignments to a json file along with information on probability of assignment, and which variants cannot be called and why.
    """
    low_freq_bound = 0.02
    high_freq_bound = 0.98
    freq_precision = 3
    assignment_output = os.path.join(output_dir, sample_name+"_assignments.json")
    reference_file = "/home/chrissy/Desktop/sequence.fasta"
    bed_file = "/home/chrissy/Desktop/sarscov2_v2_primers.bed"
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
    all_assignments_untrimmed = []
    all_not_assigned = []
    all_reason_not_assigned = []
    variants_json = os.path.join(output_dir, sample_name+"_variants.txt")   
    with open(variants_json, "r") as rfile:
        primer_dict = json.load(rfile)
    
    reference_sequence = file_util.parse_reference_sequence(reference_file)
    primer_positions = file_util.parse_bed_file(bed_file)
    frequency, nucs, positions, depth, low_depth_positions, reference_positions, ambiguity_dict, \
        total_mutated_pos, training_removed = file_util.parse_variants(primer_dict, primer_positions, reference_sequence, depth_cutoff=9)
    #here we get all variants for the file, even those exlcuded in the training process
    new_frequency, new_nucs, new_positions, add_low_depth_positions, universal_mutations, ambiguity_dict, primer_binding_issue = file_util.parse_additional_var(primer_dict, primer_positions, reference_sequence, ambiguity_dict, low_freq_bound, high_freq_bound, depth_cutoff=10)

    new_variants = [str(p) + str(n) for p,n,f in zip(new_positions, new_nucs, new_frequency) if f > low_freq_bound and f < high_freq_bound]
    new_positions = [str(p) for p,f in zip(new_positions, new_frequency) if f > low_freq_bound and f < high_freq_bound]
    new_nucs = [str(n) for n,f in zip(new_nucs, new_frequency) if f > low_freq_bound and f < high_freq_bound]
    new_frequency = [round(x,freq_precision) for x in new_frequency if x > low_freq_bound and x < high_freq_bound]
 
    #map the pos_freq combo to the posnuc combo
    mapping = {} 
    for f, n, p in zip(new_frequency, new_nucs, new_positions):
        mapping[str(p)+"_"+str(f)] = str(p)+str(n)
        mapping[str(p)+str(n)] = str(p)+"_"+str(f)
    for sol, dist, up in zip(solutions, distributions, used_peaks):
        if sol != [0.017, 0.196, 0.242, 0.242, 0.309]:
            continue
        print("\nnew solution", sol)
        print("used peaks", up)
        #we need to seperate out things that are not assigned due to peaks being too close together vs the likelihood being shit
        aic, peak_dict, not_assigned, reason_not_assigned, assignment_fail_dict = assign_variants(dist, new_positions, new_frequency, new_nucs, sol, up)
        print(peak_dict.keys())
        all_aic.append(aic)
        all_not_assigned.append(not_assigned)
        all_reason_not_assigned.append(reason_not_assigned)
        new_peak_dict = {}
        #given the list of groups where we cannot assign a variant, remove it from the assignment dictionary
        for key, value in peak_dict.items():
            new_value = []
            for var in value:
                freq_assign = mapping[var]
                #we've had a least 2 groups conflated in assignment
                if freq_assign in assignment_fail_dict:
                    overlap = assignment_fail_dict[freq_assign][0]
                    if key in overlap:
                        new_value.append(var)
                    #else:
                        #print('removed', assignment_fail_dict[freq_assign],  key, var, freq_assign)
                else:
                    new_value.append(var)
            new_peak_dict[key] = new_value
        all_assignments.append(new_peak_dict)
        all_assignments_untrimmed.append(peak_dict)

    idx = np.array(all_aic).argmin()
    original_peak_dict = all_assignments_untrimmed[idx]
    new_peak_dict = all_assignments[idx]
    kept_percentage = {} 
    for k, v in new_peak_dict.items():
        len_og = len(original_peak_dict[k])
        len_new = len(v)
        print("percent remaining", k, len_new/len_og)
        kept_percentage[k] = len_new/len_og
        for item in v:
            pos = item[:-1]
            nuc = item[-1]
            """
            for p, n, f in zip(new_positions, new_nucs, new_frequency):
                if str(p) == str(pos) and str(n) == str(nuc):
                    #print(p, n , f)
            """
    if len(all_aic) == 0:
        return(1)
    output_dict = {'variants':all_assignments[idx], 'aic':all_aic[idx], 'not_assigned':all_not_assigned[idx], "reason_not_assigned":all_reason_not_assigned[idx], "kept_percent":kept_percentage}
     
    with open(assignment_output, "w") as afile:
        json.dump(output_dict, afile)    
    return(0)

def assign_variants(all_distributions, positions, frequencies, nucs, solution, used_peaks):
    """
    Given a set of distributions, positions, frequencies and nucs create a dictionary of consensus sequences.
    """
    min_freq = 0.02
    max_freq = 0.98
    decimal = 3
    #print("\nsolution", solution)
    conflicting_frequencies = find_conflicting_frequencies(positions, frequencies)
    permutations = create_permutations(len(used_peaks)) 
    #add in additional permutations in the event we will allow variants to be assigned to the same group
    conflict_permutations = copy.deepcopy(permutations)
    conflict_permutations.extend(create_conflict_permutations(len(used_peaks)))
    x = np.linspace(0.01, 0.99, 1000)
    #the float type affects the rounding
    solution = [np.float64(x) for x in solution]
    combination_solutions, combination_sums = model_util.generate_combinations(len(solution), solution)
    combination_sums = [round(x, decimal) for x in combination_sums]

    assigned_pos, assigned_points, assigned_variants, total_prob, not_assigned, reason_not_assigned, assignment_fail_dict = assign_points_groups(positions, frequencies, conflict_permutations, conflicting_frequencies, permutations, all_distributions, x, used_peaks, True, combination_solutions, combination_sums)
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
        print(var)
        mode = round(stats.mode(points).mode, decimal)
        mean = round(float(np.mean(points)), decimal)
        peak_variants[mode] = variants
        points.sort()
        print(points, mode, mean)
    print(not_assigned)
    #HERE WE HAVE AN ERROR
    sys.exit(0)
    aic = (2 * len(used_peaks)) - (2 * total_prob)
    all_mapping, individual_peaks = map_solution_experimental_peaks(list(peak_variants.keys()), solution, combination_sums, combination_solutions, used_peaks)
    #prepopulate assignment dict
    final_peak_assign = {}
    for ip in individual_peaks:
        final_peak_assign[ip] = []

    for key, value in peak_variants.items():
        if key in final_peak_assign:
            final_peak_assign[key].extend(value)
        else:
            for individual in all_mapping[key]:
                final_peak_assign[individual].extend(value)
    
    return(aic, final_peak_assign, not_assigned, reason_not_assigned, assignment_fail_dict)

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

def assign_points_groups(positions, frequency, conflict_permutations, conflicting_frequencies, permutations, all_distributions, x, used_peaks=None, calc_ratio=False, combo_solutions=None, combo_sums=None):
    """
    x : 
        The np.linspace object where values were passed to pdf.

    Given a set of distributions, assign each variant to the most probable group given biological constraints.
    """
    seen_pos = []
    assigned_pos = []
    assigned_points = []
    assigned_variants = []
    total_prob = 0
    not_assigned = []
    reason_not_assigned = [] #either low likelihood, or indistinguishable between groups
    assignment_fail_dict = {} #for those not assigned because mulitple assignment, what's overlapping
 
    all_distributions = np.array(all_distributions)
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
        all_linspace = [np.abs(x-f).argmin() for f in all_freqs]
        all_probs = []
        for j, l in zip(all_idxs, all_linspace):
            tmp = all_distributions[:, l]
            all_probs.append(tmp)
        all_combo_likelihood = []
        for permute in used_permutations:
            tmp = 0
            if len(permute) == len(all_idxs):
                for idx, probs in zip(permute, all_probs):
                    tmp += probs[idx]
            all_combo_likelihood.append(tmp)
        best_i = all_combo_likelihood.index(max(all_combo_likelihood))
        for permute, f, k in zip(used_permutations[best_i], all_freqs, all_linspace):
            var = str(pos)+"_"+str(f)
            assigned_pos[permute].append(pos)
            assigned_points[permute].append(f)
            assigned_variants[permute].append(var)
            chosen_prob = all_distributions[permute, k]
            all_probs = all_distributions[:,k]
            if calc_ratio is True:
                #handles the case of not being able to concretely assign between multiple groups
                ratio_success, overlap, non_overlap = calculate_prob_ratio(chosen_prob, all_probs, f, used_peaks, combo_solutions, combo_sums, pos)
            else:
                ratio_success = True
            #handles the case of just having shit assignment in general
            if chosen_prob < 0.001:
                small_prob = True   
            else:
                small_prob = False
            #assigned 
            if ratio_success is True and small_prob is False:
                if chosen_prob != 0:    
                    total_prob += np.log(chosen_prob)
            #not assigned
            else:
                if ratio_success is True and small_prob is True:
                    reason_not_assigned.append("low likelihood")
                    total_prob += np.log(0.000001)
                #here we assign to 
                elif ratio_success is False and small_prob is False:
                    reason_not_assigned.append("muliple assignments")
                    assignment_fail_dict[var] = [overlap, non_overlap]
                    total_prob += np.log(chosen_prob)
                else:
                    reason_not_assigned.append("both")
                    assignment_fail_dict[var] = [overlap, non_overlap]
                    total_prob += np.log(0.000001)
                not_assigned.append(str(pos)+"_"+str(f))
        seen_pos.append(pos)
 
    return(assigned_pos, assigned_points, assigned_variants, total_prob, not_assigned, reason_not_assigned, assignment_fail_dict)

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
        #generate vector from 0.01-0.99 with steps of 0.001
        x = np.linspace(0.01, 0.99, 1000)
        dist = beta.pdf(x, alpha, betas)
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

    min_freq = 0.02
    max_freq = 0.98    
    decimal = 3

    output_dir = "/home/chrissy/Desktop/saga_spike_in_results/" + sample_id
    output_file = output_dir + "/" + sample_id + "_beta_dist.json"
    reference_file = "/home/chrissy/Desktop/sequence.fasta"
    bed_file = "/home/chrissy/Desktop/sarscov2_v2_primers.bed"
    output_dict = {"solutions":[], "distributions":[], "frequency":[], "positions":[], "nucs":[], "complexity":"", "used_peaks":[]}
    variants_file = "/home/chrissy/Desktop/saga/test_data/output_test.txt"
       
    reference_sequence = file_util.parse_reference_sequence(reference_file)
    primer_positions = file_util.parse_bed_file(bed_file)

    #reference is True/False if nuc/pos combo is reference
    frequency, nucs, positions, reference, flagged = file_util.parse_ivar_variants(variants_file, reference_sequence)
    sys.exit(0)

    frequency, nucs, positions, depth, low_depth_positions, reference_positions, ambiguity_dict, \
        total_mutated_pos, training_removed = file_util.parse_variants(primer_dict, primer_positions, reference_sequence)

    complexity, complexity_estimate = model_util.create_complexity_estimate(total_mutated_pos, reference_sequence)

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
        allow_resample = 5
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
        for mu in np.arange(0.03, 1.0, 0.1):
        #for mu in priors:
            mu = round(mu, 2)
            sigma = 0.1
            alpha, beta = get_alpha_beta(mu, sigma)  
            a, b = solve_for_log_normal_parameters(alpha, 1)  
            c, d = solve_for_log_normal_parameters(beta, 1)
            a_params.extend([a]*num)
            b_params.extend([b]*num)
            c_params.extend([c]*num)
            d_params.extend([d]*num)
            #print(mu)
        #sys.exit(0)
        n_max = len(a_params)
        r_min = 2
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

    #test points for distribtuion
    x = np.linspace(0.01, 0.99, 1000)

    #find positions with indentity problem
    conflicting_frequencies = find_conflicting_frequencies(positions, frequency)
    permutations = create_permutations(n_max) 

    #add in additional permutations in the event we will allow variants to be assigned to the same group
    conflict_permutations = copy.deepcopy(permutations)
    conflict_permutations.extend(create_conflict_permutations(n_max))
    all_distributions = run_model(frequency, n_max)
    
    assigned_pos, assigned_points, assigned_variants, total_prob, not_assigned, reason_not_assigned, removed_peak_dict = assign_points_groups(positions, frequency, conflict_permutations, conflicting_frequencies, permutations, all_distributions, x)

    means = []
    final_assigned_points = []
    for i, (ap, pos) in enumerate(zip(assigned_points, assigned_pos)):
        if len(ap) > 0:
            print("\n", i, ap)
            means.append(round(np.mean(ap), decimal))
            final_assigned_points.append(ap)
        else:
            means.append(0)

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
                    #print(combo)
                    possible_solutions.append(combo)

    output_dict['solutions'] = []
    output_dict['positions'] = positions
    output_dict['frequency'] = frequency
    output_dict['nucs'] = nucs 
    print("means", means)
    og_means = copy.deepcopy(means)
    for solution in possible_solutions:
        if solution != (0.017, 0.196, 0.242, 0.242, 0.309):
            continue
        print("\n")
        print(solution)
        solution = list(solution)
        output_dict['solutions'].append(solution)
        #we don't want to include noise in coming up with peak combos
        tmp_sol = [x for x in solution if x >= 0.02]
        combination_solutions, combination_sums = model_util.generate_combinations(len(tmp_sol), tmp_sol)
        combination_sums = [round(x, decimal) for x in combination_sums]
        #lets take only peaks from possible combos that are in the area of what the first model discovered
        arr = np.array([round(x, decimal) for x in means])
        used_peaks = []    
        for value in combination_sums:
            i = np.abs(arr-value).argmin()
            if 0.02 <= arr[i] <= 0.98 and abs(arr[i]-value) < 0.10 and value not in used_peaks:
                tmp = [value]
                if value in solution:
                    tmp = [value]*solution.count(value)
                print(tmp)
                used_peaks.extend(tmp)
        #make sure we have a "100%" peak and a noise peak     
        if len(used_peaks) > 0:
            largest = max(used_peaks)
            smallest = min(used_peaks)
            if abs(1-largest) > 0.05:        
                used_peaks.append(0.98)
            if abs(0.03-smallest) > 0.05:
                used_peaks.append(0.02)
        else:
            used_peaks = [0.02, 0.98]

        #get new params
        a_params = []
        b_params = []
        c_params = []
        d_params = []  
        m = len(used_peaks)
        fup = []
        for mu in used_peaks:
            if mu <= 0.03:
                sigma = 0.009
                osi = 0.005
            elif mu >= 0.97:
                sigma = 0.01
                osi = 0.005
            else:
                sigma = 0.03
                osi = 0.0001
            alpha, beta = get_alpha_beta(mu, sigma)  
            if DEBUG: 
                print("mu", mu, "sigma", sigma, "alpha", alpha, "beta",beta)
            a, b = solve_for_log_normal_parameters(alpha, osi)  
            c, d = solve_for_log_normal_parameters(beta, osi)
            if np.isnan(a) or np.isnan(b) or np.isnan(c) or np.isnan(d):                
                continue
            a_params.append(float(a))
            b_params.append(float(b))
            c_params.append(float(c))
            d_params.append(float(d))

        m = len(a_params)        
        #add in additional permutations in the event we will allow variants to be assigned to the same group
        all_distributions = run_model(frequency, m, num_warmup=500, num_samples=3000)
        output_dict['distributions'].append([list(x) for x in list(all_distributions)])
        output_dict['used_peaks'].append(used_peaks)
    with open(output_file, "w") as ofile:    
        json.dump(output_dict, ofile)    
    return(0)

