import os
import sys
import copy
import math
import scipy
import time
import pulp
import json
import pickle
import random
import argparse
import itertools

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy import stats
from scipy import spatial
from scipy.spatial import distance
from sklearn.mixture import GaussianMixture
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import KernelDensity
from sklearn.decomposition import PCA
from line_profiler import LineProfiler

import file_util
import generate_consensus 

from other_gmm import GMM
random.seed(10)
np.random.seed(10)


def parallel_expand_solutions(solution_space, freq_precision, error=0.03):
    kept_solutions = Parallel(n_jobs=10)(delayed(expand_solutions)(solution, freq_precision, error) for solution in solution_space)
    kept_solutions = [x for x in kept_solutions if x is not None]
    return(kept_solutions) 

def expand_solutions(solution, freq_precision, error=0.03):
    """
    Look for ways duplicates can fill gaps in the data.
    """
    solution = [round(x,freq_precision) for x in list(solution) if x > 0]
    summation = sum(solution)
    diff = 1-summation
    
    #we need something still in the ballpark of 100%, lets allow for duplicates while we enforce constraint #1
    if diff < 1-error and diff > 0:
        return(None)
        solution, include = find_duplicate(solution, error)
        if include:
            return(solution)
        else:
            return(None)
    else:
        return(solution)

def find_duplicate(solution, error):
    """
    Given the space between 1 and the solution, can duplicated peaks in the data account for this space?
    """
    diff = 1 - sum(solution)

    combination_solutions, combination_sums = generate_combinations(len(solution), solution)
    combination_sums = [round(x,5) for x in combination_sums]

    arr = np.array(combination_sums)
    idx = np.abs(arr - diff).argmin()
    duplicate = combination_sums[idx]

    #we haven't properly been able to account for the 1 - E(solution) space
    if abs(duplicate-diff) > error:
        return(solution, False)
    else:
        dups = list(combination_solutions[idx])
        if len(dups) > 3 or sum(dups) > 0.15:
            return(solution, False)
        solution.extend(dups)
        solution.sort(reverse=True)
        return(solution, True)

def parallel_eliminate_solutions(solution_space, means, error=0.02):
    accounted_points = [x for x in means if x > 0.10]
    kept_solutions = Parallel(n_jobs=25)(delayed(eliminate_solutions)(solution, accounted_points, error) for solution in solution_space)
    kept_solutions = [x for x in kept_solutions if x is not None]
    return(kept_solutions) 

def eliminate_solutions(solution, means, error=0.02):
    """
    Solutions must account for every peak in the means reported by the "preseed" GMM model within a reasonable tolerance. No, this logic isn't circular because the solution space is generated directly from the KDE while the means/all peaks value represents a GMM model output. Theoretically you COULD build your solution space from this, but the KDE allows more diversity in the solution space.
    """
    keep = True
    tmp = [x for x in solution if x < 0.01]
    solution = [x for x in solution if x >= 0.01]
    combination_solutions, combination_sums = generate_combinations(len(solution), solution)
    combination_sums = np.array(combination_sums)
    for mean in means:
        idx = (np.abs(combination_sums - mean)).argmin()
        match = combination_sums[idx]
        if abs(match - mean) > (mean * 0.03):
            keep = False
            break
    if keep:
        solution.extend(tmp)
        solution.sort(reverse=True)
        return(solution)
    else:
        return(None)

def remove_primer_binding(autoencoder_dict, problem_positions, call_ambiguity, removal_dict):
    for key, value in autoencoder_dict.items():
        new_value = []
        for variant in value:
            pos = variant[:-1]
            if pos in problem_positions:
                if pos not in call_ambiguity:
                    call_ambiguity[str(pos)] = {"reason":[], "variants":[], "population":[]}
                if 'amp_flux' not in call_ambiguity[str(pos)]["reason"]:
                    call_ambiguity[str(pos)]["reason"].append('amp_flux')
            else:
                new_value.append(variant)
        autoencoder_dict[key] = new_value
    return(autoencoder_dict, call_ambiguity)

def expanded_solution_space(original_solution_space, freq_precision):
    all_mu = []
    max_len = 1
    for x in original_solution_space:
        combination_solutions, combination_sums = generate_combinations(len(x), x)
        combination_sums = [round(x, freq_precision) for x in list(combination_sums) if x < 1]
        combinations_sums = list(np.unique(combination_sums))
        if len(combination_sums) > max_len:
            max_len = len(combination_sums)
        all_mu.append(combination_sums)

    new_X = [] 
    for x,y in zip(all_mu, original_solution_space):
        x.sort(reverse=True)
        if len(x) < max_len:
            tmp = [0.0] * (max_len-len(x))
            x.extend(tmp)
        new_X.append(x) 
        x = [str(y) for y in x]

    return(new_X)

def find_positions_exclude(solution, positions, nucs, scores, all_scores, assignments, assignment_data, complexity, ambiguity_dict, likelihood_ratio=5, threshold=0.10):
    """
    Returns
    -------
    outliers : dict
        The list of positions where we're going to call N's due to low likelihood and high linear distance.

    Under the given model, find points to exclude from the final consensus.
    """
    ###try setting linear dist threshold using nearest point ie.
    solution = [x for x in solution if x > 0.05]
    dist = []
    for s in solution:
        for ss in solution:
            if s == ss:
                continue
            dist.append(abs(s-ss))
    #threshold = min(dist)
    #print(threshold) 
    
    if complexity == "low":
        threshold = 0.05
    elif complexity == "extremely low":
        threshold = 0.10
    
    max_pos_dict = {}
    pos_count_dict = {}
    linear_dist = {}    
    largest_peak = max(list(np.squeeze(assignments)))
    best_likelihood = 0   
    for pos, score, assignment, point in zip(positions, scores, assignments, assignment_data):
        if assignment[0] == largest_peak and (complexity == "extremely low" or complexity == "singular"):
            continue
        if score > best_likelihood and score < 100:
            best_likelihood = score
    outliers = {}
    #print("best likelihood", best_likelihood)
    for pos, n, score, assignment, point, all_score in zip(positions, nucs, scores, assignments, assignment_data, all_scores):
        lin_d = abs(assignment[0]-point[0])
        ratio = score/best_likelihood
        all_score =list(all_score)
        all_score.sort(reverse=True)
        #print(ratio, 1/likelihood_ratio, "linear", lin_d, "point", point, assignment, "score", score, pos, n)
        if ratio < (1/likelihood_ratio) and lin_d >= threshold and all_score[1] == 0:
            #print("ratio")
            #print("ratio", ratio, 1/likelihood_ratio, "linear", lin_d, "point", point, assignment, "score", score, pos, n)
            #print(all_score)
            if str(pos) not in outliers:
                outliers[str(pos)] = {"reason":[], "variants":[], "population":[]}
            if 'outlier' not in outliers[str(pos)]["reason"]:
                outliers[str(pos)]["reason"].append('outlier')

        #if the likelihood is extremely low, even with a close liner distance
        elif score < 0.00001 and all_score[1] == 0:
            #print("smol score", ratio, 1/likelihood_ratio, "linear", lin_d, "point", point, assignment, "score", score, pos, n)
            if str(pos) not in outliers:
                outliers[str(pos)] = {"reason":[], "variants":[], "population":[]}
            if 'outlier' not in outliers[str(pos)]["reason"]:
                outliers[str(pos)]["reason"].append('outlier')    
    return(outliers)
 
def determine_consensus_call(assignment_data, assignments, scores, all_likelihoods, autoencoder_dict, new_nucs, new_positions, combination_sums, combination_solutions, reference_positions, primer_positions, ambiguity_dict, call_ambiguity, complexity):
    """
    Given a set of sequences, determine what we call consensus on.
    """   
    if complexity == "singular" or complexity == "extremely low":
        ratio_threshold = 5
    else:
        ratio_threshold = 5

    removal_dict = {}
    keys = []
    could_have_assigned = {}
   
    for k,v in autoencoder_dict.items():
        removal_dict[k] = []
        could_have_assigned[k] = []
        k = k.split("_")[2]     
        keys.append(float(k))
    no_call = [min(keys)]
    cluster_values = []
    
    for i in range(len(combination_sums)):
        cluster_values.append([])
    unique_positions = list(np.unique(new_positions))
    for i, (point, assignment, score, all_like) in enumerate(zip(assignment_data, assignments, scores, all_likelihoods)):
        point = point[0]
        assignment = assignment[0]
        nuc = str(new_nucs[i])
        pos = str(new_positions[i])           
        variant = pos+nuc
        location = combination_sums.index(assignment)
        combo_sol = combination_solutions[location]      
        combo_sol_copy = copy.deepcopy(combination_solutions)
        zipped = list(zip(all_like, combo_sol_copy))
        zipped.sort(reverse=True)
        all_like, combo_sol_copy = zip(*zipped)
        for cs in combo_sol_copy[0]:
            idxs = [i for i,x in enumerate(keys) if cs == x]
            names = [list(autoencoder_dict.keys())[i] for i in idxs]
            for n in names:
                if variant not in could_have_assigned[n]:
                    could_have_assigned[n].append(variant)
       
        if all_like[1] > 0: 
            ratio = all_like[0] / all_like[1]
        else:
            ratio = ratio_threshold
        print('point', point, 'variant', variant, 'assignment', assignment, 'score', score, all_like[1], combo_sol_copy[0], combo_sol_copy[1])
        dont_assign = []
        diff_assignment = []
        #we cannot concretely assign this point
        if ratio < ratio_threshold:
            print("ratio less than %s" %ratio_threshold)
            diff_assignment = [x for x in combo_sol_copy[0] if x not in combo_sol_copy[1]]
            diff_assignment.extend([x for x in combo_sol_copy[1] if x not in combo_sol_copy[0]])

            #if we can say it definitly belongs to one population at least
            for difference in diff_assignment:    
                idxs = [i for i,x in enumerate(keys) if difference == x]
                names = [list(autoencoder_dict.keys())[i] for i in idxs]
                for n in names:
                    removal_dict[n].append(variant)  
                    could_have_assigned[n].append(variant) 
                print(difference)
                print("ratio", ratio, "point", point, "assign", assignment, variant, difference, all_like[0], all_like[1], all_like[2], combo_sol_copy[0], combo_sol_copy[1])
                if str(pos) not in call_ambiguity:
                    call_ambiguity[str(pos)] = {"reason":[], "variants":[], "population":[]}
                if 'low_ratio' not in call_ambiguity[str(pos)]['reason']:
                    call_ambiguity[str(pos)]['reason'].append('low_ratio')
                if 'population' not in call_ambiguity[str(pos)]:
                    call_ambiguity[str(pos)]['population'] = []
                call_ambiguity[str(pos)]['variants'].append(variant)
                call_ambiguity[str(pos)]['population'].append(difference)
                dont_assign.append(difference)

        for cs in combo_sol_copy[0]:
            if cs in dont_assign:
                continue
            idxs = [i for i,x in enumerate(keys) if cs == x]
            names = [list(autoencoder_dict.keys())[i] for i in idxs]
            for n in names:
                if variant not in autoencoder_dict[n]:
                    autoencoder_dict[n].append(variant)
    percent_missing = {}
    #never call the smallest value, its where the noise be
    for key,value in autoencoder_dict.items():
        new_value = []
        for v in value:
            if "+" not in v and "-" not in v:
                nuc = v[:-1]
            elif "+" in v:
                nuc = v.split("+")[0]
            elif "-" in v:
                nuc = v.split("-")[0]
            if nuc in call_ambiguity.keys():
                ambig_var = call_ambiguity[nuc]['variants']
                ambig_pop = call_ambiguity[nuc]['population']
                add = True
                for av, p in zip(ambig_var, ambig_pop):
                    if float(p) == float(key.split("_")[2]) and av == v:
                        print(key, v, p)
                        add = False
                if add:
                    new_value.append(v)
            else:
                new_value.append(v)
        autoencoder_dict[key] = new_value
    for key, value in could_have_assigned.items():
        percent_missing[key] = len(value)        
       
    for key, value in removal_dict.items():
        if len(value) == 0:
            percent_missing[key] = 0
        else:
            percent_missing[key] = len(value) / percent_missing[key]
    return(removal_dict, autoencoder_dict, no_call, percent_missing, call_ambiguity)          

def generate_combinations(n, solution):
    combination_solutions = []
    combination_sums = []
    for subset in pulp.allcombinations(solution, n):
        combination_solutions.append(subset)
        combination_sums.append(sum(subset))
    return(combination_solutions, combination_sums)

def define_kde(frequencies, complexity, num, total_return_length, freq_precision, bw=0.0001):
    x = np.array(frequencies)
    x_full = x.reshape(-1, 1)
    eval_points = np.linspace(min(x), max(x), num=num)
    kde = KernelDensity(kernel='gaussian', bandwidth=bw).fit(x_full)
    a = np.exp(kde.score_samples(eval_points.reshape(-1,1)))
    peaks = a[1:-1][np.diff(np.diff(a)) < 0]
    ind = [i for i,xx in enumerate(a) if xx in peaks]
    peak = []
    peak_list = list(peaks)
   
    tmp = copy.deepcopy(peak_list)
    tmp.sort(reverse=True)

    keep_tmp = []
    keep_idx = []
    peakos = []
    for t in tmp:
        idx = peak_list.index(t)
        ev = eval_points[list(a).index(peak_list[idx])]
        if round(ev, freq_precision) != 0:
            keep_tmp.append(t)
        if round(ev, freq_precision) not in peakos:
            peakos.append(round(ev, freq_precision))
        #if len(peakos) == total_return_length:
        #    break
        print(t, round(ev,2))
    tmp = keep_tmp
    #peakos.sort()
    print(peakos)
    sys.exit(0)
    for i, xx in enumerate(eval_points):
        if i in ind:
            idx = ind.index(i)
            if peak_list[idx] in tmp:
                tmp.remove(peak_list[idx])
                val = round(xx, freq_precision)
                if val > 0 and val not in peak:
                    peak.append(val)
    return(peak)

def create_solution_space(peaks, r_min, r_max, error):
    total_value = 1
    lower_bound = total_value + error
    upper_bound = total_value - error
    overlapping_sets = []
    all_combos = pulp.allcombinations(peaks, r_max)
    
    for subset in all_combos:
        if len(subset) < r_min:
            continue
        subset = list(subset)
        total = sum(subset) #most expensive line
        if (total > lower_bound) or (total < upper_bound):
            continue
        overlapping_sets.append(subset)
    return(overlapping_sets)

def collapse_solution_space(solution_space, expanded_solution_space, freq_precision, clusters=50, error=0.03):
    print("collapsing solution space...")
    if clusters > len(expanded_solution_space):
        return(solution_space)    

    #let's now cluster the solutions to try and eliminate duplicates
    k_solutions = MiniBatchKMeans(n_clusters=clusters)
    k_solutions.fit(np.array(expanded_solution_space))
    clustered_solutions = k_solutions.cluster_centers_

    ess = np.array(expanded_solution_space)
    closest_index = distance.cdist(clustered_solutions, ess)
    am = [int(x) for x in list(np.squeeze(closest_index.argmin(axis=1)))]
    kept_solutions = [x for i,x in enumerate(solution_space) if i in am]
    return(kept_solutions)    

def parallel_train_models(solutions, training_data, freq_precision, positions, filename, pos_error):
    code = Parallel(n_jobs=20)(delayed(train_models)(solution, training_data, freq_precision, positions, filename, pos_error) for solution in solutions)
    all_saved_solutions = [x[0] for x in code]
    all_final_points = [x[1] for x in code]
    all_models = [x[2] for x in code]
    scores = [x[3] for x in code]
    assignments = [x[4] for x in code]
    all_scores = [x[5] for x in code]
    all_aic = [x[6] for x in code]
    return(all_saved_solutions, all_final_points, all_models, scores, assignments, all_scores, all_aic)

def train_models(solution, training_data, freq_precision, positions, filename, pos_error): 
    n = len(solution)
    combination_solutions, combination_sums = generate_combinations(n, solution)
    combination_sums = [round(x, freq_precision) for x in combination_sums]
    final_points = []
    for j, (tp, sol) in enumerate(zip(combination_sums, combination_solutions)):
        if j < len(solution):
            pass
        elif len(combination_solutions[j]) == 1:
            pass
        elif tp in final_points:
            continue
        final_points.append(round(tp, freq_precision))
    final_points_expand = np.expand_dims(final_points, axis=1)
    gxx = GMM(k=len(final_points), dim=1, init_mu = final_points_expand, solution=solution, fixed_means=True, filename=filename)
    gxx.init_em(training_data, positions, combination_solutions, combination_sums, pos_error)
    max_num_iters = 100
    useful_iterations = 100
    gxx.converge  = max_num_iters-1
    lower_aics = 0
    for e in range(max_num_iters):
        gxx.iteration += 1
        gxx.e_step()
        gxx.m_step()
        #we've had an error in the calculations
        if len(gxx.sigma_reset) > 0:
            useful_iterations = e
            break
        #gxx.aics.append(aic)
    if useful_iterations != 99 and max_num_iters != useful_iterations-1:
        gxx = GMM(k=len(final_points), dim=1, init_mu = final_points_expand, solution=solution, fixed_means=True, filename=filename)
        gxx.init_em(training_data, positions, combination_solutions, combination_sums, pos_error)
        for e in range(useful_iterations):
            gxx.iteration += 1
            gxx.e_step()
            gxx.m_step() 
    assignment, score, all_scores, ll, aic, linear_distance = gxx.other_score(training_data, positions)
    sll = [np.log(x) if x > 0 else 0 for x in score]
    final_points.sort(reverse=True) 
    return(solution, final_points, gxx, score, assignment, all_scores, aic)

def collapse_small_peaks(solution_space, freq_precision, r_min, r_max, collapse_value=0.03):
    """
    Iterate all solutions and collapse frequencies that occur < 0.03 into one bin, we simply don't need that level of resolution and it complicates the problem in an unnecessary way.
    """
    print("collapsing small peaks...")
    new_solution_space = []
    for solution in solution_space:
        tmp = [x for x in solution if x < collapse_value]
        binned = round(sum(tmp),freq_precision)
        solution = [round(x, freq_precision) for x in solution if x >= collapse_value]
        if binned > 0.01:
            solution.append(binned)
        solution.sort(reverse=True)
        if r_min > len(solution) or len(solution) > r_max:
            continue
        new_solution_space.append(solution)

    return(new_solution_space)    

def parallel_contract_solutions(solution_space, freq_precision, error=0.03):
    code = Parallel(n_jobs=20)(delayed(contract_solutions)(solution, freq_precision, error) for solution in solution_space)
    kept_solutions = [x[0] for x in code]
    hidden_peaks = [x[1] for x in code]
    kept_solutions = [x[0] for x in kept_solutions if x is not None and len(x) > 0]
    hidden_peaks = [x[0] for x in hidden_peaks if x is not None and len(x) > 0]
    return(kept_solutions, hidden_peaks) 

def contract_solutions(solution, freq_precision, error=0.03):
    diff = sum(solution) - 1
    if diff > error and diff > 0:
        new_solution, keep, peak = find_hidden_population(solution, error, freq_precision, diff)
        if keep:
            return(new_solution, peak)
        else:
            return(None, None)
    else:
        return([solution], None)

def find_hidden_population(solution, error, freq_precision, diff):
    """
    This assumes only one hidden population.
    """
    arr = np.array(solution)
    idx = np.abs(arr - diff).argmin()
    duplicate = solution[idx]
    hidden_peaks = []
    #we haven't properly been able to account for the 1 - E(solution) space
    if abs(duplicate-diff) > error:
        return([solution], False, hidden_peaks)
    else:
        hidden = solution[idx]
        possible_parent = [x for x in solution if x > hidden + error]
        arr = np.array(solution)
        difference = possible_parent - hidden
        new_solutions = []
        for i in range(len(possible_parent)):
            new_solutions.append(solution)
        for i,(old_peak, new_peak) in enumerate(zip(possible_parent, difference)):
            tmp = new_solutions[i]
            tmp.remove(old_peak)
            new_peak = round(new_peak, freq_precision)
            tmp.append(new_peak)
            hidden_peaks.append(new_peak)
            tmp.sort(reverse=True)
            new_solutions[i] = tmp
        return(new_solutions, True, hidden_peaks)

def find_masked_solution(solution_space, freq_precision):
    """
    Given the possiblity that we have exactly one thing that's entirely masked, add a solution for that.
    """ 
    new_solution_space = []
    for solution in solution_space:
        total = sum(solution)
        if total <= 0.97:
            diff = round(1 - total, freq_precision)
            solution.sort(reverse=True)
            tmp = copy.deepcopy(solution)
            tmp.append(diff)
            tmp.sort(reverse=True)
            new_solution_space.append(tmp)
        new_solution_space.append(solution)    
    return(new_solution_space)
             
def create_complexity_estimate(total_mut_pos, reference="", evolutionary_rate=0.001):
    """
    Estimate the complexity of the sample given a reference and an evolutionary rate.
    """ 
    length_total_mut = len([x for x in total_mut_pos if x > 0.1])
    scaled_complexity_estimate = length_total_mut/(29903*3)
    complexity = "high"
    if scaled_complexity_estimate < 0.0002:
        complexity = "singular"
    elif scaled_complexity_estimate < 0.0009:
        complexity = "extremely low"
    elif scaled_complexity_estimate < 0.02:
        complexity = "low"
    print("complexity estimate", scaled_complexity_estimate, complexity)
    return(complexity, scaled_complexity_estimate)

def run_model(output_dir, output_name, bam_file, bed_file, reference_file, freyja_file=None):
    """
    parser = argparse.ArgumentParser(description='Model variant frequencies in bam file.')
    parser.add_argument('-o','--output_name', help='Output directory.', required=True)
    parser.add_argument('-p','--prefix', help='File prefix.', required=True)
    args = vars(parser.parse_args())         
    print(args)
    sys.exit(0)
    """
    freq_lower_bound = 0.0001
    solutions_to_train = 1000
    model_location = os.path.join(output_dir, output_name+"_model.pkl")
    problem_primers = None
    remove_pos_dict = None
    variants_json = os.path.join(output_dir, output_name+"_variants.txt") 
    if os.path.isfile(variants_json):
        with open(variants_json, "r") as rfile:
            primer_dict = json.load(rfile)
            found_insertion = False
            for k, v in primer_dict['variants'].items():
                nuc_count = list(v[0].keys())
                insert = [x for x in nuc_count if "+" in x]
                if len(insert) > 0:
                    found_insertion = True
                    break
            #if found_insertion:
            bam_file = None    
    #print(primer_dict['variants']['28271'])
    #sys.exti(0)
    if bam_file is not None:
        """
        lp = LineProfiler()
        lp_wrapper = lp(file_util.parse_bam_depth_per_position)   # add additional function to profile
        lp_wrapper(bam_file, bed_file, variants_json)
        lp.print_stats()
        return(1)
        """
        primer_dict = file_util.parse_bam_depth_per_position(bam_file, bed_file, variants_json)
        return(1)
    if freyja_file is not None:
        gt_centers, gt_lineages = file_util.parse_freyja_file(freyja_file)
        gt_mut_dict = file_util.parse_usher_barcode(gt_lineages)
        print(gt_centers)
        print(gt_lineages)
    reference_sequence = file_util.parse_reference_sequence(reference_file)
    primer_positions = file_util.parse_bed_file(bed_file)
    frequency, nucs, positions, depth, low_depth_positions, reference_positions, ambiguity_dict, \
        total_mutated_pos, training_removed = file_util.parse_variants(primer_dict, primer_positions, reference_sequence)
    complexity, complexity_estimate = create_complexity_estimate(total_mutated_pos)
    collapse_value = 0.05

    #this may be expanded in future iterations for more complex samples
    if complexity == "low":
        r_min = 4
        r_max = 6
        error = 0.10
        freq_precision = 2
        total_return_length = 31
        if complexity_estimate > 0.0008 and complexity_estimate <= 0.001:
            total_return_length = 35
        elif complexity_estimate > 0.001:
            total_return_length = 41
            r_min = 5
            r_max = 7
        freq_upper_bound = 0.96
        pos_error = 0.05 
        training_lower_bound = 0.02
        solutions_to_train = 1000
    elif complexity == "extremely low":
        r_min = 3
        r_max = 4
        error = 0.05
        freq_precision = 2
        freq_upper_bound = 0.98
        total_return_length = 16
        pos_error = 0.10
        training_lower_bound = 0.02
        collapse_value = 0.01
    elif complexity == "singular":
        r_min = 2
        r_max = 3
        error = 0.05
        freq_precision = 2
        freq_upper_bound = 0.98
        total_return_length = 16
        pos_error = 0.10
        training_lower_bound = 0.01
    else:
        return(1)
    
    joint_peak, joint_nuc, joint_dict = file_util.parse_physcial_linkage(frequency, nucs, positions, depth, primer_dict, r_max)
    print("removed from the training set", training_removed)
   
    """
    zipped = list(zip(frequency, positions, nucs, depth))
    zipped.sort()
    frequency, positions, nucs, depth = zip(*zipped) 
    for f,n,p,d in zip(frequency, nucs, positions, depth):
        r_check = str(n) + "_" + str(p)
        #if r_check in reference_positions:
        #    continue
        if f > 0.03 and f < freq_upper_bound:
            var = str(p) + str(n)
            print(round(f,3), n, p)
    sys.exit(0)
    """ 
 
    joint_peak = [round(x, freq_precision) for x in joint_peak] 
    jd = {}
    for k,v in joint_dict.items():
        jd[round(k,freq_precision)] = v
    joint_dict = jd
    print("joint dict", joint_dict)
    #here we filter out really low data points, and data points we assume to be "universal mutations"
    new_frequencies = [round(x, freq_precision) for x in frequency if x > freq_lower_bound and x < freq_upper_bound]
    new_positions = [p for p,f in zip(positions, frequency) if f > freq_lower_bound and f < freq_upper_bound] 
    if len(new_frequencies) == 0:
        return(1)
    #create the dataset that we're going to actually use for training, filtering out more low level point
    training_positions = [p for p,f in zip(new_positions, new_frequencies) if f > training_lower_bound]
    training_data = np.array([x for x in new_frequencies if x > training_lower_bound])
     
    #lets look at the removed points and the new_frequencies
    kde_data = []
    arr = np.array(joint_peak)
    for f in new_frequencies:
        diff = np.abs(arr-f)
        thresh = f * 0.05
        diff = [x for x in diff if x < thresh]
        if len(diff) == 0:
            kde_data.append(f)
      
    num = 3000
    print(kde_data)
    refined_kde_peaks = define_kde(kde_data, complexity, num, total_return_length, freq_precision)
    refined_kde_peaks.sort(reverse=True)
    print(refined_kde_peaks)
    sys.exit(0)    
    #if we have no data
    if len(refined_kde_peaks) == 0 or len(new_frequencies) == 0:
        return(1)

    refined_kde_peaks.sort()
    print("complexity", complexity)

    refined_kde_peaks = [round(x, freq_precision) for x in refined_kde_peaks if x > 0]
    print("refined kde", refined_kde_peaks, len(refined_kde_peaks))
    print("joint peak", joint_peak)
    if len(refined_kde_peaks) < r_max:
        r_max = len(refined_kde_peaks)

    solution_space = create_solution_space(refined_kde_peaks, r_min, r_max, error=error)
    print(refined_kde_peaks)
    """
    #here we assume that their might be exactly 1 totally hidden solution that accounts for the difference
    solution_space = find_masked_solution(solution_space, freq_precision)
    print("masked solution space length", len(solution_space))

    #constraint #1 E(solution) ~= 1
    solution_space, hidden_peaks = parallel_contract_solutions(solution_space, freq_precision, error=0.40)
    #constraint #1 E(solution) ~= 1
    solution_space = parallel_expand_solutions(solution_space, freq_precision, error=0.40)    
    print("length of expansive solution space", len(solution_space))
    """
    print("solution space length", len(solution_space))
    original_solution_space = copy.deepcopy(solution_space)
    original_solution_space = collapse_small_peaks(original_solution_space, freq_precision, r_min, r_max, collapse_value = 0.05)
    #here we expand the solutions to encompass all possible peaks
    original_space_expand = expanded_solution_space(original_solution_space, freq_precision)
    if len(solution_space) > solutions_to_train:
        solution_space = collapse_solution_space(solution_space, original_space_expand, freq_precision, clusters=solutions_to_train) 
        print("length of collapsed solution space", len(solution_space))
   
    #prior to acutal training, collapse down tiny peaks to make training more reasonable
    solution_space = collapse_small_peaks(solution_space, freq_precision, r_min,  r_max, collapse_value = collapse_value)
    training_data = [round(x, freq_precision) for x in training_data]

    #deduplicate the solution space
    keep = []
    for sol in solution_space:
        if sol not in keep and 1-error < sum(sol) < 1+error:
            keep.append(sol)
    solution_space = keep
    print("sol length after deduplication", len(solution_space))    
    if len(solution_space) == 0:
        print("No solutions found:", output_name)
        return(1)
    """
    for sol in solution_space:
        sol.sort(reverse=True)
        if len(sol) == 4 and sol[0] == 0.83 and sol[1] == 0.1 and sol[2] == 0.08:
            print(sol)
    sys.exit(0)    
    """
    new_solution_space = solution_space
    #for sol in new_solution_space:
    #    print(sol)
    #sys.exit(0)
    print("solution space contains %s solutions..." %len(new_solution_space))
    print("training models...")
    #filter out variants that aren't being used in training, look for universal mutations 
    training_data = np.expand_dims(training_data, axis=1)
    new_positions = [y for x,y in zip(frequency, positions) if float(x) > training_lower_bound and x < freq_upper_bound]
    new_nucs = [y for x,y in zip(frequency, nucs) if float(x) > training_lower_bound and float(x) < freq_upper_bound]
    original_variants = [str(x)+y for x,y in zip(new_positions, new_nucs)]

    #train the models
    all_saved_solutions, all_final_points, all_models, \
        all_scores, assignments, all_likelihoods, all_aic \
        = parallel_train_models(new_solution_space, training_data, freq_precision, training_positions, output_name, pos_error)

    #get the best model
    sorted_scores = copy.deepcopy(all_aic)
    original_index = list(range(0, len(all_aic)))
    zipped = list(zip(sorted_scores, original_index))
    zipped.sort()
    sorted_scores, original_index = zip(*zipped)    
    best_score = 3000
    flat_training = list(np.squeeze(training_data))
    found_model = False

    seen_length = []    
    seen_models = []
    for i, (loc, aic) in enumerate(zip(original_index, sorted_scores)):
        mm = all_models[loc]
        a = assignments[loc]
        flat_assignments = list(np.squeeze(a))
        als = all_likelihoods[loc]
        scores = all_scores[loc]
        solution = all_saved_solutions[loc]
        ambiguity  = find_positions_exclude(solution, new_positions, new_nucs, scores, als, a, training_data, complexity, ambiguity_dict)
        if i < 20:
            print(i, "aic", aic, all_saved_solutions[loc], len(all_saved_solutions[loc]))
            #print(list(np.squeeze(mm.mu)))
            #for td, score, assign in zip(training_data, scores, a):
            #    print("point", td[0], "scores", score, "assign", assign)
        combination_solutions = list(mm.mu_combo)
        combination_sums = list(np.squeeze(mm.mu))
 
        combination_sums = [round(x,3) for x in combination_sums]
        useful_solution = True
        seen_sol = [x for x in solution if x in flat_assignments]
        #if len(seen_sol) != len(solution):
        #    useful_solution = False
        if len(joint_peak) > 0:
            for aa, td, all_sc, pos in zip(a, training_data, als, new_positions):
                if str(pos) in ambiguity:
                    continue
                td = td[0]
                if td in joint_peak:
                    idx = combination_sums.index(aa[0])
                    constituents = [round(x, freq_precision) for x in joint_dict[td][0]]
                    for c in constituents:
                        idxx = flat_training.index(c)
                        ca = a[idxx][0]
                        idxx = combination_sums.index(ca)
                        must_have = combination_solutions[idxx]
                        overlap = [x for x in must_have if x not in combination_solutions[idx]]                    
                        if len(overlap) > 0:                            
                            useful_solution = False
        if useful_solution is True and len(solution) not in seen_length:
            seen_length.append(len(solution))
            seen_models.append(mm)

        if useful_solution is True and aic < best_score:
            #sns.lineplot(x=list(range(0,len(mm.aics))), y=list(mm.aics))
            #plt.savefig("./figures/converge.pdf")
            #for td, score, assign in zip(training_data, scores, a):
            #    print("point", td[0], "scores", score, "assign", assign)
            print("used")
            found_model = True
            best_score = aic

    if found_model is False:
        return(1)
    loc_best_model = all_aic.index(best_score)
    gx = all_models[loc_best_model]      
    pickle.dump(gx, open(model_location, 'wb')) 
    return(0)

def call_consensus(output_dir, output_name, model_location, reference_file, bed_file):
    gx = pickle.load(open(model_location, 'rb')) 
    solution = gx.solution
    training_lower_bound = 0.02
    freq_upper_bound = 0.98
    freq_precision = 3
    
    reference_sequence = file_util.parse_reference_sequence(reference_file)
    primer_positions = file_util.parse_bed_file(bed_file)
    text_file = os.path.join(output_dir, output_name+"_model_results.txt")
    variants_json = os.path.join(output_dir, output_name+"_variants.txt")    
    if os.path.isfile(variants_json):
        with open(variants_json, "r") as rfile:
            primer_dict = json.load(rfile)
    frequency, nucs, positions, depth, low_depth_positions, reference_positions, ambiguity_dict, \
        total_mutated_pos, training_removed = file_util.parse_variants(primer_dict, primer_positions, reference_sequence, depth_cutoff=9)
    #get complexity value
    complexity, complexity_estimate = create_complexity_estimate(total_mutated_pos)

    #create a new dataset for actually assigning out variants
    #this needs to be fixed
    new_frequency, new_nucs, new_positions, add_low_depth_positions, universal_mutations, ambiguity_dict, primer_binding_issue = file_util.parse_additional_var(primer_dict, primer_positions, reference_sequence, ambiguity_dict, training_lower_bound, freq_upper_bound, depth_cutoff=10)
    #look for amplicons where we have both a mutation in the primer region and amplicon flux
    problem_positions = file_util.check_primer_binding(ambiguity_dict, primer_positions, primer_binding_issue)
     
    new_variants = [str(p) + str(n) for p,n,f in zip(new_positions, new_nucs, new_frequency) if f > training_lower_bound and f < freq_upper_bound] 
    new_positions = [str(p) for p,f in zip(new_positions, new_frequency) if f > training_lower_bound and f < freq_upper_bound] 
    new_nucs = [str(n) for n,f in zip(new_nucs, new_frequency) if f > training_lower_bound and f < freq_upper_bound] 
    new_frequency = [round(x,freq_precision) for x in new_frequency if x > training_lower_bound and x < freq_upper_bound]
    assignment_data = np.array(new_frequency).reshape(-1,1)    
    assignments, scores, all_likelihoods, ll, aic, linear_distance = gx.other_score(assignment_data, new_positions)
    combination_solutions = list(gx.mu_combo)
    combination_sums = list(gx.mu)
    print("best solution", solution)
    autoencoder_dict = {}
    print("%s universal mutations found..." %len(universal_mutations))

    call_ambiguity = find_positions_exclude(solution, copy.deepcopy(new_positions), copy.deepcopy(new_nucs), scores, all_likelihoods, copy.deepcopy(assignments), copy.deepcopy(assignment_data), complexity, ambiguity_dict)
    #handle the case where we have two peaks in the solution that are identical
    for i,s in enumerate(solution):
        s = "peak_%s_%s" %(i, s)
        autoencoder_dict[s] = copy.deepcopy(universal_mutations)    
    removal_dict, autoencoder_dict, no_call, percent_not_used, call_ambiguity = determine_consensus_call(assignment_data, assignments, scores, all_likelihoods, autoencoder_dict, new_nucs, new_positions, combination_sums, combination_solutions, reference_positions, primer_positions, ambiguity_dict, call_ambiguity, complexity)
    autoencoder_dict, call_ambiguity = remove_primer_binding(autoencoder_dict, problem_positions, call_ambiguity, removal_dict)       
    #print(problem_positions) 
    #print(call_ambiguity)
    for pos in add_low_depth_positions:
        if str(pos) not in call_ambiguity:
            call_ambiguity[str(pos)] = {"reason":[], "variants":[], "population":[]}
        call_ambiguity[str(pos)]['reason'].append("low_depth")    
    no_call.extend([x for x in solution if x <= 0.05])
    no_call = list(np.unique(no_call))
    
    print(primer_dict['variants']['28271'])
    #print(problem_positions)
    #sys.exit(0) 
    """ 
    list_check = [14788, 17883]
    for l in list_check:
        i = str(l+1)
        print(i)
        print(primer_dict['variants'][i])
        print(ambiguity_dict[i])
        print(call_ambiguity[i])
    sys.exit(0)
    """
    
    tmp_dict = {"autoencoder_dict":autoencoder_dict, "ambiguity_dict":ambiguity_dict, "assignment_data":new_frequency, "no_call":no_call, "removal_dict":removal_dict, "complexity":complexity, "total_mutated_pos":total_mutated_pos, "call_ambiguity":call_ambiguity, "percent_not_used":percent_not_used}
    with open(text_file, "w") as bfile:
        bfile.write(json.dumps(tmp_dict))
        bfile.write("\n")    

if __name__ == "__main__":
    run_model()
