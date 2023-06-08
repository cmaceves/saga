"""
Functionality for running the model.
"""
import os
import sys
import copy
import math
import scipy
import time
import json
import pickle
import random
import itertools

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy import stats
from scipy import spatial
from joblib import Parallel, delayed
from sklearn.mixture import GaussianMixture
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import KernelDensity

from line_profiler import LineProfiler

import file_util
import math_util

from other_gmm import GMM
random.seed(10)
np.random.seed(10)

DEBUG=False   


def determine_consensus_call(assignment_data, assignments, scores, all_likelihoods, autoencoder_dict, new_nucs, new_positions,combination_sums, combination_solutions, reference_positions):
    """
    Given a set of sequences, determine what we call consensus on.
    """
    removal_dict = {}
    for k,v in autoencoder_dict.items():
        removal_dict[k] = []
    for i, (point, assignment, score, all_like) in enumerate(zip(assignment_data, assignments, scores, all_likelihoods)):
        point = point[0]
        assignment = assignment[0]
        nuc = str(new_nucs[i])
        pos = str(new_positions[i])
        variant = pos+nuc

        if pos+"_"+nuc in reference_positions:
            continue
        location = combination_sums.index(assignment)
        combo_sol = combination_solutions[location]      
        all_like = [round(x,3) for x in all_like]
        combo_sol_copy = copy.deepcopy(combination_solutions)
        zipped = list(zip(all_like, combo_sol_copy))
        zipped.sort(reverse=True)
        all_like, combo_sol_copy = zip(*zipped)
        
        if all_like[1] == 0:
            continue
        ratio = all_like[0] / all_like[1]

        #we cannot concretely assign this point
        if ratio < 5:
            diff_assignment = [x for x in combo_sol_copy[0] if x not in combo_sol_copy[1]]  
            #if we can say it definitly belongs to one population at least
            for difference in diff_assignment:            
                removal_dict[difference].append(variant)   
            continue
        
        for cs in combo_sol:
            if variant not in autoencoder_dict[cs]:
                autoencoder_dict[cs].append(variant)
    no_call = []
    for k,v in removal_dict.items():
        if len(v) > 0:
            no_call.append(k)
    return(removal_dict, autoencoder_dict, no_call)          
            
def generate_combinations(n, solution):

    combination_solutions = []
    combination_sums = []
    highest_val_over_3_percent = solution[0]

    for L in range(n + 1):
        for subset in itertools.combinations(solution, L):
            subset = list(subset)
            if len(subset) < 1:
                continue
            combination_solutions.append(subset)
            combination_sums.append(sum(subset))

    return(combination_solutions, combination_sums)

def define_kde(frequencies, complexity, num, bw=0.0001, round_decimal=3):
    x = np.array(frequencies)
    x_full = x.reshape(-1, 1)
    eval_points = np.linspace(min(x), max(x), num=num)
    kde = KernelDensity(kernel='gaussian', bandwidth=bw).fit(x_full)
    #sys.exit(0)
    a = np.exp(kde.score_samples(eval_points.reshape(-1,1)))
    peaks = a[1:-1][np.diff(np.diff(a)) < 0]
    ind = [i for i,xx in enumerate(a) if xx in peaks]
    peak = []
    peak_list = list(peaks)
    #print(peak_list)
    #print(ind)
    num = 50
    tmp = copy.deepcopy(peak_list)
    tmp.sort(reverse=True)
    keep_tmp = []
    for t in tmp:
        idx = peak_list.index(t)
        ev = eval_points[list(a).index(peak_list[idx])]
        if round(ev, round_decimal) != 0:
            keep_tmp.append(t)
        if len(keep_tmp) == num:
            break
    tmp = keep_tmp
    for i, xx in enumerate(eval_points):
        if i in ind:
            idx = ind.index(i)
            if peak_list[idx] in tmp:
                tmp.remove(peak_list[idx])
                peak.append(round(xx, round_decimal))
    #print(peak, len(peak))
    #sys.exit(0)
    return(peak)

def create_solution_space(peaks, r, lower_subtract=0.40):
    """
    Parameters
    ----------
    peaks : list
        The values from which a solution may be selected.
    r : int
        The number of components in the solutions.
    total_value : int
        The value to which the solution must sum, within a certain tolerance.

    Returns
    -------
    overlap_sets : list
        The expansive solution space.

    Create a solution set using nCr things where n = kde_peaks. Return only those solutions that fall within reasonable tolerance of 1, allowing room for duplicates accounting for up to 25% of the composition.
    """
    print("creating solution with %s things..." %r)
    total_value = 1
    lower_bound = total_value + lower_subtract
    upper_bound = total_value - lower_subtract
    overlapping_sets = []
    for subset in itertools.combinations(peaks, r):
        subset = list(subset)
        total = sum(subset) #most expensive line
        if (total > lower_bound) or (total < upper_bound):
            continue
        overlapping_sets.append(subset)
    return(overlapping_sets)

def collapse_solution_space(solution_space, freq_precision, clusters=50, error=0.03):
    print("collapsing solution space...")
    new_solution_space = []
    kept_solutions = []
    longest_solution = 0

    #find the longest solution in the set
    for solution in solution_space:
        if len(solution) > longest_solution:
            longest_solution = len(solution)
        solution.sort(reverse=True)
        new_solution_space.append(solution)

    #pad the data prior to clustering
    padded_data = []
    for solution in new_solution_space:
        if len(solution) < longest_solution:
            zeros = longest_solution - len(solution)
            solution.extend([0.0] * zeros)
        padded_data.append(solution)

    padded_data = np.array(padded_data)

    #make sure the clusters passed doesn't exceed the amount of data
    if clusters > padded_data.shape[0]:
        clusters = padded_data.shape[0]    

    #let's now cluster the solutions to try and eliminate duplicates
    k_solutions = MiniBatchKMeans(n_clusters=clusters)
    k_solutions.fit(padded_data)
    clustered_solutions = k_solutions.cluster_centers_
    
    for solution in clustered_solutions:
        solution = list(solution)
        solution = [round(x, freq_precision) for x in solution]
        solution = [x for x in solution if x > 0]
        solution.sort(reverse=True)
        summation = sum(solution)
        kept_solutions.append(solution)
    return(kept_solutions)    

def parallel_expand_solutions(solution_space, freq_precision, error=0.03):
    kept_solutions = Parallel(n_jobs=25)(delayed(expand_solutions)(solution, freq_precision, error) for solution in solution_space)
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


def parallel_train_models(solutions, training_data, freq_precision, default_sigma, positions):
    code = Parallel(n_jobs=10)(delayed(train_models)(solution, training_data, freq_precision, default_sigma, positions) for solution in solutions)
    all_saved_solutions = [x[0] for x in code]
    all_final_points = [x[1] for x in code]
    all_models = [x[2] for x in code]
    maxmimum_likelihood_points = [x[3] for x in code]
    scores = [x[4] for x in code]
    assignments = [x[5] for x in code]
    all_scores = [x[6] for x in code]
    all_aic = [x[7] for x in code]
    return(all_saved_solutions, all_final_points, all_models, maxmimum_likelihood_points, scores, assignments, all_scores, all_aic)

def train_models(solution, training_data, freq_precision, default_sigma, positions): 
    tmp_solution = [x for x in solution if x > 0.03]
    other_point = round(sum([x for x in solution if x <= 0.03]), freq_precision)
    if other_point > 0.01:
        tmp_solution.append(other_point)
    solution = tmp_solution
    n = len(solution)
    combination_solutions, combination_sums = generate_combinations(n, solution)
    combination_sums = [round(x, freq_precision) for x in combination_sums]
    final_points = []
    for j, (tp, sol) in enumerate(zip(combination_sums, combination_solutions)):
        if j < len(solution):
            pass
        elif len(combination_solutions[j]) == 1:
            pass
        elif tp >= 1.0:
            continue
        elif tp in final_points:
            continue
        final_points.append(round(tp, freq_precision))
    final_points_expand = np.expand_dims(final_points, axis=1)
    gxx = GMM(k=len(final_points), dim=1, init_mu = final_points_expand, solution=solution, default_sigma=default_sigma, fixed_means=True)
    gxx.init_em(training_data, positions, combination_solutions, combination_sums)
    max_num_iters = 100
    useful_iterations = 100
    log_likelihood = []
    for e in range(max_num_iters):
        gxx.iteration += 1
        gxx.e_step()
        gxx.m_step()
        if len(gxx.sigma_reset) > 0:
            useful_iterations = e
            break
   
    if max_num_iters != useful_iterations-1:
        gxx = GMM(k=len(final_points), dim=1, init_mu = final_points_expand, solution=solution, default_sigma=default_sigma, fixed_means=True)
        gxx.init_em(training_data, positions, combination_solutions, combination_sums)
        log_likelihood = []
        for e in range(useful_iterations):
            gxx.iteration += 1
            gxx.e_step()
            gxx.m_step() 
    
    assignment, score, all_scores, ll = gxx.other_score(training_data, positions)
    sll = [np.log(x) for x in score]
    aic = (2 * len(final_points)) - (2 * sum(sll))
    final_points.sort(reverse=True)
    return(solution, final_points, gxx, sum(sll), sll, assignment, all_scores, aic)

def collapse_small_peaks(solution_space, freq_precision, collapse_value=0.03):
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
        new_solution_space.append(solution)

    return(new_solution_space)    

def determine_r_values(solution_space, r_clusters):
    """
    Given a solution space, look at the prevelance of solutions of each length r to determine rmin and rmax.
    """
    r_count = [0.0] * len(r_clusters)
    for solution in solution_space:
        i = len(solution)
        if i not in r_clusters:
            continue
        idx = r_clusters.index(i)
        r_count[idx] += 1

    total_count = sum(r_count)
    keep_r = []
    idx = r_count.index(max(r_count))
     
    #in order to be considered, a r value must account for at least 20% of the solution space
    for a,b in zip(r_clusters, r_count):
        print(a, b)
        if total_count == 0:
            continue
        if b/total_count > 0.20:
            keep_r.append(a)
    if len(keep_r) == 0:
        for a,b in zip(r_clusters, r_count):
            if total_count == 0:
                continue
            if b/total_count > 0.15:
                keep_r.append(a)
    #keep_r = [r_clusters[idx]]
    return(keep_r)

def parallel_contract_solutions(solution_space, freq_precision, error=0.03):
    code = Parallel(n_jobs=25)(delayed(contract_solutions)(solution, freq_precision, error) for solution in solution_space)
    kept_solutions = [x[0] for x in code]
    hidden_peaks = [x[1] for x in code]
    kept_solutions = [x[0] for x in kept_solutions if x is not None and len(x) > 0]
    hidden_peaks = [x[0] for x in hidden_peaks if x is not None and len(x) > 0]
    return(kept_solutions, hidden_peaks) 

def contract_solutions(solution, freq_precision, error=0.03):
    diff = sum(solution) - 1
    if diff > error and diff > 0:
        new_solution, keep, peak = find_hidden_population(solution, error, freq_precision)
        if keep:
            return(new_solution, peak)
        else:
            return(None, None)
    else:
        return([solution], None)

def find_hidden_population(solution, error, freq_precision):
    """
    This assumes only one hidden population.
    """
    diff = sum(solution) - 1
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

def round_solution_space(solution_space, freq_precision):
    new_solution_space = []
    for solution in solution_space:
        solution = [round(x, freq_precision) for x in solution]
        new_solution_space.append(solution)
    return(new_solution_space)

def define_useful_kde_peaks(frequencies, kde_reshape, complexity, freq_precision=3, genome_length=29903):
    print("finding useful kde peaks...")
    frequencies.sort()

    if complexity == "high":
        total_peaks = 20
    elif complexity == "low":
        total_peaks = 20
    else:
        total_peaks = 20
    n_components = [50, 45, 40, 35, 30, 25, 20, 15, 10, 5]
    n = len(kde_reshape)    
    alot_peaks = []
    all_peaks = []
    keep_peaks = []
    refined_kde_peaks = list(np.squeeze(kde_reshape))
    ranges = [[0, 10], [10, 20], [20, 30], [30,40], [40,50], [50,60], [60,70], [70,80], [80, 90], [90, 100]]
    for r in ranges:
        count = 0
        for peak in refined_kde_peaks:
            if r[0] <= peak*100 < r[1]:
                count +=1
        percent = count/n
        alot_peaks.append(percent)        
        kp = round(total_peaks * percent)
        keep_peaks.append(kp)
        #print(r, round(percent,2), count, kp, percent)
    for r, kp in zip(ranges, keep_peaks):
        if kp == 0:
            continue
        for n_comp in n_components:
            if n_comp >= len(kde_reshape):
                continue
            cluster_model = GaussianMixture(n_components=n_comp)
            cluster_model.fit(kde_reshape)
            small_peaks = list(np.squeeze(np.array(cluster_model.means_)))
            small_peaks.sort(reverse=True)
            small_peaks = [round(x, freq_precision) for x in small_peaks]
            small_peaks = [x for x in small_peaks if x >= r[0]/100 and x < r[1]/100]
            #print(r, small_peaks, n_comp, kp)
            if len(small_peaks) <= kp and n_comp != min(n_components):
                all_peaks.extend(small_peaks) 
                break
            elif n_comp == min(n_components):
                if len(small_peaks) == 0:
                    small_peaks = prev
                all_peaks.extend(small_peaks)
                break
            prev = copy.deepcopy(small_peaks)
        pf = [round(x,3) for x in small_peaks]
        pf.sort()  
 
    all_peaks = [round(x, freq_precision) for x in all_peaks]
    #all_peaks = [x for x in list(np.unique(all_peaks)) if x > 0]
    all_peaks.sort()
    return(all_peaks)
               
def create_complexity_estimate(total_mut_pos, ambiguity_dict):
    """
    """
    length_total_mut = len([x for x in total_mut_pos if x > 0.001])
    length_amb_mut = len(ambiguity_dict)
    scaled_complexity_estimate = length_total_mut/(29903*3)
    print("ambiguity length", length_amb_mut)
    print("scaled_complexity_estimate", scaled_complexity_estimate)
    if scaled_complexity_estimate < 0.20:
        complexity = "extremely low"
    elif scaled_complexity_estimate < 0.375:
        complexity = "low"
    else:
        complexity = "high"
    return(complexity)

def run_model(output_dir, output_name, bam_file, bed_file, reference_file, freyja_file=None):
    freq_lower_bound = 0.0001
    freq_upper_bound = 0.98
    training_lower_bound = 0.03
    freq_precision = 3
    r_value_maxima = 31
    percent_solution_space = 0.01
    lower_subtract = 0.30
    default_sigma = 1
    r_min = 3
    r_max = 14
    solutions_to_train = 3000
    text_file = os.path.join(output_dir, output_name+"_model_results.txt")
    model_location = os.path.join(output_dir, output_name+"_model.pkl")
    problem_primers = None
    remove_pos_dict = None
    
    variants_json = os.path.join(output_dir, output_name+"_variants.txt")
     
    if os.path.isfile(variants_json):
        with open(variants_json, "r") as rfile:
            primer_dict = json.load(rfile)
            bam_file = None
    if bam_file is not None:
        primer_dict = file_util.parse_bam_depth_per_position(bam_file, bed_file, variants_json)
        return(1)
    
    freyja_file = None
    if freyja_file is not None:
        gt_centers, gt_lineages = file_util.parse_freyja_file(freyja_file)
        gt_mut_dict = file_util.parse_usher_barcode(gt_lineages)
        print(gt_centers)
        print(gt_lineages)
    reference_sequence = file_util.parse_reference_sequence(reference_file)
    primer_positions = file_util.parse_bed_file(bed_file)
    frequency, nucs, positions, depth, low_depth_positions, reference_positions, ambiguity_dict, \
        total_mutated_pos, training_removed, joint_peak, joint_nuc, joint_dict  = file_util.parse_variants(primer_dict, primer_positions, reference_sequence)
    
    """
    zipped = list(zip(positions, frequency, nucs, depth))
    zipped.sort()
    positions, frequency, nucs, depth = zip(*zipped) 
    print(len(frequency)) 
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
    print(joint_dict)
    #sys.exit(0)
    print("total mutated pos", len(total_mutated_pos))
    #here we filter out really low data points, and data points we assume to be "universal mutations"
    new_frequencies = [round(x, freq_precision) for x in frequency if x > freq_lower_bound and x < freq_upper_bound]
    
    if len(new_frequencies) == 0:
        return(1)

    complexity = create_complexity_estimate(total_mutated_pos, ambiguity_dict)
    if complexity != "extremely low":
        return(1)
    
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
    refined_kde_peaks = define_kde(kde_data, complexity, num)
    refined_kde_peaks.sort(reverse=True)
    #if we have no data
    if len(refined_kde_peaks) == 0 or len(new_frequencies) == 0:
        return(1)

    tmp = sum([x for x in refined_kde_peaks if x < 0.01])
    refined_kde_peaks = [x for x in refined_kde_peaks if x >= 0.01]
    refined_kde_peaks.append(tmp)
    refined_kde_peaks.sort()
    print("complexity", complexity)

    #this may be expanded in future iterations for more complex samples
    if complexity == "low" or complexity == "extremely low":
        r_max = 7
        error = 0.05
    else:
        return(1)

    refined_kde_peaks = [round(x, freq_precision) for x in refined_kde_peaks if x > 0]
    print("refined kde", refined_kde_peaks, len(refined_kde_peaks))
    print("joint peak", joint_peak)

    #create the dataset that we're going to actually use for training, filtering out more low level point
    training_data = np.array([x for x in new_frequencies if x > training_lower_bound])
    if len(refined_kde_peaks) < r_max:
        r_max = len(refined_kde_peaks)
    r_clusters = list(np.arange(r_min, r_max))
    #define an expansive solution space from r_min, r_max where E(solution) > 0.65
    solution_space = []    
    for r in r_clusters:
        r_solution = create_solution_space(refined_kde_peaks, r, lower_subtract=lower_subtract)
        solution_space.extend(r_solution)

    print("original solution space length", len(solution_space))
    #here we assume that their might be exactly 1 totally hidden solution that accounts for the difference
    solution_space = find_masked_solution(solution_space, freq_precision)
    print("masked solution space length", len(solution_space))
    #constraint #1 E(solution) ~= 1
    solution_space, hidden_peaks = parallel_contract_solutions(solution_space, freq_precision, error=error)
    #constraint #1 E(solution) ~= 1
    solution_space = parallel_expand_solutions(solution_space, freq_precision, error=error)    
    print("length of expansive solution space", len(solution_space))
    #constraint #2 all means from original GMM must be accounted for
    if complexity != "low" and complexity != "extremely low":
        solution_space = parallel_eliminate_solutions(solution_space, refined_kde_peaks) 

    if len(solution_space) > solutions_to_train:
        solution_space = collapse_solution_space(solution_space, freq_precision, clusters=solutions_to_train) 
        print("length of collapsed solution space", len(solution_space))
    
    #prior to acutal training, collapse down tiny peaks to make training more reasonable
    solution_space = collapse_small_peaks(solution_space, freq_precision, collapse_value = 0.05)
    training_data = [round(x, freq_precision) for x in training_data]
    #deduplicate the solution space
    keep = []
    for sol in solution_space:
        if sol not in keep and len(sol) > 2:
            keep.append(sol)
    solution_space = keep

    print("sol length after deduplication", len(solution_space))    
    total = 0
    points = 0
    for sol in solution_space:
        points += 1
        total += len(sol)
    if points == 0:
        print("MESSED UP POINTS", output_name)
        return(1)
    r_average = total/points
    print("average r", r_average)
    new_solution_space = solution_space

    if DEBUG:    
        check = gt_mut_dict[gt_lineages[0]]
        check2 = gt_mut_dict[gt_lineages[1]]
        check3 = gt_mut_dict[gt_lineages[2]]
     
    all_models = []
    all_model_log_likelihood = []
    all_final_points = []    
    maximum_likelihood_points = []
    all_saved_solutions = []
    print("solution space contains %s solutions..." %len(new_solution_space))
    print("training models...")
    #filter out variants that aren't being used in training, look for universal mutations 
    training_data = np.expand_dims(training_data, axis=1)
    new_positions = [y for x,y in zip(frequency, positions) if float(x) > training_lower_bound and x < freq_upper_bound]
    new_nucs = [y for x,y in zip(frequency, nucs) if float(x) > training_lower_bound and float(x) < freq_upper_bound]
    original_variants = [str(x)+y for x,y in zip(new_positions, new_nucs)]

    #train the models
    all_saved_solutions, all_final_points, all_models, \
        maximum_likelihood_points, scores, assignments, all_scores, all_aic  \
        = parallel_train_models(new_solution_space, training_data, freq_precision, default_sigma, new_positions)

    #get the best model
    sorted_scores = copy.deepcopy(all_aic)
    original_index = list(range(0, len(maximum_likelihood_points)))
    zipped = list(zip(sorted_scores, original_index))
    zipped.sort()
    sorted_scores, original_index = zip(*zipped)    
    best_score = 1000
    flat_training = list(np.squeeze(training_data))
    for i, (loc, aic) in enumerate(zip(original_index, sorted_scores)):
        #if i > 100:
        #    break
        mm = all_models[loc]
        a = assignments[loc]
        s = scores[loc]
        als = all_scores[loc]
        n = [round(x,3) for x in list(np.squeeze(mm.mu))]
        n.sort()
        print(i, "score", sum(s), "aic", aic, all_saved_solutions[loc], len(all_saved_solutions[loc]))
        combination_solutions, combination_sums = generate_combinations(len(all_saved_solutions), all_saved_solutions[loc])
        combination_sums = [round(x,3) for x in combination_sums]
        useful_solution = True
        for aa, ss, td, all_sc in zip(a, s, training_data, als):
            td = td[0]
            if td in joint_peak:
                idx = combination_sums.index(aa[0])
                print("assignment", aa, "score", ss, "data", td, combination_solutions[idx])
                constituents = [round(x, freq_precision) for x in joint_dict[td][0]]
                for c in constituents:
                    idxx = flat_training.index(c)
                    ca = a[idxx][0]
                    idxx = combination_sums.index(ca)
                    must_have = combination_solutions[idxx]
                    print(c, "must have", must_have)
                    overlap = [x for x in must_have if x not in combination_solutions[idx]]
                    
                    if len(overlap) > 0:
                        useful_solution = False
        if useful_solution is True and aic < best_score:
            print("HERE")
            best_score = aic
    loc_best_model = all_aic.index(best_score)
    gx = all_models[loc_best_model]
    solution = all_saved_solutions[loc_best_model]
    final_points = all_final_points[loc_best_model]
    #create a new dataset for actually assigning out variants
    new_frequency, new_nucs, new_positions, add_low_depth_positions, universal_mutations = file_util.parse_additional_var(primer_dict, primer_positions, reference_sequence, ambiguity_dict, training_lower_bound, freq_upper_bound)
    new_variants = [str(p) + str(n) for p,n in zip(new_positions, new_nucs)] 
    assignment_data = np.array(new_frequency).reshape(-1,1)    
       
    print("scoring new data points")
    assignments, scores, all_likelihoods, ll  = gx.other_score(assignment_data, new_positions)
    
    combination_solutions, combination_sums = generate_combinations(len(solution), solution)
    combination_sums = [round(x,freq_precision) for x in combination_sums]
    print("best solution", solution)
    autoencoder_dict = {}
    print("%s universal mutations found..." %len(universal_mutations))
    for s in solution:
        autoencoder_dict[s] = universal_mutations
    save_variants = []
    save_scores = []
    save_assignments = []
    save_freq = []
    save_combos = []

    removal_dict, autoencoder_dict, no_call = determine_consensus_call(assignment_data, assignments, scores, all_likelihoods, autoencoder_dict, new_nucs, new_positions, combination_sums, combination_solutions, reference_positions)
    if DEBUG: 
        counter = 0
        check.sort()
        check2.sort()
        check3.sort()

        check_fail = []
        check2_fail = []
        check3_fail = [] 
        for key, value in autoencoder_dict.items(): 
            value.sort()
            if counter == 0:
                check_fail = [item for item in check if item not in value and item[:-1] not in low_depth_positions]
                print("\n", key, check_fail)
                print("extra", [item for item in value if item not in check and "-" not in item and item[:-1] not in low_depth_positions])
            elif counter == 1:
                check2_fail = [item for item in check2 if item not in value and item[:-1] not in low_depth_positions]
                print("\n", key, check2_fail)
                print("extra", [item for item in value if item not in check2 and "-" not in item and item[:-1] not in low_depth_positions])
            elif counter == 2:
                check3_fail = [item for item in check3 if item not in value and item[:-1] not in low_depth_positions]
                print("\n", key, check3_fail)
                print("extra", [item for item in value if item not in check3 and "-" not in item and item[:-1] not in low_depth_positions])
            counter += 1  

    pickle.dump(gx, open(model_location, 'wb')) 
    tmp_dict = {"autoencoder_dict":autoencoder_dict,"low_depth_positions":add_low_depth_positions, "ambiguity_dict":ambiguity_dict, "all_trained_solutions": new_solution_space, "assignment_data":new_frequency, "no_call":no_call, "removal_dict":removal_dict, "aic": best_score}

    with open(text_file, "w") as bfile:
        bfile.write(json.dumps(tmp_dict))
        bfile.write("\n")    
    return(0)

if __name__ == "__main__":
    main()
