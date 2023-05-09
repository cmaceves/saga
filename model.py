"""
Functionality for running the model.
"""
import os
import sys
import copy
import math
import time
import json
import random
import itertools

import numpy as np
import pandas as pd
from scipy import spatial
from joblib import Parallel, delayed
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.neighbors import KernelDensity

from line_profiler import LineProfiler

import file_util
import math_util

from other_gmm import GMM
random.seed(10)
np.random.seed(10)

DEBUG=False

def filter_combinations(combination_solutions, combination_sums):
    #remove things where all signals > 0.03 are the same
    filter_1_sol = []
    filter_1_sum = []
    zipped = list(zip(combination_sums, combination_solutions))
    zipped.sort()
    combination_sums, combination_solutions = zip(*zipped)
    filter_1_sol.extend(combination_solutions[:10])
    filter_1_sum.extend(combination_sums[:10])
    counter = 0
    for j, (sol, sm) in enumerate(zip(combination_solutions, combination_sums)):
        if sm > 0.97:
            continue
        if sol in filter_1_sol:
            continue
        if len(sol) == 1:
            filter_1_sol.append(sol)
            filter_1_sum.append(sm)
            continue

        tmp_sm = [x for x in sol if x > 0.03]
        tmp_sm.sort(reverse=True)
        keep = True
        for i, (s, z) in enumerate(zip(filter_1_sum, filter_1_sol)):                
            #we've passed our room for matches
            if s-sm > 0.03:
                filter_1_sol.append(sol)
                filter_1_sum.append(sm)
                break
            #we're way early
            elif s-sm < -0.03:
                continue
            else:
                #it's within a few percent and it shares all the same major peaks
                tmp_s = [x for x in z if x > 0.03]
                tmp_s.sort(reverse=True)
                if tmp_s == tmp_sm:
                    keep = False
                    break

                #lets check if they're really fing close
                same = True
                for a, b in zip(tmp_sm, tmp_s):
                    if a == b:
                        continue
                    if abs(a-b) < 0.01:
                        continue
                    if abs(a-b) < a * 0.03:
                        continue
                    same = False
                if same:
                    keep = False
                    break
        if keep is True:
            if sol not in filter_1_sol:
                #we haven't seem a solution like this
                filter_1_sol.append(sol)
                filter_1_sum.append(sm)
        counter += 1

    return(filter_1_sol, filter_1_sum)
    
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

def define_kde(frequencies, bw=0.0001, round_decimal=4, num=3000):
    x = np.array(frequencies)
    x_full = x.reshape(-1, 1)
    eval_points = np.linspace(np.min(x_full), np.max(x_full), num=num)
    kde = KernelDensity(kernel='gaussian', bandwidth=bw).fit(x_full)
    a = np.exp(kde.score_samples(eval_points.reshape(-1,1)))
    peaks = a[1:-1][np.diff(np.diff(a)) < 0]
    ind = [i for i,xx in enumerate(a) if xx in peaks]
    peak = []
    for i, xx in enumerate(eval_points):
        if i in ind:
            peak.append(round(xx, round_decimal))
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

def possible_conflicts(combination_sums, combination_solutions, solution):
    """
    Where do peaks conflict, what constitutes those conflicts?
    """
    #initialize
    conflict_dict = {}
    for csum in combination_sums:
        conflict_dict[csum] = []
    
    for i, (csum, csol) in enumerate(zip(combination_sums, combination_solutions)):
        csol = list(csol)
        for j, (csum2, csol2) in enumerate(zip(combination_sums, combination_solutions)):
            #identity
            if i == j:
                continue
            csol2 = list(csol2)
            if abs(csum - csum2) < 0.01:               
                misfire = [x for x in csol if x not in csol2]
                misfire.extend([x for x in csol2 if x not in csol])
                misfire.sort()
                current_list = conflict_dict[csum]                
                current_list.append({csum2:misfire})                
                conflict_dict[csum] = current_list

    return(conflict_dict)

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


def parallel_train_models(solutions, training_data, freq_precision, default_sigma):
    code = Parallel(n_jobs=30)(delayed(train_models)(solution, training_data, freq_precision, default_sigma) for solution in solutions)
    all_saved_solutions = [x[0] for x in code]
    all_final_points = [x[1] for x in code]
    all_models = [x[2] for x in code]
    all_conflict_dicts = [x[3] for x in code]
    maxmimum_likelihood_points = [x[4] for x in code]
    scores = [x[5] for x in code]
    assignments = [x[6] for x in code]
    return(all_saved_solutions, all_final_points, all_models, all_conflict_dicts, maxmimum_likelihood_points, scores, assignments)

def train_models(solution, training_data, freq_precision, default_sigma): 
    tmp_solution = [x for x in solution if x > 0.03]
    other_point = round(sum([x for x in solution if x <= 0.03]), freq_precision)
    if other_point > 0.01:
        tmp_solution.append(other_point)
    solution = tmp_solution
    
    n = len(solution)
    combination_solutions, combination_sums = generate_combinations(n, solution)
    combination_sums = [round(x, freq_precision) for x in combination_sums]
    conflict_dict = possible_conflicts(combination_sums, combination_solutions, solution)       
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
    gxx = GMM(k=len(final_points), dim=1, init_mu = final_points_expand, solution=solution, default_sigma=default_sigma)
    gxx.init_em(training_data)

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
    if max_num_iters != useful_iterations:
        gxx = GMM(k=len(final_points), dim=1, init_mu = final_points_expand, solution=solution, default_sigma=default_sigma)
        gxx.init_em(training_data)
        log_likelihood = []
        for e in range(useful_iterations):
            gxx.iteration += 1
            gxx.e_step()
            gxx.m_step() 
    assignment, score, all_scores, ll = gxx.score(training_data)
    sll = [np.log(x) for x in score]
  
    sigma_reset = gxx.sigma_reset
    
    #print(solution, "sll", round(sum(sll),3), len(solution))
    return(solution, final_points, gxx, conflict_dict, sum(sll), sll, assignment)

def collapse_small_peaks(solution_space, freq_precision):
    """
    Iterate all solutions and collapse frequencies that occur < 0.03 into one bin, we simply don't need that level of resolution and it complicates the problem in an unnecessary way.
    """
    print("collapsing small peaks...")
    new_solution_space = []
    for solution in solution_space:
        tmp = [x for x in solution if x < 0.03]
        binned = round(sum(tmp),freq_precision)
        solution = [round(x, freq_precision) for x in solution if x >= 0.03]
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
        if b/total_count > 0.20:
            keep_r.append(a)
    if len(keep_r) == 0:
        for a,b in zip(r_clusters, r_count):
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
            if 0.787 in new_solutions[i]:
                print(tmp, new_peak, old_peak, duplicate, diff)
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

def define_useful_kde_peaks(kde_reshape, freq_precision=3, total_peaks=20):
    print("finding useful kde peaks...")
    n = int(len(kde_reshape) / 5)
    if n < 20:
        n = 20
    print("length data", kde_reshape.shape)
    print("n", n)
    if len(kde_reshape) < n:
        n = len(kde_reshape)
    n_components = [50, 45, 40, 35, 30, 25, 20, 15, 10]
    tmp_n = []
    for value in n_components:
        if len(kde_reshape) < value:
            continue
        tmp_n.append(value)
    n_components = tmp_n
    
    #first we get a fast estimate of peaks in data, looking at ranges of 10% frequency - the way these distribute tells you something about the data complexity
    cluster_model = GaussianMixture(n_components=n)
    cluster_model.fit(kde_reshape)
    refined_kde_peaks = list(np.squeeze(np.array(cluster_model.means_)))
    refined_kde_peaks.sort(reverse=True)
    refined_kde_peaks = [round(x, freq_precision) for x in refined_kde_peaks]
    refined_kde_peaks = list(np.unique(refined_kde_peaks))    
    n = len(refined_kde_peaks)    
    alot_peaks = []
    ranges = [[0, 10], [10, 20], [20, 30], [30,40], [40,50], [50,60], [60,70], [70,80], [80, 90], [90, 100]]
    for r in ranges:
        count = 0
        for peak in refined_kde_peaks:
            if r[0] <= peak*100 < r[1]:
                count +=1
        percent = count/n
        alot_peaks.append(percent)       
        print(r, round(percent,2), count)
    q3, fifty, q1, tenth = np.percentile(refined_kde_peaks, [75 , 50, 25, 10])
    print('tenth', round(tenth,3))
    print('q1', round(q1,3))
    print("fifty", round(fifty,3))
    print('q3', round(q3,3))

    if q1 > 0.1 and (q3 > 0.90 or fifty > 0.5):
        complexity = "lowest"
    elif q3 > 0.90:
        complexity = "low"
    else:
        complexity = "high"    
    print(complexity)
    sys.exit(0)
    for r, percentage in zip(ranges, alot_peaks):
        if percentage == 0:
            continue
        keep_peaks = round(total_peaks * percentage)
        if keep_peaks == 0:
            keep_peaks += 1
        for n_comp in n_components:
            cluster_model = GaussianMixture(n_components=n_comp)
            cluster_model.fit(kde_reshape)
            small_peaks = list(np.squeeze(np.array(cluster_model.means_)))
            small_peaks.sort(reverse=True)
            small_peaks = [round(x, freq_precision) for x in small_peaks]
            small_peaks = [x for x in small_peaks if x >= r[0]/100 and x < r[1]/100]
            if len(small_peaks) <= keep_peaks:
                all_peaks.extend(small_peaks) 
                break
        pf = [round(x,3) for x in small_peaks]
        pf.sort()
        print(pf, r, round(percentage,2), keep_peaks)

    all_peaks = [round(x, freq_precision) for x in all_peaks]
    all_peaks = [x for x in list(np.unique(all_peaks)) if x > 0]
    all_peaks.sort()
    sys.exit(0)
    return(all_peaks, complexity)

               
def run_model(variants_file, output_dir, output_name, primer_mismatches=None, physical_linkage_file=None, freyja_file=None, bed_file=None, bam_file=None):
    freq_lower_bound = 0.0001
    freq_upper_bound = 0.98
    training_lower_bound = 0.03
    freq_precision = 3
    r_min = 2
    r_value_maxima = 31
    percent_solution_space = 0.01

    text_file = os.path.join(output_dir, output_name+"_model_results.txt")
    r_values_file = os.path.join(output_dir, output_name+"_solutions.txt") 

    problem_primers = None
    problem_positions = None
    remove_pos_dict = None
    if bam_file is not None:
        remove_pos_dict, problem_positions = file_util.parse_bam_depth_per_position(bam_file, bed_file)
    if freyja_file is not None:
        gt_centers, gt_lineages = file_util.parse_freyja_file(freyja_file)
        gt_mut_dict = file_util.parse_usher_barcode(gt_lineages)
        print(gt_centers)
        print(gt_lineages)
        """
        target = "AY.3"
        target_dict = gt_mut_dict[target]
        for key, value in gt_mut_dict.items():
            idx = gt_lineages.index(key)
            if gt_centers[idx] < 0.03:
                continue
            tmp = [x for x in target_dict if x not in value]
            target_dict = tmp
        print(target, "unique mutations", target_dict)
        sys.exit(0)
        """
    positions, frequency, nucs, low_depth_positions, reference_positions = file_util.parse_ivar_variants_file(
            variants_file, \
            freq_precision, \
            problem_positions, \
            bed_file, \
            problem_primers, \
            remove_pos_dict)
     
    print(len(frequency)) 
    for f,n,p in zip(frequency, nucs, positions):
        r_check = str(n) + "_" + str(p)
        #if r_check in reference_positions:
        #    continue
        if f < 0.97 and f > 0.10:
            var = str(p) + str(n)
            print(f, n, p)

    #here we filter out really low data points, and data points we assume to be "universal mutations"
    new_frequencies = [round(x, freq_precision) for x in frequency if x > freq_lower_bound and x < freq_upper_bound]

    #first, we do a kernel density estimate to find local maxmima
    kde_peaks = define_kde(new_frequencies)
    kde_peaks.sort(reverse=True)
    
    r_min = 2
    r_max = 14

    #second, we cluster the local maxmima into a useable solution set, and estimate the complexity
    kde_reshape = np.array(kde_peaks).reshape(-1,1)
    refined_kde_peaks, complexity  = define_useful_kde_peaks(kde_reshape)
    tmp = sum([x for x in refined_kde_peaks if x < 0.01])
    refined_kde_peaks = [x for x in refined_kde_peaks if x >= 0.01]
    refined_kde_peaks.append(tmp)
    refined_kde_peaks.sort()
    print(refined_kde_peaks, len(refined_kde_peaks))
    print("complexity", complexity)
    if complexity == "high":
        r_min = 5
        lower_subtract = 0.40
    elif complexity == "low":
        r_max = 5
        lower_subtract = 0.30
        error = 0.07
    elif complexity == "extremely high":
        r_min = 9
        return(1)

    refined_kde_peaks = [round(x, freq_precision) for x in refined_kde_peaks if x > 0]
    default_sigma = 1
    print("refined kde", refined_kde_peaks, len(refined_kde_peaks))
    #sys.exit(0)
    #print("len gt centers over 3", len([x for x in gt_centers if x > 0.03]))
    #print("len gt centers over 1", len([x for x in gt_centers if x > 0.01]))
    #print("freq under 3", sum([x for x in gt_centers if x < 0.03]))    

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
    #now eliminate solutions that fail to account for all peaks reported by the "preseed" GMM
    if complexity != "low":
        solution_space = parallel_eliminate_solutions(solution_space, refined_kde_peaks)
        keep = [x for x in refined_kde_peaks if x < 0.10 and x > 0]
        other = []
        for sol in solution_space:
            tmp = [x for x in keep if x not in sol]
            if len(tmp) > 0:
                continue
            other.append(sol)
        solution_space = other
    
    if complexity == "low":
        keep = []
        for solution in solution_space:
            if len(solution) <= r_max and solution not in keep:
                keep.append(solution)
        solution_space = keep
    r_values = range(r_min, r_max)  

    print("length of solution space accounting for individual/shared peak presence", len(solution_space))
    if len(solution_space) > 0:
        #here we determine a tighter rmin, rmax based on the number of solutions at each r value
        r_values = determine_r_values(solution_space, r_clusters)        
        print("looking for solutions of ", r_values, "lengths...")
        keep_solutions = []
        for sol in solution_space:
            if len(sol) in r_values:
                keep_solutions.append(sol)
        print("total of %s solutions..." %(len(keep_solutions))) 
        solution_space = keep_solutions
   
    solutions_to_train = 100
    if len(solution_space) > solutions_to_train:
        print("exploring %s percent of the solution space which is %s solutions..." %(str(percent_solution_space*100), str(solutions_to_train)))
        solution_space = collapse_solution_space(solution_space, freq_precision, clusters=solutions_to_train) 
        print("length of collapsed solution space", len(solution_space))
   
    if complexity != "low":
        freq_precision = 2 
    #prior to acutal training, collapse down tiny peaks to make training more reasonable
    solution_space = collapse_small_peaks(solution_space, freq_precision)
    training_data = [round(x, freq_precision) for x in training_data]
   
    #deduplicate the solution space
    keep = []
    for sol in solution_space:
        if sol not in keep:
            print(sol)
            keep.append(sol)
    solution_space = keep
    print("sol length after deduplication", len(solution_space))    
    total = 0
    points = 0
    for sol in solution_space:
        points += 1
        total += len(sol)
    r_average = total/points
    print("average r", r_average)
    r_values = [int(x) for x in r_values]
    new_solution_space = solution_space

    if DEBUG:    
        check = gt_mut_dict[gt_lineages[0]]
        check2 = gt_mut_dict[gt_lineages[1]]
        check3 = gt_mut_dict[gt_lineages[2]]
     
    all_models = []
    all_model_log_likelihood = []
    all_final_points = []    
    all_conflict_dicts = []    
    maximum_likelihood_points = []
    all_saved_solutions = []
    print("solution space contains %s solutions..." %len(new_solution_space))
    print("training models...")
    #training_data.sort()
    #filter out variants that aren't being used in training, look for universal mutations 
    training_data = np.expand_dims(training_data, axis=1)
    new_positions = [y for x,y in zip(frequency, positions) if float(x) > training_lower_bound and x < freq_upper_bound]
    new_nucs = [y for x,y in zip(frequency, nucs) if float(x) > training_lower_bound and float(x) < freq_upper_bound]
    universal_mutations = [str(x)+str(y) for (x,y,z) in zip(positions, nucs, frequency) if float(z) > freq_upper_bound and str(y) != '0']
    tmp_universal_mutations = []
    for um in universal_mutations:
        tum = um[-1] + "_" + um[:-1]
        if tum not in reference_positions:
            tmp_universal_mutations.append(um)
    universal_mutations = tmp_universal_mutations

    #train the models
    all_saved_solutions, all_final_points, all_models, all_conflict_dicts, \
        maximum_likelihood_points, scores, assignments \
        = parallel_train_models(new_solution_space, training_data, freq_precision, default_sigma)
 

    #get the best model
    sorted_scores = copy.deepcopy(maximum_likelihood_points)
    sorted_scores.sort(reverse=True)

    all_sigma_resets = []
    for i, ll in enumerate(sorted_scores):
        if i < 10000:
            loc = maximum_likelihood_points.index(ll)
            
            mm = all_models[loc]
            print(i, "score", ll, all_saved_solutions[loc], len(all_saved_solutions[loc]))
            all_sigma_resets.extend(list(mm.sigma_reset.keys()))
            lll = scores[loc]
            llll = assignments[loc]
            """
            for td, sl, asi in zip(training_data, lll, llll):
                if asi[0] > 0.10:
                    print("point", td, sl, "assignment", asi)
            """
    loc_best_model = maximum_likelihood_points.index(sorted_scores[0])
    solution = all_saved_solutions[loc_best_model]
    final_points = all_final_points[loc_best_model]
    gx = all_models[loc_best_model]
    conflict_dict = all_conflict_dicts[loc_best_model]
    assignments, scores, all_likelihoods, ll  = gx.score(training_data)
 
    combination_solutions, combination_sums = generate_combinations(len(solution), solution)
    combination_sums = [round(x,freq_precision) for x in combination_sums]
    #combination_solutions, combination_sums = filter_combinations(combination_solutions, combination_sums)
    print("best solution", solution)
    autoencoder_dict = {}
    print("%s universal mutations found..." %len(universal_mutations))
    for s in solution:
        autoencoder_dict[s] = universal_mutations
    print(universal_mutations) 
    save_variants = []
    save_scores = []

    cluster_scores = [0.0] * len(solution)
    num_points = [0.0] * len(solution)
    print("nucs:", len(new_nucs), "pos:", len(new_positions), "training data:", len(training_data))
    for i, (point, assignment, score, all_score) in enumerate(zip(training_data, assignments, scores, all_likelihoods)):
        assignment = assignment[0]
        point = point[0]
        nuc = str(new_nucs[i])
        if nuc == '0':
            continue
        pos = str(new_positions[i])
        variant = pos+nuc
        if str(nuc) + "_"+ str(pos) in reference_positions:
            continue
        save_variants.append(variant)
        save_scores.append(score)
        location = combination_sums.index(assignment)
                
        #print(variant, point, assignment, score, combination_solutions[location])
        #print(conflict_dict[assignment])
        #print(i, combination_solutions[location])
        for individual in combination_solutions[location]:
            idx = solution.index(individual)
            cluster_scores[idx] += np.log(score)        
            num_points[idx] += 1
            tmp_list = copy.deepcopy(autoencoder_dict[individual])
            tmp_list.append(variant)
            autoencoder_dict[individual] = tmp_list

    print(variants_file)
    print(solution)
    #print(gt_centers)
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
    sys.exit(1)
    tmp_dict = {"autoencoder_dict":autoencoder_dict, "problem_positions":problem_positions, "low_depth_positions":low_depth_positions, "variants":save_variants, "scores":save_scores, "conflict_dict": conflict_dict, "cluster_metrics": cluster_scores, "cluster_count":num_points, "r_values":r_values, "sigma_reset":gx.sigma_reset}

    with open(text_file, "w") as bfile:
        bfile.write(json.dumps(tmp_dict))
        bfile.write("\n")    
    return(0)

if __name__ == "__main__":
    main()
