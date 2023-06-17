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
import itertools

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy import stats
from scipy import spatial
from scipy.spatial import distance
from joblib import Parallel, delayed
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import KernelDensity
from sklearn.decomposition import PCA
from line_profiler import LineProfiler

import file_util
import math_util
import generate_consensus 

from other_gmm import GMM
random.seed(10)
np.random.seed(10)

DEBUG=False   

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

def closest_node(node, nodes):   
    closest_index = distance.cdist([node], nodes).argmin()
    return(nodes[closest_index], closest_index)


def closest_nodes(node, nodes, num_nodes=10):
    distances = distance.cdist([node], nodes)[0]
    indexes = list(range(len(distances)))
    zipped = list(zip(distances, indexes))
    zipped.sort()
    distances, indexes = zip(*zipped)
    distances = list(distances)
    indexes = list(indexes)[:num_nodes]
    selected_nodes = [x for i,x in enumerate(nodes) if i in indexes]
    return(selected_nodes, indexes)

def determine_consensus_call(assignment_data, assignments, scores, all_likelihoods, autoencoder_dict, new_nucs, new_positions,combination_sums, combination_solutions, reference_positions):
    """
    Given a set of sequences, determine what we call consensus on.
    """
    removal_dict = {}
    for k,v in autoencoder_dict.items():
        removal_dict[k] = []

    call_ambiguity = []    

    #look at what scores may be lower outliers
    arr_scores = np.array(scores)
    med = np.median(arr_scores)
    abs_diff = np.abs(arr_scores-med)
    med_abs_diff = np.median(abs_diff)
    z_scores = 0.6745 * (arr_scores - med) / med_abs_diff

    for i, (point, assignment, score, all_like, z) in enumerate(zip(assignment_data, assignments, scores, all_likelihoods, z_scores)):
        point = point[0]
        assignment = assignment[0]
        nuc = str(new_nucs[i])
        pos = str(new_positions[i])
        variant = pos+nuc
        if pos+"_"+nuc in reference_positions:
            continue
        if z < -1.65: #this is the 0.05 percent cutoff in a normal distribution
            print("not used", variant, z)
            call_ambiguity.append(pos)
            continue
        location = combination_sums.index(assignment)
        combo_sol = combination_solutions[location]      
        all_like = [round(x,3) for x in all_like]
        combo_sol_copy = copy.deepcopy(combination_solutions)
        zipped = list(zip(all_like, combo_sol_copy))
        zipped.sort(reverse=True)
        all_like, combo_sol_copy = zip(*zipped)
        #print('point', point, 'variant', variant, 'assignment', assignment, 'score', score, all_like[0], all_like[1], all_like[2], combo_sol_copy[0], "z", z)
       
        ratio = all_like[0] / all_like[1]
        #we cannot concretely assign this point
        if ratio < 5:
            diff_assignment = [x for x in combo_sol_copy[0] if x not in combo_sol_copy[1]] 
            #if we can say it definitly belongs to one population at least
            for difference in diff_assignment:            
                removal_dict[difference].append(variant)   
                #print('point', point, 'variant', variant, 'assignment', assignment, 'score', score, all_like[0], all_like[1], all_like[2], combo_sol_copy[0], combo_sol_copy[1], "z", z)
            continue
        
        for cs in combo_sol_copy[0]:
            if variant not in autoencoder_dict[cs]:
                autoencoder_dict[cs].append(variant)
        
    no_call = []
    for k,v in removal_dict.items():
        if len(v) > 0:
            no_call.append(k)
    return(removal_dict, autoencoder_dict, no_call, call_ambiguity)          
            
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
    for t in tmp:
        idx = peak_list.index(t)
        ev = eval_points[list(a).index(peak_list[idx])]
        if round(ev, freq_precision) != 0:
            keep_tmp.append(t)
        if len(keep_tmp) == total_return_length:
            break
    tmp = keep_tmp
    for i, xx in enumerate(eval_points):
        if i in ind:
            idx = ind.index(i)
            if peak_list[idx] in tmp:
                tmp.remove(peak_list[idx])
                peak.append(round(xx, freq_precision))
    return(peak)

def create_solution_space(peaks, r_min, r_max, lower_subtract=0.40):
    total_value = 1
    lower_bound = total_value + lower_subtract
    upper_bound = total_value - lower_subtract
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
    new_solution_space = []
    kept_solutions = []
    #let's now cluster the solutions to try and eliminate duplicates
    k_solutions = MiniBatchKMeans(n_clusters=clusters)
    k_solutions.fit(np.array(expanded_solution_space))
    clustered_solutions = k_solutions.cluster_centers_
    for solution in clustered_solutions:
        solution = list(solution)
        selected_nodes, index = closest_node(solution, expanded_solution_space)           
        kept_solutions.append(solution_space[index])
    return(kept_solutions)    

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


def parallel_train_models(solutions, training_data, freq_precision, positions):
    code = Parallel(n_jobs=10)(delayed(train_models)(solution, training_data, freq_precision, positions) for solution in solutions)
    all_saved_solutions = [x[0] for x in code]
    all_final_points = [x[1] for x in code]
    all_models = [x[2] for x in code]
    all_linear_distances = [x[3] for x in code]
    scores = [x[4] for x in code]
    assignments = [x[5] for x in code]
    all_scores = [x[6] for x in code]
    all_aic = [x[7] for x in code]
    return(all_saved_solutions, all_final_points, all_models, all_linear_distances, scores, assignments, all_scores, all_aic)

def train_models(solution, training_data, freq_precision, positions): 
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
    gxx = GMM(k=len(final_points), dim=1, init_mu = final_points_expand, solution=solution, fixed_means=True)
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
        gxx = GMM(k=len(final_points), dim=1, init_mu = final_points_expand, solution=solution, fixed_means=True)
        gxx.init_em(training_data, positions, combination_solutions, combination_sums)
        log_likelihood = []
        for e in range(useful_iterations):
            gxx.iteration += 1
            gxx.e_step()
            gxx.m_step() 
    
    assignment, score, all_scores, ll, aic, linear_distance = gxx.other_score(training_data, positions)
    sll = [np.log(x) for x in score]
    final_points.sort(reverse=True)
    return(solution, final_points, gxx, linear_distance, score, assignment, all_scores, aic)

def collapse_small_peaks(solution_space, freq_precision, r_max, collapse_value=0.03):
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
        if len(solution) > r_max:
            continue
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

def round_solution_space(solution_space, freq_precision):
    new_solution_space = []
    for solution in solution_space:
        solution = [round(x, freq_precision) for x in solution]
        new_solution_space.append(solution)
    return(new_solution_space)
              
def create_complexity_estimate(total_mut_pos, ambiguity_dict):
    """
    """
    length_total_mut = len([x for x in total_mut_pos if x > 0.1])
    scaled_complexity_estimate = length_total_mut/(29903*3)
    print("scaled_complexity_estimate", scaled_complexity_estimate)
    complexity = "high"
    if scaled_complexity_estimate < 0.0008:
        complexity = "extremely low"
    elif scaled_complexity_estimate < 0.02:
        complexity = "low"
    return(complexity)

def run_model(output_dir, output_name, bam_file, bed_file, reference_file, freyja_file=None):
    freq_lower_bound = 0.0001
    freq_upper_bound = 0.97
    training_lower_bound = 0.03
    lower_subtract = 0.30
    solutions_to_train = 1000
    model_location = os.path.join(output_dir, output_name+"_model.pkl")
    problem_primers = None
    remove_pos_dict = None

    variants_json = os.path.join(output_dir, output_name+"_variants.txt")
    
    if os.path.isfile(variants_json):
        with open(variants_json, "r") as rfile:
            primer_dict = json.load(rfile)
            bam_file = None
    
    if bam_file is not None:
        """
        lp = LineProfiler()
        lp_wrapper = lp(file_util.parse_bam_depth_per_position)
        lp_wrapper(bam_file, bed_file, variants_json)
        lp.print_stats()
        sys.exit(0)
        """
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
        total_mutated_pos, training_removed = file_util.parse_variants(primer_dict, primer_positions, reference_sequence)
    complexity = create_complexity_estimate(total_mutated_pos, ambiguity_dict)
    #this may be expanded in future iterations for more complex samples
    if complexity == "low":
        r_min = 3
        r_max = 6
        error = 0.05
        freq_precision = 2
        total_return_length = 40
    elif complexity == "extremely low":
        r_min = 2
        r_max = 4
        error = 0.05
        freq_precision = 2
        total_return_length = 30
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
    print("total mutated pos", len(total_mutated_pos))
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
    refined_kde_peaks = define_kde(kde_data, complexity, num, total_return_length, freq_precision)
    refined_kde_peaks.sort(reverse=True)
    
    #if we have no data
    if len(refined_kde_peaks) == 0 or len(new_frequencies) == 0:
        return(1)

    tmp = sum([x for x in refined_kde_peaks if x < 0.01])
    refined_kde_peaks = [x for x in refined_kde_peaks if x >= 0.01]
    refined_kde_peaks.append(tmp)
    refined_kde_peaks.sort()
    print("complexity", complexity)

    refined_kde_peaks = [round(x, freq_precision) for x in refined_kde_peaks if x > 0]
    print("refined kde", refined_kde_peaks, len(refined_kde_peaks))
    print("joint peak", joint_peak)
    if len(refined_kde_peaks) < r_max:
        r_max = len(refined_kde_peaks)

    solution_space = create_solution_space(refined_kde_peaks, r_min, r_max, lower_subtract=lower_subtract)
    print("original solution space length", len(solution_space))
    #here we assume that their might be exactly 1 totally hidden solution that accounts for the difference
    solution_space = find_masked_solution(solution_space, freq_precision)
    print("masked solution space length", len(solution_space))

    #constraint #1 E(solution) ~= 1
    solution_space, hidden_peaks = parallel_contract_solutions(solution_space, freq_precision, error=error)
    #constraint #1 E(solution) ~= 1
    solution_space = parallel_expand_solutions(solution_space, freq_precision, error=error)    
    print("length of expansive solution space", len(solution_space))

    keep = []
    for sol in solution_space:
        if sol not in keep and len(sol) > 2:
            keep.append(sol)

    solution_space = keep
    original_solution_space = copy.deepcopy(solution_space)
    original_solution_space = collapse_small_peaks(original_solution_space, freq_precision, r_max, collapse_value = 0.05)
    #here we expand the solutions to encompass all possible peaks
    original_space_expand = expanded_solution_space(original_solution_space, freq_precision)
    if len(solution_space) > solutions_to_train:
        solution_space = collapse_solution_space(solution_space, original_space_expand, freq_precision, clusters=solutions_to_train) 
        print("length of collapsed solution space", len(solution_space))
    
    #prior to acutal training, collapse down tiny peaks to make training more reasonable
    solution_space = collapse_small_peaks(solution_space, freq_precision, r_max, collapse_value = 0.05)
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
        all_linear_distances, all_scores, assignments, all_likelihoods, all_aic  \
        = parallel_train_models(new_solution_space, training_data, freq_precision, training_positions)

    #get the best model
    sorted_scores = copy.deepcopy(all_aic)
    original_index = list(range(0, len(all_aic)))
    zipped = list(zip(sorted_scores, original_index))
    zipped.sort()
    sorted_scores, original_index = zip(*zipped)    
    best_score = 3000
    flat_training = list(np.squeeze(training_data))

    X = []
    color = []
    sol_order =[]
   
    for i, (loc, aic) in enumerate(zip(original_index, sorted_scores)):
        mm = all_models[loc]
        a = assignments[loc]
        als = all_likelihoods[loc]
        linear_dist = all_linear_distances[loc]
        scores = all_scores[loc]
        #look at what scores may be lower outliers
        arr_scores = np.array(scores)
        med = np.median(arr_scores)
        abs_diff = np.abs(arr_scores-med)
        med_abs_diff = np.median(abs_diff)
        z_scores = 0.6745 * (arr_scores - med) / med_abs_diff
        n = [round(x,3) for x in list(np.squeeze(mm.mu))]
        n.sort()
        color.append(aic)
        X.append(list(np.squeeze(mm.mu)))
        sol_order.append(all_saved_solutions[loc])
        if i < 15:
            print(i, "aic", aic, "linear", linear_dist, all_saved_solutions[loc], len(all_saved_solutions[loc]))
        
        combination_solutions, combination_sums = generate_combinations(len(all_saved_solutions), all_saved_solutions[loc])
        combination_sums = [round(x,3) for x in combination_sums]
        useful_solution = True
        for aa, td, all_sc, z in zip(a, training_data, als, z_scores):
            td = td[0]
            if td in joint_peak:
                if z < -1.65:
                    continue
                idx = combination_sums.index(aa[0])
                constituents = [round(x, freq_precision) for x in joint_dict[td][0]]
                for c in constituents:
                    idxx = flat_training.index(c)
                    ca = a[idxx][0]
                    idxx = combination_sums.index(ca)
                    must_have = combination_solutions[idxx]
                    overlap = [x for x in must_have if x not in combination_solutions[idx]]
                    
                    if len(overlap) > 0:
                        if i < 15:
                            print("assignment", aa, "data", td, combination_solutions[idx])
                            all_sc = list(all_sc)
                            all_sc.sort(reverse=True)
                            if all_sc[0] / all_sc[1] < 5:
                                print("cont")
                                continue
                            #print(all_sc[0], all_sc[1], z)
                            #print(c, "must have", must_have)
                        useful_solution = False
        if useful_solution is True and aic < best_score:
            print("used")
            best_score = aic
    if best_score == 3000:
        return(1)
  
    loc_best_model = all_aic.index(best_score)
    gx = all_models[loc_best_model]      
    print("solution", gx.solution)
    print("score", all_aic.index(best_score))
    pickle.dump(gx, open(model_location, 'wb')) 

def call_consensus(output_dir, output_name, model_location, reference_file, bed_file):
    gx = pickle.load(open(model_location, 'rb')) 
    solution = gx.solution
    training_lower_bound = 0.03
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
        total_mutated_pos, training_removed = file_util.parse_variants(primer_dict, primer_positions, reference_sequence)

    #get complexity value
    complexity = create_complexity_estimate(total_mutated_pos, ambiguity_dict)
 
    #create a new dataset for actually assigning out variants
    new_frequency, new_nucs, new_positions, add_low_depth_positions, universal_mutations = file_util.parse_additional_var(primer_dict, primer_positions, reference_sequence, ambiguity_dict, training_lower_bound, freq_upper_bound)
    new_variants = [str(p) + str(n) for p,n,f in zip(new_positions, new_nucs, new_frequency) if f > training_lower_bound and f < freq_upper_bound] 
    new_positions = [str(p) for p,f in zip(new_positions, new_frequency) if f > training_lower_bound and f < freq_upper_bound] 
    new_nucs = [str(n) for n,f in zip(new_nucs, new_frequency) if f > training_lower_bound and f < freq_upper_bound] 
    new_frequency = [x for x in new_frequency if x > training_lower_bound and x < freq_upper_bound]
    assignment_data = np.array(new_frequency).reshape(-1,1)    
 
    print("scoring new data points")
    assignments, scores, all_likelihoods, ll, aic, linear_distance = gx.other_score(assignment_data, new_positions)
    combination_solutions, combination_sums = generate_combinations(len(solution), solution)
    combination_sums = [round(x,freq_precision) for x in combination_sums]
    print("best solution", solution)
    autoencoder_dict = {}
    print("%s universal mutations found..." %len(universal_mutations))
    for s in solution:
        autoencoder_dict[s] = copy.deepcopy(universal_mutations)

    removal_dict, autoencoder_dict, no_call, call_ambiguity = determine_consensus_call(assignment_data, assignments, scores, all_likelihoods, autoencoder_dict, new_nucs, new_positions, combination_sums, combination_solutions, reference_positions)

    call_ambiguity = [int(x) for x in call_ambiguity]
    call_ambiguity.sort()
    tmp_dict = {"autoencoder_dict":autoencoder_dict,"low_depth_positions":add_low_depth_positions, "ambiguity_dict":ambiguity_dict, "assignment_data":new_frequency, "no_call":no_call, "removal_dict":removal_dict, "complexity":complexity, "total_mutated_pos":total_mutated_pos, "call_ambiguity":call_ambiguity}

    with open(text_file, "w") as bfile:
        bfile.write(json.dumps(tmp_dict))
        bfile.write("\n")    
    return(0)

if __name__ == "__main__":
    main()
