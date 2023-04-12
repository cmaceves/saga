"""
Functionality for running the model.
"""
import os
import sys
import copy
import math
import json
import random
import itertools

import numpy as np
import pandas as pd
import networkx as nx
from scipy import spatial
from joblib import Parallel, delayed
from networkx.algorithms.components import connected_components
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.cluster import KMeans
from sklearn.neighbors import KernelDensity

from line_profiler import LineProfiler

import file_util
import math_util

from other_gmm import GMM
random.seed(10)
np.random.seed(10)

DEBUG=True

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
    
def determine_close_individuals(solution, tolerance = 0.01):
    pairs = []
    seen = []
    for value in solution:
        if value in seen:
            continue
        for second_value in solution:
            if second_value in seen or value in seen:
                continue
            if value == second_value:
                continue
            if abs(value - second_value) < tolerance and value > 0.03:
                minimum = min(value, second_value)
                maxmimum = max(value, second_value)
                seen.append(value)
                seen.append(second_value)
                tmp_list = [minimum, maxmimum, np.mean([minimum, maxmimum])]
                if tmp_list not in pairs:
                    pairs.append(tmp_list)
            elif value < 0.03 and abs(value - second_value) < value * 0.10:
                seen.append(value)
                seen.append(second_value)
                minimum = min(value, second_value)
                maxmimum = max(value, second_value)
                tmp_list = [minimum, maxmimum, np.mean([minimum, maxmimum])]
                if tmp_list not in pairs:
                    pairs.append(tmp_list)
    return(pairs)

def generate_combinations(n, solution):
    combination_solutions = []
    combination_sums = []
    highest_val_over_3_percent = 1
    for s in solution:
        if s < 0.03:
            continue
        elif s < highest_val_over_3_percent:
            highest_val_over_3_percent = s
    for L in range(n + 1):
        #print("L", L, "out of", n+1)
        for subset in itertools.combinations(solution, L):
            things_over_3_percent = 0
            things_under_3_percent_sum = 0
            
            for thing in subset:
                if thing < 0.03:
                    things_under_3_percent_sum += thing
                else:
                    things_over_3_percent += 1
            if things_over_3_percent > 1:
                pass
            elif things_under_3_percent_sum > highest_val_over_3_percent:
                pass
            elif len(subset) == 1:
                pass
            else:
                continue            
            if len(subset) < 1:
                continue                   
            
            #sum_val = round(sum(subset), 5)
            combination_solutions.append(subset)
            combination_sums.append(sum(subset))
    return(combination_solutions, combination_sums)

def assign_clusters(frequencies, transformed_centers):
    #recalculate centers
    cluster_points = []
    return_points = []
    
    arr = np.array(transformed_centers)

    for n in range(len(transformed_centers)):
        cluster_points.append([])
    
    #asign points to clusters and recalculate
    for f in frequencies:
        closest_index = np.abs(arr-f).argmin()
        cluster_points[closest_index].append(f)

    return(cluster_points, return_points)

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

def apply_physical_linkage(solution_space, G, kde_peaks, positions, nucs, frequencies):
    arr = np.asarray(kde_peaks)
    variants = [str(n) + "_" + str(p) for n,p in zip(nucs, positions)]
    remove_solutions = []
    frequencies_remove = []
    linked = []
    #print(kde_peaks)
    #sys.exit(0)
    for freq, var in zip(frequencies, variants):
        if not G.has_node(var):
            continue
        node = G.nodes[var]
        subgraphs = [G.subgraph(var) for var in connected_components(G)]
        n_subgraphs = len(subgraphs)
        for i in range(n_subgraphs):
            node_names = subgraphs[i].nodes()
            if var not in node_names:
                continue
            for ov in node_names:
                ov_freq = float(G.nodes[ov]['frequency'])
                if ov == var:
                    continue
                count = G.get_edge_data(var, ov)
                if count is None:
                    continue
                
                #TODO: this should be a proportion of the reads covering this area
                #if int(count['count']) < 3:
                #    continue
                
                if ov_freq == 0 or freq == 0:
                    continue
                idx = (np.abs(arr - ov_freq)).argmin()
                assigned1 = arr[idx]
                idx = (np.abs(arr - freq)).argmin()
                assigned2 = arr[idx]
                
                if assigned1 == assigned2:
                    continue
                
                diff = abs(ov_freq - freq)
                print(ov_freq, freq)
                continue
                #print(round(diff,3), assigned1, assigned2, max(assigned1, assigned2))
                if diff > 0.05:
                    #print(var, ov)
                    #print(assigned1, assigned2, max(assigned1, assigned2))
                    remove_solutions.append(max(assigned1, assigned2))
                    if max(assigned1, assigned2) == assigned1:
                        frequencies_remove.append(ov_freq)
                    else:
                        frequencies_remove.append(freq)
                else:
                    linked.append([min(assigned1, assigned2), max(assigned1, assigned2)])
    sys.exit(0)
    new_linked = []
    for l in linked:
        if l not in new_linked:
            new_linked.append(l)

    remove_solutions = list(np.unique(remove_solutions))
    added = []
    for rs in remove_solutions:
        for l in new_linked:
            if rs in l:
                added.extend(l)
    remove_solutions.extend(added)
    remove_solutions = np.unique(remove_solutions)

    new_solutions = []
    for sol in solution_space:
        nsol = [x for x in sol if x not in remove_solutions]
        if len(nsol) == len(sol):
            new_solutions.append(sol)

    for f in frequencies:
        idx = (np.abs(arr - f)).argmin()
        assigned = arr[idx]
        if assigned in remove_solutions:
            frequencies_remove.append(f)
    
    return(new_solutions, frequencies_remove)

def create_solution_space(kde_peaks, n, total_value=1):
    print("creating solution space for %s things..." %n)
    overlap_sets = []
    flat = []
    lower_bound = total_value + 0.03
    upper_bound = total_value - 0.25
    for subset in itertools.combinations(kde_peaks, n):
        subset = list(subset)
        total = sum(subset) #most expensive line
        if (total > lower_bound) or (total < upper_bound):
            continue
        overlap_sets.append(subset)
    new_overlap_set = []

    #each data peak must be represented in the data
    for subset in overlap_sets:
        combination_solutions, combination_sums = generate_combinations(n, subset) 
        combination_sums.sort(reverse=True)
        combination_sums = np.array(combination_sums)    
        found = True
        for peak in kde_peaks:
            #we don't care too much about the tiny peaks
            if peak < 0.03:
                continue
            index = (np.abs(combination_sums-peak)).argmin()
            match = combination_sums[index]
            if abs(match-peak) > 0.02:
                #print(peak, match, combination_sums)
                found = False
                break
        if found:
            new_overlap_set.append(subset)

    return(overlap_sets)

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

def collapse_solution_space(solution_space, threshold = 0.005, n_clusters=50):
    """
    Treat all points under threshold as same population, eliminate solution duplicates.
    """
    print("collapsing solution space...")
    new_solution_space = []
    longest_solution = 0
    #here we collapse points less than the threshold into one category
    for solution in solution_space:
        tmp_sol = [x for x in solution if x >= threshold]
        tmp_sol_low = [x for x in solution if x < threshold]
        solution = tmp_sol
        other = round(sum(tmp_sol_low), 5)
        if other > 0:
            solution.append(other)
        if len(solution) > longest_solution:
            longest_solution = len(solution)
        solution.sort(reverse=True)
        new_solution_space.append(solution)
    padded_data = []
    for solution in new_solution_space:
        if len(solution) < longest_solution:
            zeros = longest_solution - len(solution)
            solution.extend([0.0] * zeros)
        padded_data.append(solution)

    padded_data = np.array(padded_data)
    #let's now cluster the solutions to try and eliminate duplicates
    k_solutions = KMeans(n_clusters=n_clusters)
    k_solutions.fit(padded_data)
    clustered_solutions = k_solutions.cluster_centers_
    
    new_solution_space = []    
    for solution in clustered_solutions:
        summation = sum(solution)
        #if summation > 1:
        #    continue
        solution = [round(x,5) for x in list(solution) if x > threshold]
        diff = 1-summation
        #we need something still in the ballpark of 100%
         
        if diff < 0.97 and diff > 0:
            solution, include = find_duplicate(solution)
            if include:
                new_solution_space.append(solution)
        else:
            new_solution_space.append(solution)
    return(new_solution_space)

def find_duplicate(solution):
    diff = 1 - sum(solution)
    combination_solutions, combination_sums = generate_combinations(len(solution), solution)
    combination_sums = [round(x,5) for x in combination_sums]

    arr = np.array(combination_sums)
    idx = np.abs(arr - diff).argmin()
    duplicate = combination_sums[idx]

    if abs(duplicate-diff) > 0.03:
        return(solution, False)
    else:
        dups = list(combination_solutions[idx])
        solution.extend(dups)
        solution.sort(reverse=True)
        return(solution, True)

def eliminate_solutions(solution_space, means):
    """
    Solutions must account for every peak in the mean.
    """
    means.sort()
    means = [round(x, 5) for x in means]

    #we want to find everything over 10%
    accounted_points = [x for x in means if x > 0.05]
    kept_solutions = []
    for solution in solution_space:
        #print(solution)
        keep = True
        combination_solutions, combination_sums = generate_combinations(len(solution), solution)
        combination_sums = [round(x,5) for x in combination_sums]
        combination_sums = np.array(combination_sums)
        for mean in accounted_points:
            idx = (np.abs(combination_sums - mean)).argmin()
            match = combination_sums[idx]
            if abs(match - mean) > 0.02:
                keep = False
                break
        if keep:
            #print(solution)
            kept_solutions.append(solution)
    return(kept_solutions)        
              
def expand_for_duplicates(solution_space, threshold=0.03):
    new_solution_space = []    
    for solution in solution_space:
        summation = sum(solution)
        solution = [round(x,5) for x in list(solution) if x > threshold]
        diff = 1-summation
         
        if diff < 0.97 and diff > 0:
            solution, include = find_duplicate(solution)
            if include:
                new_solution_space.append(solution)
        else:
            new_solution_space.append(solution)
    return(new_solution_space)


def parallel_train_models(solutions, training_data, freq_precision):
    #all_saved_solutions, all_final_points, all_models, all_model_log_likelihood, all_conflict_dicts, maximum_likelihood_points \
    code = Parallel(n_jobs=20)(delayed(train_models)(solution, training_data, freq_precision) for solution in solutions)
    all_saved_solutions = [x[0] for x in code]
    all_final_points = [x[1] for x in code]
    all_models = [x[2] for x in code]
    all_model_log_likelihood = [x[3] for x in code]
    all_conflict_dicts = [x[4] for x in code]
    maxmimum_likelihood_points = [x[5] for x in code]
    return(all_saved_solutions, all_final_points, all_models, all_model_log_likelihood, all_conflict_dicts, maxmimum_likelihood_points)

def train_models(solution, training_data, freq_precision): 
    tmp_solution = [x for x in solution if x > 0.03]
    other_point = round(sum([x for x in solution if x <= 0.03]), freq_precision)
    if other_point > 0.01:
        tmp_solution.append(other_point)
    solution = tmp_solution
    n = len(solution)
    combination_solutions, combination_sums = generate_combinations(n, solution)
    combination_sums = [round(x, freq_precision) for x in combination_sums]
    combination_solutions, combination_sums = filter_combinations(combination_solutions, combination_sums)
    conflict_dict = possible_conflicts(combination_sums, combination_solutions, solution)       
    final_points = []
    for j, tp in enumerate(combination_sums):
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
    gx = GMM(k=len(final_points), dim=1, init_mu = final_points_expand)
    gx.init_em(training_data)
    num_iters = 100
    log_likelihood = []
    for e in range(num_iters):
        gx.e_step()
        gx.m_step()

    ll = gx.log_likelihood(training_data)
    log_likelihood.append(ll)
    assignment, score, all_scores = gx.score(training_data)
    sll = [np.log(x) for x,y in zip(score, training_data)]
    major_cluster = 0
    second_cluster = 0 
    for a, s, ass, td  in zip(assignment, score, all_scores, training_data):
        point = a[0]
        idx = combination_sums.index(point)
        sol = combination_solutions[idx]
        if solution[0] in sol:
            major_cluster += np.log(s)
        if solution[1] in sol:
            second_cluster += np.log(s)
    """ 
    if DEBUG:
        print("major cluster", major_cluster, "second cluster", second_cluster)
        print("sll", sum(sll))
        print("final points", len(final_points))
        print("solution length", len(solution))
        print("i", i, "solution", solution, variants_file.split("/")[2])
        print("log likelihood", round(log_likelihood[-1],4), "\n")
    """
    return(solution, final_points, gx, log_likelihood[-1], conflict_dict, sum(sll))

def run_model(variants_file, output_dir, output_name, primer_mismatches=None, physical_linkage_file=None, freyja_file=None):
    freq_lower_bound = 0.0001
    freq_upper_bound = 0.95
    solutions_to_train = 6000
    training_lower_bound = 0.03
    freq_precision = 3 
    text_file = os.path.join(output_dir, output_name+"_model_results.txt")
 
    if freyja_file is not None:
        gt_centers, gt_lineages = file_util.parse_freyja_file(freyja_file)
        gt_mut_dict = file_util.parse_usher_barcode(gt_lineages)
        
    if primer_mismatches is not None:
        problem_positions = file_util.parse_primer_mismatches(primer_mismatches)
        print("problem positions:", problem_positions)
    else:
        problem_positions = None
    positions, frequency, nucs, low_depth_positions, reference_positions = file_util.parse_ivar_variants_file(
            variants_file, \
            freq_precision, \
            problem_positions)
    new_frequencies = [round(x, freq_precision) for x in frequency if x > freq_lower_bound and x < freq_upper_bound]
    solution_space = []
    kde_peaks = define_kde(new_frequencies)
    kde_peaks.sort(reverse=True)
    print("kde peaks", kde_peaks, len(kde_peaks))
    """
    To acheive a KDE that recognizes low frequencies, we must set the linspace such that it returns many local maxima, however often these values are repetitive. A simple linear clustering model (KMeans) helps refine the kde peaks to summarize the local maxima in the frequencies.
    """
    kde_reshape = np.array(kde_peaks).reshape(-1,1)
    cluster_model = GaussianMixture(n_components=30)
    cluster_model.fit(kde_reshape)
    refined_kde_peaks = np.array(cluster_model.means_)
    refined_kde_peaks = list(np.squeeze(refined_kde_peaks))
    refined_kde_peaks.sort(reverse=True)
    refined_kde_peaks = [round(x, freq_precision) for x in refined_kde_peaks]
    print("original refined", refined_kde_peaks)
    print("gt_centers", gt_centers)
    
    aic_list = []
    aic_means = []
    training_data = np.array([x for x in new_frequencies if x > training_lower_bound])
    tmp_reshape = np.array(training_data).reshape(-1,1)
    n_clusters = list(np.arange(2, 30))
   
    for n_val in n_clusters:
        gx = GaussianMixture(n_components = n_val)
        gx.fit(tmp_reshape)
        aic = gx.aic(tmp_reshape)
        aic_list.append(aic)
        aic_means.append(np.squeeze(gx.means_))

    idx = aic_list.index(min(aic_list))
    n = n_clusters[idx]
    points_account_for = aic_means[idx]
    print(n, points_account_for)
    points_under_50 = [x for x in points_account_for if x < 0.5]
    maximum_individual = len(points_under_50)
    if maximum_individual < 11:
        upper_bound = maximum_individual + 1
    else:
        upper_bound = 11

    n_clusters = list(np.arange(2, upper_bound))
    print(n_clusters)
    solution_space = []    
    #THIS NEEDS TO BE CHANGED TO BE ESTIMATED IN ADVANCE
    for n in n_clusters:
        n_solution = create_solution_space(refined_kde_peaks, n)
        solution_space.extend(n_solution)

    print("length of solution space", len(solution_space))
    new_solution_space = collapse_solution_space(solution_space, n_clusters=solutions_to_train) 
    print("length of collapse solution space", len(new_solution_space))
    sol = eliminate_solutions(new_solution_space, points_account_for)
    new_solution_space = sol
    n_count = [0.0] * len(n_clusters)
    for s in new_solution_space:
        i = len(s)
        try:
            idx = n_clusters.index(i)
            n_count[idx] += 1
        except:
            continue
        #print(s)
    
    zipped = list(zip(n_count, n_clusters))
    zipped.sort(reverse=True)
    n_count, n_clusters = zip(*zipped)
    total_count = sum(n_count)
    keep_lengths = []
    for a,b in zip(n_clusters, n_count):
        if b/total_count > 0.20:
            keep_lengths.append(a)
    print("looking for solutions of ", keep_lengths, "lengths...")
    keep_solutions = []
    for sol in new_solution_space:
        if len(sol) in keep_lengths:
            keep_solutions.append(sol)
    
    percent = 1.0
    print("total of %s solutions..." %(len(keep_solutions)))
    explore = int(len(keep_solutions) * percent)
    random.shuffle(keep_solutions)
    print("exploring %s percent of the solution space which is %s solutions..." %(str(percent*100), str(explore)))
    new_solution_space = keep_solutions[:explore]

    """
    G = file_util.parse_physical_linkage_file(physical_linkage_file, new_positions, \
            new_frequencies, \
            new_nucs)

    solution_space = []
    new_solution_space, frequencies_remove = apply_physical_linkage(solution_space, \
            G, \
            kde_peaks, \
            new_positions, new_nucs, new_frequencies)
    sys.exit(0)
    """

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
   
    #we really only need to fit the model for data over 3% 
    training_data = np.array([x for x in new_frequencies if x > training_lower_bound])
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
    all_saved_solutions, all_final_points, all_models, all_model_log_likelihood, all_conflict_dicts, maximum_likelihood_points \
        = parallel_train_models(new_solution_space, training_data, freq_precision)

    #get the best model
    sorted_scores = copy.deepcopy(maximum_likelihood_points)
    sorted_log_likelihoods = copy.deepcopy(all_model_log_likelihood)
    zipped = list(zip(sorted_scores, sorted_log_likelihoods))
    zipped.sort(reverse=True)
    sorted_scores, sorted_log_likelihoods = zip(*zipped)
    highest_score = sorted_scores[0]
    for i,(score, ll) in enumerate(zip(sorted_scores, sorted_log_likelihoods)):
        loc = all_model_log_likelihood.index(ll)
        print(i, "score", score, all_saved_solutions[loc], "ll", round(all_model_log_likelihood[loc],4), len(all_saved_solutions[loc]))

    loc_best_model = maximum_likelihood_points.index(highest_score)
    log_likelihood = all_model_log_likelihood[loc_best_model]
    solution = all_saved_solutions[loc_best_model]
    final_points = all_final_points[loc_best_model]
    gx = all_models[loc_best_model]
    conflict_dict = all_conflict_dicts[loc_best_model]
    assignments, scores, all_likelihoods  = gx.score(training_data)
     
    combination_solutions, combination_sums = generate_combinations(len(solution), solution)
    combination_sums = [round(x,freq_precision) for x in combination_sums]
    combination_solutions, combination_sums = filter_combinations(combination_solutions, combination_sums)
    print("best solution", solution)
    autoencoder_dict = {}
    print("%s universal mutations found..." %len(universal_mutations))
    for s in solution:
        autoencoder_dict[s] = universal_mutations
    print(universal_mutations) 
    save_variants = []
    save_scores = []

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
        #if variant == "22599A":
        #    for sig, p, ass in zip(gx.sigma, gx.mu, all_score):
        #        if ass > 0:
        #            print(p, ass, sig)
        for individual in combination_solutions[location]:
            tmp_list = copy.deepcopy(autoencoder_dict[individual])
            tmp_list.append(variant)
            autoencoder_dict[individual] = tmp_list
    print(variants_file)
   
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
                print("\nextra", [item for item in value if item not in check3 and "-" not in item and item[:-1] not in low_depth_positions])
            counter += 1
        
    print("low depth positions:", np.unique(low_depth_positions))
    tmp_dict = {"autoencoder_dict":autoencoder_dict, "log_likelihood": log_likelihood, "problem_positions":problem_positions, "low_depth_positions":low_depth_positions, "variants":save_variants, "scores":save_scores, "conflict_dict": conflict_dict}
    with open(text_file, "a") as bfile:
        bfile.write(json.dumps(tmp_dict))
        bfile.write("\n")    
    return(0)

if __name__ == "__main__":
    main()
