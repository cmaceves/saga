"""
Functionality for running the model.
"""
import os
import sys
import copy
import json
import itertools

import numpy as np
import pandas as pd
import networkx as nx
from scipy import spatial
from networkx.algorithms.components import connected_components
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.cluster import KMeans
from sklearn.neighbors import KernelDensity

from line_profiler import LineProfiler

import file_util
import math_util

from other_gmm import GMM

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

def define_kde(frequencies, bw=0.0001, round_decimal=4, num=2000):
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
    upper_bound = total_value - 0.05
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
        solution = [round(x,5) for x in list(solution) if x > threshold]
        new_solution_space.append(solution)
    return(new_solution_space)

def run_model(variants_file, output_dir, output_name, primer_mismatches=None, physical_linkage_file=None, freyja_file=None):
    freq_lower_bound = 0.0001
    freq_upper_bound = 0.95
    solutions_to_train = 20
    training_lower_bound = 0.03
    freq_precision = 5 
    text_file = os.path.join(output_dir, output_name+"_model_results.txt")
 
    if freyja_file is not None:
        gt_centers, gt_lineages = file_util.parse_freyja_file(freyja_file)
        gt_mut_dict = file_util.parse_usher_barcode(gt_lineages)
        #lets try and see if this file is even reasonable to try and parse
        
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
    cluster_model = GaussianMixture(n_components=25)
    cluster_model.fit(kde_reshape)
    refined_kde_peaks = np.array(cluster_model.means_)
    refined_kde_peaks = list(np.squeeze(refined_kde_peaks))
    refined_kde_peaks.sort(reverse=True)
    refined_kde_peaks = [round(x,5) for x in refined_kde_peaks]
    print("original refined", refined_kde_peaks)

    #TESTLINE FOR PRINTING DOWNSTREAM
    check = gt_mut_dict[gt_lineages[0]]
    check2 = gt_mut_dict[gt_lineages[1]]
    check3 = gt_mut_dict[gt_lineages[2]]
    check4 = gt_mut_dict[gt_lineages[3]]
    check5 = gt_mut_dict[gt_lineages[4]]
    check6 = gt_mut_dict[gt_lineages[5]]
    
    #we can only reasonably pass like 25 peaks to the solution space 
    print("gt_centers", gt_centers)
    n_clusters = [3,4,5,6,7,8,9, 10]
    for n in n_clusters:
        """
        lp = LineProfiler()
        lp_wrapper = lp(create_solution_space)
        lp_wrapper(refined_kde_peaks, n)
        lp.print_stats()
        sys.exit(0)
        """
        n_solution = create_solution_space(refined_kde_peaks, n)
        solution_space.extend(n_solution)  
    new_solution_space = collapse_solution_space(solution_space, n_clusters=solutions_to_train) 
   
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
    all_models = []
    all_model_log_likelihood = []
    all_final_points = []    
    all_conflict_dicts = []    
    all_minimum_scores = []
    print("solution space contains %s solutions..." %len(new_solution_space))
    print("training models...")
   
    #we really only need to fit the model for data over 3% 
    training_data = np.array([x for x in new_frequencies if x > training_lower_bound])
    training_data = np.expand_dims(training_data, axis=1)
    new_positions = [y for x,y in zip(frequency, positions) if x > training_lower_bound and x < freq_upper_bound]
    new_nucs = [y for x,y in zip(frequency, nucs) if float(x) > training_lower_bound and float(x) < freq_upper_bound]
    universal_mutations = [str(x)+str(y) for (x,y,z) in zip(positions, nucs, frequency) if float(z) > freq_upper_bound and str(y) != '0']
    
    #fit a model for each possible solution 
    for i, solution in enumerate(new_solution_space):      
        n = len(solution)
        combination_solutions, combination_sums = generate_combinations(n, solution)
        combination_sums = [round(x,5) for x in combination_sums]
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
            final_points.append(tp)

        final_points_expand = np.expand_dims(final_points, axis=1)
        gx = GMM(k=len(final_points), dim=1, init_mu = final_points_expand)
        gx.init_em(training_data)
        num_iters = 30
        log_likelihood = [gx.log_likelihood(training_data)]
        for e in range(num_iters):
            gx.e_step()
            gx.m_step()
            ll = gx.log_likelihood(training_data)
            log_likelihood.append(ll)
            #print("Iteration: {}, log-likelihood: {:.4f}".format(e+1, log_likelihood[-1]))
        assignment, score, all_scores = gx.score(training_data)

        minimum_score = min(score) 
        #save relevant information
        all_final_points.append(final_points)
        all_models.append(gx)
        all_model_log_likelihood.append(log_likelihood[-1])
        all_conflict_dicts.append(conflict_dict)
        all_minimum_scores.append(minimum_score)
        print("i", i, "solution", solution)
        print("log likelihood", log_likelihood[-1], "minimum_score", minimum_score)


    #here we do something spciy. we look at the top few scores within 0.02 and see which then has the highest overall
    sorted_scores = copy.deepcopy(all_minimum_scores)
    sorted_log_likelihoods = copy.deepcopy(all_model_log_likelihood)
    zipped = list(zip(sorted_scores, sorted_log_likelihoods))
    zipped.sort(reverse=True)
    sorted_scores, sorted_log_likelihoods = zip(*zipped)
    highest_score = sorted_scores[0]
    for i,(score, ll) in enumerate(zip(sorted_scores, sorted_log_likelihoods)):
        if sorted_scores[0] - score < 0.02 and ll - sorted_log_likelihoods[0] > 75:
            highest_score = score
        loc = all_minimum_scores.index(score)
        if i < 5:
            print(score, new_solution_space[loc], all_model_log_likelihood[loc])
    
    sys.exit(0)

    loc_best_model = all_minimum_scores.index(highest_score)
    log_likelihood = all_model_log_likelihood[loc_best_model]
    solution = new_solution_space[loc_best_model]
    final_points = all_final_points[loc_best_model]
    gx = all_models[loc_best_model]
    conflict_dict = all_conflict_dicts[loc_best_model]
    assignments, scores, all_likelihoods  = gx.score(training_data)
     
    combination_solutions, combination_sums = generate_combinations(len(solution), solution)
    combination_sums = [round(x,5) for x in combination_sums]
    combination_solutions, combination_sums = filter_combinations(combination_solutions, combination_sums)

    print("best solution", solution)
    autoencoder_dict = {}
    tmp_universal_mutations = []

    for um in universal_mutations:
        tum = um[-1] + "_" + um[:-1]
        if tum not in reference_positions:
            tmp_universal_mutations.append(um)
    universal_mutations = tmp_universal_mutations
    print("%s universal mutations found..." %len(universal_mutations))
    for s in solution:
        autoencoder_dict[s] = universal_mutations
    
    save_variants = []
    save_scores = []

    print("nucs:", len(new_nucs), "pos:", len(new_positions), "training data:", len(training_data))
    for i, (point, assignment, score) in enumerate(zip(training_data, assignments, scores)):
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
        print(variant, point, assignment, score)
        print(conflict_dict[assignment])
        for individual in combination_solutions[location]:
            tmp_list = copy.deepcopy(autoencoder_dict[individual])
            tmp_list.append(variant)
            autoencoder_dict[individual] = tmp_list

    counter = 0
    check.sort()
    check2.sort()
    check3.sort()
    check4.sort()
    check5.sort()
    check6.sort()

    check_fail = []
    check2_fail = []
    check3_fail = []
    check4_fail = []
    check5_fail = []
    check6_fail = []
    print(variants_file)
    
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
        elif counter == 3:
            check4_fail = [item for item in check4 if item not in value and item[:-1] not in low_depth_positions]
            print("\n", key, check4_fail)
            print("\nextra", [item for item in value if item not in check4 and "-" not in item and item[:-1] not in  low_depth_positions])
        elif counter == 4:
            check5_fail = [item for item in check5 if item not in value and item[:-1] not in low_depth_positions]
            print("\n", key, check5_fail)
            print("\nextra", [item for item in value if item not in check5 and "-" not in item and item[:-1] not in  low_depth_positions])
        elif counter == 5:
            check6_fail = [item for item in check6 if item not in value and item[:-1] not in low_depth_positions]
            print("\n", key, check6_fail)
            print("\nextra", [item for item in value if item not in check6 and "-" not in item and item[:-1] not in  low_depth_positions])
           
        counter += 1
    print("low depth positions:", np.unique(low_depth_positions))
    tmp_dict = {"autoencoder_dict":autoencoder_dict, "log_likelihood": log_likelihood, "problem_positions":problem_positions, "low_depth_positions":low_depth_positions, "variants":save_variants, "scores":save_scores, "conflict_dict": conflict_dict}
    with open(text_file, "a") as bfile:
        bfile.write(json.dumps(tmp_dict))
        bfile.write("\n")    
    return(0)

if __name__ == "__main__":
    main()
