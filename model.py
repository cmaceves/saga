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
from networkx.algorithms.components import connected_components
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.neighbors import KernelDensity

import file_util
import math_util

from gmm import GMM

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
        #if j % 100000 == 0:
        #    print(j)
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

    print(len(filter_1_sol))
    """
    print(filter_1_sol[-1])
    print(filter_1_sol[-2])
    print(filter_1_sum[-1])
    print(filter_1_sum[-2])
    """
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
        print("L", L, "out of", n+1)
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
            
            sum_val = round(sum(subset), 5)
            combination_solutions.append(subset)
            combination_sums.append(sum_val)
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

def define_kde(frequencies, bw=0.0001, round_decimal=4, num=1000):
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

    kde_peaks = list(np.unique(peak))
    return(kde_peaks)

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
    print("creating solution space...")

    overlap_sets = []
    flat = []
    for subset in itertools.combinations(kde_peaks, n):
        subset = list(subset)
        overlap_sets.append(subset)
    print("parsing down solution space...")    
    for os in overlap_sets:
        if np.sum(os) < total_value+0.03 and np.sum(os) > total_value-0.03:
            os = list(os)
            combos = []
            #recombine again
            for L in range(len(os) + 1):
                for subset in itertools.combinations(os, L):
                    if 0 in list(subset):
                        continue
                    if len(list(subset)) < 2:
                        continue
                    combos.append(round(sum(list(subset)), 3))
            outer_found = True
            for peak in kde_peaks:
                found = False 
                #the peak could be a noise peak
                if peak < min(os) or peak >= 0.98:
                    continue
                if peak in os:
                    continue
                for item in combos:
                    if(abs(item-peak) <= 0.03):
                        found = True
                for item in os:
                    if(abs(item-peak) <= 0.03):
                        found = True
                if found is False:
                    pass
            if outer_found is True:
                flat.append(os)
    return(flat)

def run_model(variants_file, output_dir, output_name, primer_mismatches, physical_linkage_file=None, freyja_file=None):
    freq_lower_bound = 0.01
    freq_upper_bound = 0.95
    freq_precision = 5 
    text_file = os.path.join(output_dir, output_name+"_model_results.txt")

    
    if freyja_file is not None:
        gt_centers, gt_lineages = file_util.parse_freyja_file(freyja_file)
        gt_mut_dict = file_util.parse_usher_barcode(gt_lineages)
    problem_positions = file_util.parse_primer_mismatches(primer_mismatches)
    print("problem positions:", problem_positions)
    #problem_positions = None
    positions, frequency, nucs, low_depth_positions, reference_positions = file_util.parse_ivar_variants_file(
            variants_file, \
            freq_precision, \
            problem_positions)
    new_frequencies = [round(x, freq_precision) for x in frequency if x > freq_lower_bound and x < freq_upper_bound]
    solution_space = [] 
    kde_peaks = define_kde(new_frequencies)
    
    #TESTLINE
    check2 = gt_mut_dict[gt_lineages[0]]
    check = gt_mut_dict[gt_lineages[1]]
    check3 = gt_mut_dict[gt_lineages[2]]
    """
    sys.exit(0)
    for n in n_clusters:
        solution_space.extend(create_solution_space(kde_peaks, n))    
    """
    new_positions = [y for x,y in zip(frequency, positions) if x > freq_lower_bound and x < freq_upper_bound]
    new_nucs = [y for x,y in zip(frequency, nucs) if float(x) > freq_lower_bound and float(x) < freq_upper_bound]
    universal_mutations = [str(x)+str(y) for (x,y,z) in zip(positions, nucs, frequency) if float(z) > freq_upper_bound and str(y) != '0']
    #print("universal mutations:", universal_mutations)
    print(len(universal_mutations), "universal mutations found")
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
    all_model_scores = []
    all_final_points = []
    new_solution_space = [gt_centers]
    
    all_combination_sums = []
    all_combination_solutions = []
    all_junk_points = []
    impossible_solutions = []
    print("solution space contains %s solutions..." %len(new_solution_space))
    print("training models...")
    print(new_solution_space)
    #fit a model for each possible solution
    for i, solution in enumerate(new_solution_space):
        tmp_sol = [x for x in solution if x > 0.01]
        tmp_sol_low = [x for x in solution if x < 0.01]
        solution = tmp_sol
        other = sum(tmp_sol_low)
        n = len(solution)

        #looking for places we might not be able to see individual pops
        individual_overlaps = determine_close_individuals(solution)
        print(solution)
        solution.append(other)
        n = len(solution)
        combination_solutions, combination_sums = generate_combinations(n, solution)
        combination_solutions, combination_sums = filter_combinations(combination_solutions, combination_sums)
        print(len(combination_sums))               
        all_junk_points.append(other)
        all_combination_sums.append(combination_sums)
        all_combination_solutions.append(combination_solutions)
        clustered_data, filtered_points_2 = assign_clusters(new_frequencies, combination_sums)
        final_points = []
        for i, (cd, tp) in enumerate(zip(clustered_data, combination_sums)):
            if i < len(solution):
                pass
            elif len(combination_solutions[i]) == 1:
                pass
            elif tp >= 1.0:
                continue
            elif (tp / len(combination_solutions[i])) < 0.05:
                  #print(tp, combination_solutions[i])
                  continue
            #elif len(cd) == 0:
            #    continue
            final_points.append(tp)
        #print("initial mean length:", len(final_points))
        #print(final_points)
        print(len(final_points), len(combination_solutions))
        #sys.exit(0)
        all_final_points.append(final_points)
      
        gx = GMM(n_components=len(final_points), mean_init=final_points)
        gx.fit(new_frequencies)
        likelihoods, all_likelihoods = gx.score(new_frequencies)

        sorted_likelihood = copy.deepcopy(likelihoods)
        deep_freq = copy.deepcopy(new_frequencies)
        zipped = list(zip(sorted_likelihood, deep_freq, all_likelihoods))
        zipped.sort()
        sorted_likelihood, deep_freq, all_likelihoods = zip(*zipped)
        min_likelihood = sorted_likelihood[0]
        
        mod_means = [round(x,5) for x in list(gx.mean_vector)]
        again_final_points = []
        all_final_points.append(final_points)
        all_models.append(gx)
        #all_model_scores.append(min_likelihood)
        all_model_scores.append(np.mean(sorted_likelihood))

    #TESTLINE
    sorted_scores = copy.deepcopy(all_model_scores)
    sorted_scores.sort(reverse=True)
    for score in sorted_scores[:5]:
        loc = all_model_scores.index(score)
        print(score, new_solution_space[loc])

    highest_score = max(all_model_scores)

    loc_best_model = all_model_scores.index(highest_score)
    solution = new_solution_space[loc_best_model]
    final_points = all_final_points[loc_best_model]

    combination_solutions = all_combination_solutions[loc_best_model]
    combination_sums = all_combination_sums[loc_best_model]
    junk_point = all_junk_points[loc_best_model]
    other_solution = [x for x in solution if x > 0.03]
    throwaway = [x for x in solution if x < 0.03]
    other_solution.append(junk_point)
    gx = all_models[loc_best_model]
    likelihoods, all_likelihoods  = gx.score(new_frequencies)
    
    print("minimum likelihood", min(likelihoods)) 
    print("highest model score", highest_score)
    print("solution", solution)
    print("means", gx.mean_vector)    
    autoencoder_dict = {}
    tmp_universal_mutations = []
    for um in universal_mutations:
        tum = um[-1] + "_" + um[:-1]
        if tum not in reference_positions:
            tmp_universal_mutations.append(um)
    universal_mutations = tmp_universal_mutations

    for s in solution:
        autoencoder_dict[s] = universal_mutations
    
    save_variants = []
    save_scores = []
    for i, (point, all_like) in enumerate(zip(new_frequencies, all_likelihoods)):
        if point < 0.03:
            continue
        nuc = str(new_nucs[i])
        if nuc == '0':
            continue

        pos = str(new_positions[i])
        variant = pos+nuc
        if str(nuc) + "_"+ str(pos) in reference_positions:
            continue
        tmp_means = copy.deepcopy(gx.mean_vector)
        adjusted_scores = []
        passed = False
        for t, al in zip(tmp_means, all_like):
            loc = combination_sums.index(t)
            tmp_sol = combination_solutions[loc]
            total = len(tmp_sol)
            if junk_point in tmp_sol:
                total -= 1
                total += len(throwaway)
            #total += len([x for x in tmp_sol if x < 0.03]) * 3
            new_score = al / (total)
            adjusted_scores.append(new_score)
            #if point == 0.19698 and new_score > 0.0001:
            #    print(t, point, al, combination_solutions[combination_sums.index(t)], new_score, total)
 
        zipped = list(zip(adjusted_scores, tmp_means))
        zipped.sort(reverse=True)
        adjusted_scores, tmp_means = zip(*zipped)
        location = -1
        score = -1
        for t, al in zip(tmp_means, adjusted_scores):
            if ((abs(point-t) < (point * 0.05)) or (abs(point-t) < 0.01)) and al > 0.0:
                if passed is False:
                    location = combination_sums.index(t)
                    score = al
                    passed = True
            #if point == 0.844:
            #    print(t, point, al, combination_solutions[combination_sums.index(t)])
    
        
        if location != -1:
            print("\nlocation", location, combination_sums[location], combination_solutions[location])
            print("point:", point)
            print(variant, combination_solutions[location])
            save_variants.append(variant)
            save_scores.append(score)
        else:
            print("\nFAILED", point, variant)
            for t, al in zip(tmp_means[:5], adjusted_scores[:5]):
                if al > 0:
                    print(t, al)
            continue
        
        for individual in combination_solutions[location]:
            if individual == junk_point:
                continue
            tmp_list = copy.deepcopy(autoencoder_dict[individual])
            tmp_list.append(variant)
            autoencoder_dict[individual] = tmp_list
    counter = 0
    check.sort()
    check2.sort()
    check3.sort()
    check_fail = []
    check2_fail = []
    check3_fail = []
    print(variants_file)
    for key, value in autoencoder_dict.items(): 
        value.sort()
        if counter == 0:
            check_fail = [item for item in check if item not in value]
            print("\n", key, check_fail)
            print("extra", [item for item in value if item not in check and "-" not in item])
        elif counter == 1:
            check2_fail = [item for item in check2 if item not in value]
            print("\n", key, check2_fail)
            print("extra", [item for item in value if item not in check2 and "-" not in item])
        elif counter == 2:
            check3_fail = [item for item in check3 if item not in value]
            print("\n", key, check3_fail)
            print("\nextra", [item for item in value if item not in check3 and "-" not in item])
        counter += 1
    tmp_dict = {"autoencoder_dict":autoencoder_dict, "score": min(likelihoods), "problem_positions":problem_positions, "low_depth_positions":low_depth_positions, "variants":save_variants, "scores":save_scores}
    with open(text_file, "a") as bfile:
        bfile.write(json.dumps(tmp_dict))
        bfile.write("\n")
    
    return(0)

if __name__ == "__main__":
    main()
