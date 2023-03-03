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

def generate_combinations(n, solution):
    combination_solutions = []
    combination_sums = []
    for L in range(n + 1):
        for subset in itertools.combinations(solution, L):
            if len(subset) < 1:
                continue
            combination_solutions.append(list(subset))
            combination_sums.append(round(sum(subset),4))
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

def define_kde(frequencies, bw=0.01, round_decimal=3):
    x = np.array(frequencies)
    x_full = x.reshape(-1, 1)
    eval_points = np.linspace(np.min(x_full), np.max(x_full))
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
    """
    Parameters
    ----------
    solution_space : list
    G : networkx
    kde_peaks : list
    positions : list
    nucs : list
    frequencies : list
    Eliminate kde peaks from the solution space. 
    Given the solution space, use physical linkage and kde to eliminate possible
    solutions. Logically if we have two variants that look like they come from different kde 
    peaks, and are decently far apart > 0.05 (and don't belong to the noise peak) 
    we know that the smaller must be a component of the larger, and therefore the larger peak
    is not a member of the solution.
    """
    min_value = min(kde_peaks)
    arr = np.asarray(kde_peaks)
    variants = [str(n) + "_" + str(p) for n,p in zip(nucs, positions)]
    remove_solutions = []
    frequencies_remove = []
    linked = []
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
                if int(count['count']) < 3:
                    continue
                
                idx = (np.abs(arr - ov_freq)).argmin()
                assigned1 = arr[idx]
                idx = (np.abs(arr - freq)).argmin()
                assigned2 = arr[idx]

                #if abs(assigned1 - ov_freq) > 0.03:
                #    continue
                #if abs(assigned2 - freq) > 0.03:
                #    continue

                if assigned1 == min_value or assigned2 == min_value:
                    continue
                if assigned1 == assigned2:
                    continue
                
                diff = abs(ov_freq - freq)
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
    #we don't accidently want to include more than one noise peak
    #kde_peaks = [x for x in kde_peaks if x > 0.05]
    overlap_sets = []
    flat = []
    for subset in itertools.combinations(kde_peaks, n):
        subset = list(subset)
        #subset.append(0.03) #manually set noise peak
        overlap_sets.append(subset)
    
    for os in overlap_sets:
        noise = 0.03
        if np.sum(os)-noise < total_value+0.05 and np.sum(os)-noise > total_value-0.05:
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
                if peak < min(os) or peak >= 0.97:
                    continue
                if peak in os:
                    continue
                for item in combos:
                    #print(item, peak)
                    if(abs(item-peak) <= 0.03):
                        found = True
                for item in os:
                    #print(item, peak)
                    if(abs(item-peak) <= 0.03):
                        found = True
                if found is False:
                    #print(os)
                    #print(combos)
                    #print(peak)
                    pass
                    #outer_found = False
            if outer_found is True:
                #print("found", os)
                flat.append(os)
    return(flat)

def run_model(variants_file, output_dir, output_name, primer_mismatches, physical_linkage_file=None, freyja_file=None):
    freq_lower_bound = 0.03
    freq_upper_bound = 0.97
    freq_precision = 4
    n_clusters = [3, 4, 5, 6] 
    text_file = os.path.join(output_dir, output_name+"_model_results.txt")

    if freyja_file is not None:
        gt_centers, gt_lineages = file_util.parse_freyja_file(freyja_file)

    problem_positions = file_util.parse_primer_mismatches(primer_mismatches) 
    positions, frequency, nucs, low_depth_positions, reference_positions = file_util.parse_ivar_variants_file(
            variants_file, \
            freq_precision, \
            problem_positions)

    new_frequencies = [round(x, freq_precision) for x in frequency if x > freq_lower_bound and x < freq_upper_bound]
    solution_space = [] 
    kde_peaks = define_kde(new_frequencies)
    for n in n_clusters:
        solution_space.extend(create_solution_space(kde_peaks, n))    
    
    new_positions = [y for x,y in zip(frequency, positions) if x > freq_lower_bound and x < freq_upper_bound]
    new_nucs = [y for x,y in zip(frequency, nucs) if float(x) > freq_lower_bound and float(x) < freq_upper_bound]
    universal_mutations = [str(x)+str(y) for (x,y,z) in zip(positions, nucs, frequency) if float(z) > freq_upper_bound and str(y) != '0']
    print(len(universal_mutations), "universal mutations found")

    G = file_util.parse_physical_linkage_file(physical_linkage_file, new_positions, \
            new_frequencies, \
            new_nucs)

    new_solution_space, frequencies_remove = apply_physical_linkage(solution_space, \
            G, \
            kde_peaks, \
            new_positions, new_nucs, new_frequencies)

    filter_freq_first = [x for x in new_frequencies if x not in frequencies_remove]    

    all_models = []
    all_model_scores = []
    all_final_points = []
    
    print("solution space contains %s solutions..." %len(new_solution_space))
    print("training models...")
    #fit a model for each possible solution
    for i, solution in enumerate(new_solution_space):
        n = len(solution)
        combination_solutions, combination_sums = generate_combinations(n, solution)
        
        clustered_data, filtered_points_2 = assign_clusters(new_frequencies, combination_sums)
        final_points = []
        for cd, tp in zip(clustered_data, combination_sums):
            if tp >= 1:
                continue
            if len(cd) == 0:
                final_points.append(tp)
                continue
            final_points.append(tp)
            
        all_final_points.append(final_points)
        final_points_reshape = np.array(final_points).reshape(-1, 1)
        all_data_reshape = np.array(new_frequencies).reshape(-1, 1)

        x_data = np.array(new_frequencies).reshape(-1,1)

        #sklean implementation
        #gx = GaussianMixture(n_components=len(final_points), means_init=final_points_reshape, \
        #        max_iter=1000, n_init=1, random_state=10, init_params='k-means++') 
        
        gx = GMM(n_components=len(final_points), mean_init=final_points)
        gx.fit(new_frequencies)
        likelihoods = gx.score(new_frequencies)
        loc = likelihoods.index(min(likelihoods))
        
        print(solution, min(likelihoods), new_frequencies[loc])
        print(likelihoods)
        #print(gx.mean_vector)
        sorted_likelihood = copy.deepcopy(likelihoods)
        sorted_likelihood.sort()
        
        #print(sorted_likelihood[:3])
        #print(gx.mean_vector)   
        all_models.append(gx)
      
        #HELPME: this is a weak point, need to be more emperical
        if str(sorted_likelihood[0]) != 'nan':
            all_model_scores.append(sorted_likelihood[0])
        else:
            all_model_scores.append(0)
    
    #TESTLINE
    sorted_scores = copy.deepcopy(all_model_scores)
    sorted_scores.sort(reverse=True)
    for score in sorted_scores[:5]:
        print(score)
        print(new_solution_space[all_model_scores.index(score)])
    
    if freyja_file is None:
        highest_score = max(all_model_scores)
    else:
        sorted_model_scores = copy.deepcopy(all_model_scores)
        sorted_model_scores.sort(reverse=True)

        for score in sorted_model_scores:
            loc = all_model_scores.index(score)
            tmp_sol = new_solution_space[loc]
            valid = True
            #handle the case where they're the same length 
            if len(tmp_sol) == len(gt_centers):
                gt_centers.sort(reverse=True)
                tmp_sol.sort(reverse=True)
                for g, t in zip(gt_centers, tmp_sol):
                    if abs(g - t) > 0.05:
                        valid = False
                if valid is True:
                    highest_score = score
            if len(tmp_sol) != len(gt_centers):
                print("You haven't tested this code yet.")
                tmp_sol.sort(reverse=True)
                gt_centers.sort(reverse=True)
                if tmp_sol[0] > gt_centers[0]:
                    print("error in solution calling")
                    highest_score = max(all_model_scores)
                else:
                    highest_score = max(all_model_scores)

            if valid is True:
                break

    loc_best_model = all_model_scores.index(highest_score)
    solution = new_solution_space[loc_best_model]
    final_points = all_final_points[loc_best_model]
    final_points_reshape = np.array(final_points).reshape(-1, 1)
    combination_solutions, combination_sums = generate_combinations(len(solution), solution)
    
    #retrain with higher number of iterations
    gx = all_models[loc_best_model]
    likelihood = gx.score(new_frequencies)
    clusters = gx.predict(new_frequencies)  

    print("minimum likelihood", min(likelihood)) 
    print("highest model score", highest_score)
    print("final_points", final_points)
    print("means", gx.mean_vector) 
    print("solution", solution)
    hard_to_classify = []
    autoencoder_dict = {}
    for s in solution:
        autoencoder_dict[s] = universal_mutations
    
    for i, (cluster, point, score) in enumerate(zip(clusters, new_frequencies, likelihood)):
        arr = np.asarray(combination_sums)
        location = (np.abs(arr - cluster)).argmin()
       
        nuc = str(new_nucs[i])
        pos = str(new_positions[i])
        
        variant = pos+nuc

        #this is true for the ref nucs 
        if nuc == '0':
            continue
        
        #print("combo sum:", arr[location], "mean:", mean_label, \
        #        "point:", point, \
        #        "combo solution:", \
        #        combination_solutions[location], "var:", variant)
 
        for individual in combination_solutions[location]:
            tmp_list = copy.deepcopy(autoencoder_dict[individual])
            tmp_list.append(variant)
            autoencoder_dict[individual] = tmp_list

    tmp_dict = {"autoencoder_dict":autoencoder_dict, "score": min(likelihood)}
    """
    for k, v in autoencoder_dict.items():
        print(k, v[len(universal_mutations):])
    """ 
    with open(text_file, "a") as bfile:
        bfile.write(json.dumps(tmp_dict))
        bfile.write("\n")
    
if __name__ == "__main__":
    main()
