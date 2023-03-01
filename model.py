"""
Functionality for running the model.
"""
import os
import sys
import json
import itertools

import numpy as np
import pandas as pd
import networkx as nx
from networkx.algorithms.components import connected_components
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity

import file_util

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

    for n in range(len(transformed_centers)):
        cluster_points.append([])
    
    #asign points to clusters and recalculate
    for f in frequencies:
        closest_index = 0
        closest_value = 1
        for i,cc in enumerate(transformed_centers):
            dist = abs(cc-f)
            if dist < closest_value:
                closest_value = dist
                closest_index = i
        cluster_points[closest_index].append(f)

    return(cluster_points)

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
                
                if int(count['count']) < 5:
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

def run_model(variants_file, output_dir, output_name, primer_mismatches, physical_linkage_file=None):
    freq_lower_bound = 0.03
    freq_upper_bound = 0.97
    freq_precision = 4
    n_clusters = [3, 4, 5, 6] 
    text_file = os.path.join(output_dir, output_name+"_model_results.txt")
       
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
    universal_mutations = [str(x)+str(y) for (x,y,z) in zip(positions, nucs, frequency) if float(z) > freq_upper_bound]
    
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

    #fit a model for each possible solution
    for i, solution in enumerate(new_solution_space):
        n = len(solution)
        combination_solutions, combination_sums = generate_combinations(n, solution)

        clustered_data = assign_clusters(filter_freq_first, combination_sums)
        final_points = []
        for cd, tp in zip(clustered_data, combination_sums):
            if len(cd) == 0 and tp in solution:
                print("orphan")
            if len(cd) == 0:
                continue
            final_points.append(tp)
    
        final_points_reshape = np.array(final_points).reshape(-1, 1)
        all_data_reshape = np.array(new_frequencies).reshape(-1, 1)
        x_data = np.array(filter_freq_first).reshape(-1,1)
        gx = GaussianMixture(n_components=len(final_points), means_init=final_points_reshape) 
        gx.fit(x_data)
        labels = gx.predict(all_data_reshape)
        score_samples = list(np.squeeze(gx.score_samples(all_data_reshape)))
        score_samples.sort()
        means = [round(x,4) for x in list(np.squeeze(gx.means_))]
        all_models.append(gx)
        
        #this is a weak point, need to be more emperical
        all_model_scores.append(np.mean(score_samples[:3]))
    
    lowest_score = max(all_model_scores)
    loc_best_model = all_model_scores.index(lowest_score)
    best_model = all_models[loc_best_model]

    solution = new_solution_space[loc_best_model]

    combination_solutions, combination_sums = generate_combinations(len(solution), solution)
    
    hard_to_classify = []
    autoencoder_dict = {}
    frequency_dict = {}

    for s in solution:
        autoencoder_dict[s] = universal_mutations
        frequency_dict[s] = []

    for label, point, score in zip(labels, new_frequencies, score_samples):
        mean_label = means[label]
        arr = np.asarray(combination_sums)
        location = (np.abs(arr - mean_label)).argmin()
        
        #this is also a weak point, need to find a way to exclude points in the first pass
        if score < 0.75 and abs(combination_sums[location] - point) > 0.05:
            print(combination_solutions[location], "point:", point, "score:" , round(score,3))
            hard_to_classify.append(point)
        else:
            for individual in combination_solutions[location]:
                frequency_dict[individual].append(point)
                variant_location = new_frequencies.index(point)
                check_var = str(new_nucs[variant_location]) + "_" + str(new_positions[variant_location])
                if check_var in reference_positions:
                    continue
                variant = str(new_positions[variant_location]) + str(new_nucs[variant_location])
                autoencoder_dict[individual].append(variant)
    
    """
    tmp_dict = {"autoencoder_dict":autoencoder_dict, "frequency_dict":frequency_dict}
    with open(text_file, "a") as bfile:
        bfile.write(json.dumps(tmp_dict))
        bfile.write("\n")
    """
    #hard_classify_kde = get_kde(hard_to_classify, 0.1)  
    #print(hard_classify_kde)
    #sol = create_solution_space(hard_classify_kde, 2, max(solution))
    #print(sol)

if __name__ == "__main__":
    main()
