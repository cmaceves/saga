"""
Functionality for running the model.
"""
import os
import sys
import numpy as np
import pandas as pd

from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity

import file_util

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


def main():
    pass

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
    positions, frequency, nucs, low_depth_positions, reference_positions = file_utilparse_ivar_variants_file(
            variants_file, \
            problem_positions)

    new_frequencies = [round(x, freq_precision) for x in frequency if x > freq_lower_bound and x < freq_upper_bound]
    solution_space = [] 
    kde_peaks = define_kde(new_frequencies)
    for n in n_clusters:
        solution_space.extend(create_solution_space(kde_peaks, n))    

    print(solution_space)
    sys.exit(0)
     
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
    
    for i, solution in enumerate(new_solution_space):
        n = len(solution)
        combination_solutions = []
        combination_sums = []
        total_points = []
        for L in range(n + 1):
            for subset in itertools.combinations(solution, L):
                if len(subset) < 1:
                    continue
                combination_solutions.append(list(subset))
                combination_sums.append(round(sum(subset),4))
        
        clustered_data = assign_clusters([filter_freq_first], combination_sums)
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
        if score_samples[0] > -1.25:
            print(i, solution)
            print(score_samples[:5])
            #print(gx.means_)
            print(np.mean(score_samples))
        
        if i == 32:
            #print(universal_mutations)
            hard_to_classify = []
            autoencoder_dict = {}
            frequency_dict = {}
            for s in solution:
                #autoencoder_dict[s]  = []
                autoencoder_dict[s] = universal_mutations
                frequency_dict[s] = []

            for label, point, score in zip(labels, new_frequencies, score_samples):
                mean_label = means[label]
                arr = np.asarray(combination_sums)
                location = (np.abs(arr - mean_label)).argmin()
                if score < 0.5 and abs(combination_sums[location] - point) > 0.05:
                    print(combination_solutions[location], point, label, score)
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
            #print(autoencoder_dict)
            print(solution)
            #continue
            #sys.exit(0)
            
            tmp_dict = {"autoencoder_dict":autoencoder_dict, "frequency_dict":frequency_dict}
            with open(text_file, "a") as bfile:
                bfile.write(json.dumps(tmp_dict))
                bfile.write("\n")
             
            #hard_classify_kde = get_kde(hard_to_classify, 0.1)  
            #print(hard_classify_kde)
            #sol = create_solution_space(hard_classify_kde, 2, max(solution))
            #print(sol)
    sys.exit(0)
 
if __name__ == "__main__":
    main()
