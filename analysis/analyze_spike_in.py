"""
Author: Chrissy Aceves
Email: caceves@scripps.edu
"""
import os
import ast
import sys
import copy
import json
import scipy
import pickle
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from Levenshtein import distance
from scipy import stats
from scipy.spatial.distance import cosine
from line_profiler import LineProfiler
sys.path.insert(0, "../")
from model import run_model, generate_combinations
import file_util
from generate_consensus import write_fasta

sns.set_style("whitegrid")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 12

#this is for primer pess ups 
palette = ['#66c2a5','#fc8d62','#8da0cb','#e78ac3','#a6d854','#ffd92f','#e5c494']
lineage_palette= {"A":"#e66101", "B.1.1.7":"#fdb863", "B.1.351":"#f7f7f7", "P.1":"#b2abd2", "B.1.617.2":"#5e3c99"}

#for mixture categories
categories = ["100", "95/5", "90/10", "80/20", "60/40", "50/50", \
    "33/33/33", "25/25/25/25", "20/20/20/20/20"]
mix_palette = {"100":'#fff5eb',"95/5":'#fee6ce',"90/10":'#fdd0a2',"80/20":'#fdae6b',"60/40":'#fd8d3c', "50/50":'#f16913',"33/33/33":'#d94801',"25/25/25/25":'#a63603',"20/20/20/20/20":'#7f2704'}

#wastewater versus simulated wastewater versus spike in
sample_type_palette = ['#7570b3', '#d95f02', '#1b9e77']

def main():
    #simulated data location
    simulated_dir = "./simulated_data_results/"
    simulated_files = os.listdir(simulated_dir)  
    
    directory_bam = "/home/chrissy/Desktop/spike_in"
    directory_variants = "/home/chrissy/Desktop/spike_in_variants_saga"
    reference_file = "/home/chrissy/Desktop/sequence.fasta"
    bed_file = "/home/chrissy/Desktop/sarscov2_v2_primers.bed"

    finished_files = "/home/chrissy/Desktop/saga_spike_in_results"
    finished_spikes = os.listdir(finished_files)

    sample_ids = os.listdir(directory_bam)           
    all_files = [os.path.join(directory_bam, x) for x in sample_ids if x.endswith(".bam")]
    sample_ids = [x.split("_sorted")[0] for x in sample_ids if x.endswith(".bam")]
    sample_ids = list(np.unique(sample_ids))
   
    gt_dict = parse_spike_in_metadata()
    
    pure_sample_dict = find_100_percent(gt_dict)
    exclude_positions, consensus_calls, ambiguous_nts = pure_consensus(finished_spikes, pure_sample_dict)
    pure_consensus_dict = subpopulation_pure_consensus(finished_spikes, pure_sample_dict)

    
    """
    done_files = []
    with open("stdout.txt", "r") as tfile:
        for line in tfile:
            line = line.strip()
            if "file_" in line:
                ll = line.split("file_")[-1]
                number = ll.split(" ")[0].strip()
                done_files.append("file_"+number)
    done_files = list(np.unique(done_files))
    """

    """
    - which things do we call consensus on
    """
    #barplot_consensus_boolean(finished_spikes, gt_dict)

    """
    - comparison of technical replicates consensus calls
    """
    #compare_technical_consensus(exclude_positions, consensus_calls, finished_spikes, gt_dict)
    #sys.exit(0)
    """
    - boxplot of edit distance vs. lineage vs. frequency
    - shows that distance from gt consensus varies with lineage and frequency
    """
    finished_spikes = ["file_300"]
    load_compare_consensus(exclude_positions, consensus_calls, finished_spikes, gt_dict, ambiguous_nts, pure_consensus_dict) 

    #build_centroid_plot(finished_spikes, gt_dict)   
    #print(finished_spikes)
    #finished_spikes = finished_spikes[:10]
    #finished_spikes = ['file_236']
    #analyze(finished_spikes, directory_bam, directory_variants, reference_file, bed_file, gt_dict)

    #build_complexity_plot(finished_spikes, gt_dict, simulated_files)
    #build_r_value_plot(finished_spikes, directory_bam, gt_dict)

def barplot_consensus_boolean(samples, gt_dict):
    """
    Given the model outputs which would we reasonably call consensus on?
    """
    mix_type = [] #the type of mixture (categorical)
    num_consensus = [] #number of consensus called (categorical)
    
    for sample_id in samples:
        #if sample_id != "file_227":
        #    continue
        if sample_id not in gt_dict:
            continue
        #print("\nanalyze...", sample_id)
        output_dir = "/home/chrissy/Desktop/saga_spike_in_results/" + sample_id
        fasta = output_dir + "/" + sample_id + ".fa"
        mix_consensus = []
        gt_lineages = gt_dict[sample_id]['gt_lineages']
        gt_frequencies = gt_dict[sample_id]['gt_centers']
        zipped = list(zip(gt_frequencies, gt_lineages))
        zipped.sort(reverse=True)
        gt_frequencies, gt_lineages = zip(*zipped)
        mix = "/".join([str(int(x*100)) for x in gt_frequencies])
        
        model_json = os.path.join(output_dir, sample_id+"_model_results.txt")

        if os.path.isfile(model_json):
            with open(model_json, "r") as mfile:
                model_dict = json.load(mfile)
                if "removal_points" not in model_dict:
                    continue
                autoencoder_dict = model_dict['autoencoder_dict']
                consensus_dict = model_dict['consensus_dict']
                predicted_centers = [float(x) for x in list(autoencoder_dict.keys())]
                assignments = model_dict['assignments']
                variants = model_dict['variants']
                removal_points = model_dict['removal_points']
                assignment_data = model_dict['assignment_data']
                if mix != "33/33/33":
                    continue
                print(sample_id)
                print(predicted_centers)        
                print(mix)
                mix_type.append(mix)
                """
                combination_solutions, combination_sums = generate_combinations(len(predicted_centers), predicted_centers)
                conflict_dict = define_conflicting_peaks(combination_solutions, combination_sums)
                unable_to_assign = []
                print(conflict_dict)
                print(removal_points)
                for data, assign, in zip(assignment_data, assignments):
                    if data in removal_points:
                        continue
                    try:
                        possible_conflict = conflict_dict[assign]
                        print(assign, possible_conflict)
                    except:
                        print("FAILED HERE")
                        continue
                    if len(possible_conflict) == 0:
                        continue
                    unable_to_assign.extend(possible_conflict)
                unable_to_assign = list(np.unique(unable_to_assign))
                unable_to_assign.sort(reverse=True)
                print("can't assign", unable_to_assign)
                called = [x for x in predicted_centers if x not in unable_to_assign and x != min(predicted_centers)]
                """
                called = [ x for x in consensus_dict.keys() if x != min(autoencoder_dict.keys())]
                print("called", called)
                num_consensus.append(len(called)) 

    consensus_palette = {0:'#f2f0f7', 1:'#dadaeb', 2:'#bcbddc', 3:'#9e9ac8',4:'#807dba',5:'#6a51a3',6:'#4a1486'}
    df = pd.DataFrame({"mix_type": mix_type, "num_consensus_called":num_consensus})
    print(df)
    sns.histplot(data=df, x="mix_type", hue="num_consensus_called", multiple="fill", stat="proportion", discrete=True, shrink=.8)
    plt.tight_layout()
    plt.savefig("consensus_barplot.png", bbox_inches='tight')


def count_nucleotide_differences(consensus, gt_consensus, excluded_positions):
    """
    Parameters
    ----------
    consensus : str
    gt_consensus : str
    """
    count = 0
    for i, (gc, c) in enumerate(zip(gt_consensus, consensus)):
        if gc != c:
            if c != "N" and gc != "N" and i+1 not in excluded_positions:
                count += 1                
    return(count)

def match_lineages_consensus(mix_keys, gt_lineages, gt_frequencies, consensus_calls, mix_consensus, exclude_positions, pure_consensus_dict):
    count_dictionary = {}
    for frequency, consensus in mix_consensus.items():
        count_dictionary[frequency] = {}
        for lineage in gt_lineages:
            gt_consensus_dict = pure_consensus_dict[lineage]
            for key, gt_consensus in gt_consensus_dict.items(): 
                excluded_positions = exclude_positions[lineage]
                gt_freq = gt_frequencies[gt_lineages.index(lineage)]
                count = count_nucleotide_differences(lineage, mix_freq, gt_freq, consensus, gt_consensus, excluded_positions)
                print(key, lineage, count, mix_freq)
    
    sys.exit(0)    
    return(best_permutation)

def find_all_excluded_pos(gt_lineages, ambiguous_nts, excluded_positions):
    """
    Given a set of lineages return all positions which NT is the ground truth.
    """
    flat_excluded_pos = []
    for lineage in gt_lineages:
        ambig = list(ambiguous_nts[lineage].keys())
        excluded_pos = excluded_positions[lineage]
        flat_excluded_pos.extend(ambig)
        flat_excluded_pos.extend(excluded_pos)
    flat_excluded_pos = list(np.unique(flat_excluded_pos))
    flat_excluded_pos.sort()
    return(flat_excluded_pos)

def find_lineage_consensus(gt_lineages, consensus_calls, mix_consensus, ambiguous_nts, excluded_positions):
    """
    Parameters
    ----------
    ambiguous_nts : dict 
        Nucleotides by lineage where the frequency is not over the consensus threshold to call a base.
    excluded_positions : dict
        Nucleotides by lineage where the technical replicates are not aligned on the base called.

    Returns
    -------
    matched_lineages : list
        In order of the mix consensus dictionary, the closest matching lineage.

    Find the lineage that most closely matches the consensus sequences, ignoring all positions that are called as N or are inconsistent between technical replicates.
    """
    matched_lineages = []
    nt_diff_count = []

    flat_exclude_pos = find_all_excluded_pos(gt_lineages, ambiguous_nts, excluded_positions)
    for frequency, consensus in mix_consensus.items():
        tmp_track_count = []
        for lineage in gt_lineages:
            gt_consensus = consensus_calls[lineage][0]
            count = count_nucleotide_differences(consensus, gt_consensus, flat_exclude_pos)
            tmp_track_count.append(count)
            print(frequency, count, lineage)
        idx = tmp_track_count.index(min(tmp_track_count))   
        matched_lineages.append(gt_lineages[idx])
        nt_diff_count.append(min(tmp_track_count))
    return(matched_lineages, nt_diff_count)

def load_compare_consensus(exclude_positions, consensus_calls, samples, gt_dict, ambiguous_nts, pure_consensus_dict):
    x = [] #frequency (categorical)
    y = [] #edit distance (continuous)
    hue = [] #lineage (categorical)
    seen_samples = []
    all_percent_not = []
    centroid = [] #experimental centroid


    perc_not_used = []
    nt_diff = []
    f = []
    all_samples = []
    for sample_id in samples:
        if sample_id not in gt_dict:
            continue
        output_dir = "/home/chrissy/Desktop/saga_spike_in_results/" + sample_id
        fasta = output_dir + "/" + sample_id + ".fa"
        mix_consensus = {}
        gt_lineages = gt_dict[sample_id]['gt_lineages']
        gt_frequencies = gt_dict[sample_id]['gt_centers']
        zipped = list(zip(gt_frequencies, gt_lineages))
        zipped.sort(reverse=True)
        gt_frequencies, gt_lineages = zip(*zipped)

        output_name = sample_id
        model_json = os.path.join(output_dir, output_name+"_model_results.txt")
        if os.path.isfile(model_json):
            with open(model_json, "r") as mfile:
                model_dict = json.load(mfile)
                ambiguity_dict = model_dict["ambiguity_dict"]
                autoencoder_dict = model_dict['autoencoder_dict']
                no_call = model_dict['no_call']
                call_ambiguity = model_dict['call_ambiguity']
                low_depth = model_dict['low_depth_positions']
                percent_not_used = model_dict['percent_not_used']
                removal_dict = model_dict['removal_dict']
        else:
            continue
        
        print("\nanalyze...", sample_id)
        print("percent not used...", percent_not_used)
        
        print(gt_frequencies)
        print(autoencoder_dict.keys())
        print("no call", no_call)
        #continue
        variants_json = os.path.join(output_dir, output_name+"_variants.txt")
        if os.path.isfile(variants_json):
            with open(variants_json, "r") as rfile:
                primer_dict = json.load(rfile)
                variants = primer_dict['variants']
        print("call ambiguity", len(call_ambiguity))
        print("low depth", len(low_depth))
        with open(fasta, "r") as ffile:
            tmp = ""
            for j, line in enumerate(ffile):
                if line.startswith(">"):
                    if tmp != "":
                        mix_consensus[name] += tmp
                        tmp = ""
                    name = line.strip().replace(">","")
                    mix_consensus[name] = ""
                    continue
                line = line.strip()
                tmp += line
        try:    
            mix_consensus[name] += tmp
        except:
            print("empty fasta")
            continue
        mix_keys = list(mix_consensus.keys())        
        matched_lineages, nt_diff_count = find_lineage_consensus(gt_lineages, consensus_calls, mix_consensus, ambiguous_nts, exclude_positions)
        
        print("percent not used", percent_not_used)
        print("gt lineages", gt_lineages)
        print("matched_lineages", matched_lineages, "nt diff", nt_diff_count) 
        
        for i,(k,v)  in enumerate(mix_consensus.items()):
            f.append(float(k))
            perc_not_used.append(percent_not_used[k])
            nt_diff.append(nt_diff_count[i])
            all_samples.append(sample_id)
        #continue 
        #best_permutation = match_lineages_consensus(mix_keys, gt_lineages, gt_frequencies, consensus_calls, mix_consensus, exclude_positions, pure_consensus_dict)
        best_permutation_freq = []
        best_permutation = matched_lineages
        for lineage in best_permutation:
            idx = gt_lineages.index(lineage)
            best_permutation_freq.append(gt_frequencies[idx])
        #print(best_permutation)
        #print(gt_lineages)
        #print(gt_frequencies)
        #sys.exit(0)
        #we classify something as incorreclty assigned if it goes to the wrong group even once
        for consensus_key, lineage, frequency in zip(mix_keys, best_permutation, best_permutation_freq):
            seen_samples.append(sample_id)
            all_percent_not.append(percent_not_used[consensus_key])
            centroid.append(float(consensus_key))         
            consensus = mix_consensus[consensus_key]
            gt_consensus = consensus_calls[lineage][0]
            print(consensus_key)
            print(lineage, frequency)
            x.append(frequency)
            hue.append(lineage)
            count = 0
            ground_truth_n = 0
            consensus_n = 0            
            for i, (gc, c) in enumerate(zip(gt_consensus, consensus)):
                if gc != c:
                    if c != "N" and gc != "N" and i+1 not in exclude_positions[lineage]:
                        count += 1
                        freq = variants[str(i+1)][0]
                        print(lineage, "ground truth", gc, "consens", c, i+1)
                        #print(freq)
                    if c == "N":
                        consensus_n += 1
                    if gc == "N":
                        ground_truth_n += 1
            print("differs by:", count)
            print("ground truth n:", ground_truth_n)
            print("consensus n:", consensus_n)
            y.append(count)
   
    #print(perc_not_used)
    #print(nt_diff)
    #print(f)
    zipped = list(zip(nt_diff, perc_not_used, f, all_samples))
    zipped.sort(reverse=True)
    nt_diff, perc_not_used, f, all_samples = zip(*zipped)
    for pnu, nt, fr,s in zip(perc_not_used, nt_diff, f, all_samples):
        print(pnu, nt, fr, s)
    sns.scatterplot(x=f, y=nt_diff, hue=perc_not_used)
    plt.xlabel("predicted population frequency")
    plt.ylabel("# nts incorrectly assigned")
    plt.savefig("./figures/regression_pernotused_wrong.pdf")
    # x frequency (categorical)
    # y edit distance (continuous)
    # hue lineage (categorical)
    #plt.tight_layout()
    #sns.boxplot(x=x, y=y, hue=hue, palette=lineage_palette)
    #plt.ylabel("Number of Nucleotide Differences")
    #plt.savefig("./figures/consensus_boxplot.pdf", bbox_inches='tight')

def subpopulation_pure_consensus(samples, gt_dict):
    possible_consensus_dict = {}
    for sample_id in samples:
        if sample_id not in gt_dict:
            continue
        lineage = gt_dict[sample_id]['gt_lineages'][0]
        if lineage not in possible_consensus_dict:
            possible_consensus_dict[lineage] = {}
        output_dir = "/home/chrissy/Desktop/saga_spike_in_results/" + sample_id
        output_name = sample_id
        model_json = os.path.join(output_dir, output_name+"_model_results.txt")
        consensus = os.path.join(output_dir, output_name+".fa")
        if not os.path.isfile(consensus):
            continue
        tmp = ""
        with open(consensus, "r") as cfile:
            for line in cfile:
                if line.startswith(">"):
                    if tmp != "":
                        possible_consensus_dict[lineage][name] = tmp
                    tmp = ''
                    name = line.strip()                     
                else:
                    tmp += line.strip()
            possible_consensus_dict[lineage][name] = tmp

    return(possible_consensus_dict)
    
def pure_consensus(samples, gt_dict, consensus_threshold=0.95):
    consensus_calls = {}
    ambiguous_nts = {}

    for sample_id in samples:
        if sample_id not in gt_dict:
            continue
        output_dir = "/home/chrissy/Desktop/saga_spike_in_results/" + sample_id
        output_name = sample_id
        model_json = os.path.join(output_dir, output_name+"_model_results.txt")
        
        variants_json = os.path.join(output_dir, output_name+"_variants.txt")
        if os.path.isfile(variants_json):
            with open(variants_json, "r") as rfile:
                primer_dict = json.load(rfile)
                variants = primer_dict['variants']
        else:
            continue
        gt_lineage = gt_dict[sample_id]['gt_lineages'][0]
        if gt_lineage not in ambiguous_nts:
            ambiguous_nts[gt_lineage] = {}
        #print(gt_lineage)
        if gt_lineage not in consensus_calls:
            consensus_calls[gt_lineage] = []
        sequence = ""
        for i, (position, value) in enumerate(variants.items()):
            if i == 0:
                continue
            nuc_counts = value[1]
            nuc_freq = value[0]
            total_depth = sum(list(nuc_counts.values())) 
            if total_depth < 50:
                sequence += "N"
                continue
            canon = ""
            for n, f in nuc_freq.items():
                if f > consensus_threshold:
                    canon += n
            if len(canon) > 1 or len(canon) == 0:
                if i not in ambiguous_nts[gt_lineage]:
                    ambiguous_nts[gt_lineage][i] = []
                #print(gt_lineage, i, nuc_freq)
                ambiguous_nts[gt_lineage][i].append(max(list(nuc_freq.values())))
                canon = "N"
            if i == 14786 and gt_lineage == "B.1.617.2":
                print(i, canon)
                print(nuc_freq)
            sequence += canon
        consensus_calls[gt_lineage].append(sequence)

    exclude_positions = {} 
    for lineage, consensus in consensus_calls.items():
        exclude_positions[lineage] = [] 
        if len(consensus) ==  4:
            for i, (a, b, c, d) in enumerate(zip(consensus[0], consensus[1], consensus[2], consensus[3])):
                if a != b != c != d:                    
                    exclude_positions[lineage].append(i)
        elif len(consensus) == 3:
             for i, (a, b, c) in enumerate(zip(consensus[0], consensus[1], consensus[2])):
                if a != b != c:                    
                    exclude_positions[lineage].append(i)
    """
    print(exclude_positions)
    print(ambiguous_nts)
    sys.exit(0)
    """
    return(exclude_positions, consensus_calls, ambiguous_nts)

def find_100_percent(gt_dict):
    pure_sample_dict = {}
    for key, value in gt_dict.items():
        gt_centers = value['gt_centers']
        if 1.0 in gt_centers:
            pure_sample_dict[key] = value
    return(pure_sample_dict)

def build_centroid_plot(samples, gt_dict):
    mix_type = [] # x (categorical)
    cosine_score = [] # y (continuous)
    
    for sample_id in samples:
        if sample_id not in gt_dict:
            continue
        #print(sample_id)
        output_dir = "/home/chrissy/Desktop/saga_spike_in_results/" + sample_id
        output_name = sample_id
        model_json = os.path.join(output_dir, output_name+"_model_results.txt")
        
        if os.path.isfile(model_json):
            with open(model_json, "r") as rfile:
                model_dict = json.load(rfile)
                predicted_centers = [float(x) for x in list(model_dict['autoencoder_dict'].keys())]
                predicted_centers.sort(reverse=True)
                centers = gt_dict[sample_id]['gt_centers']
                centers.sort(reverse=True)
                mix = "/".join([str(int(x*100)) for x in centers])
                mix_type.append(mix)
                diff = len(centers) - len(predicted_centers)                
                if diff > 0:
                    tmp = [0.0] * diff
                    predicted_centers.extend(tmp)
                elif diff < 0:
                    tmp = [0.0] * abs(diff)
                    centers.extend(tmp)
                cos_score = cosine(predicted_centers, centers)
                                
                print("\n", sample_id, mix)
                print(predicted_centers)
                cosine_score.append(1-cos_score)

    g = sns.boxplot(x=mix_type, y=cosine_score, order=categories, palette=mix_palette, hue_order=categories)
    plt.tight_layout()
    plt.ylabel("Cosine Similarity")
    plt.setp(g.axes, xticks=[], xlabel='') # remove x ticks and xlabel
    handles = []
    for category,color in zip(categories, mix_palette.values()): 
        patch = mpatches.Patch(color=color, label=category)
        handles.append(patch)
    plt.legend(handles=handles, loc="upper right", bbox_to_anchor=(1.4, 1.0))
    plt.savefig("./figures/cosine_plot.pdf", bbox_inches='tight')
    sys.exit(0)

def build_complexity_plot(samples, gt_dict, simulated_files):
    """
    Complexity estimate proof
    """
    gt_length = [] #ground truth number of populations present
    ambiguity_length = [] #the length of NTs that are ambiguous
    hue = [] #simulated ww or spike-in
    percent_mutations = [] #normalized # muts per sample
    cutoff = 0.001
    """
    for sample_id in samples:
        if sample_id not in gt_dict:
            continue
        output_dir = "/home/chrissy/Desktop/saga_spike_in_results/" + sample_id
        output_name = sample_id
        model_json = os.path.join(output_dir, output_name+"_model_results.txt")
        if os.path.isfile(model_json):
            with open(model_json, "r") as rfile:
                model_dict = json.load(rfile)
                gt_centers = gt_dict[sample_id]['gt_centers']
                ambiguity = model_dict['ambiguity_dict']
                complexity = model_dict['complexity']
                tmp = len([x for x in model_dict['total_mutated_pos'] if x > cutoff])
                gt_length.append(len(gt_centers))
                ambiguity_length.append(len(ambiguity))
                percent_mutations.append(tmp/(29903*3))
                hue.append("spike-in")
    """
    for sample_id in simulated_files:
        output_dir = "./simulated_data_results/" + sample_id
        output_name = sample_id
        freyja = "./data/" + sample_id + "_L001_L002_freyja_results.tsv"
        gt_centers, gt_lineages = file_util.parse_freyja_file(freyja)
        gt_centers = [x for x in gt_centers if x > 0.03]
        #print(gt_centers)
        model_json = os.path.join(output_dir, output_name+"_model_results.txt")
        if os.path.isfile(model_json):
            with open(model_json, "r") as rfile:
                model_dict = json.load(rfile)
                ambiguity = model_dict['ambiguity_dict']
                complexity = model_dict['complexity']
                tmp = len([x for x in model_dict['total_mutated_pos'] if x > cutoff])
                gt_length.append(len(gt_centers))
                ambiguity_length.append(len(ambiguity))
                percent_mutations.append(tmp/(29903*3))
                hue.append("simulated wastewater")
    
    df = pd.DataFrame({"hue":hue, "gt_length":gt_length, "percent_mutated":percent_mutations, "number_ambiguities":ambiguity_length}) 
    plt.clf()
    plt.close()
    sns.lmplot(y="percent_mutated", x="gt_length", hue="hue", data=df, fit_reg=False, legend=False, palette=sample_type_palette[:2])
    ax = sns.regplot(y="percent_mutated", x="gt_length", data =df, scatter_kws={"zorder":-1}, line_kws={"color": "black"})

    slope, intercept, r, p, sterr = stats.linregress(x=ax.get_lines()[0].get_xdata(),
                                                       y=ax.get_lines()[0].get_ydata())
    print("slope", slope)
    ax.legend()
    plt.tight_layout()
    plt.ylabel("Percent of Positions Mutated (>%s frequency)" %str(cutoff))
    plt.xlabel("Ground Truth Number of Populations")
    plt.savefig("./figures/complexity_plot_2.pdf", bbox_inches='tight')

def build_r_value_plot(spike_ins, directory_bam, gt_dict):
    """
    Looking at the spike in samples alone, what did we estimate the complexity to be? What was the length of the solution settled on?
    """
    complexity_estimate = []
    mix_type = []
    for sample_id in spike_ins:
        if sample_id not in gt_dict:
            continue
        output_dir = "/home/chrissy/Desktop/saga_spike_in_results/" + sample_id
        output_name = sample_id
        output_fasta_name = "/home/chrissy/Desktop/saga_spike_in_results/%s/%s.fa" %(sample_id, sample_id)
        bam_file = directory_bam + "/%s_sorted.calmd.bam" %sample_id

        model_json = os.path.join(output_dir, output_name+"_model_results.txt")
        if os.path.isfile(model_json):
            with open(model_json, "r") as rfile:
                model_dict = json.load(rfile)
                print(model_dict.keys())
                sys.exit(0)
    sys.exit(0)

def parse_spike_in_metadata():
    lineages = ['A', 'B.1.1.7', 'B.1.351', 'P.1', 'B.1.617.2']
    names = ['Aaron', "Alpha", "Beta", "Gamma", "Delta"]
    filename = "/home/chrissy/Desktop/spike-in_bams_spikein_metadata.csv" 
    gt_dict = {}
    df = pd.read_csv(filename)
    for index, row in df.iterrows():
        sample_id = row['filename'].replace(".bam","")
        gt_centers = [round(float(x)/100,2) for x in ast.literal_eval(row['abundance(%)'])]
        gt_lineages = [] 
        try:
            var = ast.literal_eval(row['variant'])
        except:
            continue
        for x in var:
            idx = names.index(x)
            gt_lineages.append(lineages[idx])

        gt_dict[sample_id] = {"gt_centers":gt_centers, "gt_lineages":gt_lineages}

    return(gt_dict)

def analyze(spike_ins, directory_bam, directory_variants, reference_file, bed_file, gt_dict):
    y = []
    hue = []
    mix_type = []
    reference_sequence = file_util.parse_reference_sequence(reference_file)
    primer_positions = file_util.parse_bed_file(bed_file)
    barcode = pd.read_csv("/home/chrissy/Desktop/usher_barcodes.csv")
    accuracy = [] 
    master_l_dict = {}

    #values for the breakdown of falsely removed things in the barplot
    plot_a_removal = []
    plot_a_mix_type = []     

    for sample_id in spike_ins:
        tmp = sample_id.split("_")[:2]
        sample_id = "_".join(tmp)
        if sample_id not in gt_dict:
            print(sample_id, "not found in metadata")
            continue

        gt_frequencies = gt_dict[sample_id]["gt_centers"]
        gt_lineages = gt_dict[sample_id]["gt_lineages"]
        mut_dict = {}
        mix_freq = copy.deepcopy(gt_frequencies)
        mix_freq.sort(reverse=True)
        mix = "/".join([str(int(x*100)) for x in mix_freq])
        #if '100' not in mix and '20' not in mix:
        #    continue
        output_dir = "/home/chrissy/Desktop/saga_spike_in_results/" + sample_id
        output_name = sample_id
        output_fasta_name = "/home/chrissy/Desktop/saga_spike_in_results/%s/%s.fa" %(sample_id, sample_id)
        bam_file = directory_bam + "/%s_sorted.calmd.bam" %sample_id

        variants_json = os.path.join(output_dir, output_name+"_variants.txt")
        if os.path.isfile(variants_json):
            with open(variants_json, "r") as rfile:
                primer_dict = json.load(rfile)
                if 'total_unique_links' not in primer_dict:
                    continue
        else:
            continue
        flagged_dist = primer_dict["flagged_dist"]
        variants = primer_dict['variants']
        for l in gt_lineages:
            if l not in master_l_dict:
                tmp = file_util.parse_usher_barcode([l], barcode)
                for k,v in tmp.items():
                    master_l_dict[k] = v
                    mut_dict[k] = v
            else:
                mut_dict[l] = master_l_dict[l]
        lower_bound = 0.03
        upper_bound = 0.98
        depth_cutoff = 50
        frequencies, nucs, positions, depth, low_depth_positions, reference_variants, ambiguity_dict, total_muts, possible_training_removed, joint_peak, joint_nuc, joint_dict, test = file_util.parse_variants(primer_dict, primer_positions, reference_sequence)
        
        positions_mut_dict = list(mut_dict.values())
        positions_mut_dict = [item for sublist in positions_mut_dict for item in sublist]
        positions_mut_dict = [item[:-1] for item in positions_mut_dict]
        for position, value in variants.items():
            if position not in positions_mut_dict:
                continue
            nuc_frequency = value[0]
            for nuc, freq in nuc_frequency.items():
                #find out which strain this position/nuc combo belongs to
                var = str(position)+str(nuc)
                belong_lineage = []
                for lineage, key_mutations in mut_dict.items():
                    if var in key_mutations:
                        belong_lineage.append(lineage)
                if len(belong_lineage) < 1:
                    continue
            
                total_freq_expected = 0
                for thing in belong_lineage:
                    idx = gt_lineages.index(thing)
                    total_freq_expected += gt_frequencies[idx]                    
                dist = round(abs(float(freq) - total_freq_expected),3)
                y.append(dist)
                #print("\n", position)
                if str(position) in ambiguity_dict:
                    flagged_reasons = ambiguity_dict[str(position)]
                    #if flagged_reasons == ['primer_binding']:
                        #print(flagged_reasons)
                        #if str(position) in test:
                        #    print(test[position])
                    flagged_reasons = "\n&".join(flagged_reasons)
                    if dist < 0.05:
                        #print("falsely removed")
                        #print(flagged_reasons)
                        accuracy.append("falsely removed")
                        plot_a_removal.append(flagged_reasons)
                        plot_a_mix_type.append(mix)                        
                    else:
                        #print("correctly removed")
                        accuracy.append("correctly removed")
                    hue.append(flagged_reasons)
                else:
                    flagged_reasons = "not flagged"
                    hue.append("not flagged")
                    if dist > 0.20:
                        accuracy.append("falsely kept")
                        #print("false kept", freq, belong_lineage, position, gt_lineages, gt_frequencies)
                    else:
                        #print("correctly kept")
                        accuracy.append("correctly kept")
                mix_type.append(mix)
    colors = list(np.unique(hue))
    flag_color_dict = {}
    for h, c in zip(colors, palette[:len(colors)]):
        flag_color_dict[h] = c

    #this is plot a, a breakdown of what got falsely removed and why     
    plt.clf()
    plt.close()
    plt.figure(figsize=(10, 6), dpi=80)
    colors_needed = len(np.unique(plot_a_removal))
    print("plot a the number of colors needed, ", colors_needed)
    df = pd.DataFrame({"mix_type": plot_a_mix_type, "removal_reason":plot_a_removal})
    tmp_colors = {}
    for val in np.unique(plot_a_removal):
        tmp_colors[val] = flag_color_dict[val]
    sns.histplot(data=df, x="mix_type", hue="removal_reason", multiple="fill", stat="proportion", discrete=True, shrink=.8, palette=tmp_colors)
    plt.xticks(rotation=45)
    plt.xlabel("mix type")
    plt.ylabel("proportion")
    plt.savefig("./figures/false_removal_data_cleaning.pdf", bbox_inches='tight')    
    plt.clf()
    plt.close()

    #heatmap
    df = pd.DataFrame({"mix_type":mix_type, "flag_status":hue, "y":y, "accuracy":accuracy})
    hue_order = ["correctly kept", "correctly removed", "falsely kept", "falsely removed"]

    unique_accuracy = list(np.unique(df['accuracy']))
    unique_mixtures = list(np.unique(df['mix_type']))
    mixture_counts = [0] * len(unique_mixtures)
    counts = np.zeros((len(unique_mixtures), len(hue_order)))
    for index, row in df.iterrows():
        x = unique_mixtures.index(row['mix_type'])
        mixture_counts[x] += 1
        y = unique_accuracy.index(row['accuracy'])
        counts[x,y] += 1
    new_df = pd.DataFrame({"mix_type":"", "accuracy":"", "ratio":0}, index=[0])
    for i, x in enumerate(counts):
        x = x / mixture_counts[i]
        for a,b in zip(unique_accuracy, x):
            row = [unique_mixtures[i], a, b]
            row = pd.DataFrame({"mix_type":unique_mixtures[i], "accuracy":a, "ratio":b}, index=[0])
            new_df = pd.concat([new_df, row], ignore_index=True)

    new_df = new_df.tail(-1)
    used_mix = new_df['mix_type'].tolist()
    used_mix = [x for x in categories if x in used_mix]
    new_df = new_df.pivot("accuracy", "mix_type", "ratio")
    new_df = new_df.reindex(used_mix, axis=1)
    print(new_df)
    sns.heatmap(data=new_df, annot=True,  linewidths=.5, vmin=0, vmax=1, cmap="Purples", fmt='.2f')
    plt.xticks(rotation=90)
    plt.xlabel("mix type")
    plt.ylabel("")
    plt.savefig("./figures/heatmap_confusion_matrix.pdf", bbox_inches='tight')  
    plt.close()
    plt.clf() 

    new_category = ["100", "95/5", "90/10"]
    new_df = df[df['mix_type'].isin(new_category)]
    g = sns.boxplot(x='mix_type', y='y', hue='flag_status', data =new_df, palette=flag_color_dict)
    sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
    #plt.setp(g.axes, xticks=[], xlabel='') # remove x ticks and xlabel
    plt.tight_layout()
    plt.ylabel("Distance from Expected Frequency")
    plt.xlabel("mix type")
    plt.savefig("./figures/output1.pdf", bbox_inches='tight')

    plt.close()
    plt.clf()
    new_category = ["80/20", "60/40", "50/50"]
    new_df = df[df['mix_type'].isin(new_category)]
    g = sns.boxplot(x='mix_type', y='y', hue='flag_status', data =new_df, palette=flag_color_dict)
    sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
    #plt.setp(g.axes, xticks=[], xlabel='') # remove x ticks and xlabel
    plt.tight_layout()
    plt.ylabel("Distance from Expected Frequency")
    plt.xlabel("mix type")
    plt.savefig("./figures/output2.pdf", bbox_inches='tight')

    plt.close()
    plt.clf()
    new_category = ["33/33/33", "25/25/25/25", "20/20/20/20/20"]
    new_df = df[df['mix_type'].isin(new_category)]
    g = sns.boxplot(x='mix_type', y='y', hue='flag_status', data =new_df, palette=flag_color_dict)
    sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
    #plt.setp(g.axes, xticks=[], xlabel='') # remove x ticks and xlabel
    plt.tight_layout()
    plt.ylabel("Distance from Expected Frequency")
    plt.xlabel("mix type")
    plt.savefig("./figures/output3.pdf", bbox_inches='tight')


if __name__ == "__main__":
    main()
