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

from scipy import stats
from scipy.spatial.distance import cosine
from line_profiler import LineProfiler
sys.path.insert(0, "../")
from model_util import run_model, generate_combinations
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
def r2(x, y):
    return stats.pearsonr(x, y)[0] ** 2

def beta_distribution_analysis(gt_dict, samples):
    """
    Looking at the models built using beta distributions.
    """

    samples = ["file_348"]
    centroid_reg_x = []
    centroid_reg_y = []

    for i, sample_id in enumerate(samples):
        if sample_id not in gt_dict:
            continue
        output_dir = "/home/chrissy/Desktop/saga_spike_in_results/" + sample_id
        beta_filename = os.path.join(output_dir, "%s_beta_dist.json"%sample_id)
        #assignment filename 
        #example: file_0_assignments.json 
        assignment_filename = os.path.join(output_dir, "%s_assignments.json"%sample_id)
        if not os.path.isfile(beta_filename):
            continue
        if not os.path.isfile(assignment_filename):
            continue
        print(sample_id)
        gt = gt_dict[sample_id]
        gt_centroids = gt['gt_centers']
        with open(beta_filename, 'r') as bfile:
            data = json.load(bfile)
            solutions = data['solutions']
            distributions = data['distributions']
            frequency = data['frequency']
        with open(assignment_filename, 'r') as bfile:
            assignment_data = json.load(bfile)
            variants_dict = assignment_data['variants']

        ##PLOT A
        #process data for the linear regression plot
        predicted_centroids = [float(x) for x in list(variants_dict.keys())]
        predicted_centroids.sort(reverse=True)
        gt_centroids.sort(reverse=True)
        if len(gt_centroids) != len(predicted_centroids):
            diff = len(gt_centroids) - len(predicted_centroids)
            tmp = [0.0] * abs(diff)
            if diff < 0:
                gt_centroids.extend(tmp)    
            else:
                predicted_centroids.extend(tmp)
        if np.isnan(predicted_centroids).any():
            continue
        print(predicted_centroids)
        print(solutions)
        print(gt_centroids)
        centroid_reg_x.extend(gt_centroids)
        centroid_reg_y.extend(predicted_centroids)
    ##PLOT A
    plt.clf()
    plt.close()
    sns.regplot(x=centroid_reg_x, y=centroid_reg_y)
    r = r2(centroid_reg_x, centroid_reg_y)
    print("r2", r)
    plt.plot( [0,1],[0,1], color="purple")
    plt.tight_layout()
    plt.savefig("./figures/beta/beta_plot_a.pdf")    
    plt.clf()
    plt.close()
      
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
    beta_distribution_analysis(gt_dict, finished_spikes)
    sys.exit(0)
    
    pure_sample_dict = find_100_percent(gt_dict)
    exclude_positions, consensus_calls, ambiguous_nts = pure_consensus(finished_spikes, pure_sample_dict)
    pure_consensus_dict = subpopulation_pure_consensus(finished_spikes, pure_sample_dict)
        
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
    ivar_spike_dict = ivar_spike_in(gt_dict, pure_sample_dict, consensus_calls, exclude_positions, ambiguous_nts)
    #sys.exit(0)
    #finished_spikes = ["file_124"]
    #compare_ivar_saga(ivar_spike_dict, gt_dict, consensus_calls, ambiguous_nts, exclude_positions)
    lower_population_ivar_saga(ivar_spike_dict, gt_dict, consensus_calls, ambiguous_nts, exclude_positions)
   
    #load_compare_consensus(exclude_positions, consensus_calls, finished_spikes, gt_dict, ambiguous_nts, pure_consensus_dict) 

    #build_centroid_plot(finished_spikes, gt_dict)   
    #print(finished_spikes)
    #finished_spikes = finished_spikes[:10]
    #finished_spikes = ['file_236']
    #analyze(finished_spikes, directory_bam, directory_variants, reference_file, bed_file, gt_dict)

    #build_complexity_plot(finished_spikes, gt_dict, simulated_files)
    #build_r_value_plot(finished_spikes, directory_bam, gt_dict)

def lower_population_ivar_saga(ivar_spike_dict, gt_dict, consensus_calls, ambiguous_nts, exclude_positions):
    #how well does saga recover the lower population?
    ivar_wrong_saga_n = []
    ivar_right_saga_n = []

    ivar_wrong_saga_wrong = []
    ivar_right_saga_wrong = []

    ivar_n_saga_right = []
    ivar_n_saga_wrong = []
    seen_samples = []

    ivar_wrong = []
    status = []
    all_mix_types = []
    for filename, i_consensus in ivar_spike_dict.items():
        if "file" not in filename:
            continue
        sample_id = filename.split("_sorted")[0].split("Consensus_")[1]

        #EXCEPTION
        if sample_id == "file_0" or sample_id == "file_200" or sample_id == "file_208" or sample_id == "file_60":
            continue
        if sample_id not in gt_dict:
            continue

        gt_centers = gt_dict[sample_id]['gt_centers']
        gt_lineages = gt_dict[sample_id]['gt_lineages']
        zipped = list(zip(gt_centers, gt_lineages))
        zipped.sort(reverse=True)
        gt_centers, gt_lineages = zip(*zipped)
        if len(gt_centers) != 2:
            continue
        mix_type = "/".join([str(int(x*100)) for x in gt_centers])
        
        model_json = os.path.join("/home/chrissy/Desktop/saga_spike_in_results", sample_id, sample_id+"_model_results.txt")
        if os.path.isfile(model_json):
            with open(model_json, "r") as mfile:
                model_dict = json.load(mfile)
                percent_not_used = model_dict['percent_not_used']
        output_dir = "/home/chrissy/Desktop/saga_spike_in_results/" + sample_id
        fasta = output_dir + "/" + sample_id + ".fa"
        if not os.path.isfile(fasta):
            continue
        print("\n", gt_centers)
        print(gt_lineages)
        print(sample_id)
        print(percent_not_used)
        noise = False
        for i,(key, value) in enumerate(percent_not_used.items()):
            if value > 0.15:
                noise=True
                print("too much noise")
                break
            if i == 0:
                break 
        if noise:
            status.append("no consensus call")
        else:
            status.append("call consensus")
        all_mix_types.append(mix_type)
        s_consensus = ""
        name = ""
        all_con_exp = {}
        print(ambiguous_nts['B.1.617.2'])
        with open(fasta, "r") as ffile:
            for i, line in enumerate(ffile):
                if line.startswith(">"):
                    name = line.strip()
                    all_con_exp[name] = ""
                    continue
                #elif line.startswith(">") and i != 0 and i != 2 and "peak_2" not in line:
                #    break
                else:
                    #if name != "":
                    s_consensus += line.strip()
                    all_con_exp[name] += line.strip()
        diff_count = 0
        pos_diff = []
        print(name)
        print(all_con_exp.keys())
        con_1 = all_con_exp['>peak_0_0.92']
        con_2 = all_con_exp['>peak_2_0.07']
        for j,(a, b) in enumerate(zip(con_1, con_2)):
            if a != b:
                print(j, a ,b)
        sys.exit(0)
        diff = 29000
        lineage_use = ""
        for lineage in gt_lineages:
            gt_consensus = consensus_calls[lineage][0]
            count_lin = count_nucleotide_differences(s_consensus, gt_consensus, exclude_positions[lineage])
            if count_lin < diff:
                lineage_use = lineage
                diff = count_lin
            print(lineage, count_lin)
        print(lineage_use) 
        gt_consensus = consensus_calls[lineage_use][0]           
        exclude = exclude_positions[lineage_use]
        amb = ambiguous_nts[lineage_use]
        exclude.extend(list(amb.keys()))
        exclude.sort()
        exclude = [int(x) for x in exclude]
        civar_wrong_saga_n = 0
        civar_right_saga_n = 0
        print(ambiguous_nts)
        civar_wrong_saga_wrong = 0
        civar_right_saga_wrong = 0

        civar_n_saga_right = 0
        civar_n_saga_wrong = 0
        civar_wrong = 0
        for j, (i, s, truth) in enumerate(zip(i_consensus, s_consensus, gt_consensus)):
            truth = truth.upper()
            i = i.upper()
            s = s.upper()
            if j+1 in exclude:
                continue
            if truth == "N":
                continue
            if i == s and s == truth:
                continue
            if i != truth and i != "N":
                civar_wrong += 1
            if i != truth and s == "N":
                civar_wrong_saga_n += 1
            elif i == truth and s == "N":
                print(j, "ivar", i, "saga", s, "truth", truth)
                civar_right_saga_n += 1
            elif i != truth and s != truth and i != "N" and s != "N":
                print(j, "ivar", i, "saga", s, "truth", truth)
                civar_wrong_saga_wrong += 1
            elif i == truth and s != truth and s != "N":
                print(j, "ivar", i, "saga", s, "truth", truth)
                civar_right_saga_wrong += 1
            elif i == "N" and s == truth:
                civar_n_saga_right += 1
            elif i == "N" and s != truth:
                civar_n_saga_wrong += 1
        sys.exit(0)        
        ivar_wrong_saga_n.append(civar_wrong_saga_n)
        ivar_right_saga_n.append(civar_right_saga_n)

        ivar_wrong_saga_wrong.append(civar_wrong_saga_wrong)
        ivar_right_saga_wrong.append(civar_right_saga_wrong)

        ivar_n_saga_right.append(civar_n_saga_right)
        ivar_n_saga_wrong.append(civar_n_saga_wrong)       
        ivar_wrong.append(civar_wrong)
        seen_samples.append(sample_id)
    print("ivar wrong saga n", ivar_wrong_saga_n)
    #print("ivar right saga n", ivar_right_saga_n)
    
    df = pd.DataFrame({"ivar_wrong":ivar_wrong, "filename":seen_samples, "status":status, "mix_type":all_mix_types})
    df = df[df['status'] == "no consensus call"]
    colors = [mix_palette[x] for x in df['mix_type'].tolist()]
    df['color'] = colors
    purple_patch = mpatches.Patch(color='#f16913', label='50/50')
    orange_patch = mpatches.Patch(color='#fd8d3c', label='60/40')
    red_patch = mpatches.Patch(color='#fdae6b', label='80/20')
    plt.legend(handles=[purple_patch, orange_patch, red_patch], title="mixture type")
    plt.bar(x=df['filename'].tolist(), height=df['ivar_wrong'].tolist(), color=df['color'].tolist(), width = 0.5)
    plt.xticks(rotation=45)
    plt.ylabel("# NT Wrongly Called by iVar")
    plt.tight_layout()
    #plt.savefig("./figures/no_consensus_call.png")
    plt.close()
    plt.clf()


    df = pd.DataFrame({"ivar_wrong_saga_n":ivar_wrong_saga_n, "filename":seen_samples, "status":status, "mix_type":all_mix_types})
    df = df[df['status'] != "no consensus call"]
    cut_categories = [x for x in categories if x in df['mix_type'].tolist()]
    sns.boxplot(y="ivar_wrong_saga_n", x="mix_type", data=df, order=cut_categories, palette=mix_palette)
    plt.ylabel("# NT Wrongly Called by iVar, N in Saga")
    plt.tight_layout()
    #plt.savefig("./figures/consensus_corrections.png")
    plt.close()
    plt.clf()    

    #print("both wrong", ivar_wrong_saga_wrong)
    print("ivar right saga wrong", ivar_right_saga_wrong)
    for a, b, c, d in zip(seen_samples, ivar_right_saga_wrong, ivar_wrong_saga_n, status):
        if status == "no consensus call":
            continue
        if c > 0:
            print(a, b, c)   
    
    #print(ivar_n_saga_right)
    #print(ivar_n_saga_wrong)
 
def compare_ivar_saga(ivar_spike_dict, gt_dict, consensus_calls, ambiguous_nts, exclude_positions):
    """
    On a file by file basis, compare the highest value consensus sequences to the "ground truth".
    """
    #plot 1: how does saga "clean up" the higher consensus sequence in mixtures of two things
    ivar_wrong_saga_n = []
    ivar_right_saga_n = []

    ivar_wrong_saga_wrong = []
    ivar_right_saga_wrong = []

    ivar_n_saga_right = []
    ivar_n_saga_wrong = []
    seen_samples = []

    ivar_wrong = []
    status = []
    all_mix_types = []
    for filename, i_consensus in ivar_spike_dict.items():
        if "file" not in filename:
            continue
        sample_id = filename.split("_sorted")[0].split("Consensus_")[1]

        #EXCEPTION
        if sample_id == "file_0" or sample_id == "file_200" or sample_id == "file_208" or sample_id == "file_60":
            continue
        if sample_id not in gt_dict:
            continue
        gt_centers = gt_dict[sample_id]['gt_centers']
        gt_lineages = gt_dict[sample_id]['gt_lineages']
        zipped = list(zip(gt_centers, gt_lineages))
        zipped.sort(reverse=True)
        gt_centers, gt_lineages = zip(*zipped)
        if len(gt_centers) != 2:
            continue
        mix_type = "/".join([str(int(x*100)) for x in gt_centers])
        
        model_json = os.path.join("/home/chrissy/Desktop/saga_spike_in_results", sample_id, sample_id+"_model_results.txt")
        if os.path.isfile(model_json):
            with open(model_json, "r") as mfile:
                model_dict = json.load(mfile)
                percent_not_used = model_dict['percent_not_used']
        output_dir = "/home/chrissy/Desktop/saga_spike_in_results/" + sample_id
        fasta = output_dir + "/" + sample_id + ".fa"
        if not os.path.isfile(fasta):
            continue
        print("\n", gt_centers)
        print(gt_lineages)
        print(sample_id)
        print(percent_not_used)
        noise = False
        for i,(key, value) in enumerate(percent_not_used.items()):
            if value > 0.15:
                noise=True
                print("too much noise")
                break
            if i == 0:
                break 
        if noise:
            status.append("no consensus call")
        else:
            status.append("call consensus")
        all_mix_types.append(mix_type)
        s_consensus = ""
        with open(fasta, "r") as ffile:
            for i, line in enumerate(ffile):
                if line.startswith(">") and i == 0:
                    name = line.strip()
                    continue
                elif line.startswith(">") and i != 0:
                    break
                else:
                    s_consensus += line.strip()
        diff_count = 0
        pos_diff = []

        diff = 29000
        lineage_use = ""
        for lineage in gt_lineages:
            gt_consensus = consensus_calls[lineage][0]
            count_lin = count_nucleotide_differences(s_consensus, gt_consensus, exclude_positions[lineage])
            if count_lin < diff:
                lineage_use = lineage
                diff = count_lin
            print(lineage, count_lin)
        print(lineage_use) 
        gt_consensus = consensus_calls[lineage_use][0]           
        exclude = exclude_positions[lineage_use]
        amb = ambiguous_nts[lineage_use]
        exclude.extend(list(amb.keys()))
        exclude.sort()
        exclude = [int(x) for x in exclude]
        #print(exclude)    
        civar_wrong_saga_n = 0
        civar_right_saga_n = 0

        civar_wrong_saga_wrong = 0
        civar_right_saga_wrong = 0

        civar_n_saga_right = 0
        civar_n_saga_wrong = 0
        civar_wrong = 0
        for j, (i, s, truth) in enumerate(zip(i_consensus, s_consensus, gt_consensus)):
            truth = truth.upper()
            i = i.upper()
            s = s.upper()
            if j+1 in exclude:
                continue
            if truth == "N":
                continue
            if i == s and s == truth:
                continue
            if i != truth and i != "N":
                civar_wrong += 1
            if i != truth and s == "N":
                civar_wrong_saga_n += 1
            elif i == truth and s == "N":
                #print(j, "ivar", i, "saga", s, "truth", truth)
                civar_right_saga_n += 1
            elif i != truth and s != truth and i != "N" and s != "N":
                #print(j, "ivar", i, "saga", s, "truth", truth)
                civar_wrong_saga_wrong += 1
            elif i == truth and s != truth and s != "N":
                print(j, "ivar", i, "saga", s, "truth", truth)
                civar_right_saga_wrong += 1
            elif i == "N" and s == truth:
                civar_n_saga_right += 1
            elif i == "N" and s != truth:
                civar_n_saga_wrong += 1
        
        ivar_wrong_saga_n.append(civar_wrong_saga_n)
        ivar_right_saga_n.append(civar_right_saga_n)

        ivar_wrong_saga_wrong.append(civar_wrong_saga_wrong)
        ivar_right_saga_wrong.append(civar_right_saga_wrong)

        ivar_n_saga_right.append(civar_n_saga_right)
        ivar_n_saga_wrong.append(civar_n_saga_wrong)       
        ivar_wrong.append(civar_wrong)
        seen_samples.append(sample_id)
    print("ivar wrong saga n", ivar_wrong_saga_n)
    #print("ivar right saga n", ivar_right_saga_n)
    
    df = pd.DataFrame({"ivar_wrong":ivar_wrong, "filename":seen_samples, "status":status, "mix_type":all_mix_types})
    df = df[df['status'] == "no consensus call"]
    colors = [mix_palette[x] for x in df['mix_type'].tolist()]
    df['color'] = colors
    purple_patch = mpatches.Patch(color='#f16913', label='50/50')
    orange_patch = mpatches.Patch(color='#fd8d3c', label='60/40')
    red_patch = mpatches.Patch(color='#fdae6b', label='80/20')
    plt.legend(handles=[purple_patch, orange_patch, red_patch], title="mixture type")
    plt.bar(x=df['filename'].tolist(), height=df['ivar_wrong'].tolist(), color=df['color'].tolist(), width = 0.5)
    plt.xticks(rotation=45)
    plt.ylabel("# NT Wrongly Called by iVar")
    plt.tight_layout()
    plt.savefig("./figures/no_consensus_call.png")
    plt.close()
    plt.clf()


    df = pd.DataFrame({"ivar_wrong_saga_n":ivar_wrong_saga_n, "filename":seen_samples, "status":status, "mix_type":all_mix_types})
    df = df[df['status'] != "no consensus call"]
    cut_categories = [x for x in categories if x in df['mix_type'].tolist()]
    sns.boxplot(y="ivar_wrong_saga_n", x="mix_type", data=df, order=cut_categories, palette=mix_palette)
    plt.ylabel("# NT Wrongly Called by iVar, N in Saga")
    plt.tight_layout()
    plt.savefig("./figures/consensus_corrections.png")
    plt.close()
    plt.clf()    
    sys.exit(0)
    #print("both wrong", ivar_wrong_saga_wrong)
    print("ivar right saga wrong", ivar_right_saga_wrong)
    for a, b, c, d in zip(seen_samples, ivar_right_saga_wrong, ivar_wrong_saga_n, status):
        if status == "no consensus call":
            continue
        if c > 0:
            print(a, b, c)   
    
    #print(ivar_n_saga_right)
    #print(ivar_n_saga_wrong)
    
  
def ivar_spike_in(gt_dict, pure_sample_dict, consensus_calls, exclude_positions, ambiguous_nts):
    """
    Parse multialignment file to get all sequences called using ivar.
    """
    aligned_filename = "./multi_aligned.fa"
    ivar_spike_dict = {}
    with open(aligned_filename, "r") as ffile:
        name = ""
        tmp = ""
        for line in ffile:
            line = line.strip()
            if line.startswith(">"):
                if tmp != "":
                    ivar_spike_dict[name] = tmp                
                name = line
                tmp = ""
            else:
                tmp += line
            
    return(ivar_spike_dict)

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
                if "low_depth_positions" in model_dict:
                    continue
                print(model_dict.keys())
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
    #plot 1 : 
    mix_types_gt = []
    unplaced_nts = []

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
        mix_string = "/".join([str(x) for x in gt_frequencies])

        output_name = sample_id
        model_json = os.path.join(output_dir, output_name+"_model_results.txt")
        if not os.path.isfile(model_json):
            continue
        with open(model_json, "r") as mfile:
            model_dict = json.load(mfile)
        if "low_depth_positions" in model_dict:
            continue
        ambiguity_dict = model_dict["ambiguity_dict"]
        autoencoder_dict = model_dict['autoencoder_dict']
        no_call = model_dict['no_call']
        call_ambiguity = model_dict['call_ambiguity']
        percent_not_used = model_dict['percent_not_used']
        removal_dict = model_dict['removal_dict']
   
        print("\nanalyze...", sample_id)
        print("percent not used...", percent_not_used)

        #for plot 1
        consensuses = [x for x in list(autoencoder_dict.keys()) if float(x.split("_")[2]) not in no_call]
        call_ambiguity_count = 0
        #count number of NTs that can't be assigned on basis of bad model fit/messy data
        for key, value in call_ambiguity.items():
            reasons = value['reason']
            variants = value['variants']
            populations = value['population']
            if "outlier" in reasons:
                call_ambiguity_count += len(consensuses)
        total_assigned_mutations = []
        for k, v in autoencoder_dict.items():
            total_assigned_mutations.extend(v)
        percent_not_assigned = call_ambiguity_count / (call_ambiguity_count+len(total_assigned_mutations))
        print(mix_string)
        print(percent_not_assigned) 
        if percent_not_assigned >= 0.5:
            print("HERE", percent_not_assigned, mix_string)
        unplaced_nts.append(percent_not_assigned)
        mix_types_gt.append(mix_string)        
        continue

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
        all_variants_pos = []
        for k,v in autoencoder_dict.items():
            p = [x[:-1] for x in v]
            all_variants_pos.extend(p)
        all_variants_pos = list(np.unique(all_variants_pos))
        print(len(all_variants_pos))
        print(len(call_ambiguity))
        for i,(k,v)  in enumerate(mix_consensus.items()):
            f.append(float(k))
            perc_not_used.append(percent_not_used[k])
            nt_diff.append(nt_diff_count[i])
            all_samples.append(sample_id)
            matches.append(matched_lineages)
            mixes.append(gt_frequencies)
        continue 
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

    #plot 1   
        unplaced_nts.append(call_ambiguity_count)
        mix_types_gt.append(mix_string)        

    sns.boxplot(x=mix_types_gt, y=unplaced_nts)
    plt.xlabel("mixture type")
    plt.ylabel("# nts unable to be assigned")
    plt.tight_layout()
    plt.savefig("./figures/regression_pernotused_wrong.pdf")


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
            #if (i == 22812) and gt_lineage == "B.1.617.2":
            #    print(i, canon)
            #    print(nuc_freq)
            #    print(total_depth)
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
    gt_centroid = []
    predicted_centroid = []
    mix_type = [] 
    for sample_id in samples:
        if sample_id not in gt_dict:
            continue
        centers = gt_dict[sample_id]['gt_centers']
        centers.sort(reverse=True)
        if len(centers) != 2:
            continue
        #print(sample_id)
        output_dir = "/home/chrissy/Desktop/saga_spike_in_results/" + sample_id
        output_name = sample_id
        model_json = os.path.join(output_dir, output_name+"_model_results.txt")
        
        if os.path.isfile(model_json):
            with open(model_json, "r") as rfile:
                model_dict = json.load(rfile)
                predicted_centers = list(model_dict['autoencoder_dict'].keys())
                predicted_centers = [float(x.split("_")[-1]) for x in predicted_centers]
                #print(predicted_centers)
                predicted_centers.sort(reverse=True)
                mix = "/".join([str(int(x*100)) for x in centers])
                mix_type.append(mix)
                diff = len(centers) - len(predicted_centers)                
                if diff > 0:
                    tmp = [0.0] * diff
                    predicted_centers.extend(tmp)
                elif diff < 0:
                    tmp = [0.0] * abs(diff)
                    centers.extend(tmp)
                print(centers)
                print(predicted_centers)                                
                print("\n", sample_id, mix)
                #print(predicted_centers)
                predicted_centroid.extend(predicted_centers)
                gt_centroid.extend(centers)

    g = sns.regplot(x=gt_centroid, y=predicted_centroid, color="orange")
    orange_patch = mpatches.Patch(color='orange', label='regression line')
    red_patch = mpatches.Patch(color='purple', label='identity line')
    plt.legend(handles=[orange_patch, red_patch])
    plt.plot( [0,1],[0,1], color="purple")
    plt.tight_layout()
    plt.ylabel("Predicted Population Frequency")
    plt.xlabel("Ground Truth Population Frequency")
    """
    plt.setp(g.axes, xticks=[], xlabel='') # remove x ticks and xlabel
    handles = []
    for category,color in zip(categories, mix_palette.values()): 
        patch = mpatches.Patch(color=color, label=category)
        handles.append(patch)
    plt.legend(handles=handles, loc="upper right", bbox_to_anchor=(1.4, 1.0))
    """
    plt.savefig("./figures/centroid.png", bbox_inches='tight')

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
