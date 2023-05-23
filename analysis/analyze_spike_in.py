"""
Author: Chrissy Aceves
Email: caceves@scripps.edu
"""
import os
import ast
import sys
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from Levenshtein import distance
from scipy.spatial.distance import cosine
from line_profiler import LineProfiler
sys.path.insert(0, "../")
from model import run_model
import file_util
from generate_consensus import write_fasta

sns.set_style("whitegrid")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 12

#this is for primer pess ups 
palette = ['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c','#fdbf6f','#ff7f00','#cab2d6','#6a3d9a','#ffff99','#b15928']

lineage_palette= {"A":"#e66101", "B.1.1.7":"#fdb863", "B.1.351":"#f7f7f7", "P.1":"#b2abd2", "B.1.617.2":"#5e3c99"}

#for mixture categories
categories = ["100", "95/5", "90/10", "80/20", "60/40", "50/50", \
    "33/33/33", "25/25/25/25", "20/20/20/20/20"]
mix_palette = {"100":'#fff5eb',"95/5":'#fee6ce',"90/10":'#fdd0a2',"80/20":'#fdae6b',"60/40":'#fd8d3c', "50/50":'#f16913',"33/33/33":'#d94801',"25/25/25/25":'#a63603',"20/20/20/20/20":'#7f2704'}

def main():
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
    exclude_positions, consensus_calls = pure_consensus(finished_spikes, pure_sample_dict)

    """
    - comparison of technical replicates consensus calls
    """
    #compare_technical_consensus(exclude_positions, consensus_calls, finished_spikes, gt_dict)

    """
    - boxplot of edit distance vs. lineage vs. frequency
    - shows that distance from gt consensus varies with lineage and frequency
    """
    load_compare_consensus(exclude_positions, consensus_calls, finished_spikes, gt_dict) 


    #build_centroid_plot(finished_spikes, gt_dict)   
    #print(finished_spikes)
    #finished_spikes = finished_spikes[:10]
    #finished_spikes = ['file_181']
    #analyze(finished_spikes, directory_bam, directory_variants, reference_file, bed_file, gt_dict)
    #analyze(finished_spikes, directory_bam, directory_variants, reference_file, bed_file, filtered_dict)

    #build_complexity_plot(finished_spikes, gt_dict)
    #build_r_value_plot(finished_spikes, directory_bam, gt_dict)

def compare_technical_consensus(exclude_positions, consensus_calls, finished_spikes, gt_dict):
    """
    PCA using euclidean distance between encoded technical replicates.
    """
    nt_encoding = {"A":0, "C":1, "G":2, "T":3, "N":4, "-":5}
    x = []
    y = [] 
    shape = [] #mixture type
    hue = [] #frequency 

    matrix = []
     
    
def load_compare_consensus(exclude_positions, consensus_calls, sample_id, gt_dict):
    x = [] #frequency (categorical)
    y = [] #edit distance (continuous)
    hue = [] #lineage (categorical)

    for sample_id in samples:
        output_dir = "/home/chrissy/Desktop/saga_spike_in_results/" + sample_id
        fasta = output_dir + "/" + sample_id + ".fa"
        mix_consensus = []
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
       
        variants_json = os.path.join(output_dir, output_name+"_variants.txt")
        if os.path.isfile(variants_json):
            with open(variants_json, "r") as rfile:
                primer_dict = json.load(rfile)
                variants = primer_dict['variants']

        with open(fasta, "r") as ffile:
            tmp = ""
            for line in ffile:
                if line.startswith(">"):
                    if tmp != "":
                        mix_consensus.append(tmp)
                        tmp = ""
                    continue
                line = line.strip()
                tmp += line
            mix_consensus.append(tmp)

        target_points = [] #points we've mis-assigned

        #we classify something as incorreclty assigned if it goes to the wrong group even once
        for consensus, lineage, frequency in zip(mix_consensus, gt_lineages, gt_frequencies):
            gt_consensus = consensus_calls[lineage][0]
            edit_distance = distance(gt_consensus, consensus)
            print(lineage, frequency, edit_distance)
            x.append(frequency)
            y.append(edit_distance)
            hue.append(lineage)
             
            """
            for i,(gc, c) in enumerate(zip(gt_consensus, consensus)):
                colors=False
                if gc != c:
                    if c != "N" and gc != "N" and i+1 not in exclude_positions[lineage]:
                        depth = sum(list(variants[str(i+1)][1].values()))
                        freq = variants[str(i+1)][0]
                        print(lineage, gc, c, i+1) #, "depth", depth, freq)
                        if str(i+1) in ambiguity_dict:
                            print(ambiguity_dict[str(i+1)])
                        target_points.append(str(i+1))
                    elif c == "N":
                        print(lineage, i+1, gc, c)
            """
        print(autoencoder_dict.keys()) 
    
    # x frequency (categorical)
    # y edit distance (continuous)
    # hue lineage (categorical)

    plt.tight_layout()
    sns.boxplot(x=x, y=y, hue=hue, palette=lineage_palette)
    plt.ylabel("Number of Nucleotide Differences")
    plt.savefig("consensus_boxplot.png", bbox_inches='tight')

def pure_consensus(samples, gt_dict, consensus_threshold=0.90):
    consensus_calls = {}
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
        gt_lineage = gt_dict[sample_id]['gt_lineages'][0]
        if gt_lineage not in consensus_calls:
            consensus_calls[gt_lineage] = []
        sequence = ""
        for i, (position, value) in enumerate(variants.items()):
            if i == 0:
                continue
            nuc_counts = value[1]
            nuc_freq = value[0]
            total_depth = sum(list(nuc_counts.values())) 
            if total_depth < 10:
                sequence += "N"
                continue
            canon = ""
            for n, f in nuc_freq.items():
                if f > consensus_threshold:
                    canon += n
            if len(canon) > 1 or len(canon) == 0:
                sequence += "N"
                #print(nuc_freq, position)
            else:
                sequence += canon
        consensus_calls[gt_lineage].append(sequence)
    
    exclude_positions = {} 
    for lineage, consensus in consensus_calls.items():
        exclude_positions[lineage] = []
        for i, (a, b, c, d) in enumerate(zip(consensus[0], consensus[1], consensus[2], consensus[3])):
            if a != b != c != d:                    
                exclude_positions[lineage].append(i)

    return(exclude_positions, consensus_calls)

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
    model_scores = [] # hue (continuous)
    for sample_id in samples:
        if sample_id not in gt_dict:
            continue
        output_dir = "/home/chrissy/Desktop/saga_spike_in_results/" + sample_id
        output_name = sample_id
        model_json = os.path.join(output_dir, output_name+"_model_results.txt")
        if os.path.isfile(model_json):
            with open(model_json, "r") as rfile:
                model_dict = json.load(rfile)
                if 'assignment_data' not in model_dict:
                    continue
                scores = model_dict['scores']   
                assignments = model_dict['assignments']
                assignment_data = model_dict['assignment_data']
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
                scores = [x for x in scores]
                model_score = sum(scores)
                model_scores.append(model_score)
                if 0.2 in centers and 0.80 not in centers:
                    print(predicted_centers)
                    print(centers)
                    print(model_score)
                    print(len(assignment_data))
                    """ 
                    for a, s, p in zip(assignments, scores, assignment_data):
                        print(a, s, p)
                    """
                cos_score = cosine(predicted_centers, centers)
                #print(cos_score)
                cosine_score.append(1-cos_score)
                

    g = sns.boxplot(x=mix_type, y=model_scores, palette=palette, hue_order=categories)
    plt.tight_layout()
    #plt.title("Spike In")
    plt.ylabel("Cosine Similarity")
    plt.setp(g.axes, xticks=[], xlabel='') # remove x ticks and xlabel
    handles = []
    for category,color in zip(categories, palette):
        patch = mpatches.Patch(color=color, label=category)
        handles.append(patch)
    plt.legend(handles=handles, loc="upper right", bbox_to_anchor=(1.1, 1.05))
    plt.savefig("cosine_plot.png", bbox_inches='tight')

def build_complexity_plot(samples, gt_dict):
    gt_length = []
    ambiguity_length = []
    for sample_id in samples:
        if sample_id not in gt_dict:
            continue
        output_dir = "/home/chrissy/Desktop/saga_spike_in_results/" + sample_id
        output_name = sample_id
        model_json = os.path.join(output_dir, output_name+"_model_results.txt")
        if os.path.isfile(model_json):
            with open(model_json, "r") as rfile:
                model_dict = json.load(rfile)
                if 'amb_length' not in model_dict or "filtered_mut_length" not in model_dict:
                    continue
                gt_centers = gt_dict[sample_id]['gt_centers']
                print(model_dict.keys())
                gt_length.append(len(gt_centers))
                ambiguity_length.append(model_dict['amb_length'])

    sns.regplot(x=gt_length, y=ambiguity_length)
    plt.tight_layout()
    plt.title("Spike In, Relationship between # Populations and Number of Ambiguous Positions")
    plt.xlabel("Ground Truth Number of Populations")
    plt.ylabel("Number of Ambiguous Positions")
    plt.savefig("output5.png", bbox_inches='tight')

def build_r_value_plot(spike_ins, directory_bam, gt_dict):
    finished = 0
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
                if "percent_mut" in model_dict:
                    percent_mut = model_dict['percent_mut']
                    print(percent_mut)
                    print(gt_dict[sample_id])
                    finished += 1
                if "all_trained_solutions" in model_dict:
                    all_trained_solutions = model_dict['all_trained_solutions']
    print("files finished processing...", finished, "...out of total...", len(gt_dict))

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

        print("creating results for %s sample..." %sample_id)
        output_dir = "/home/chrissy/Desktop/saga_spike_in_results/" + sample_id
        output_name = sample_id
        output_fasta_name = "/home/chrissy/Desktop/saga_spike_in_results/%s/%s.fa" %(sample_id, sample_id)
        bam_file = directory_bam + "/%s_sorted.calmd.bam" %sample_id

        variants_json = os.path.join(output_dir, output_name+"_variants.txt")
        if os.path.isfile(variants_json):
            with open(variants_json, "r") as rfile:
                primer_dict = json.load(rfile)
        else:
            continue
        flagged_primers = primer_dict["flagged_primers"]
        flagged_dist = primer_dict["flagged_dist"]
        single_primers = primer_dict["single_primers"]
        variants = primer_dict['variants']
        gt_frequencies = gt_dict[sample_id]["gt_centers"]
        #if 0.10 not in gt_frequencies and 0.05 not in gt_frequencies:
        #    continue
        gt_lineages = gt_dict[sample_id]["gt_lineages"]
        unique_primers = primer_dict['unique_primers']
        z_scores = primer_dict['z_scores']
        mut_dict = {}
        for l in gt_lineages:
            if l not in master_l_dict:
                tmp = file_util.parse_usher_barcode([l], barcode)
                for k,v in tmp.items():
                    master_l_dict[k] = v
                    mut_dict[k] = v
            else:
                mut_dict[l] = master_l_dict[l]
        lower_bound = 0.01
        upper_bound = 0.99
        low_depth_positions = []       
        depth_cutoff = 50
        """ 
        lp = LineProfiler()
        lp_wrapper = lp(file_util.parse_variants)
        lp_wrapper(primer_dict, primer_positions, reference_sequence)
        lp.print_stats()
        sys.exit(0)
        """
        frequencies, nucs, positions, low_depth_positions, \
            reference_variants, \
            ambiguity_dict, total_muts = file_util.parse_variants(primer_dict, primer_positions, reference_sequence)
        print("start iterations")
        for position, value in variants.items():
            nuc_frequency = value[0]
            for nuc, freq in nuc_frequency.items():
                if freq < lower_bound or freq > upper_bound:
                    continue            
                #find out which strain this position/nuc combo belongs to
                var = str(position)+str(nuc)
                belong_lineage = []
                for lineage, key_mutations in mut_dict.items():
                    if var in key_mutations:
                        belong_lineage.append(lineage)
                if len(belong_lineage) < 1:
                    continue
                idx = gt_lineages.index(belong_lineage[0])
                mix = "/".join([str(int(x*100)) for x in gt_frequencies])
                dist = round(abs(float(freq) - gt_frequencies[idx]),3)
                y.append(dist)
                if str(position) in ambiguity_dict:
                    flagged_reasons = ambiguity_dict[str(position)]
                    flagged_reasons = "\n&".join(flagged_reasons)
                    if dist < 0.05:
                        accuracy.append("falsely removed")
                        plot_a_removal.append(flagged_reasons)
                        plot_a_mix_type.append(mix)                        
                        #print("flag_not_exp", nuc, freq, belong_lineage, "dist", dist, position, gt_lineages, gt_frequencies, ambiguity_dict[position])
                    else:
                        accuracy.append("correctly removed")
                    hue.append(flagged_reasons)
                else:
                    flagged_reasons = "not flagged"
                    hue.append("not flagged")
                    if dist > 0.20:
                        accuracy.append("falsely kept")
                        print("false kept", freq, belong_lineage, position, gt_lineages, gt_frequencies)
                    else:
                        accuracy.append("correctly kept")
                mix_type.append(mix)
    colors = list(np.unique(hue))
    categories = ["5/95", "10/90", "20/80"]

    
    #this is plot a, a breakdown of what got falsely removed and why     
    plt.clf()
    plt.close()
    plt.figure(figsize=(10, 6), dpi=80)
    colors_needed = len(np.unique(plot_a_removal))
    print("plot a the number of colors needed, ", colors_needed)
    df = pd.DataFrame({"mix_type": plot_a_mix_type, "removal_reason":plot_a_removal})
    sns.histplot(data=df, x="mix_type", hue="removal_reason", multiple="fill", stat="proportion", discrete=True, shrink=.8, palette=palette[:colors_needed])
    plt.savefig("false_removal_data_cleaning.png", bbox_inches='tight')
    #sys.exit(0)
    
    #plots b
    plt.clf()
    plt.close()
    df = pd.DataFrame({"mix_type":mix_type, "flag_status":hue, "y":y, "accuracy":accuracy})
    print(df)
    hue_order = ["correctly kept", "correctly removed", "falsely kept", "falsely removed"]
    sns.histplot(data=df, x="mix_type", hue="accuracy", multiple="fill", stat="proportion", discrete=True, shrink=.8, hue_order = hue_order, palette=palette[:4])
    plt.savefig("output1.png", bbox_inches='tight')
    sys.exit(0)
    
    df = df[df['mix_type'].isin(categories)]
    g = sns.boxplot(x='mix_type', y='y', hue='flag_status', data =df, palette=sns.color_palette("husl", len(colors)))
    sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
    #plt.setp(g.axes, xticks=[], xlabel='') # remove x ticks and xlabel
    plt.tight_layout()
    plt.ylabel("Distance from Ground Truth Population Frequency")
    plt.savefig("output2.png", bbox_inches='tight')


if __name__ == "__main__":
    main()
