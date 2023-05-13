import os
import sys
import json
import pysam
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from line_profiler import LineProfiler

def parse_reference_sequence(ref_file="/home/chrissy/Desktop/sequence.fasta"):
    """
    Open and parse a reference sequence to a string.
    """
    reference_sequence = ""
    with open(ref_file, "r") as rfile:
        for line in rfile:
            line = line.strip()
            if line.startswith(">"):
                continue
            reference_sequence += line
    return(reference_sequence)

def remove_depth(read, remove_pos_dict, reference_sequence, ref_pos, query_seq):
    """
    Given a query alignment sequence where the primer has a mutation, find and record mutations byt position.
    """
    #not going to bother with insertions
    if len(query_seq) != len(reference_sequence[ref_pos[0]:ref_pos[-1]+1]):
        return(remove_pos_dict)
    
    for i, (quer, p) in enumerate(zip(query_seq, ref_pos)):
        if i > 0 and p-1 != ref_pos[i-1]:
            i += 1
        if i >= len(ref_pos):
            continue            
        re = reference_sequence[ref_pos[i]]
        if quer != re and quer != "N":
            remove_pos_dict[p+1][quer] += 1
    return(remove_pos_dict)

def parse_variants(bam_dict, bed_file, reference_file, depth_cutoff=50):
    """
    """
    single_primers = bam_dict["single_primers"]
    flagged_dist = bam_dict["flagged_dist"]
    flagged_primers = bam_dict["flagged_primers"]
    variants = bam_dict["variants"]

    ambiguity_dict = {}
    model_frequencies = []
    low_depth_positions = []
    primer_binding_issue = []    
    reference_variants = []

    primer_positions = parse_bed_file(bed_file)
    reference_sequence = parse_reference_sequence(reference_file)

    #look for low depth
    for position, value in variants.items():
        nuc_counts = value[1]
        total_depth = sum(list(nuc_counts.values()))
        if total_depth < depth_cutoff:
            low_depth_positions.append(position)

    #find primers with binding mutations
    for position, value in variants.items():
        if position in low_depth_positions:
            continue
        #its in a primer binding region
        in_primer = False
        for primer in primer_positions:
            if primer[0] < int(position) < primer[1] or primer[2] < int(position) < primer[3]:
                in_primer = True
                break
        if in_primer is False:
            continue
        #it doesn't match the nuc and occurs > 10%
        nuc_frequency = value[0]
        ref_nuc = reference_sequence[int(position)-1]
        for nuc, freq in nuc_frequency.items():
            if nuc == ref_nuc:
                continue
            if freq > 0.10:
                amplicon = [primer[1], primer[2]]
                if amplicon not in primer_binding_issue:
                    primer_binding_issue.append(amplicon)
                break
    nucs = []
    frequencies = []
    positions = []
    first_primer_index = [x[0] for x in primer_positions]
    for position, value in variants.items():
        nuc_frequency = value[0]
        nuc_counts = value[1]
        ref_nuc = reference_sequence[int(position)-1]
        primer_issue = False
        for primer in primer_binding_issue:
            if int(primer[0]) < int(position) < int(primer[1]) and int(position) in single_primers:
                primer_issue = True  
                ambiguity_dict[position] = "primer_binding"
                break
        if primer_issue:
            continue 
        if int(position) in flagged_dist:
            ambiguity_dict[position] = "amplicon_flux"
            continue

        flagged = False
        for primer in flagged_primers:
            idx = first_primer_index.index(primer)
            amplicon = primer_positions[idx]
            if amplicon[1] < int(position) < amplicon[2]:
                flagged=True
                ambiguity_dict[position] = "primer_mutation"
                break
        if flagged is True:
            continue 
        for nuc, freq in nuc_frequency.items():
            if nuc == "N":
                continue
            if float(freq) == 0:
                continue
            frequencies.append(float(freq))
            nucs.append(nuc)
            positions.append(int(position))
            if nuc == ref_nuc:
                reference_variants.append(str(position) + "_" + nuc)
    return(frequencies, nucs, positions, low_depth_positions, reference_variants, ambiguity_dict)

def parse_bam_depth_per_position(bam_file, bed_file, variants_json, contig="NC_045512.2"):
    read_dict = []
    mut_dict = []
    reference_sequence = parse_reference_sequence()
    for i in range(1, len(reference_sequence)+1):       
        mut_dict.append({"A":[], "C":[], "G":[], "T":[], "N":[]})
        read_dict.append({"A":[], "C":[], "G":[], "T":[], "N":[]})
                

    samfile = pysam.AlignmentFile(bam_file, "rb")
    primer_positions = parse_bed_file(bed_file)
    
    #for quickly indexing which primer is closest 
    last_pos_val = np.array([x[2] for x in primer_positions])
    first_pos_val = np.array([x[1] for x in primer_positions])
    messed_up_primers = {}
    all_primer_muts = [] 
    all_primer_names = []
    
    for i, read in enumerate(samfile.fetch(contig)):
        if i % 100000 == 0:
            print(i)
        
        #if i < 1500000 or i > 1800000:
        #    continue
        
        ref_pos = read.get_reference_positions()
        found = False
        if read.is_reverse: 
            last_base = ref_pos[-1]
            idx = np.abs(last_pos_val-last_base).argmin()
            for primer in primer_positions[idx:]:
                if found:
                    break
                if (last_base >= primer[2] - 10 and last_base <= primer[2] + 10):
                    name = read.query_name + "_R"
                    query_align_seq = read.query_alignment_sequence
                    #not touching reads with insertions
                    if len(query_align_seq) == len(reference_sequence[ref_pos[0]:ref_pos[-1]+1]): 
                        for quer, p in zip(query_align_seq, ref_pos):                            
                            k = p + 1
                            mut_dict[k][quer].append(primer[0])
                            read_dict[k][quer].append(name)

                    found = True
                    align_end = read.query_alignment_end
                    query_seq = read.query_sequence
                    clipped_positions = list(range(last_base, len(query_seq)+last_base))
                    soft_clipped_bases = query_seq[align_end:]
                    ref_sc = reference_sequence[last_base+1:last_base+(len(query_seq)-align_end)+1]     
                    mut_count = 0
                    if ref_sc != soft_clipped_bases:
                        for r,s in zip(ref_sc, soft_clipped_bases):
                            if r != s:
                                mut_count += 1
                    messed_up_primers[name] = mut_count
                    all_primer_muts.append(mut_count)
                    all_primer_names.append(primer[0])
        else:
            first_base = ref_pos[0]
            idx = np.abs(first_pos_val-first_base).argmin()
            for primer in primer_positions[idx-1:]:
                if found:
                    break
                if (first_base >= primer[1] - 10 and first_base <= primer[1] + 10):
                    name = read.query_name + "_F"
                    query_align_seq = read.query_alignment_sequence
                    if len(query_align_seq) == len(reference_sequence[ref_pos[0]:ref_pos[-1]+1]): 
                        for quer, p in zip(query_align_seq, ref_pos):
                            k = p + 1
                            mut_dict[k][quer].append(primer[0])
                            read_dict[k][quer].append(name)

                    found = True    
                    align_start = read.query_alignment_start
                    query_seq = read.query_sequence
                    clipped_positions = list(range(first_base - align_start, first_base))
                    soft_clipped_bases = query_seq[:align_start]
                    ref_sc = reference_sequence[first_base-align_start:first_base]
                    mut_count = 0
                    if ref_sc != soft_clipped_bases:
                        for r,s in zip(ref_sc, soft_clipped_bases):
                            if r != s:
                                mut_count += 1
                    messed_up_primers[name] = mut_count
                    all_primer_muts.append(mut_count)
                    all_primer_names.append(primer[0])
    
    percentile = 97 
    ninty_seven = np.percentile(all_primer_muts, [percentile])
    unique_primers, counts = np.unique(all_primer_names, return_counts=True)
    unique_primers = list(unique_primers)
    removal = [0] * len(unique_primers)
    for pm, pn in zip(all_primer_muts, all_primer_names):
        if pm >= ninty_seven:
            idx = unique_primers.index(pn)
            removal[idx] += 1
    percent_primer_mut = []
    for c, r, p in zip(counts, removal, unique_primers):
        percent_primer_mut.append(r/c)
    
    outliers = []
    flagged_primers = []
    threshold = 3
    z_scores = np.abs(stats.zscore(percent_primer_mut))
    for z, ppm in zip(z_scores, percent_primer_mut):
        idx = percent_primer_mut.index(ppm)
        if z > threshold:
            outliers.append(ppm)
            flagged_primers.append(unique_primers[idx])

    flagged_dist = []
    single_primers= []
    variants = {}
    for key, value in enumerate(mut_dict):
        #if key != 27627 and key != 28698:
        #    continue 
        num_primers = []
        nuc_counts = {"A":0, "C":0, "G":0, "T":0, "N":0}
        for k,v in value.items():
            num_primers.extend(v)
            nuc_counts[k] += len(v)

        total_depth = len(num_primers)
        nuc_freq = {}
        for k,v in nuc_counts.items():
            if total_depth > 0:
                nuc_freq[k] = v/total_depth
        
        variants[key] = [nuc_freq, nuc_counts]
        num_primers = list(np.unique(num_primers))
        effective_num_primers = []
        tracking_depth = []

        for i in range(len(num_primers)):
            tracking_depth.append([0,0,0,0])
        for i, (nuc, primer) in enumerate(value.items()):
            associated_reads = read_dict[key][nuc]
            if len(associated_reads) < 50 or len(associated_reads)/total_depth < 0.03:
                continue            
            print("\n", key, nuc)
            unique, counts = np.unique(primer, return_counts=True)
            unique = list(unique)
            counts = list(counts)
            if len(counts) == 0:
                continue
            for u, c in zip(unique, counts):
                print("primer", u, "count", c)
                idx = num_primers.index(u)
                tracking_depth[idx][i] += c
                if c > 10:
                    if u not in effective_num_primers:
                        effective_num_primers.append(u)
        if len(effective_num_primers) > 1:
            td = np.array(tracking_depth)
            if np.count_nonzero(td) == 0:
                continue
            denom = np.sum(td, axis=0)
            a = td/denom
            a[np.isnan(a)] = 0
            a = a.T 
            a = a[~np.all(a == 0, axis=1)]
            threshold = 0.05
            flag = False
            for i in range(a.shape[1]):
                v = list(a[:,i])
                dist = [abs(i-j) for i in v for j in v if i != j]
                dist = np.mean(np.unique(dist))           
                if dist > threshold:
                    flag = True
                print(v)
                print(dist)
            if flag:
                print("dist flagged", key)
                flagged_dist.append(key)
        else:
            single_primers.append(key)
    #change types for json dump
    flagged_primers = [int(x) for x in flagged_primers] 
    z_scores = list(z_scores)
    unique_primers = [int(x) for x in unique_primers]
    
    tmp_dict = {"flagged_primers":flagged_primers, "flagged_dist":flagged_dist, "single_primers":single_primers, \
        "z_scores":z_scores, "percent_primer_mut":percent_primer_mut, "unique_primers":unique_primers, "variants":variants}
    
    with open(variants_json, "w") as bfile:
        bfile.write(json.dumps(tmp_dict))    
    
    return(tmp_dict)

def parse_usher_barcode(lineages, usher_file = "/home/chrissy/Desktop/usher_barcodes.csv", return_ref=False):
    print("parsing usher barcodes...")
    barcode = pd.read_csv(usher_file)
    
    #initialize dictionary for mutation pos
    l_dict = {}
    for l in lineages:
        l_dict[l] = []
        tmp = barcode[barcode['Unnamed: 0'] == l]
        tmp = tmp.loc[tmp.index.tolist()[0]].to_frame(name=l)[1:]
        tmp = tmp[tmp[l] == 1].index.tolist()
        if not return_ref:
            tmp = [x[1:] for x in tmp]
        l_dict[l] = tmp
    return(l_dict)


def parse_freyja_file(file_path, tolerance=0.99):
    """
    Open freyja results .tsv file and get the ground truth lineages and frequencies.
    """
    print("parsing freyja file...")

    actual_centers = []
    actual_lineages = []

    df = pd.read_table(file_path)
    df.columns = ["attribute", "value"]
    freyja_res = df.to_dict()
    
    lineages = freyja_res['value'][1].split(" ")
    centers = freyja_res['value'][2].split(" ")
    
    total_accounted = 0
    for l, c in zip(lineages, centers):
        total_accounted += float(c)
        if total_accounted > tolerance:
            break
        actual_centers.append(float(c))
        actual_lineages.append(str(l))
    return(actual_centers, actual_lineages)

def parse_bed_file(bed_file):
    """
    Parse bed file and paired primers to defined amplicon start and end positions.
    """
    print("parsing bed file...")
    primer_positions = []
    primer_pairs = "/home/chrissy/Desktop/primer_pairs.tsv"
    pdf = pd.read_table(primer_pairs, names=["left", "right"])
    for i in range(len(pdf)):
        primer_positions.append([0, 0, 0, 0])

    with open(bed_file, "r") as bfile:
        for line in bfile:
            line = line.strip()
            line_list = line.split("\t")
            start = line_list[1]
            end = line_list[2] 
            name = line_list[3]
            idx_l = pdf.index[pdf["left"] == name].tolist()
            idx_r = pdf.index[pdf["right"] == name].tolist()
            if len(idx_l) == 0 and len(idx_r) == 0:
                continue
            if len(idx_l) > 0:
                primer_positions[idx_l[0]][0] = int(start)
                primer_positions[idx_l[0]][1] = int(end)
            if len(idx_r) > 0:
                primer_positions[idx_r[0]][2] = int(start)
                primer_positions[idx_r[0]][3] = int(end)
    return(primer_positions)
