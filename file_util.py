import os
import sys
import pysam
import itertools
import numpy as np
import pandas as pd
#import networkx as nx
from line_profiler import LineProfiler

def parse_reference_sequence(ref_file="/home/chrissy/Desktop/sequence.fasta"):
    reference_sequence = ""
    with open(ref_file, "r") as rfile:
        for line in rfile:
            line = line.strip()
            if line.startswith(">"):
                continue
            reference_sequence += line
    return(reference_sequence)

def remove_depth(read, remove_pos_dict, reference_sequence):
    ref_pos = read.get_reference_positions()
    for quer, re, p in zip(read.query_alignment_sequence, reference_sequence[ref_pos[0]:ref_pos[-1]+1], ref_pos):        
        if quer != re and quer != "N":
            p += 1
            if p not in remove_pos_dict:
                remove_pos_dict[p] = {"A":0, "C":0, "G":0, "T":0}
            remove_pos_dict[p][quer] += 1
    return(remove_pos_dict)

def parse_bam_depth_per_position(bam_file, bed_file, contig="NC_045512.2"):
    total_coverage = []
    for i in range(1, 30000):
        total_coverage.append({"A":0, "C":0, "G":0, "T":0, "N":0})

    reference_sequence = parse_reference_sequence()
    remove_pos_dict = {}
    ref_dict = {} #track how many times the ref appears with the mut
    samfile = pysam.AlignmentFile(bam_file, "rb")
    primer_positions = parse_bed_file(bed_file)
    
    last_pos_val = np.array([x[2] for x in primer_positions])
    first_pos_val = np.array([x[1] for x in primer_positions])
    test = []
    messed_up_primers = []
    
    for i, read in enumerate(samfile.fetch(contig)):
        if i % 100000 == 0:
            print(i)
        found = False
        ref_pos = read.get_reference_positions()
        query_seq = read.query_alignment_sequence
        for rp, qs in zip(ref_pos, query_seq):
            total_coverage[rp+1][qs] += 1
            
        if read.is_reverse:
            last_base = ref_pos[-1]
            idx = np.abs(last_pos_val-last_base).argmin()
            for primer in primer_positions[idx:]:
                if found:
                    break
                if (last_base >= primer[2] - 10 and last_base <= primer[2] + 10):
                    found = True
                    align_end = read.query_alignment_end
                    query_seq = read.query_sequence
                    clipped_positions = list(range(last_base, len(query_seq)+last_base))
                    soft_clipped_bases = query_seq[align_end:]
                    ref_sc = reference_sequence[last_base+1:last_base+(len(query_seq)-align_end)+1]
                    read_mut_count = 0
                    #experimental
                    first_base = ref_pos[0]
                    align_start = read.query_alignment_start
                    clipped_positions_start = list(range(first_base - align_start, first_base))
                    soft_clipped_bases_start = query_seq[:align_start]
                    ref_sc_start = reference_sequence[first_base-align_start:first_base]

                    if ref_sc == soft_clipped_bases:
                        diff = [x for x,y in zip(ref_sc_start, soft_clipped_bases_start) if x != y]
                        if len(diff) > 4:
                            messed_up_primers.append(read.query_name)
                        continue
                    for ref_nuc, sc_nuc, pos in zip(ref_sc, soft_clipped_bases, clipped_positions):
                        #if primer[2] <= pos <= primer[3]:
                        #    pass
                        #else:
                        #    continue
                        if ref_nuc == sc_nuc:
                            continue
                        read_mut_count += 1
                    if read_mut_count >= 1:
                        messed_up_primers.append(read.query_name)
        else:
            first_base = ref_pos[0]
            idx = np.abs(first_pos_val-first_base).argmin()

            for primer in primer_positions[idx:]:
                if found:
                    break
                if (first_base >= primer[1] - 10 and first_base <= primer[1] + 10):
                    found = True
                    align_start = read.query_alignment_start
                    query_seq = read.query_sequence
                    clipped_positions = list(range(first_base - align_start, first_base))
                    soft_clipped_bases = query_seq[:align_start]
                    ref_sc = reference_sequence[first_base-align_start:first_base]
                    read_mut_count = 0              
                    if ref_sc == soft_clipped_bases:
                        continue
                    for ref_nuc, sc_nuc, pos in zip(ref_sc, soft_clipped_bases, clipped_positions):
                        #if primer[0] <= pos <= primer[1]:
                        #    pass
                        #else:
                        #    continue
                        if ref_nuc == sc_nuc:
                            continue
                        read_mut_count += 1
                    if read_mut_count >= 1:
                        messed_up_primers.append(read.query_name)
        if found is False:
            messed_up_primers.append(read.query_name)
    messed_up_primers = set(np.unique(messed_up_primers))
    #now we remove mutations based on this messed up primer list
    for i, read in enumerate(samfile.fetch(contig)):
        if read.query_name not in messed_up_primers:
            continue
        ref_pos = read.get_reference_positions()
        if read.query_alignment_sequence != reference_sequence[ref_pos[0]:ref_pos[-1]+1]:
            remove_pos_dict = remove_depth(read, remove_pos_dict, reference_sequence)
        else:
            for p in ref_pos:
                p += 1
                if p not in ref_dict:
                    ref_dict[p] = {"A":0, "C":0, "G":0, "T":0}
                ref_dict[p][reference_sequence[p-1]] += 1
    #print(remove_pos_dict[28881])
    #print(total_coverage[28881])
    problem_positions = [] 
    for key, value in remove_pos_dict.items():
        keep = False
        total_dict = total_coverage[key]
        total_depth = sum(list(total_dict.values()))
        for k, v in value.items():
            if  v/total_depth > 0.05 and total_depth > 10:
                keep = True
                break
        if keep:
            if key in ref_dict:
                value_list = list(value.values())
                coverage_specific = list(total_coverage[key].values())
                coverage_mut_ref = list(ref_dict[key].values())
                norm_mut_mut = [round(x/y,2) if y > 10 else 0 for x,y in zip(value_list, coverage_specific)]
                norm_mut_ref = [round(x/y,2) if y > 10 else 0 for x,y in zip(coverage_mut_ref, coverage_specific)]
                
                keep = False
                high_mut_ref = [x for x in norm_mut_ref if x > 0.10]
                high_mut_mut = [x for x in norm_mut_mut if x > 0.50]
                if len(high_mut_ref) == 0 and len(high_mut_mut) > 0:
                    problem_positions.append(key)
                    print("problem") 
                print(key, "mut", norm_mut_mut)
                print("ref", norm_mut_ref)
                print("mut", value_list)
                print("ref", coverage_mut_ref)
                #for mm, mr in zip(norm_mut_mut, norm_mut_ref):
                #    print(mm, mr)
            else:
                print("not found ref", key, value)
                value_list = list(value.values())
                coverage_specific = list(total_coverage[key].values())
                norm_mut_mut = [round(x/y,2) if y > 10 else 0 for x,y in zip(value_list, coverage_specific)]
                high_mut_mut = [x for x in norm_mut_mut if x > 0.50]
                print(norm_mut_mut)
                if len(high_mut_mut) > 0.50:
                    problem_positions.append(key)

    return(remove_pos_dict, problem_positions)

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
       
def parse_ivar_variants_file(file_path, frequency_precision=4, problem_positions=None, bed_file=None, problem_primers = None, remove_pos_dict=None):
    """
    Use the variants .tsv output from ivar to create a list of variants and frequencies for 
    downstream use.
    """
    print("parsing ivar variants file...")
    positions = []
    frequency = []
    nucs = []
    depth = []
    primer_positions = []
    problem_amplicons = []

    if problem_primers is not None:
        for primer in problem_primers:
            problem_amplicons.append([primer[0], primer[3]])

    if bed_file is not None:
        primer_positions = parse_bed_file(bed_file)

    df = pd.read_table(file_path)

    low_depth_positions = []
    reference_variants = []

    #keep track of the reference positions/frequencies/nucs
    unique_pos = []
    unique_freq = []
    ref_nucs = []
    for index, row in df.iterrows():
        if "N" in row['ALT'] or "+" in row['ALT']:
            continue
        p = row['POS']
        n = row['ALT']
        ad = int(row['ALT_DP'])
        rd = int(row['REF_DP'])
        total_depth = int(row['TOTAL_DP']) 
        
        if total_depth < 50:
            low_depth_positions.append(str(p))
            continue
        if problem_positions is not None and int(p) in problem_positions:
            if row['ALT_FREQ'] > 0.03:
                print("removing...", p, n, row['ALT_FREQ'])
            continue    
        f = float(row['ALT_FREQ'])
        if bed_file is not None:
            #here we check if this exists in a primer position
            if f > 0.10:
                tmp_p = int(p)
                for amplicon in primer_positions:
                    if amplicon[0] < tmp_p < amplicon[1] or amplicon[2] < tmp_p < amplicon[3]:
                        tmp = [amplicon[0], amplicon[3]]
                        if tmp not in problem_amplicons:
                            problem_amplicons.append(tmp)                 
        positions.append(p)
        frequency.append(round(f, frequency_precision))
        nucs.append(n)
        
        if p not in unique_pos:
            unique_pos.append(p)
            unique_freq.append(round(f, frequency_precision))
            ref_nucs.append(row['REF'])
        else:
            loc = unique_pos.index(p)
            unique_freq[loc] += round(f, frequency_precision)

    ref_freq = [1-x for x in unique_freq]
    frequency.extend(ref_freq)
    positions.extend(unique_pos)
    nucs.extend(ref_nucs)
   
    
    for pa in problem_amplicons:
        print(pa)

    reference_variants = [str(n) + '_' + str(p) for p,n in zip(unique_pos, ref_nucs) if n != 0]
    final_positions = [] 
    final_frequency = []
    final_nucs = []
    for p, f, n in zip(positions, frequency, nucs):
        tmp_p = int(p)
        keep = True
        for amplicon in problem_amplicons:
            if amplicon[0] < tmp_p < amplicon[1]:
                keep = False
                break
        if keep:
            final_positions.append(p)
            final_frequency.append(f)
            final_nucs.append(n)
    
    return(final_positions, final_frequency, final_nucs, low_depth_positions, reference_variants)
