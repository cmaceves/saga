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

def parse_bam_depth_per_position(bam_file, bed_file, contig="NC_045512.2"):
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
        
        #if i < 1500000 or i > 1600000:
        #    continue
        
        ref_pos = read.get_reference_positions()
        #we have a deletion in the read
        #if ref_pos != list(range(ref_pos[0], ref_pos[-1]+1)):
        #    continue
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
                            """
                            n = ref_pos[i]
                            if i > 0 and p-1 != n-1:
                                i += 1
                                if i >= len(ref_pos):
                                    continue            
                                re = reference_sequence[ref_pos[i]]
                                mut_dict[k][quer].append(primer[0])
                                read_dict[k][quer].append(name)

                            else:
                            """
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
    eighty_five, ninty, ninty_five, ninty_seven, ninty_nine = np.percentile(all_primer_muts, [85, 90, 95, 97, 99])
    print("85", eighty_five)
    print("90", ninty)
    print("95", ninty_five)
    print("97", ninty_seven)
    print("99", ninty_nine)
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
    threshold = 1 
    z_scores = np.abs(stats.zscore(percent_primer_mut))
    for z, ppm in zip(z_scores, percent_primer_mut):
        if z > threshold:
            outliers.append(ppm)
            idx = percent_primer_mut.index(ppm)
            flagged_primers.append(unique_primers[idx])
    flagged_dist = []
    single_primers= []
    for key, value in enumerate(mut_dict):
        #if key != 27627 and key != 28461:
        #    continue 
        num_primers = []
        for k,v in value.items():
            num_primers.extend(v)
        total_depth = len(num_primers)
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
            threshold = 0.10
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
                flagged_dist.append(key)
        else:
            single_primers.append(key)
    
    tmp_dict = {"flagged_primers":flagged_primers, "flagged_dist":flagged_dist, "single_primers":single_primers}
    with open("primer_issues.txt", "w") as bfile:
        bfile.write(json.dumps(tmp_dict))    
    return(flagged_dist, flagged_primers, single_primers)

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


  
def parse_ivar_variants_file(file_path, frequency_precision=4, bed_file=None, primer_dict=None):
    """
    Use the variants .tsv output from ivar to create a list of variants and frequencies for 
    downstream use.
    """
    print("parsing ivar variants file...")
    positions = []
    frequency = []
    nucs = []
    primer_positions = []
    problem_amplicons = []

    if bed_file is not None:
        primer_positions = parse_bed_file(bed_file)

    df = pd.read_table(file_path)

    low_depth_positions = []
    reference_variants = []
    call_ambiguity = []
    #keep track of the reference positions/frequencies/nucs
    unique_pos = []
    unique_freq = []
    ref_nucs = []

    if primer_dict is not None:
        single_primers = primer_dict["single_primers"]
        flagged_dist = primer_dict["flagged_dist"]
        flagged_primers = primer_dict["flagged_primers"]
        per_messed = primer_dict["per_messed"]
    
    """
    eighty_five, ninty, ninty_five, ninty_nine = np.percentile(per_messed, [85, 90, 95, 99])
    print("85", eighty_five)
    print("90", ninty)
    print("95", ninty_five)
    print("99", ninty_nine)
    #sns.kdeplot(per_messed)
    #plt.savefig("output.png") 
    sys.exit(0) 
    """
    for index, row in df.iterrows():
        #if it's a N, an insertion, or deletion we don't care
        if "N" in row['ALT'] or "+" in row['ALT'] or "-" in row['ALT']:
            continue
        p = row['POS']
        n = row['ALT']
        ad = int(row['ALT_DP'])
        rd = int(row['REF_DP'])
        total_depth = int(row['TOTAL_DP']) 
       
        #depth too low 
        if total_depth < 50:
            low_depth_positions.append(str(p))
            continue
        f = float(row['ALT_FREQ'])

        #this mutation heavily co-occurs with > 3 mutations in a primer binding site 
        if int(p) in flagged_primers and f > 0.03 and f < 0.98:
            print(p, n, round(f,2), "flagged primers")
            call_ambiguity.append(p)
            continue
            
        if bed_file is not None:
            #here we check if this exists in a primer position
            if f > 0.10:
                tmp_p = int(p)
                for amplicon in primer_positions:
                    if amplicon[0] < tmp_p < amplicon[1] or amplicon[2] < tmp_p < amplicon[3]:
                        tmp = [amplicon[1], amplicon[2]]
                        if tmp not in problem_amplicons:
                            print("problem amp", amplicon, p, n, round(f,2))
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
       
        #mutation frequency affected, reason unknown
        if tmp_p in flagged_dist:
            call_ambiguity.append(tmp_p)
            print("removing due to flagged dist...", tmp_p, n, round(f,2))
            keep = False
        else:
        #primer binding issues affecting mutation frequency
            for amplicon in problem_amplicons:
                if amplicon[0] < tmp_p < amplicon[1]:
                    if f > 0.03 and f < 0.98:
                        print('h', tmp_p, round(f,2))
                    if (tmp_p in single_primers):
                        if f > 0.03 and f < 0.98:
                            call_ambiguity.append(tmp_p)
                            print("removing due to primer binding...", tmp_p, n, round(f,2), amplicon[0], amplicon[1])
                        keep = False
                        break
        if keep:
            final_positions.append(p)
            final_frequency.append(f)
            final_nucs.append(n)
    call_ambiguity = list(np.unique(call_ambiguity))
    call_ambiguity.sort()
    print("call ambiguity", call_ambiguity)
    print(flagged_dist)
    return(final_positions, final_frequency, final_nucs, low_depth_positions, reference_variants)
