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
DEBUG=False

def parse_additional_var(bam_dict, primer_positions, reference_sequence, ambiguity_dict, \
    lower_bound, upper_bound, depth_cutoff=50):
    flagged_dist = bam_dict["flagged_dist"]
    value_flagged_dist = bam_dict['value_flagged_dist']
    variants = bam_dict["variants"]
    model_frequencies = []
    low_depth_positions = []
    primer_binding_issue = []    
    call_ambiguity = []
    #look for low depth
    for position, value in variants.items():
        nuc_counts = value[1]
        #ungapped depth
        total_depth = sum(list(nuc_counts.values())) - nuc_counts['-']
        if total_depth < depth_cutoff:
            low_depth_positions.append(int(position))
            continue
    #find primers with binding mutations
    for position, value in variants.items():
        if int(position) in low_depth_positions:
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
            if freq > 0.05:
                amplicon = [primer[1], primer[2]]
                if amplicon not in primer_binding_issue:
                    #print(nuc, freq, amplicon, position)
                    primer_binding_issue.append(amplicon)
                break
    nucs = []
    frequencies = []
    positions = []
    first_primer_index = [x[0] for x in primer_positions]
    for position, value in variants.items():
        if int(position) in low_depth_positions:
            continue           
        nuc_frequency = value[0]
        nuc_counts = value[1]
        ref_nuc = reference_sequence[int(position)-1]
      
        if int(position) in flagged_dist:
            idx = flagged_dist.index(int(position))
            distance = value_flagged_dist[idx]
            dist = np.array(distance)
            for v in dist:
                a = [abs(i-j) for i in v for j in v if i != j]
                b = np.mean(np.unique(a))
            if b > 0.15:
                call_ambiguity.append(int(position))
    
    #make a list of primers we think are subject to amplicon flux
    amplicon_flux_amplicons = []
    """
    for primer in primer_positions:
        for ca in call_ambiguity:
            if primer[1] < ca < primer[2]:
                tmp = list(range(primer[1], primer[2]))
                amplicon_flux_amplicons.extend(tmp)
                break
    """
    universal_mutations = []
    for position, value in variants.items():
        if int(position) in call_ambiguity or int(position) in low_depth_positions:
            continue 
        nuc_frequency = value[0]
        nuc_counts = value[1]
        ref_nuc = reference_sequence[int(position)-1]
        if len(nuc_frequency) == 0:
            continue
        for nuc, freq in nuc_frequency.items():
            #if nuc == ref_nuc:
            #    continue
            if nuc == "N":
                continue
            if float(freq) < lower_bound:
                continue
            #in suspect amplicon
            if int(position) in amplicon_flux_amplicons:
                call_ambiguity.append(int(position))
                continue
            if float(freq) > upper_bound and nuc != ref_nuc:
                universal_mutations.append(position+nuc)
                continue
            frequencies.append(float(freq))
            nucs.append(nuc)
            positions.append(int(position))

    low_depth_positions.extend(call_ambiguity)    
    return(frequencies, nucs, positions, low_depth_positions, universal_mutations)

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

def parse_variants(bam_dict, primer_positions, reference_sequence, depth_cutoff=50):
    flagged_dist = bam_dict["flagged_dist"]
    variants = bam_dict["variants"]
    ambiguity_dict = {}
    model_frequencies = []
    low_depth_positions = []
    primer_binding_issue = []    
    reference_variants = []
    first_primer_index = [x[0] for x in primer_positions]
    
    test = {}

    depth_cutoff = 50
    flagged_dist_amp = []
    for position in flagged_dist:
        position = int(position)
        for primer in primer_positions:
            if primer[1] < position < primer[2]:
                if primer not in flagged_dist_amp:
                    flagged_dist_amp.append(primer)
    primer_pos = []
    for primer in primer_positions:
        tmp = list(range(primer[0], primer[1]+1))
        primer_pos.extend(tmp)
        tmp = list(range(primer[2], primer[3]+1))
        primer_pos.extend(tmp)
    primer_pos = set(primer_pos)

    #look for low depth
    for position, value in variants.items():
        position = int(position)
        nuc_counts = value[1]
        total_depth = sum(list(nuc_counts.values()))
        if total_depth < depth_cutoff:
            low_depth_positions.append(position)
            continue

        #its in a primer binding region
        in_primer = False
        if int(position) in primer_pos:
            in_primer = True
        if in_primer is False:
            continue

        #it doesn't match the nuc and occurs > 10%
        nuc_frequency = value[0]
        ref_nuc = reference_sequence[int(position)-1]
        for nuc, freq in nuc_frequency.items():
            if nuc == ref_nuc:
                continue
            if freq > 0.05:
                for primer in primer_positions:
                    tmp = list(range(primer[0], primer[1]+1))
                    if position in tmp:
                        amplicon = [primer[1], primer[2]]
                        break
                    tmp = list(range(primer[2], primer[3]+1))
                    if position in tmp:
                        amplicon = [primer[1], primer[2]]
                        break
                tmp = list(range(amplicon[0], amplicon[1]))
                for pos in tmp:
                    if str(pos) not in test:
                        test[str(pos)] = []
                    test[str(pos)].append([freq, position, primer])
                          
                if amplicon not in primer_binding_issue:
                    primer_binding_issue.append(amplicon)
                break

    nucs = []
    frequencies = []
    positions = []
    depth = []
    total_mutated_frequencies = [] #without ambiguity removal
    possible_training_removed = [] 
    possible_train_freq_removed = []
    possible_train_nuc_removed = []
    for position, value in variants.items():
        if int(position) in low_depth_positions:
            if position not in ambiguity_dict:
                ambiguity_dict[position] = []
            ambiguity_dict[position].append("low_depth")
            
        nuc_frequency = value[0]
        nuc_counts = value[1]
        ref_nuc = reference_sequence[int(position)-1]
        primer_issue = False
        
        for primer in primer_binding_issue:
            if int(primer[0]) < int(position) < int(primer[1]): #and int(position) in single_primers:
                primer_issue = True 
                if position not in ambiguity_dict:
                    ambiguity_dict[position] = []
                ambiguity_dict[position].append("primer_mutation")
                break
        
        for primer in flagged_dist_amp:
            if int(primer[1]) < int(position) < int(primer[2]):
                if position not in ambiguity_dict:
                    ambiguity_dict[position] = []
                ambiguity_dict[position].append("amplicon_flux")
                break
        if len(nuc_frequency) == 0:
            continue

        deletion_impacted = False
        if nuc_frequency['-'] > 0.03:
             if position not in ambiguity_dict:
                ambiguity_dict[position] = []
             ambiguity_dict[position].append("deletion_position")
           
        for nuc, freq in nuc_frequency.items():
            if nuc == "N":
                continue
            if float(freq) == 0:
                continue
            if float(freq) > 0.0 and nuc != ref_nuc and nuc != "-":
               total_mutated_frequencies.append(float(freq))
            if position in ambiguity_dict:
                if nuc == ref_nuc:
                    reference_variants.append(str(position) + "_" + nuc)
                if freq > 0.03 and freq < 0.97:
                    #print(nuc, freq, position, ambiguity_dict[position])
                    if int(position) not in possible_training_removed:
                        possible_training_removed.append(int(position))
                        possible_train_freq_removed.append(float(freq))
                        possible_train_nuc_removed.append(nuc)
                continue
            if nuc == "-":
                continue
            frequencies.append(float(freq))
            nucs.append(nuc)    
            positions.append(int(position))
            depth.append(sum(list(nuc_counts.values())))
            if nuc == ref_nuc:
                reference_variants.append(str(position) + "_" + nuc)
    return(frequencies, nucs, positions, depth, low_depth_positions, reference_variants, ambiguity_dict, total_mutated_frequencies, possible_training_removed)

def parse_physcial_linkage(frequencies, nucs, positions, depth, bam_dict, r_max, freq_threshold=0.10):
    unique_link = bam_dict['total_unique_links']
    link_counts = bam_dict['total_link_counts']
    total_link_counts = bam_dict['total_possible_links']    

    #assumption is that one point is noise peak
    threshold = 1/(r_max-1)

    phys_link_fewer = []
    phys_counts_fewer = []
    phys_total_fewer = []

    joint_peak = []
    joint_nuc = []
    joint_dict = {}
    all_query = [str(p)+str(n) for p,n in zip(positions,nucs)]
    for ul, lc, tlc in zip(unique_link, link_counts, total_link_counts):
        if lc > 50 and len(ul) > 1 and tlc > 0:
            if lc/tlc > threshold:
                phys_link_fewer.append(ul)
                phys_counts_fewer.append(lc)
                phys_total_fewer.append(tlc)
                #print(ul, lc, tlc, lc/tlc)
                """
                for l in ul:
                    if l in all_query:
                        idx = all_query.index(l)
                        print(frequencies[idx])
                """
    for link,counts, total_count in zip(phys_link_fewer, phys_counts_fewer, phys_total_fewer):
        freq = [0] * len(link)
        for i,l in enumerate(link):
            if l in all_query:
                idx = all_query.index(l)
                freq[i] += frequencies[idx]
        idx = list(unique_link).index(link)        
       
        link = [x for x,y in zip(link, freq) if y > freq_threshold and y < 0.97]
        freq = [x for x in freq if x > freq_threshold and x < 0.97]
        count_over_thresh = len(freq)
        if count_over_thresh > 1:
            arr = np.array(freq)
            for l,f in zip(link, freq):
                if f > 0.97:
                    continue
                thresh = f * 0.20
                diff = f - arr
                kover = [i for i,x in enumerate(diff) if x > thresh]
                over = [x for x in diff if x > thresh]
                if len(over) > 0:
                    if l not in joint_nuc:
                        joint_peak.append(f)
                        if f not in joint_dict:
                            joint_dict[f] = []
                        joint_dict[f].append([x for i,x in enumerate(freq) if i in kover])
                        joint_nuc.append(l)
    return(joint_peak, joint_nuc, joint_dict)

def parse_bam_depth_per_position(bam_file, bed_file, variants_json, contig="NC_045512.2"):
    read_dict = []
    mut_dict = []
    phys_link_list = []    
    total_co_occurences = []

    reference_sequence = parse_reference_sequence()
    for i in range(1, len(reference_sequence)+1):       
        mut_dict.append({"A":[], "C":[], "G":[], "T":[], "N":[], "-": []})
        read_dict.append({"A":[], "C":[], "G":[], "T":[], "N":[], "-": []})
    samfile = pysam.AlignmentFile(bam_file, "rb")
    primer_positions = parse_bed_file(bed_file)

    #for quickly indexing which primer is closest 
    last_pos_val = np.array([x[2] for x in primer_positions])
    first_pos_val = np.array([x[1] for x in primer_positions])
    test = []
    for i, read in enumerate(samfile.fetch(contig)):
        test.append(read.query_name)
    read_names, read_name_counts = np.unique(test, return_counts=True)
    read_names = set([x for x,y in zip(read_names, read_name_counts) if y > 2])
    for i, read in enumerate(samfile.fetch(contig)):
        if i % 100000 == 0:
            print(i)
        if DEBUG:
            if i % 100000 == 0:
                print(i)
            if i < 600000 or i > 800000:
                continue   
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
                    if name[:-2] in read_names:
                        continue
                    query_align_seq = read.query_alignment_sequence                        
                    found = True
                    align_end = read.query_alignment_end
                    query_seq = read.query_sequence
                    clipped_positions = list(range(last_base, len(query_seq)+last_base))
                    soft_clipped_bases = query_seq[align_end:]
                    ref_sc = reference_sequence[last_base+1:last_base+(len(query_seq)-align_end)+1]    
                    #not touching reads with insertions
                    muts_linked = []
                    total_co_occurences.append(ref_pos)
                    if len(query_align_seq) == len(ref_pos): 
                        for i, (quer, p) in enumerate(zip(query_align_seq, ref_pos)):                
                            k = p + 1
                            #deletion
                            if ref_pos[i] - 1 != ref_pos[i-1] and i != 0:
                                diff = ref_pos[i] - ref_pos[i-1]
                                for j in range(1, diff):
                                    deleted_pos = ref_pos[i] - j
                                    mut_dict[deleted_pos+1]["-"].append(primer[0])
                                    read_dict[deleted_pos+1]["-"].append(name)
                             
                            mut_dict[k][quer].append(primer[0])
                            read_dict[k][quer].append(name)
                            if quer != reference_sequence[p]:
                                muts_linked.append(str(k)+str(quer))
                    if len(muts_linked) > 1:
                        phys_link_list.append(muts_linked)

        else:
            first_base = ref_pos[0]
            idx = np.abs(first_pos_val-first_base).argmin()
            for primer in primer_positions[idx-1:]:
                if found:
                    break
                if (first_base >= primer[1] - 10 and first_base <= primer[1] + 10):
                    name = read.query_name + "_F"
                    if name[:-2] in read_names:
                        continue
                    query_align_seq = read.query_alignment_sequence
                    found = True    
                    align_start = read.query_alignment_start
                    query_seq = read.query_sequence
                    clipped_positions = list(range(first_base - align_start, first_base))
                    soft_clipped_bases = query_seq[:align_start]
                    ref_sc = reference_sequence[first_base-align_start:first_base]
                                        
                    muts_linked = []
                    total_co_occurences.append(ref_pos)
                    if len(query_align_seq) == len(reference_sequence[ref_pos[0]:ref_pos[-1]+1]): 
                        for i, (quer, p) in enumerate(zip(query_align_seq, ref_pos)):
                            k = p + 1
                            if ref_pos[i] - 1 != ref_pos[i-1] and i != 0:
                                diff = ref_pos[i] - ref_pos[i-1]
                                for j in range(1, diff):
                                    deleted_pos = ref_pos[i] - j
                                    mut_dict[deleted_pos+1]["-"].append(primer[0])
                                    read_dict[deleted_pos+1]["-"].append(name)
                            mut_dict[k][quer].append(primer[0])
                            read_dict[k][quer].append(name)
                            if quer != reference_sequence[p]:
                                muts_linked.append(str(k)+str(quer))
                           
                    if len(muts_linked) > 1:
                        phys_link_list.append(muts_linked)
                    

    flagged_dist = []
    value_flagged_dist = []
    variants = {}
    for key, value in enumerate(mut_dict):
        #if key != 22281 and key != 22813 and key != 22206 and key != 25563:
        #    continue 
        num_primers = []
        nuc_counts = {"A":0, "C":0, "G":0, "T":0, "N":0, "-":0}
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
            tracking_depth.append([0,0,0,0,0,0])
        for i, (nuc, primer) in enumerate(value.items()):
            associated_reads = read_dict[key][nuc]
            if len(associated_reads) < 50 or len(associated_reads)/total_depth < 0.03:
                continue            
            if DEBUG:
                print("\n", key, nuc)
            unique, counts = np.unique(primer, return_counts=True)
            unique = list(unique)
            counts = list(counts)
            if len(counts) == 0:
                continue
            for u, c in zip(unique, counts):
                if DEBUG:
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
            b = []
            for i in range(a.shape[1]):
                v = list(a[:,i])
                if len(v) > 1:
                    dist = [abs(i-j) for i in v for j in v if i != j]
                    dist = np.mean(np.unique(dist))          
                    if DEBUG: 
                        print(v)
                        print(dist)
                    if dist > threshold:
                        flag = True
                else:
                    across_amplicons = a[0]
                    dist = [abs(i-j) for i in across_amplicons for j in across_amplicons if i != j]
                    dist = list(np.unique(dist))
                    b.append([float(x) for x in across_amplicons])
                    break
                
                b.append([float(x) for x in v])
            if flag:
                flagged_dist.append(key)
                value_flagged_dist.append(b)

    unique_link, link_counts = np.unique(phys_link_list, return_counts=True)
    total_link_counts = []
    total_unique_links = []
    total_possible_links = []
    
    start_total_co = [x[0] for x in total_co_occurences]
    arr = np.array(start_total_co)
    for i, (ul, lc) in enumerate(zip(unique_link, link_counts)):
        if len(ul) <= 1 or lc < 20:
            continue
        tlc = 0
        ul_pos = [int(x[:-1]) for x in ul]
        for tco_tmp in total_co_occurences:
            if all(item in tco_tmp for item in ul_pos):
                tlc += 1
        total_possible_links.append(float(tlc))
        total_unique_links.append(ul)
        total_link_counts.append(float(lc))
        
    if DEBUG:
        sys.exit(0) 
    tmp_dict = {"flagged_dist":flagged_dist, "variants":variants, "total_possible_links":total_possible_links, "total_unique_links":total_unique_links, "total_link_counts":total_link_counts, "value_flagged_dist":value_flagged_dist}
    
    with open(variants_json, "w") as bfile:
        bfile.write(json.dumps(tmp_dict))        
    return(tmp_dict)

def parse_usher_barcode(lineages, barcode, return_ref=False):
    print("parsing usher barcodes...")    
    #initialize dictionary for mutation pos
    l_dict = {}
    for l in lineages:
        l_dict[l] = []
        tmp = barcode[barcode['Unnamed: 0'] == l]
        tmp = tmp.loc[:, (tmp !=0).any(axis=0)]
        tmp = tmp.columns.tolist()
        if not return_ref:
            tmp = [x[1:] for x in tmp if x != "Unnamed: 0"]
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
