import os
import sys
import pysam
import itertools
import numpy as np
import pandas as pd
import networkx as nx


def parse_reference_sequence(ref_file="/home/chrissy/Desktop/sequence.fasta"):
    reference_sequence = ""
    with open(ref_file, "r") as rfile:
        for line in rfile:
            line = line.strip()
            if line.startswith(">"):
                continue
            reference_sequence += line
    return(reference_sequence)

def parse_bam_depth_per_position(bam_file, bed_file, contig="NC_045512.2"):
    reference_sequence = parse_reference_sequence()
    problematic_positions = []
    problematic_primers = []

    samfile = pysam.AlignmentFile(bam_file, "rb")
    primer_positions = parse_bed_file(bed_file)    
    for i, primer in enumerate(primer_positions):
        primer_left_count = [0] * (primer[1]-primer[0])
        primer_left_mut = [0] * (primer[1]-primer[0])
        primer_left_index = list(range(primer[0], primer[1]))

        primer_right_count = [0] * (primer[3]-primer[2])
        primer_right_mut = [0] * (primer[3]-primer[2])
        primer_right_index = list(range(primer[2], primer[3]))
        #deal with left primer
        for read in samfile.fetch(contig, primer[1]-10, primer[1]+10):
            ref_pos = read.get_reference_positions()
            first_base = ref_pos[0]
            align_start = read.query_alignment_start #start index in query of the first base NOT soft clipped
            
            if (first_base >= primer[1] - 10 and first_base <= primer[1] + 10) and not read.is_reverse:
                query_seq = read.query_sequence
                clipped_positions = list(range(first_base - align_start, first_base))
                
                soft_clipped_bases = query_seq[:align_start]
                ref_sc = reference_sequence[first_base-align_start:first_base]
               
                for ref_nuc, sc_nuc, pos in zip(ref_sc, soft_clipped_bases, clipped_positions):
                    if pos not in primer_left_index:
                        continue
                    idx = primer_left_index.index(pos)
                    primer_left_count[idx] += 1
                    if ref_nuc == sc_nuc:
                        continue
                    primer_left_mut[idx] += 1
        #deal with right primer, yes this is repetitive
        for read in samfile.fetch(contig, primer[2]-10, primer[2]+10):
            ref_pos = read.get_reference_positions()
            last_base = ref_pos[-1]
            align_end = read.query_alignment_end #start index in query of the first base NOT soft clipped
            if (last_base >= primer[2] - 10 and last_base <= primer[2] + 10) and read.is_reverse:
                query_seq = read.query_sequence
                clipped_positions = list(range(last_base, len(query_seq)+last_base))
                soft_clipped_bases = query_seq[align_end:]
                ref_sc = reference_sequence[last_base+1:last_base+(len(query_seq)-align_end)+1]
                for ref_nuc, sc_nuc, pos in zip(ref_sc, soft_clipped_bases, clipped_positions):
                    if pos not in primer_right_index:
                        continue
                    idx = primer_right_index.index(pos)
                    primer_right_count[idx] += 1
                    if ref_nuc == sc_nuc:
                        continue
                    primer_right_mut[idx] += 1
        primer_left_per = [x/y if y > 5 else 0 for x,y in zip(primer_left_mut, primer_left_count)]
        primer_right_per = [x/y if y > 5 else 0 for x,y in zip(primer_right_mut, primer_right_count)]
        
        for pos, per in zip(primer_right_index, primer_right_per):
            if per > 0.50:
                problematic_positions.append(pos)
                if primer not in problematic_primers:
                    problematic_primers.append(primer)
        for pos, per in zip(primer_left_index, primer_left_per):
            if per > 0.50:
                problematic_positions.append(pos)
                if primer not in problematic_primers:
                    problematic_primers.append(primer)
    
    return(problematic_positions, problematic_primers)

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

def parse_physical_linkage_file(filename, positions, frequencies, nucs):
    print("parsing physical linkage file...")
    #initialize a graph
    G = nx.Graph()

    with open(filename, "r") as pfile:
        for line in pfile:
            useful = False
            #check if it's in our positions list
            split_line = line.strip().split("\t")
            #if there isn't more than 1 position who cares
            if(len(split_line) < 2):
                continue

            for sl in split_line:
                tmp_pos = int(sl.split("_")[1])
                if tmp_pos in positions:
                    useful = True
                    break
            if useful is True:
                #handle the graph side of things
                for sl in split_line:
                    #get position
                    tmp_n = sl.split("_")[0]
                    tmp_pos = int(sl.split("_")[1])
                    try:
                        loc = -1
                        for i,(p, n) in enumerate(zip(positions, nucs)):
                            if tmp_n == n and tmp_pos == p:
                                loc = i
                        if loc == -1:
                            tmp_freq = 0
                        else:
                            tmp_freq = frequencies[loc]
                    except:
                        tmp_freq = 0.0

                    #check if it's already a node
                    if G.has_node(sl) is False:
                        G.add_nodes_from([(sl, {"frequency":str(tmp_freq)})])

                #now that we have all nodes accounted for, add the path
                #turn the list into a list of sets
                list_of_edges = list(itertools.combinations(split_line, 2))
                for edge in list_of_edges:
                    if G.has_edge(edge[0], edge[1]) is False:
                        G.add_edges_from([(edge[0], edge[1], {"count":"1"})])
                    else:
                        original_count = G[edge[0]][edge[1]]['count']
                        G[edge[0]][edge[1]]['count'] = str(int(original_count) + 1)
   
    #nx.write_graphml(G,'g.xml')
    return(G)

def parse_primer_mismatches(primer_mismatches):
    """
    Given a primer mismatch text file parse out all the positions.
    """
    print("parsing primer mismatch file...")
    df = pd.read_table(primer_mismatches)
    temp_list = df['suspect_positions'].tolist()
    problem_positions = []
    for tl in temp_list:
        if "_" in str(tl):
            tl = tl.split("_")
        else:
            tl = [tl]
        tl = [int(x) for x in tl]
        problem_positions.extend(tl)

    return(problem_positions)

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
       
def parse_ivar_variants_file(file_path, frequency_precision=4, problem_positions=None, bed_file=None, problem_primers = None):
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
        if int(row['TOTAL_DP']) < 50:
            low_depth_positions.append(str(p))
            continue
        if problem_positions is not None and int(p) in problem_positions:
            continue       
        f = float(row['ALT_FREQ'])

        if bed_file is not None:
            #here we check if this exists in a primer position
            if f > 0.10:
                tmp_p = int(p)
                for amplicon in primer_positions:
                    if amplicon[0] < tmp_p < amplicon[1] or amplicon[2] < tmp_p < amplicon[3]:
                        problem_amplicons.append([amplicon[0], amplicon[3]])                 
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
