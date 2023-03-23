import os
import sys
import itertools
import numpy as np
import pandas as pd
import networkx as nx


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

def parse_ivar_variants_file(file_path, frequency_precision=4, problem_positions=None):
    """
    Use the variants .tsv output from ivar to create a list of variants and frequencies for 
    downstream use.
    """
    print("parsing ivar variants file...")
    positions = []
    frequency = []
    nucs = []
    depth = []

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
    
    return(positions, frequency, nucs, low_depth_positions, reference_variants)
