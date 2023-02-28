import os
import sys
import itertools
import numpy as np
import pandas as pd
import networkx as nx

def parse_physical_linkage_file(filename, positions, frequencies, nucs):
    """
    Parameters
    ----------
    filename : str
        The full file path to the file containing physical linkage information line by line.
    positions : list
        List of all positions containing variants being considered.
    frequencies : list
    nucs : list
    
    Returns
    -------
    G : networkx
        Graph object holding physical linkages.
    
    Takes in a physical linkage file as output by autoconsensus and write the physical linkage information to a graph. This is important because it allows interpretation based on mulitple linkages. Each node represents a mutation and each edge represents the count of that linkage. Output in addition to function returns is a graphxml file that represents the mutation connections.
    """
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

    for index, row in df.iterrows():
        if "N" in row['ALT'] or "+" in row['ALT']:
            continue
        
        p = row['POS']
        n = row['ALT']
        if int(row['TOTAL_DP']) < 100:
            low_depth_positions.append(str(n) + "_" + str(p))
            continue
        if problem_positions is not None and int(p) in problem_positions:
            continue
        
        f = float(row['ALT_FREQ'])
        positions.append(p)
        frequency.append(round(f, frequency_precision))
        nucs.append(n)

    unique_pos = list(np.unique(positions))
    unique_freq = [0] * len(unique_pos)
    ref_nucs = [0] * len(unique_pos)
    
    for p, f in zip(positions, frequency):
        loc = unique_pos.index(p)
        unique_freq[loc] += f
    
    ref_freq = [1-x for x in unique_freq]
    for index, row in df.iterrows():
        if not p in unique_pos:
            continue
        loc = unique_pos.index(p)
        ref_nucs[loc] = row['REF']
        n = row['REF']
        reference_variants.append(str(n) + "_" + str(p))

    
    frequency.extend(ref_freq)
    positions.extend(unique_pos)
    nucs.extend(ref_nucs)

    return(positions, frequency, nucs, low_depth_positions, reference_variants)
