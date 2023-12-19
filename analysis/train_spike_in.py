import os
import sys
import json
import argparse
import numpy as np
from joblib import Parallel, delayed
sys.path.insert(0, "../")
from model_util import run_model, call_consensus
from generate_consensus import write_fasta


def train_parallel(sample_ids, directory_bam, directory_variants, reference_file, bed_file):
    code = Parallel(n_jobs=10)(delayed(train)(sample_id, directory_bam, directory_variants, \
        reference_file, bed_file) for sample_id in sample_ids)

def main():
    directory_bam = "/home/chrissy/Desktop/spike_in"
    directory_variants = "/home/chrissy/Desktop/spike_in_variants_saga"
    reference_file = "/home/chrissy/Desktop/sequence.fasta"
    bed_file = "/home/chrissy/Desktop/sarscov2_v2_primers.bed"

    extensions = list(range(0, 380, 4))
    #extensions = list(range(320, 340, 1))
    file_extensions = ['file_'+str(x)+"_sorted.calmd.bam" for x in extensions]
    sample_ids = os.listdir(directory_bam)           

    sample_ids = [x for x in sample_ids if x in file_extensions]
    all_files = [os.path.join(directory_bam, x) for x in sample_ids if x.endswith(".bam")]
    sample_ids = [x.split("_sorted")[0] for x in sample_ids if x.endswith(".bam")]
    sample_ids = list(np.unique(sample_ids))

    print(sample_ids)
    #sys.exit(0)
    #train_parallel(sample_ids, directory_bam, directory_variants, reference_file, bed_file)
    #352, 220, 368, 136, 360
    #292, 136
    train("file_340", directory_bam, directory_variants, reference_file, bed_file)

def train(sample_id, directory_bam, directory_variants, reference_file, bed_file, run_new_model=True):
    tmp = sample_id.split("_")[:2]
    sample_id = "_".join(tmp)
    print("creating results for %s sample..." %sample_id)
    output_dir = "/home/chrissy/Desktop/saga_spike_in_results/" + sample_id
    output_name = sample_id
    output_fasta_name = "/home/chrissy/Desktop/saga_spike_in_results/%s/%s.fa" %(sample_id, sample_id)
    bam_file = directory_bam + "/%s_sorted.calmd.bam" %sample_id
    if not os.path.isdir(output_dir):
        os.system("mkdir %s" %output_dir)

    text_file = os.path.join(output_dir, output_name+"_model_results.txt")    
    model_location = os.path.join(output_dir, output_name+"_model.pkl")
    """
    if os.path.isfile(model_location):
        print("model built already")
        return(0)
    """
    if run_new_model:
        exit_code = run_model(output_dir, output_name, bam_file, bed_file, reference_file)
        print("finished %s" %sample_id)
        if exit_code == 1:
            return(1)
    
    if not os.path.isfile(model_location):
        print("model not found")
        return(1)
    exit_code = call_consensus(output_dir, output_name, model_location, reference_file, bed_file)
    if exit_code == 1:
        return(1)
    with open(text_file, "r") as tfile:
        for i, line in enumerate(tfile):
            line = line.strip()
            model_dictionary = json.loads(line)
            model_dict = model_dictionary['autoencoder_dict']
            no_call = [str(x) for x in model_dictionary['no_call']]
            removal_dict = model_dictionary['removal_dict']
            print("no call", no_call)
            tmp_dict = {}
            for k,v in model_dict.items():
                if k.split("_")[2] not in no_call:
                    tmp_dict[k] = v
            model_dict = tmp_dict
            problem_positions = []
            for k, v in model_dictionary['call_ambiguity'].items():
                v = v['reason'] 
                if "low_depth" in v or "amp_flux" in v or "outlier" in v:
                    problem_positions.append(k)
    write_fasta(model_dict, output_fasta_name, reference_file, problem_positions, removal_dict)
    print("finished %s" %output_dir)
    return(0)

if __name__ == "__main__":
    main()
