import os
import sys
import json
import pandas as pd
import analyze_spike_in
from line_profiler import LineProfiler
sys.path.insert(0, "../")
import numpyro_model
import file_util
import generate_consensus

def main():
    reference_file = "/home/chrissy/Desktop/sequence.fasta"       
    sample_id = "file_0"
    output_dir = "/home/chrissy/Desktop/saga_spike_in_results/%s" %sample_id
    output = "/home/chrissy/Desktop/saga_spike_in_results/%s/%s_beta_dist.json"%(sample_id,sample_id)
    status = numpyro_model.run_numpyro_model(sample_id)
    sys.exit(0)
    numpyro_model.choose_solution(output, output_dir, sample_id)        
    #sys.exit(0)
    assignment_output = os.path.join(output_dir, sample_id+"_assignments.json")
    with open(assignment_output, 'r') as afile:
        data = json.load(afile)
    var = data['variants']
    for k, v in var.items():
        print(k, v)
    #sys.exit(0)
    removal_dict = {}
    for k, v in var.items():
        removal_dict[k] = []
    problem_positions = []
    output_name = os.path.join(output_dir, sample_id+"_numpyro.fa")
    print(output_name)
    generate_consensus.write_fasta(var, output_name, reference_file, problem_positions, removal_dict)
    sys.exit(0)
    
    sample_ids = ["file_" + str(x) for x in list(range(0, 379, 4))]
    for sample_id in sample_ids:
        print(sample_id)
        status = numpyro_model.run_numpyro_model(sample_id)
        if status != 1:
            output_dir = "/home/chrissy/Desktop/saga_spike_in_results/%s" %sample_id
            output = "/home/chrissy/Desktop/saga_spike_in_results/%s/%s_beta_dist.json"%(sample_id,sample_id)
            choose_status = numpyro_model.choose_solution(output, output_dir, sample_id)
            print("choose status", choose_status)
        #sys.exit(0)
if __name__ == "__main__":
    main()
