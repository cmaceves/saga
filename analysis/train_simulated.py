"""
Script to analyze the wastewater data taken from point loma during the delta -> omicron transition.
"""
import os
import sys
import json
sys.path.insert(0, "../")
from model import run_model
from generate_consensus import write_fasta
from joblib import Parallel, delayed

def main():
    sample_ids = os.listdir("./simulated_data")           
    all_files = [os.path.join("./simulated_data", x) for x in sample_ids]
    parallel(sample_ids, all_files)

def parallel(sample_ids, all_files):
    code = Parallel(n_jobs=1)(delayed(train)(sample_id, data_folder) for sample_id, data_folder in zip(sample_ids, all_files))

def train(sample_id, data_folder):
    if "SEARCH-63701" not in sample_id:
        return(1)
    variants_file = os.path.join(data_folder, sample_id + "_variants.tsv")
    freyja_file = "./data/" + sample_id + "_L001_L002_freyja_results.tsv"
    print(variants_file)        
    print(freyja_file)

    reference = "/home/chrissy/Desktop/saga/example_data/sequence.fasta" 
    output_dir = "./simulated_data_results/" + sample_id
    output_name = sample_id
    output_fasta_name = "./simulated_data_results/%s/%s.fa" %(sample_id, sample_id)

    if not os.path.isdir(output_dir):
        os.system("mkdir %s" %output_dir)
    #else:
        #return(1)
        #continue
    print("creating results for ", sample_id)
    exit_code = run_model(variants_file, output_dir, output_name, None, None, freyja_file)
    if exit_code == 1:
        return(1)
        #continue
    text_file = os.path.join(output_dir, output_name+"_model_results.txt")
    with open(text_file, "r") as tfile:
        for i, line in enumerate(tfile):
            line = line.strip()
            model_dictionary = json.loads(line)
            model_dict = model_dictionary['autoencoder_dict']
            problem_positions = model_dictionary['problem_positions']
            if problem_positions is not None:
                problem_positions.extend(model_dictionary['low_depth_positions'])
            else:
                problem_positions = model_dictionary['low_depth_positions']
    write_fasta(model_dict, output_fasta_name, reference, problem_positions)
    #sys.exit(0)
    return(0)

if __name__ == "__main__":
    main()