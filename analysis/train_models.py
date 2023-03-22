"""
Script to analyze the wastewater data taken from point loma during the delta -> omicron transition.
"""
import os
import sys
import json
sys.path.insert(0, "../")
from model import run_model
from generate_consensus import write_fasta

def main():
    sample_ids = []
    with open("./sample_names.txt", "r") as sfile:
        for line in sfile:
            line = line.strip()            
            line_list = line.split("\t")
            sample_ids.append(line_list[1])
            
    all_files = os.listdir("./data")
    for sample_id in sample_ids:
        associated_files = []
        for filename in all_files:
            if sample_id in filename:
                associated_files.append(filename)

        #we need all info to continue 
        if len(associated_files) < 4:
            continue
        #if not sample_id.startswith("SEARCH-61644"):
        #    continue
        associated_files = [os.path.join("./data", x) for x in associated_files]
        variants_file = [x for x in associated_files if "variants" in x][0]
        problem_positions = [x for x in associated_files if "mismatches" in x][0]
        physical_linkage_file = [x for x in associated_files if "phys" in x][0]
        freyja_file = [x for x in associated_files if "freyja" in x][0]
        reference = "/Users/caceves/Desktop/saga/example_data/sequence.fasta" 
        output_dir = "./output_results/" + sample_id
        output_name = sample_id
        output_fasta_name = "./output_results/%s/%s.fa" %(sample_id, sample_id)

        if not os.path.isdir(output_dir):
            os.system("mkdir %s" %output_dir)
        else:
            continue
        print("creating results for ", sample_id)
        exit_code = run_model(variants_file, output_dir, output_name, problem_positions, physical_linkage_file, freyja_file)
        if exit_code == 1:
            continue

        text_file = os.path.join(output_dir, output_name+"_model_results.txt")
        with open(text_file, "r") as tfile:
            for i, line in enumerate(tfile):
                line = line.strip()
                model_dictionary = json.loads(line)
                model_dict = model_dictionary['autoencoder_dict']
                problem_positions = model_dictionary['problem_positions']
                problem_positions.extend(model_dictionary['low_depth_positions'])
        write_fasta(model_dict, output_fasta_name, reference, problem_positions)
        #sys.exit(0)

if __name__ == "__main__":
    main()
