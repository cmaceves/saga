import os
import sys
import json
from model import run_model
from generate_consensus import write_fasta

reference = "./example_data/sequence.fasta"

#example 1
variants_file = "./example_data/SEARCH-59774_L001_L002_variants.tsv"
output_dir = "./example_models"
output_name = "SEARCH-59774_L001_L002"
problem_positions = "./example_data/SEARCH-59774_L001_L002_primer_mismatches.txt"
physical_linkage_file = "./example_data/SEARCH-59774_L001_L002_phys_linkage.txt"
output_fasta_name = "./example_data/SEARCH-59774.fa"

run_model(variants_file, output_dir, output_name, problem_positions, physical_linkage_file)

text_file = os.path.join(output_dir, output_name+"_model_results.txt")
with open(text_file, "r") as tfile:
    for i, line in enumerate(tfile):
        line = line.strip()
        model_dictionary = json.loads(line)
        model_dict = model_dictionary['autoencoder_dict']
write_fasta(model_dict, output_fasta_name, reference)
