from model import run_model


#example 1
variants_file = "./example_data/SEARCH-59774_L001_L002_variants.tsv"
output_dir = "./example_models"
output_name = "SEARCH-59774_L001_L002"
problem_positions = "./example_data/SEARCH-59774_L001_L002_primer_mismatches.txt"
physical_linkage_file = "./example_data/SEARCH-59774_L001_L002_phys_linkage.txt"

run_model(variants_file, output_dir, output_name, problem_positions, physical_linkage_file)

