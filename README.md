# Saga
Tool for building consensus sequences from complex viral mixtures.

## Use Instructions

### Input 
- The input for this is a bed file, a reference fasta, a primer pairs tsv, and a variants file. The variants file can be output from the new ivar functionality in branch v_20 (experimental). 
- I've provided some test data in the folder test_data
- The variants file is in output_test.txt and the rest should be self explanatory

### Run
- Use file ./analysis/run_numpyro_model.py to run the model
- Modify this in-line, I don't have argparse set up yet

### Enviornment
- Environment specifics found in enviornment.yml
