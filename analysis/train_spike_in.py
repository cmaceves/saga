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
    directory_bam = "/home/chrissy/Desktop/spike_in"
    directory_variants = "/home/chrissy/Desktop/spike_in_variants_saga"
    reference_file = "/home/chrissy/Desktop/sequence.fasta"
    bed_file = "/home/chrissy/Desktop/sarscov2_v2_primers.bed"

    sample_ids = os.listdir(directory_bam)           
    all_files = [os.path.join(directory_bam, x) for x in sample_ids if x.endswith(".bam")]
    train("file_148", directory_bam, directory_variants, reference_file, bed_file)

def train(sample_id, directory_bam, directory_variants, reference_file, bed_file):
    tmp = sample_id.split("_")[:2]
    sample_id = "_".join(tmp)
    print("creating results for %s sample..." %sample_id)
    output_dir = "/home/chrissy/Desktop/saga_spike_in_results/" + sample_id
    output_name = sample_id
    output_fasta_name = "/home/chrissy/Desktop/saga_spike_in_results/%s/%s.fa" %(sample_id, sample_id)
    bam_file = directory_bam + "/%s_sorted.calmd.bam" %sample_id
    if not os.path.isdir(output_dir):
        os.system("mkdir %s" %output_dir)

    #output_dir, output_name, bam_file, bed_file, reference_file, freyja_file=None
    exit_code = run_model(
        output_dir, \
        output_name, \
        bam_file, \
        bed_file, \
        reference_file)

    if exit_code == 1:
        return(1)
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
    write_fasta(model_dict, output_fasta_name, reference_file, problem_positions)
    return(0)

if __name__ == "__main__":
    main()
