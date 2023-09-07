import os
import sys
import pandas as pd
import analyze_spike_in
from line_profiler import LineProfiler
sys.path.insert(0, "../")
import numpyro_model
import file_util

def main():
    """    
    ww = "/home/chrissy/Desktop/saga/analysis/simulated_data/SEARCH-59742/SEARCH-59742.bam"
    output_dir = "/home/chrissy/Desktop/saga/analysis/simulated_ww_preprocessed"
    variants_json = os.path.join(output_dir, output_name+"_variants.txt")
    primer_dict = file_util.parse_bam_depth_per_position(bam_file, bed_file, variants_json)
    sys.exit(0) 
    """

    """
    sample_id = "file_196"
    output_dir = "/home/chrissy/Desktop/saga_spike_in_results/%s" %sample_id
    output = "/home/chrissy/Desktop/saga_spike_in_results/%s/%s_beta_dist.json"%(sample_id,sample_id)
    #numpyro_model.choose_solution(output, output_dir, sample_id)
    status = numpyro_model.run_numpyro_model(sample_id)
    sys.exit(0)
    """
    sample_ids = ["file_" + str(x) for x in list(range(0, 379, 4))]
    for sample_id in sample_ids:
        #if sample_id != "file_352":
        #    continue
        print(sample_id)
        """
        lp = LineProfiler()
        lp_wrapper = lp(numpyro_model.run_numpyro_model)
        lp_wrapper(sample_id)
        lp.print_stats()
        sys.exit(0)
        """
        status = numpyro_model.run_numpyro_model(sample_id)
if __name__ == "__main__":
    main()
