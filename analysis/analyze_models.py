import os
import sys
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sys.path.insert(0, "../")
import file_util

def top_group(model_dictionary, gt_mt_dict, gt_centers):    
    exclude = model_dictionary['low_depth_positions']
    exclude.extend([str(x) for x in range(28300,30000)])
    autoencoder_dict = model_dictionary["autoencoder_dict"]    
    auto_centers = list(autoencoder_dict.keys())
    mut_dict_keys = list(gt_mt_dict.keys())

    scores = model_dictionary['scores']
    variants = model_dictionary['variants']
    for center in auto_centers[:1]:
        center = float(center)
        predicted = autoencoder_dict[str(center)]
        loc = gt_centers.index(center)
        keyoi = mut_dict_keys[loc]
        actual = gt_mt_dict[keyoi]
        
        actual = [item for item in actual if item[:-1] not in exclude]    
        predicted = [item for item in predicted if item[:-1] not in exclude]
        #print("exclude", exclude) 
        print("center", center)
        #print(predicted)
        #print(actual)
        missing = [item for item in predicted if item not in actual]
        extra = [item for item in actual if item not in predicted]
        print("missing", missing)
        print("extra", extra)
        for s,v in zip(scores, variants):
            if v in missing or v in extra:
                print(s,v)
    #sys.exit(0)
    return(missing, extra, center, float(auto_centers[0])-float(auto_centers[1])) 

def main():
    print("analyzing simulated models...")
    sample_ids = os.listdir("./simulated_data_results")
    sample_consensus = [os.path.join("./simulated_data_results", x , x+".fa") for x in sample_ids]    
    sample_lineages = [os.path.join("./simulated_data_results", x, "lineage_report.csv") for x in sample_ids]
    sample_freyja = [os.path.join("./data", x+"_L001_L002_freyja_results.tsv") for x in sample_ids]


    len_missing = []
    len_extra = []
    all_centers = []
    all_misplaced = []
    all_distance = []
    for sample_id, lineages, freyja_file in zip(sample_ids, sample_lineages, sample_freyja):
        #print("\n", sample_id, lineages, freyja_file)
        try:
            df = pd.read_csv(lineages)
        except:
            continue
        #print(df)
        gt_centers, gt_lineages = file_util.parse_freyja_file(freyja_file)
        #print(gt_centers[:5], gt_lineages[:5])        

    #sys.exit(0)
    for sample_id, freyja_file in zip(sample_ids, sample_freyja):
        path = os.path.join("./simulated_data_results", sample_id, "%s_model_results.txt" %sample_id)
        with open(path, "r") as mfile:
            for line in mfile:
                line = line.strip()
                model_dictionary = json.loads(line)
        gt_centers, gt_lineages = file_util.parse_freyja_file(freyja_file)
        gt_mut_dict = file_util.parse_usher_barcode(gt_lineages)
        missing, extra, center, distance_to_second = top_group(model_dictionary, gt_mut_dict, gt_centers)
        len_missing.append(len(missing))
        len_extra.append(len(extra))
        all_centers.append(center)
        all_misplaced.append(len(missing)+len(extra))
        all_distance.append(distance_to_second)
        #sys.exit(0)
    df = pd.DataFrame({"center":all_centers, "misplaced":all_misplaced, "distance_to_second":all_distance})
    print(df)
    sns.set_style("whitegrid")
    sns.scatterplot(data=df, x="center", y="misplaced") #, hue = "distance_to_second")
    plt.savefig('output.png')

if __name__ == "__main__":
    main()
