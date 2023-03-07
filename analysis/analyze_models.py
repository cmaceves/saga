import os
import sys
import json


def main():
    print("analyzing models...")
    sample_ids = os.listdir("./output_results")
    print(sample_ids)

    for sample_id in sample_ids:
        path = os.path.join("./output_results", sample_id, "%s_model_results.txt" %sample_id)
        with open(path, "r") as mfile:
            for line in mfile:
                line = line.strip()
                model_dictionary = json.loads(line)

        print(model_dictionary)
        sys.exit(0)

if __name__ == "__main__":
    main()
