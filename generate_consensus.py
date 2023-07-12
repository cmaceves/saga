import os
import sys

def write_fasta(autoencoder_dict, output_name, reference, problem_positions, removal_dict):
    ref_seq = ""
    problem_positions = [int(x) for x in problem_positions]

    with open("%s" %output_name, "w") as ffile:
        pass

    with open(reference, "r") as rfile:
        for line in rfile:
            if line.startswith(">"):
                continue
            line = line.strip()
            ref_seq += line

    min_key = autoencoder_dict.keys()
    for key, value in autoencoder_dict.items():
        if float(key) < 0.05:
            continue
        removal_variants = [int(x[:-1])  for x in removal_dict[key]]
        trial_muts = autoencoder_dict[key]
        positions = []
        mutations = []
        for tm in trial_muts:
            if "+" in tm:
                continue
                p = int(tm.split("+")[0])
                positions.append(p)
                m = "+"+tm.split("+")[1]
                mutations.append(m)
            elif "-" in tm:
                p = int(tm.split("-")[0])
                positions.append(p)
                m = "-"+tm.split("-")[1]
                mutations.append(m)
           
            else:
                positions.append(int(tm[:-1]))
                mutations.append(tm[-1:])

        final_seq = ""
        for i in range(1, len(ref_seq)+1):
            if i in problem_positions:
                final_seq += "N"
            elif i in removal_variants:
                final_seq += "N"
            elif i in positions:
                loc = positions.index(i)
                mut = mutations[loc]
                if "+" in mut:
                    final_seq += mut[1:]
                    i -= 1
                elif "-" in mut:
                    final_seq += "-"    
                else:
                    final_seq += mut
            else:
                final_seq += ref_seq[i-1]

        with open("%s" %output_name, "a") as ffile:
            ffile.write(">%s" %str(round(float(key),2)))
            ffile.write("\n")
            ffile.write(final_seq)
            ffile.write("\n")
        print("written.")


