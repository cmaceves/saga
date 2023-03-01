import os
import sys

def write_fasta(autoencoder_dict, output_name, reference):
    ref_seq = ""
    with open(reference, "r") as rfile:
        for line in rfile:
            if line.startswith(">"):
                continue
            line = line.strip()
            ref_seq += line
    for key, value in autoencoder_dict.items():
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
                continue
                p = int(tm.split("-")[0])
                positions.append(p)
                m = "-"+tm.split("-")[1]
                mutations.append(m)
            
            else:
                positions.append(int(tm[:-1]))
                mutations.append(tm[-1:])
        final_seq = ""
        for i in range(len(ref_seq)):
            if i+1 in positions:
                loc = positions.index(i+1)
                mut = mutations[loc]
                if "+" in mut:
                    final_seq += mut[1:]
                    i -= 1
                elif "-" in mut:
                    i += len(mut)-1
                else:
                    final_seq += mut
            else:
                final_seq += ref_seq[i]

        with open("%s" %output_name, "a") as ffile:
            ffile.write(">%s" %str(round(float(key),2)))
            ffile.write("\n")
            ffile.write(final_seq)
            ffile.write("\n")
        print("written.")


