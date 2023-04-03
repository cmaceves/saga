import os
import sys

sys.path.insert(0, "../")
import file_util


def create_vcf(vcf_lineage, template_vcf, output_vcf_name):
    gt_dict = file_util.parse_usher_barcode([vcf_lineage], "./usher_barcodes.csv", return_ref=True)
    mutations_list = gt_dict[vcf_lineage]

    lines_write = []
    with open(template_vcf, "r") as tvcf:
        for line in tvcf:
            line = line.strip()
            if line.startswith("#"):
                lines_write.append(line)
            else:
                template_line = line.split("\t")
                break

    #the mutations need to be in ascending position order
    tmp_nucs = []
    tmp_pos = []
    for mutation in mutations_list:
        tmp_nucs.append(mutation[0] + mutation[-1])
        tmp_pos.append(int(mutation[1:-1]))

    zipped = list(zip(tmp_pos, tmp_nucs)) 
    zipped.sort()
    tmp_pos, tmp_nucs = zip(*zipped)
    mutations_list = []
    for pos, nuc in zip(tmp_pos, tmp_nucs):
        mutations_list.append(nuc[0] + str(pos) + nuc[1])

    for mutation in mutations_list:
        reference = mutation[0]
        alt = mutation[-1]
        position = mutation[1:-1]
        
        template_line[1] = position
        template_line[3] = reference
        template_line[4] = alt
        write_line = "\t".join(template_line)
        lines_write.append(write_line)

    with open(output_vcf_name, "w") as vfile:
        for line in lines_write:
            vfile.write(line)
            vfile.write("\n")
 
def main():
    template_vcf = "./template.vcf"
    base_num_reads = 2000000

    all_freyja_files = [os.path.join("./data", x) for x in os.listdir("./data") if "freyja" in x]
    for filename in all_freyja_files:
        #if "SEARCH-63546" not in filename:
        #    continue
        actual_centers, actual_lineages = file_util.parse_freyja_file(filename)         
        
        file_id = os.path.basename(filename).replace("_L001_L002_freyja_results.tsv", "")
        simulated_output = os.path.join("./simulated_data", file_id)
        if not os.path.isdir(simulated_output):
            os.system("mkdir %s" %(simulated_output))

                
        #create a vcf for every lineage
        for center, vcf_lineage in zip(actual_centers, actual_lineages):
            output_vcf_name = "./vcfs/%s.vcf" %vcf_lineage
            percent = round(center, 4)   
            if percent < 0.0005:
                continue
            if not os.path.isfile(output_vcf_name): 
                create_vcf(vcf_lineage, template_vcf, output_vcf_name)  
    
            basename = "%s_%s" %(vcf_lineage, percent)
            percent_str = str(int(percent*100))
            
            output_filename = '%s/simulated_%s_%s.bam' %(simulated_output, vcf_lineage, percent_str)
            if os.path.isfile(output_filename):
                continue

            output_1 = "%s_1.fq" %(basename)
            output_2 = "%s_2.fq" %(basename)
            num_reads = int((base_num_reads*percent)/2)
            #print("center:", center, "lineage:", vcf_lineage, "num_reads:", \
            #    num_reads, "percent:", float(base_num_reads*percent))
            #continue
            #sys.exit(0) 
            cmd = 'reseq illuminaPE -j 32 -r ../../sequence.fasta -b aaron.preprocessed.bam -V %s -1 %s -2 %s --numReads %s -v %s --noBias' %(output_vcf_name, output_1, output_2, str(num_reads), output_vcf_name)
            cmd2 = 'bwa mem -t 32 ../../sequence.fasta %s %s | samtools view -b -F 4 -F 2048 | samtools sort -o %s' %(output_1, output_2, output_filename)

            os.system(cmd)
            os.system(cmd2)
            os.system("rm %s" %output_1)
            os.system("rm %s" %output_2)
        bam_name = os.path.join(simulated_output, file_id+".bam")
        if not os.path.isfile(bam_name):
            #samtools merge 
            cmd = "samtools merge %s %s/*.bam" %(bam_name, simulated_output)
            os.system(cmd)

        variants_name = os.path.join(simulated_output, file_id+"_variants")
        #ivar variants
        #if not os.path.isfile(variants_name + ".tsv"):
        cmd = "samtools mpileup -aa -A -d 0 -B -Q 0 --reference %s  %s | ivar variants -p %s -q 20 -t 0 -m 0 -r %s" %("../../sequence.fasta", bam_name, variants_name, "../../sequence.fasta")
        os.system(cmd)
        #sys.exit(0)

if __name__ == "__main__":
    main()
