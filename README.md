# Saga

Tool for building consensus sequences from complex viral mixtures.

### Methods

I've marked some HELPME and TODO lines in the file ```model.py```
Deletions and insertions have not yet been handled.

Files Required:

1. variants file
2. primer mismatch file
3. physical linkage text file (super basic, literally every linkage on a line)
4. freyja file (optional but useful to resolve issues where pop1 + pop2 = pop3

Essentially you parse the variants file and read in position/nuc/frequency, also using the frequency of the allele that matches the reference at positions where you have indels. Remove any nucleotides with frequencies likely modified due to primer binding errors.

Remove any low level noise < 0.03 and high level variants > 0.97 - these high level variants will become universal mutations.

Create a kernel density estimate. Here we select bw to be 0.01 or 0.001 based on the definition of how that related to frequency in a sequencing sample. Find local maxima of the KDE, these represent potential populations and/or overlaps of populations within the data.

Create a solution space of variable n values that could potentially exist given the local maxima determined from the KDE. 

Use the physical linkage file to build a network (just a convient way to store this data), and then use that network to decide which frequencies / maxmima are clearly composed of a smaller individual component. This won't be comprehensive, but it will allow us to narrow our solution space.

Generate X number of GMM models using sklearn, where X = # of possible solutions and the means are initialized to the individual components of the proposed solution and their possible overlaps. Take the points with the lowest log-likehood under the given model and use the average of these points as a general metric for how well the model is performing. Definitly this needs work.

Once a model is selected, retrain it using more iterations and initializations, and then assign out variants.

Write the fasta file.    

### How to run
I've placed test data in the example_data directory (including freyja files). Comment out/in which example you'd like to look at in train_example and then just run.

```
python train_example.py
```

### Examples in train_example
