Paper Title:
Authors: Michele Gentili, John Platig, Ed Silverman, et al.
Last update:

The code is structured in 3 main parts:

1) main.py: it generates all the partial correlations, including the permutations (shuffeling cases and controls, and bootstrap)
2) analysis.py: it generates all the statistically significant partial correlations used for analysis
3) notebooks/*.ipyjnb : it's a set of 4 python jupyter notebooks, containing the final analysis to generate the tables and figures.


1) To run the main you can set 3 main parameters:
    DIR_OUTPUT = "../data/Pcorr_output_1/" # where do you want to save the output
    list_nreg = [0,1,5,10,25,50,75,100] # Partial Correlation parameters, number of controlling genes
    list_minL = [0,0.1,1,10] # Partial Correlation parameters, min lambda
    GEX_NAME='LTRC' # what dataset to use
    n_permutations = 100 #this is the number of permutations to run ( it will be used later on to compute the pvalues)
    
This script will generate the partial correlations for each sub population (cases and controls) for each combination of parameters. The Gene expression file is handled and loaded in the module Preprocessing.py

2) Once all the partial correlations are generated, this script will compute 2 statistical significant networks:
    - Significant Partial Correlation for both cases and controls separately 
    - Differential Partial Correlation, cases vs controls
   
For each of these networks it will compute clustering and enrichment analysis.

3) This folder contains the notebooks that will produce the figures and tables.

    
    
Order to run the code:

- notebooks/1.Preprocessing.ipynb: 
	notebook to create, align and clear gene expression and Phenotype
- main.py
- analysis.py
- notebooks/* 
	all the notebooks but the 1. LTRC_Phenotype.ipynb.
    
    
Origin of the files:
    - PPI/
        - 9606.*
            From:    https://string-db.org/cgi/download?sessionId=bEfg6jMA9utr&species_text=Homo+sapiens
            Download: v11.5, 11/17/2022
        -ppi_stable.csv
            From : notebooks/1.Preprocessing.ipynb
            
    - 
    - gene_mapping_position.csv
            From: biomart (GRCh38.p13), download 02/15/2023
                
The example/ folder contains the files used for the public git folder: https://github.com/michelegentili93/Partial_Correlation
