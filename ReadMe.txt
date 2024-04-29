Paper Title: Partial correlation network analysis identifies coordinated gene expression within a regional cluster of COPD genome-wide association signals
Authors: Michele Gentili, John Platig, Ed Silverman, et al.

In this repo we share the code associated to the algorithm (Gene-Specific Ridge Partial Correlation - PCORR), and the figures presented in the paper.

./Example_PCORR.ipynb	   : Notebook with an example run of the PCORR algorithm

./Data/
	./GSE/ and ./LTRC/ : contain the output of the PCORR from the LTRC dataset (Manuscript) and GSE (Supplementary)
	./Review_response/ : contains the output of the PCORR with the suggestions presented by the reviewers
	./expression.csv   : is a gene expression file, subset from the GSE dataset, used to test the PCORR
	./List_gene_ch4_COPD.txt and List_gene_ch5_COPD.txt    : the list of COPD GWAS genes obtained from Sakornsakolpat et al. 2019 
	./ppi.csv          : sample Protein Protein interaction network to compute the Prior.csv used in the PCORR algorithm
	./prior.csv        : saved Prior computed in ../Example PCORR.ipynb

./src/
	./PartialCorrelation.py : code to run the PCORR algorithm
	./Prior/py		: code to compute the prior matrix given a network

./figures/ : Paper figures

./Paper Analysis Notebook/ : 3 Notebook to generate the paper figures from the PCORR output in ./Data/GSE/, ./Data/LTRC/ and ./Data/Review_response/.

./Gwas_hits/ Folder containing notebook and data to reproduce the simulation on the clustering probability of the COPD GWAS loci. 



