import multiprocessing
import os
import pickle
from functools import partial
import numpy as np
import pandas as pd
import scipy
import networkx as nx
from scipy.sparse import csr_matrix,lil_matrix
from abc import ABCMeta, abstractmethod
import itertools
from Preprocessing import load_STRING_ppi
from tqdm import tqdm
import ray,psutil
    
class PPI_prior():

    def __init__(self, nt_ppi, path_prior_file_output,verbose=True,parallel = False,recompute=False):
        """
        threshold_ppi_edge: how to filter PPI interaction, if STRING (default)
        path_ppi_dir = name file with the PPI network
        
        df_prior is a nxn dataframe. Not symmetric. Index and columns are genes. 
            The i-th row is the inverse of the personalized PageRank of gene i in the PPI.
        
        
        """
        self.verbose = verbose
        self.parallel = parallel

        
        if verbose: print('path name prior: ',path_prior_file_output)
        # If already computed
        
        if os.path.exists(path_prior_file_output) and not recompute:
            if verbose: print('Loading Prior')
            self.df_prior = pd.read_csv(path_prior_file_output,index_col=0)#, engine="pyarrow")

        else:
            self.df_prior = self._compute_matrix_prior(network=nt_ppi)
            self.df_prior = 1/pd.DataFrame.from_dict(self.df_prior,orient='index')
            self.df_prior.to_csv(path_prior_file_output)
        

    def to_iterator(self,obj_ids):
        while obj_ids:
            done, obj_ids = ray.wait(obj_ids)
            yield ray.get(done[0])
            
    def _compute_matrix_prior(self, network):
        """

        Create the matrix prior
            NB: IDs in ppi have to be a range(n_genes), and they have to be coherent with gene row_id in gene expression
            NBNB:
            Nodes That have only an edje, may have their prior greater than the node in the edge (because is a bottleneck for pagerank)
        """
        
        if self.verbose: print('Number of genes in PPI:', network.number_of_nodes())
        
        n_genes = network.number_of_nodes()
        matrix_prior = {}
        M = nx.to_scipy_sparse_matrix(network)
        nodelist = network.nodes()
        
        S = np.array(M.sum(axis=1), dtype=int).flatten()
        S = np.divide(1, S, out=np.zeros(n_genes), where= S!=0)
        is_dangling = np.where(S == 0)[0]
        Q = scipy.sparse.spdiags(S.T, 0, *M.shape, format='csr')
        M = Q * M
        M = M.transpose()
        if self.parallel:
            if not ray.is_initialized():
                ### Ray Initialisation
                num_cpus = psutil.cpu_count(logical=False) - 1
                print('Using ',num_cpus,'cpus')
                ray.init(log_to_driver=False, num_cpus=num_cpus)
                @ray.remote
                def _pers_pagerank_ray(gene_i, M, nodelist,is_dangling):
                    return gene_i, _pers_pagerank(gene_i, M, nodelist,is_dangling)

            id_M = ray.put(M)
            id_nodelist = ray.put(nodelist)
            id_is_dangling = ray.put(is_dangling)
            id_res = [self._pers_pagerank_ray.remote(gene_i, M = id_M, nodelist = id_nodelist, 
                                                is_dangling = id_is_dangling) for gene_i in nodelist ]
            for gene_i, result in tqdm(self.to_iterator(id_res), total=len(id_res)):
                matrix_prior[gene_i] = result

        else:
            # loop on all the other genes
            for gene_i in tqdm(nodelist):
                matrix_prior[gene_i] = _pers_pagerank(gene_i, M = M, nodelist = nodelist, is_dangling = is_dangling)
        
        return matrix_prior
    
    
def _pers_pagerank(gene,M, nodelist, is_dangling, alpha=0.85, max_iter=100, tol=1.0e-6, dangling=None):
    '''
        Compute personalige page rank wrt to input gene in input network.
        gene: id of the node coherent with the network
        network: nx network
    '''
    # gene,network=x
    personalization = {node: 0 for node in nodelist} #0.25/len(network)
    personalization[gene] = 1
    
    # the nx implementation is rather slow
    #personalization_vector = nx.pagerank(network, personalization=personalization)
    
    # implementation from ADRI, same result much faster
    current_node = [str(x) for x,v in personalization.items() if v==1]
    N =len(nodelist)

    # initial vector
    x = np.ones(N)/N

    # Personalization vector
    if personalization is None:
        p = np.ones(N)/N
    else:
        missing = set(nodelist) - set(personalization)
        if missing:
            raise nx.exception.NetworkXError('Personalization vector dictionary must have a value for every node. '
                                             'Missing nodes %s' % missing)
        p = np.array([personalization[n] for n in nodelist], dtype=float)
        p = p / p.sum()

    # Dangling nodes
    if dangling is None:
        dangling_weights = p
    else:
        missing = set(nodelist) - set(dangling)
        if missing:
            raise nx.exception.NetworkXError('Dangling node dictionary must have a value for every node. Missing '
                                             'nodes %s' % missing)
        # Convert the dangling dictionary into an array in nodelist order
        dangling_weights = np.array([dangling[n] for n in nodelist], dtype=float)
        dangling_weights /= dangling_weights.sum()
    # print(x,p,M.sum())
    # power iteration: make up to max_iter iterations
    for _ in range(max_iter):
        xlast = x
        x = alpha * (M*x + sum(x[is_dangling]) * dangling_weights) + (1 - alpha) * p
        # check convergence, l1 norm
        err = np.abs(x - xlast).sum()
        if err < N * tol:
            return dict(zip(nodelist, map(float, x))) # ,x,M)
    raise nx.exception.NetworkXError('power iteration failed to converge in %d iterations. Node: ' % max_iter + ' '.join(current_node))



        
  

