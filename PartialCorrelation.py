import pandas as pd
import numpy as np
import time
from shutil import rmtree
from tqdm import tqdm
from scipy.sparse import csr_matrix,lil_matrix
from sklearn.preprocessing import StandardScaler
import ray

def to_iterator(obj_ids):
    while obj_ids:
        done, obj_ids = ray.wait(obj_ids)
        yield ray.get(done[0])


class PartialCorrelation:

    def __init__(self, df_exp: pd.DataFrame, df_prior: pd.DataFrame, n_controlling_genes,min_scaled_lambda,
                 parallel=False, verbose=False, list_genes_to_analyze=None):
        '''
        list_genes_to_analyze = list of genes for which you want to compute to partial correlation. If a gene is not in the index of the df_exp is discarded
        '''
        self.verbose = verbose
        self.df_exp = df_exp
        
        if list_genes_to_analyze is None:
            self.list_genes_to_analyze = df_exp.index
        else:
            self.list_genes_to_analyze = list(set(list_genes_to_analyze).intersection(df_exp.index))
            print('Analyzing %i genes'%len(self.list_genes_to_analyze))
            
            
        self.df_prior = self.fix_prior(df_prior)
        self.parallel = parallel
        self.n_controlling_genes = n_controlling_genes
        self.min_scaled_lambda = min_scaled_lambda        
        
        if parallel:
            import ray,psutil
            if not ray.is_initialized():
                ### Ray Initialisation
                num_cpus = psutil.cpu_count(logical=False) - 1
                print('Using ',num_cpus,'cpus')
                ray.init(log_to_driver=True, num_cpus=num_cpus)
            
        
        self.df_partial_correlation = self.compute_partial_correlation()
        # self.list_genes_to_analyze imposes the order on the genes,
        # will be kept in mapping_geneID__position_result, and in computing _single_partial_correlation

    def fix_prior(self,df_prior):
        # Get the prior only of the genes to analyze (if missing use max value of penalty)
        # Use all the genes in df_exp as controlling genes
        df_prior_new  = pd.DataFrame(df_prior.values.max(),columns= self.df_exp.index, index=self.list_genes_to_analyze)
        list_inters_genes = list(set(self.list_genes_to_analyze).intersection(df_prior.index))
        list_inters_genes_cont = list(set(self.df_exp.index).intersection(df_prior.index))
        df_prior_new.loc[list_inters_genes,list_inters_genes_cont]=df_prior.loc[list_inters_genes,list_inters_genes_cont]
        
        return df_prior_new

    

    def _extract_results_partial_corr(self,gene_i,dict_result_i,mapping_gene_ID):
        pos_gene_i = mapping_gene_ID[gene_i]
        pos_genes_j = []
        pcorrs_gene_i = []
        for gene_j in dict_result_i:
            pos_genes_j.append(mapping_gene_ID[gene_j])
            pcorrs_gene_i.append(dict_result_i[gene_j])
        self.partial_corr[pos_gene_i,pos_genes_j] = self.partial_corr[pos_genes_j,pos_gene_i] = pcorrs_gene_i

    def compute_partial_correlation(self):
        '''
        @Input:
            gene_expression: (n,m) numpy array  of n genes and m patients
            matrix_prior: (n,n) numpy array,
                           entry (i,j) is the coefficient (lambda_ij) to regress out the j-th gene to compute
                           the partial correlation of i-th gene with an other gene. Can be asymmetrical.
            function_prior: a function that given a gene return the array of the prior lambda on all the genes.
                            Keep the same ordering as for the gene expression matrix, the i-th entry won't be considered

        NB: the prior is a score such that the lower the less will be penalized in the lasso regression


        '''

        start_time = time.time()
        mapping_gene_ID = {gene:i for i,gene in enumerate(self.list_genes_to_analyze)}
        
        if self.verbose:
            print('\t gene expression shape:', self.df_exp.shape)
        # global variables to enter the parallel funciton
        

        self.df_prior = 1/self.df_prior
        self.partial_corr = np.zeros([len(self.list_genes_to_analyze),len(self.list_genes_to_analyze)])

            
        if self.verbose: print("TIME LOOKUP TABLE: %s seconds " % (time.time() - start_time))

        if self.parallel:
            res_id = [self._single_partial_correlation_ray.remote(self,gene_i) for gene_i in self.list_genes_to_analyze]
            for gene_i, dict_result_i in tqdm(to_iterator(res_id), total=len(res_id)):
                self._extract_results_partial_corr(gene_i,dict_result_i,mapping_gene_ID)
        else:
            if self.verbose:
                res_id = [self._single_partial_correlation(gene_i) for gene_i in tqdm(self.list_genes_to_analyze)]
            else:
                res_id = [self._single_partial_correlation(gene_i) for gene_i in self.list_genes_to_analyze]        
            for gene_i, dict_result_i in res_id:
                self._extract_results_partial_corr(gene_i,dict_result_i,mapping_gene_ID)

        if self.verbose: print("TIME Partial Computed: %s seconds " % (time.time() - start_time))
        
        return pd.DataFrame(self.partial_corr, columns=self.list_genes_to_analyze, index=self.list_genes_to_analyze)


    def _work_lookup_table(self, gene_i):

        lambda_i = self.df_prior.loc[gene_i]
        residuals, indx_used_genes = self._residual(gene_i, gene_i, lambda_i)

        return residuals, indx_used_genes, gene_i
    
    @ray.remote
    def _work_lookup_table_ray(self, gene_i):
        return self._work_lookup_table(gene_i)
    
    def _extract_results_lookup_table(self,residuals, indx_used_genes, gene_i):
        self.look_up_residuals[gene_i]['used_genes'] = set(indx_used_genes)
        self.look_up_residuals[gene_i]['used_genes_list'] = indx_used_genes
        self.look_up_residuals[gene_i]['residuals'] = residuals
        
    def _create_lookup_table(self):
        '''
        Compute the residuals lookup table and the matrix_prior. NB the lookup table will be used only for parital correlation with a gene that is not used in the lasso regression (top 30 genes from prior in used_genes)

        '''

        if self.verbose:
            print('Computing Lookup table for residuals')

        n_genes = self.df_exp.shape[0]
        self.look_up_residuals = {gene: {'used_genes': None, 'residuals': None} for gene in self.list_genes_to_analyze}

        if self.parallel:
            res_id = [self._work_lookup_table_ray.remote(self,gene_i) for gene_i in self.list_genes_to_analyze]
            for residuals, indx_used_genes, gene_i in tqdm(to_iterator(res_id), total=len(res_id)):
                self._extract_results_lookup_table(residuals, indx_used_genes, gene_i)
        else:
            res_id = [self._work_lookup_table(gene_i) for gene_i in tqdm(self.list_genes_to_analyze)]
            for residuals, indx_used_genes, gene_i in res_id:
                self._extract_results_lookup_table(residuals, indx_used_genes, gene_i)

        if self.verbose: print("TIME Partial Computed: %s seconds " % (time.time() - start_time))

        for gene_j in self.look_up_residuals.keys():
            if self.look_up_residuals[gene_j]['used_genes'] is None:
                print('problem', gene_j)
                exit()
        #save files for further use or debug
        #import pickle,sys
        #with open("tmp_look_up_residualsdf.pckl", "wb") as write_file:
        #    pickle.dump(self.look_up_residuals, write_file)
        


    def compute_pcorr_inverse_cov(self):
        m = np.cov(self.df_exp.values)
        Vi = np.linalg.pinv(m)  # Inverse covariance matrix
        D = np.diag(np.sqrt(1 / np.diag(Vi)))
        pcor = -1 * (D @ Vi @ D)
        pcor[np.diag_indices_from(pcor)] = 1
        # pcor = pd.DataFrame(pcor,index=pcorr_30.index,columns=pcorr_30.index)
        return pcor

    
    
    @ray.remote
    def _single_partial_correlation_ray(self,gene_i):
        return _single_partial_correlation(self, gene_i)
    
    def _single_partial_correlation(self, gene_i):
        '''
        Compute partial correlation for (gene_i,gene_j) in the expression matrix usinge lambda_(i,j)
        as lambda coefficient of the weigths for the ridge regression to compute residuals

        gene_(i,j): int
        self.gex_exp: matrix(n,m) with the expression, n genes and m observations
        lambda_(i,j): array of length n

        '''
        
        # this is the page rank score
        lambda_i = self.df_prior.loc[gene_i]

        # use the lookup table only if the other gene in partial correlation is not used for the regression
        results = {}

        for gene_j in self.list_genes_to_analyze:
            if gene_j>=gene_i:
                continue
            lambda_ij = 1/((lambda_i +self.df_prior.loc[gene_j])/2)
            ri_tmp, _ = self._residual(gene_i, gene_j, lambda_ij)
            rj_tmp, _ = self._residual(gene_j, gene_i, lambda_ij)
            results[gene_j] = np.corrcoef(ri_tmp, rj_tmp)[0, 1]

        return (gene_i, results)
    
        
        
    def _single_partial_correlation_old(self, gene_i):
        '''
        Compute partial correlation for (gene_i,gene_j) in the expression matrix usinge lambda_(i,j)
        as lambda coefficient of the weigths for the ridge regression to compute residuals

        gene_(i,j): int
        self.gex_exp: matrix(n,m) with the expression, n genes and m observations
        lambda_(i,j): array of length n

        '''
        lambda_i = self.df_prior.loc[gene_i]
        # use the lookup table only if the other gene in partial correlation is not used for the regression
        results = {}

        for gene_j in self.look_up_residuals[gene_i]['used_genes']:
            if gene_j not in self.list_genes_to_analyze:
                continue
            if gene_j > gene_i and gene_i in self.look_up_residuals[gene_j]['used_genes']:
                continue  # the remaining will be compute when processing them

            r1_tmp, _ = self._residual(gene_i, gene_j, lambda_i)

            # extract residuals gene_j
            if gene_i in self.look_up_residuals[gene_j]['used_genes']:
                lambda_j = self.df_prior.loc[gene_j]
                r2, _ = self._residual(gene_j, gene_i, lambda_j)
            else:
                r2 = self.look_up_residuals[gene_j]['residuals']

            results[gene_j] = np.corrcoef(r1_tmp, r2)[0, 1]

        return (gene_i, results)

    def _residual(self, gene_i, gene_j, lambda_i):
        '''
        compute the residuals predicting gene_i given all the others genes but gene_j
        more info http://statweb.stanford.edu/~owen/courses/305a/Rudyregularization.pdf
        '''

        #gene_debug_1='EMCN'
        #gene_debug_1 = self.mapping_geneName_id[gene_debug_1]

        #if (gene_i == gene_debug_1 and gene_j == gene_debug_1):
        #    print(gene_i,gene_j,gene_debug_1,gene_debug_2)
        #    with open('Debug1','wb') as f:
        #        pickle.dump({'lambda_i':lambda_i,
        #                   'mapping_id_geneName':self.mapping_id_geneName},f)

        
        y = self.df_exp.loc[gene_i].values
        
        if self.n_controlling_genes ==0:
            return y,[]
        n_patients = len(y)

        # use only the first n_controlling_genes genes for the regression
        # don't sort the entire array, take the index of smallest n_controlling_genes
        # sort only those and then take the index wtr to the entire vector

        # get id genes with smallest penalty prior. remove gene_i and gene_j afterwards
        n_controlling_genes = min(self.n_controlling_genes,len(lambda_i)-3)
        controlling_genes_top_k, lambda_i_top_k = top_k(lambda_i.drop([gene_i,gene_j]),n_controlling_genes)
        if self.min_scaled_lambda is None:
            scale_lambda = 1
        else:
            
            # scale_lambda is a scaling factor, so that the smallest lambda used is equal to the min_scaled_lambda
            scale_lambda = self.min_scaled_lambda / min(lambda_i_top_k)

        lambda_i_top_k = lambda_i_top_k * scale_lambda
        
        gene_expression = self.df_exp.loc[controlling_genes_top_k].transpose()
        gene_expression = StandardScaler().fit_transform(gene_expression)
        scaler_y = StandardScaler()
        y = scaler_y.fit_transform(y.reshape(-1,1)).reshape(-1)

        res = fast_ridge_residuals(gene_expression,y,lambda_i_top_k).reshape(-1,1)
        res = scaler_y.inverse_transform(res.reshape(-1,1)).reshape(-1)
        
        return (res, controlling_genes_top_k)

    
def fast_ridge_residuals(X,y,lambda_i_top_k):
    # faster regression residuals
    beta = np.linalg.inv(np.dot(X.T,X)+np.diag(lambda_i_top_k))
    beta = np.dot(np.dot(beta,X.T),y)
    res_i_check = y-np.dot(X,beta)
    return res_i_check


def top_k(x,k):
    # extract top k values and index from x pandas DataFrame
    values = x.values
    genes = x.index
    index = values.argpartition(k)
    final_index = values[index[:k]].argsort() 
    return genes[index[final_index]],values[index[final_index]]
