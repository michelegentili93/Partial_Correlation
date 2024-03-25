import pandas as pd
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

def read_data(dict_params,dir_pcorr_output):
    '''
    Read all the Partial Correlation obtained with different run of the algorithm.
    '''    
    df_melt_pcor = pd.DataFrame()
    list_genes_ch4 = None

    list_params = []
    array_pcors = []

    gex_name,list_pop,list_ch,list_n_genes,list_min_lambda = dict_params['gex_name'],dict_params['list_pop'],dict_params['list_ch'],dict_params['list_n_genes'],dict_params['list_min_lambda']
    for pop in list_pop:#['case']:#['case','control']:
        array_pcors_pop = []
        for ch in list_ch:
            for ng in tqdm(list_n_genes):
                for ml in list_min_lambda:
                    if ng==0 and ml != 0:
                            continue
                    list_params.append((pop,ch,ng,ml))

                    df = pd.read_csv(dir_pcorr_output+'%s/nreg_%i_minL_%s_ch_%s/obs_%s.csv' %(gex_name,ng,str(ml),ch,pop),index_col=0)
                    if list_genes_ch4 is None:
                        list_genes_ch4 = list(df.index)
                    array_pcors_pop.append(df.loc[list_genes_ch4,list_genes_ch4])
                    df = df.melt(ignore_index=False).reset_index()
                    df['population'] = pop
                    df['chromosomes'] = ch
                    df['n_genes'] = ng
                    df['min_lambda'] = ml
                    df_melt_pcor = pd.concat([df_melt_pcor,df])
        array_pcors.append(array_pcors_pop)

    array_pcors = np.array(array_pcors)

    #remove the diagonal
    df_melt_pcor = df_melt_pcor.rename({'index':'gene1','variable':'gene2'},axis=1)
    df_melt_pcor = df_melt_pcor.query('gene1!=gene2').copy()
    df_melt_pcor['edge'] = df_melt_pcor['gene1'] + '-' + df_melt_pcor['gene2']
    df_melt_pcor['value_abs']= df_melt_pcor.value.abs()
    df_melt_pcor = df_melt_pcor.sort_values('edge')
    
    return df_melt_pcor,array_pcors,list_params

import seaborn as sns
from matplotlib.ticker import MultipleLocator

def plot_edges_distr(df_melt_pcor,list_pop,set_target_genes,ch='4'):
    plt.figure(figsize=[25,10*len(list_pop)])
    # to have better labeling for the plot
    dict_pop_name_plots_title = {'control':'Control','case':'COPD','Laval':'Laval','UBC':'UBC','GRNG':'GRNG'}
    dict_pop_name_plots_xlabel = {'control':'(b)','case':'(a)','Laval':'Laval','UBC':'UBC','GRNG':'GRNG'}

    for i,tmp_pop in enumerate(list_pop):
        
        df_melt_pcor_filt = df_melt_pcor.query('n_genes>=1 and population == @tmp_pop and chromosomes==@ch and gene1 in @set_target_genes and gene2 in @set_target_genes')
        df_melt_pcor_filt = df_melt_pcor_filt.sort_values(['gene1','gene2'])
        ax = plt.subplot(len(list_pop),1,i+1)
        plt.axhline(0,c='black',linestyle='-',alpha=.5,lw=.5,zorder=0)
        plt.title(dict_pop_name_plots_title[tmp_pop],{'fontsize': 25})
        
        sns.boxplot(x="gene1", y="value", hue='gene2',data=df_melt_pcor_filt,ax=ax,hue_order = list(set_target_genes),order = list(set_target_genes))
        
        ax.set_xticklabels(ax.get_xticklabels(),rotation = 30)
        ax.xaxis.set_minor_locator(MultipleLocator(0.5))
        ax.xaxis.grid(True, which='minor', color='black', lw=1,linestyle='--')
        plt.xlabel(dict_pop_name_plots_xlabel[tmp_pop],{'fontsize': 30})
        plt.ylabel('Partial Correlation',{'fontsize': 25})
        xlabel_order = [x.get_text() for x in plt.xticks()[1]]
        plt.legend(ncol=len(set_target_genes), fancybox=True, shadow=True, prop={'size': 20},loc='upper center')
        plt.axhline(0,c='black',linestyle='dashed',alpha=.3)
    plt.subplots_adjust(hspace=.3)



def get_triu(df,offset=1):
    triu = np.triu_indices(df.shape[0],offset)
    return df.values[triu]