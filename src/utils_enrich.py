import ray
import gseapy as gp
import pandas as pd
from tqdm import tqdm
import numpy as np
def to_iterator(obj_ids):
    while obj_ids:
        done, obj_ids = ray.wait(obj_ids)
        yield ray.get(done[0])


        
def get_GSEA_library_name():
    # clean enrichment library name
    library_name = sorted(gp.get_library_name())
    enr_lib_name = {}

    observed = set()
    substrings = ["jensen tissues", "gtex", "coexp", "enrichr", "archs4", "old", "_up", "_down"]
    library_name = [x for x in library_name if all([substring not in x.lower() for substring in substrings])]

    library_name = [x for x in library_name if 'exp' not in x.lower() or 'OMIM' in x]
    library_name = [x for x in library_name if 'kegg' not in x.lower() or '2021' in x]
    library_name = [x for x in library_name if 'wiki' not in x.lower() or '2021' in x]
    library_name = [x for x in library_name if 'mgi' not in x.lower() or '2021' in x]
    library_name = [x for x in library_name if 'GO_' != x[:3] or 'b' != x[-1]]

    for x in library_name[1:]:
        y = x.lower()
        if 'mouse' in y or 'grant' in y or 'chromosome_location' in y or 'covid' in y or 'nih_fu' in y or 'cell-lines' in y:
            continue

        # remove old versione same db
        last_num = x.split('_')[-1]

        if not last_num.isdigit() or len(last_num) <= 2:
            enr_lib_name[x] = x
        else:
            name_x = '_'.join(x.split('_')[:-1])
            year = int(x.split('_')[-1][:-1])

            if name_x in enr_lib_name:
                try:
                    if year < int(enr_lib_name[name_x].split('_')[-1][:-1]):
                        continue
                except:
                    pass  # print(x,name_x,enr_lib_name[name_x])
            enr_lib_name[name_x] = x

    return list(enr_lib_name.values())

@ray.remote
def ray_parall_singl_enr(x):
    gene_list, library_name, background_gene = x
    try:
        df_result = gp.enrichr(gene_list=gene_list,
                               gene_sets=library_name,
                               background=background_gene,
                               verbose=False)
    except Exception as e:
        print(e)
        return library_name
    
    df_result = df_result.results.sort_values('Adjusted P-value').drop(
        ['Old Adjusted P-value', 'Old P-value', 'Combined Score'], axis=1)
    df_result = df_result[df_result['Adjusted P-value']<0.01]
    df_result['n_found']=df_result.Overlap.map(lambda x: x.split('/')[0])
    df_result.Genes = df_result.Genes.str.replace(';', ' ')
    df_result.Gene_set = df_result.Gene_set.str.replace('_', ' ')
    df_result = df_result[~df_result['Gene_set'].str.contains("NIH Funded ")]
    return df_result

def clean_parallel_output_enr(output):
    missing_chunck = []
    list_df_results = []
    for i in output:
        if  isinstance(i, str):
            # list of library query that returned an error
            missing_chunck.append(i)
        else:
            # append the df of results
            list_df_results.append(i)
    return missing_chunck, list_df_results

def parallel_gseapy_enr(gene_list, background_gene, list_library_name=None):
    '''
    Need the while loop because some calls to gseapy return errors.
    '''
    if not ray.is_initialized():
        ray.init(log_to_driver=False)
    list_df_results = []
    id_background_gene = ray.put(background_gene)

    if list_library_name is None:
        list_library_name = get_GSEA_library_name()
    while len(list_library_name) != 0:
        res_id = [ray_parall_singl_enr.remote([gene_list, library_name, id_background_gene]) for library_name in
                  list_library_name]
        results = [x for x in tqdm(to_iterator(res_id), total=len(res_id))]
        list_library_name, new_results = clean_parallel_output_enr(results)
        list_df_results = list_df_results + new_results
    df_enr = pd.concat(list_df_results).sort_values('Adjusted P-value').reset_index(drop=True)
    #ray.shutdown()
    return df_enr
