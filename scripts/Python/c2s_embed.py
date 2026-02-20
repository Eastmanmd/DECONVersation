import anndata
import scanpy as sc
import pandas as pd
import cell2sentence as cs
from cell2sentence.tasks import embed_cells
import os
import sys
from pathlib import Path
import logging
import tracemalloc
import linecache
import torch

file1 = "/gpfs/commons/groups/compbio/projects/rf_projects/deconv_data/deconvBench/tm_full_signature_matrix_symbol.csv"
c2s_save_dir1="/gpfs/commons/groups/compbio/projects/rf_projects/deconv_data/GSE220608/c2s/410m"
c2s_save_name1 = "tm_full_signature_9090"
model_path1 = "/gpfs/commons/groups/compbio/projects/rf_projects/deconv_data/deconvbench/c2s/410mmouse/2026-02-08-08_41_26_finetune_cell_type_prediction/checkpoint-9090"
model_save_dir1 = "/gpfs/commons/groups/compbio/projects/rf_projects/deconv_data/GSE220608/c2s/410m/mouse"
model_save_name1 = "mouse_embedding_prediction"

def display_top(snapshot, key_type='lineno', limit=10):
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        print("#%s: %s:%s: %.1f KiB"
              % (index, frame.filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))
    return(total / 1024 / 1024)

def get_embedding_c2s(file,
                      c2s_save_dir, c2s_save_name,
                      model_path, model_save_dir, model_save_name,
                      embedding_save_dir = None,
                      transpose = False,
                      index_name = "Unnamed: 0",
                      gene_name_rm = "\\..+",
                      use_genes = None,
                      gene_name = None, reorder_obs_name = None,
                      ngenes = 200,
                      log = True, log_path = None):
    # Set up log
    if log:
        logging.basicConfig(format='%(asctime)s %(message)s', level=logging.WARNING,
                            filename=os.path.dirname(file1) + "/c2s_embeddings_" + Path(file1).stem + ".log",
                            filemode='w', force = True)
        logger = logging.getLogger(__name__)
        sys.stdout.write = logger.info
        sys.stderr.write = logger.error
        logger.warning('Prep data')
        tracemalloc.start()
    # Load data
    data = pd.read_csv(file, header = 0)
    data = data.set_index(index_name)
    if transpose:
        data = data.T
    if gene_name_rm is not None:
        data.columns = data.columns.str.replace(gene_name_rm, "", regex= True)
        #data[index_name] = data[index_name].str.replace(gene_name_rm, "", regex= True)
    if use_genes is not None:
        data = data[use_genes]
        #data = data.loc[use_genes]
    data = data.loc[:, ~data.columns.duplicated()]
    adata = sc.AnnData(data)
    if gene_name is not None:
        adata.var_names = gene_name
    adata.obs["organism"]="Homo sapiens"
    adata.obs["cell_type"]="unknown"
    adata.obs["tissue"]="unknown"
    adata.obs["sex"]="unknown"
    adata.obs["batch_condition"]="unknown"
    adata_obs_cols_to_keep = ["organism","cell_type","tissue", "sex", "batch_condition"]

    # Create CSData object
    if log:
        logger.warning('Prep model')
    arrow_ds, vocabulary = cs.CSData.adata_to_arrow(
        adata=adata,
        random_state=42,
        sentence_delimiter=' ',
        label_col_names=adata_obs_cols_to_keep
    )
    csdata = cs.CSData.csdata_from_arrow(
        arrow_dataset=arrow_ds,
        vocabulary=vocabulary,
        save_dir=c2s_save_dir,
        save_name=c2s_save_name,
        dataset_backend="arrow"
    )

    # Define CSModel object
    csmodel = cs.CSModel(
        model_name_or_path=model_path,
        save_dir=model_save_dir,
        save_name=model_save_name
    )

    # Get embeddings
    if log:
        logging.warning('Extract embedding')
    embedded_cells = embed_cells(
        csdata=csdata,
        csmodel=csmodel,
        n_genes=ngenes
    )
    df = pd.DataFrame(embedded_cells)
    if reorder_obs_name is None:
        df["name"] = adata.obs_names.to_list()
    else:
        df["name"] = ["sample_" + str(x) for x in list(range(1, df.shape[0]+1))]
    if log:
        logging.warning('Save embedding')
    if embedding_save_dir is None:
        df.set_index('name').to_csv(os.path.dirname(file1) + "/c2s_embeddings_" + Path(file1).stem + ".csv")
    else:
        df.set_index('name').to_csv(embedding_save_dir + "/c2s_embeddings_" + c2s_save_name + ".csv")
    if log:
        logger.warning('Done')
        snapshot = tracemalloc.take_snapshot()
        mem=display_top(snapshot)
        logger.warning('Memory used: ' + str(mem) + " MiB")
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            logger.warning('GPU Memory used: ' + str((total - free) / 1024 / 1024) + " MiB")

get_embedding_c2s(file1,
                  c2s_save_dir1, c2s_save_name1, 
                  model_path1, model_save_dir1, model_save_name1,
                  embedding_save_dir = model_save_dir1, ngenes = 200,
                  index_name = "cell_type")
