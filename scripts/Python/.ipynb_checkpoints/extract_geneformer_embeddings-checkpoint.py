import pickle
import numpy as np
import anndata as ad
import pandas as pd
import scanpy as sc
import scipy.sparse as sp 
import matplotlib.pyplot as plt
import anndata


# For PCA
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# For umap
import umap

# SCVI
import scvi
import cellxgene_census
import cellxgene_census.experimental

import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from geneformer import EmbExtractor
import numpy as np


# Tokenize data
from geneformer import TranscriptomeTokenizer

tk = TranscriptomeTokenizer({"type": "type", 
                             "cell_index":"cell_index",
                             "cell_id": "cell_id"})


tk.tokenize_data("/gpfs/commons/groups/compbio/projects/rf_projects/deconv_data/GSE220608/geneformer/", 
                 "/gpfs/commons/groups/compbio/projects/rf_projects/deconv_data/GSE220608/geneformer/tokenize/", 
                 "token_czi_cell_lines", 
                 file_format="h5ad")


### Create Geneformer Embeddings
from geneformer import EmbExtractor

#initiate EmbExtractor
embex = EmbExtractor(forward_batch_size=50,
                     emb_label=["type", "cell_id", "cell_index"],
                     emb_layer=-1,
                     max_ncells = None)


# extracts embedding from input data
embs = embex.extract_embs("ctheodoris/Geneformer",
                          "/gpfs/commons/groups/compbio/projects/rf_projects/deconv_data/GSE220608/geneformer/tokenize/token_czi_cell_lines.dataset/",
                          "/gpfs/commons/groups/compbio/projects/rf_projects/deconv_data/GSE220608/geneformer/output/",
                          "cell_line_gf_embed")

#/nfs/sw/geneformer/geneformer-1.0
#ctheodoris/Geneformer
#/nfs/sw/easybuild/software/custom-conda/geneformer-1.0/Geneformer/gf-12L-95M-i4096
#/nfs/sw/easybuild/software/custom-conda/geneformer-1.0/Geneformer/gf-20L-95M-i4096/