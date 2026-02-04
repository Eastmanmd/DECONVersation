#!/nfs/sw/easybuild/software/custom-conda/geneformer-1.0/bin/python

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
#import scvi
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


tk.tokenize_data("/gpfs/commons/groups/compbio/projects/rf_projects/deconv_data/deconvBench/geneformer/complete/", 
                 "/gpfs/commons/groups/compbio/projects/rf_projects/deconv_data/deconvBench/geneformer/complete/tokenize/", 
                 "token_hoek", 
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
                          "/gpfs/commons/groups/compbio/projects/rf_projects/deconv_data/deconvBench/geneformer/complete/tokenize/token_hoek.dataset/",
                          "/gpfs/commons/groups/compbio/projects/rf_projects/deconv_data/deconvBench/geneformer/complete/output/",
                          "hoek_gf_embed")
