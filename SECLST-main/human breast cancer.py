import os
import torch
import pandas as pd
import scanpy as sc
from sklearn import metrics
import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from SECLST import SECLST


sc.settings.verbosity = 4


# Run device, by default, the package is implemented on 'cpu'. We recommend using GPU.
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')



# the location of R, which is necessary for mclust algorithm. Please replace the path below with local R installation path
os.environ['R_HOME'] = 'D:\software\R-4.2.2'

# the number of clusters
n_clusters = 20




file_fold = 'E:/code/GraphST/GraphST-main/Data/Human_Breast_Cancer/'

adata = sc.read_visium(file_fold, count_file='filtered_feature_bc_matrix.h5', load_images=True)
adata.var_names_make_unique()



plt.rcParams["figure.figsize"] = (3, 4)
wspace_value = 0.65


adata.obsm['spatial'][:, 1] = -1*adata.obsm['spatial'][:, 1]
plot_color=["#F56867","#556B2F","#C798EE","#59BE86","#006400","#8470FF",
            "#CD69C9","#EE7621","#B22222","#FFD700","#CD5555","#DB4C6C",
            "#8B658B","#1E90FF","#AF5F3C","#CAFF70", "#F9BD3F","#DAB370",
           "#877F6C","#268785", '#82EF2D', '#B4EEB4']

wspace_value = 0.65

adata#4226*33538

# define model
model = SECLST.SECLST(adata,device=device)

# train model
adata = model.train()


adata

# set radius to specify the number of neighbors considered during refinement
# set radius to specify the number of neighbors considered during refinement

radius = 50

tool = 'mclust' # mclust, leiden, and louvain

# clustering
from SECLST.utils import clustering


if tool == 'mclust':
   clustering(adata, n_clusters, radius=radius, method=tool, refinement=True) # For DLPFC dataset, we use optional refinement step.
elif tool in ['leiden', 'louvain']:
   clustering(adata, n_clusters, radius=radius, method=tool, start=0.1, end=2.0, increment=0.01, refinement=False)#"E:\code\STAGATE_pyG-main\data\DLPFC\151676_truth.txt"


# add ground_truth
df_meta = pd.read_csv(file_fold + '/metadata.tsv', sep='\t')
df_meta_layer = df_meta['ground_truth']
adata.obs['ground_truth'] = df_meta_layer.values




# filter out NA nodes
adata = adata[~pd.isnull(adata.obs['ground_truth'])]

# calculate metric ARI
#ARI = metrics.adjusted_rand_score(labels, adata.obs['ground_truth'])
ARI = metrics.adjusted_rand_score(adata.obs['domain'], adata.obs['ground_truth'])
adata.uns['ARI'] = ARI

print("ARI=%.4f" % ARI)

print("ARI=%.2f" % ARI)


# plotting spatial clustering result
# plotting spatial clustering result
# 添加第二个图例

wspace_value = 1.5


adata.obsm['spatial'][:, 1] = -1*adata.obsm['spatial'][:, 1]

plt.rcParams["figure.figsize"] = (10, 4)
sc.pl.spatial(adata,
              img_key="hires",
              color=["ground_truth","domain"],
              title=["ground_truth","ARI=%.4f"%ARI],
              show=False,
              wspace=wspace_value)
