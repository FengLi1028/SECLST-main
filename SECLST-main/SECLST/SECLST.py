
import torch
from .preprocess import preprocess_adj, preprocess_adj_sparse, preprocess, construct_interaction, \
    construct_interaction_KNN, add_contrastive_label, get_feature, permutation, fix_seed,shuffle_adjacency_matrix
import time
import random
import numpy as np
from .model import Encoder, Encoder_sparse
from tqdm import tqdm
from torch import nn
import torch.nn.functional as F
from scipy.sparse.csc import csc_matrix
from scipy.sparse.csr import csr_matrix
import pandas as pd
from .layers import NBLoss, MeanAct, DispAct
import csv


class SECLST():
    def __init__(self,
                 adata,
                 adata_sc = None,
                 device= torch.device('cpu'),
                 learning_rate=0.001,
                 weight_decay=0.00,
                 epochs=400,
                 dim_input=3000,
                 dim_output=64,
                 random_seed = 0,
                 mean=0.2,
                 std=0.3,
                 alpha=8,
                 beta=3,
                 theta=0.01,
                 lamda=0.01,
                 datatype = '10X'
                 ):

        self.adata = adata.copy()
        self.device = device
        self.learning_rate =learning_rate
        self.weight_decay =weight_decay
        self.epochs =epochs
        self.random_seed = random_seed
        self.alpha = alpha
        self.beta = beta
        self.theta = theta
        self.lamda = lamda

        self.datatype = datatype

        fix_seed(self.random_seed)

        if 'highly_variable' not in adata.var.keys():
            preprocess(self.adata)

        if 'adj' not in adata.obsm.keys():
            if self.datatype in ['Stereo', 'Slide']:
                construct_interaction_KNN(self.adata)
            else:
                construct_interaction(self.adata)

        if 'label_CSL' not in adata.obsm.keys():
            add_contrastive_label(self.adata)

        if 'feat' not in adata.obsm.keys():
            get_feature(self.adata,mean,std)

        self.features = torch.FloatTensor(self.adata.obsm['feat'].copy()).to(self.device)
        self.features_a = torch.FloatTensor(self.adata.obsm['feat_a'].copy()).to(self.device)
        self.label_CSL = torch.FloatTensor(self.adata.obsm['label_CSL']).to(self.device)
        self.adj = self.adata.obsm['adj']
        self.adj_a = shuffle_adjacency_matrix(self.adj)
        self.graph_neigh = torch.FloatTensor(self.adata.obsm['graph_neigh'].copy() + np.eye(self.adj.shape[0])).to(self.device)

        self.label_CSL1 = torch.FloatTensor(self.adata.obsm['label_CSL'][:, 0].reshape(-1, 1)).to(self.device)
        self.label_CSL0 = torch.FloatTensor(self.adata.obsm['label_CSL'][:, 1].reshape(-1, 1)).to(self.device)

        self.dim_input = self.features.shape[1]
        self.dim_output = dim_output

        if self.datatype in ['Stereo', 'Slide']:
            # using sparse
            print('Building sparse matrix ...')
            self.adj = preprocess_adj_sparse(self.adj).to(self.device)
            self.adj_a = preprocess_adj_sparse(self.adj_a).to(self.device)
        else:
            # standard version
            self.adj = preprocess_adj(self.adj)
            self.adj = torch.FloatTensor(self.adj).to(self.device)

            self.adj_a = preprocess_adj(self.adj_a)
            self.adj_a = torch.FloatTensor(self.adj_a).to(self.device)

    def train(self):
        if self.datatype in ['Stereo', 'Slide']:
            self.model = Encoder_sparse(self.dim_input, self.dim_output, self.graph_neigh).to(self.device)
        else:
            self.model = Encoder(self.dim_input, self.dim_output, self.graph_neigh).to(self.device)
        self.loss_CSL = nn.BCEWithLogitsLoss()
        self.nb_loss = NBLoss().cuda()

        self.optimizer = torch.optim.Adam(self.model.parameters(), self.learning_rate,
                                          weight_decay=self.weight_decay)

        print('Begin to train ST data...')
        self.model.train()
        loss_values = []
        for epoch in tqdm(range(self.epochs)):
            self.model.train()


            self.hiden_feat, self.emb,ret,ret_a,mean,disp= self.model(self.features, self.features_a, self.adj,self.adj_a)


            self.loss_feat = F.mse_loss(self.features, self.emb)
            self.loss_sl_1 = self.loss_CSL(ret, self.label_CSL0)#正和弱负是拉远
            self.loss_sl_2 = self.loss_CSL(ret_a, self.label_CSL1)#强负和弱负是拉进
            self.label_CSL = torch.FloatTensor(self.adata.obsm['label_CSL']).to(self.device)

            self.loss_nb = self.nb_loss(x=self.features, mean=mean, disp=disp)

            #loss = self.loss_feat +self.loss_sl + self.loss_nb
            loss = self.alpha * self.loss_feat + self.beta * self.loss_sl_1 + self.theta * self.loss_sl_2 + self.lamda * self.loss_nb

            #loss = 6 * self.loss_feat + 0.9* self.loss_sl_1+0.01* self.loss_sl_2  + 0.5* self.loss_nb
            #loss = 10 * self.loss_feat + 0.7 * self.loss_sl_1 + 0* self.loss_sl_2 + 5 * self.loss_nb
            #loss =  2 *self.loss_feat + 0.1*self.loss_sl_1 + 9*self.loss_sl_2+0.5*self.loss_nb

            loss_values.append(loss)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        print("Optimization finished for ST data!")

        with torch.no_grad():
            self.model.eval()

            if self.datatype in ['Stereo', 'Slide']:
                self.emb_rec = self.model(self.features, self.features_a, self.adj,self.adj_a)[1]
                self.emb_rec = F.normalize(self.emb_rec, p=2, dim=1).detach().cpu().numpy()
            else:
                self.emb_rec = self.model(self.features, self.features_a, self.adj,self.adj_a)[1].detach().cpu().numpy()
            self.adata.obsm['emb'] = self.emb_rec

            return self.adata
