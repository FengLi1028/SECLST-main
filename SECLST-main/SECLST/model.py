import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from .layers import MeanAct, DispAct

class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, s_bias1=None, s_bias2=None):
        c_x = c.expand_as(h_pl)

        sc_1 = self.f_k(h_pl, c_x)


        if s_bias1 is not None:
            sc_1 += s_bias1


        logits = sc_1

        return logits




class Encoder(Module):
    def __init__(self, in_features, out_features, graph_neigh, dropout=0.0, act=F.relu):
        super(Encoder, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.graph_neigh = graph_neigh
        self.dropout = dropout
        self.act = act

        self.weight1 = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        self.weight2 = Parameter(torch.FloatTensor(self.out_features, self.in_features))
        self.reset_parameters()

        self.disc = Discriminator(self.out_features)

        self.sigm = nn.Sigmoid()


    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight1)
        torch.nn.init.xavier_uniform_(self.weight2)

    def forward(self, feat, feat_a, adj,adj_a):
        z = F.dropout(feat, self.dropout, self.training)
        z = torch.mm(z, self.weight1)

        nb_input = z
        nb_out = torch.mm(nb_input, self.weight2)

        z = torch.mm(adj, z)


        hiden_emb = z

        h = torch.mm(z, self.weight2)
        h = torch.mm(adj, h)

        emb = self.act(z)

        mean_act = MeanAct()
        # mean = torch.mm(z, self.weight2)
        # mean = torch.mm(adj, mean)
        mean = mean_act(nb_out)

        disp_act = DispAct()
        # disp = torch.mm(z, self.weight2)
        # disp = torch.mm(adj, disp)
        disp = disp_act(nb_out)

        z_a = F.dropout(feat_a, self.dropout, self.training)
        z_a = torch.mm(z_a, self.weight1)
        z_a = torch.mm(adj, z_a)
        emb_a = self.act(z_a)

        z_s = F.dropout(feat_a, self.dropout, self.training)
        z_s = torch.mm(z_s, self.weight1)
        z_s = torch.mm(adj_a, z_s)
        emb_s = self.act(z_s)



        ret = self.disc(emb, emb_a)#正和弱负
        ret_a = self.disc(emb_a, emb_s)#强负和弱负



        return hiden_emb, h,ret,ret_a,mean,disp


class Encoder_sparse(Module):
    """
    Sparse version of Encoder
    """

    def __init__(self, in_features, out_features, graph_neigh, dropout=0.0, act=F.relu):
        super(Encoder_sparse, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.graph_neigh = graph_neigh
        self.dropout = dropout
        self.act = act

        self.weight1 = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        self.weight2 = Parameter(torch.FloatTensor(self.out_features, self.in_features))
        self.reset_parameters()

        self.disc = Discriminator(self.out_features)

        self.sigm = nn.Sigmoid()


    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight1)
        torch.nn.init.xavier_uniform_(self.weight2)

    def emb_fusion(self, adj_fusion, z_ae, z_igae,z_sgae):
        # z_ae=torch.spmm(adj, z)
        # z_igae=torch.spmm(adj1, z)
        #self.c=1-self.a-self.b
        z_i = self.A * z_ae + self.B * z_igae + self.C * z_sgae#不同视角的信息的融合
        # 可要可不要
        z_l = torch.spmm(adj_fusion, z_i)  # 局部想惯性
        s = torch.mm(z_l, z_l.t())
        s = F.softmax(s, dim=1)  # 类似于全局相关性,给一个全局相关性的评分
        z_g = torch.mm(s, z_l)  # 根据局部相关和全局相关计算全局表示#这样的全局信息更依赖节点的关系，如果换成z_i依赖的的是（abc）三个参数
        z_tilde = self.alpha * z_g + z_l  # 全局信息加局部信息

        return z_tilde

    def forward(self, feat, feat_a, adj, adj_a):
        z = F.dropout(feat, self.dropout, self.training)
        z = torch.mm(z, self.weight1)

        zinb_input = z
        zinb_out = torch.mm(zinb_input, self.weight2)

        z = torch.spmm(adj, z)

        hiden_emb = z

        hiden= self.emb_fusion(self, adj, z_ae, z_igae,z_sgae)

        h = torch.mm(z, self.weight2)
        h = torch.spmm(adj, h)

        emb = self.act(z)

        mean_act = MeanAct()
        # mean = torch.mm(z, self.weight2)
        # mean = torch.mm(adj, mean)
        mean = mean_act(zinb_out)

        disp_act = DispAct()
        # disp = torch.mm(z, self.weight2)
        # disp = torch.mm(adj, disp)
        disp = disp_act(zinb_out)

        z_a = F.dropout(feat_a, self.dropout, self.training)
        z_a = torch.mm(z_a, self.weight1)
        z_a = torch.spmm(adj, z_a)
        emb_a = self.act(z_a)

        z_s = F.dropout(feat_a, self.dropout, self.training)
        z_s = torch.mm(z_s, self.weight1)
        z_s = torch.spmm(adj_a, z_s)
        emb_s = self.act(z_s)

        ret = self.disc(emb, emb_a)
        ret_a = self.disc(emb_a, emb_s)

        return hiden_emb, h, ret, ret_a, mean, disp

'''
class loss_fusion(nn.Module):
    def __init__(self,in_features, out_features):
        super(loss_fusion, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        
        #stacked_tensor = torch.stack([self.a, self.b, self.c])
        #weights = nn.Parameter(F.softmax(stacked_tensor, dim=0),requires_grad=True)
        #self.A = nn.Parameter(nn.init.constant_(torch.zeros(n_node, opt.args.n_z), self.a), requires_grad=True)
        #self.B = nn.Parameter(nn.init.constant_(torch.zeros(n_node, opt.args.n_z), self.b), requires_grad=True)
        #self.C = nn.Parameter(nn.init.constant_(torch.zeros(n_node, opt.args.n_z), self.c), requires_grad=True)
        self.A = nn.Parameter(torch.FloatTensor(in_features, out_features).uniform_(0, 1).cuda())
        self.B = nn.Parameter(torch.FloatTensor(in_features, out_features).uniform_(0, 1).cuda())
        self.C = nn.Parameter(torch.FloatTensor(in_features, out_features).uniform_(0, 1).cuda())
        # Z_ae, Z_igae
        self.alpha = Parameter(torch.zeros(1),requires_grad=True)# ZG, ZL




    def forward(self, loss1, loss2):
        # z_ae=torch.spmm(adj, z)
        # z_igae=torch.spmm(adj1, z)
        #self.c=1-self.a-self.b
        loss = self.A * loss1 + self.B *loss2#+ self.C * z_sgae#不同视角的信息的融合


'''