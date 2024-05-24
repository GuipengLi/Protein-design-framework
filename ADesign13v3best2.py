import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_sum, scatter_mean
#from torch_scatter import scatter_sum, scatter_mean, scatter_softmax
import numpy as np
import math


########### fix apex bug
from typing import Optional

from torch_scatter import scatter_sum, scatter_max
from torch_scatter.utils import broadcast


def scatter_softmax(src: torch.Tensor, index: torch.Tensor,
                    dim: int = -1,
                    dim_size: Optional[int] = None) -> torch.Tensor:
    if not torch.is_floating_point(src):
        raise ValueError('`scatter_softmax` can only be computed over tensors '
                         'with floating point data types.')

    index = broadcast(index, src, dim)

    max_value_per_index = scatter_max(
        src, index, dim=dim, dim_size=dim_size)[0]
    max_per_src_element = max_value_per_index.gather(dim, index)

    recentered_scores = src - max_per_src_element
    #recentered_scores_exp = recentered_scores.exp_()
    recentered_scores_exp = recentered_scores.exp()

    sum_per_index = scatter_sum(
        recentered_scores_exp, index, dim, dim_size=dim_size)
    normalizing_constants = sum_per_index.gather(dim, index)

    return recentered_scores_exp.div(normalizing_constants)


############################ Graph Encoder ########################
###################################################################

def get_attend_mask(idx, mask):
    mask_attend = gather_nodes(mask.unsqueeze(-1), idx).squeeze(-1) # 一阶邻居节点的mask: 1代表节点存在, 0代表节点不存在
    mask_attend = mask.unsqueeze(-1) * mask_attend # 自身的mask*邻居节点的mask
    return mask_attend

class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''
    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.w_3 = nn.Linear(d_in, d_hid) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        #self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        #x = self.w_2(F.relu(self.w_1(x)))
        x = self.w_2(F.silu(self.w_1(x))* self.w_3(x))
        #x = self.dropout(x)
        x += residual
        x = self.layer_norm(x)
        return x


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear( dim, hidden_dim )
        self.w2 = nn.Linear( hidden_dim, dim )
        self.w3 = nn.Linear( dim, hidden_dim )
        #torch.nn.init.ones_( self.w1.weight)
        #torch.nn.init.ones_( self.w2.weight)
        #torch.nn.init.ones_( self.w3.weight)
        self.layer_norm = nn.LayerNorm(dim, eps=1e-6)


    def forward(self, x):
        x = self.w2(F.silu(self.w1(x)) * self.w3(x))
        x = self.layer_norm(x)
        return x



class NeighborAttentionV2(nn.Module):
    def __init__(self, num_hidden, num_in, num_heads=4, edge_drop=0):
        super(NeighborAttentionV2, self).__init__()
        self.num_heads = num_heads
        self.num_hidden = num_hidden
        self.edge_drop = edge_drop
        
        #self.W_Q = nn.Linear(num_hidden, num_hidden, bias=False)
        #self.W_K = nn.Linear(num_in, num_hidden, bias=False)
        self.W_V = nn.Sequential(
                                nn.Linear(num_in, num_hidden),
                                nn.GELU(),
                                nn.Linear(num_hidden, num_hidden),
                                nn.GELU(),
                                nn.Linear(num_hidden, num_hidden)
                                )
        self.mlp = nn.Sequential(
                                nn.Linear(num_hidden*3, num_hidden*2),
                                nn.GELU(),
                                nn.Linear(num_hidden*2,num_hidden),
                                nn.GELU(),
                                nn.Linear(num_hidden,num_heads)
                                )
        self.W_O = nn.Linear(num_hidden, num_hidden, bias=False)

    #def forward(self, h_V, h_E, center_id, batch_id):
    def forward(self, h_V, h_E, center_id ): # batch_id not used
        N = h_V.shape[0]
        E = h_E.shape[0]
        n_heads = self.num_heads
        d = int(self.num_hidden / n_heads)
        #print( h_V.shape, center_id.shape, h_E.shape) 
        w = self.mlp(torch.cat([h_V[center_id], h_E],dim=-1)).view(E, n_heads, 1) 
        attend_logits = w/np.sqrt(d) 
        attend = scatter_softmax(attend_logits, index=center_id, dim=0)

        V = self.W_V(h_E).view(-1, n_heads, d) 
        #print(attend.shape, V.shape)
        h_V = scatter_sum(attend*V, center_id, dim=0).view([N, self.num_hidden])

        h_V_update = self.W_O(h_V)
        #print(h_V.shape, h_V_update.shape)
        return h_V_update


class GNNModuleV2(nn.Module):
    def __init__(self, num_hidden, num_in, num_heads=4, dropout=0):
        super(GNNModuleV2, self).__init__()
        self.num_heads = num_heads
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.dropout = nn.Dropout(dropout)
        #self.norm = nn.ModuleList([nn.BatchNorm1d(num_hidden) for _ in range(2)])
        self.norm = nn.ModuleList([nn.LayerNorm(num_hidden, eps=1e-6) for _ in range(2)])
        self.attention = NeighborAttentionV2(num_hidden, num_in, num_heads, edge_drop=0) # TODO: edge_drop
        #self.attention2 = NeighborAttentionV2(num_hidden, num_in, num_heads, edge_drop=0.0) # TODO: edge_drop
        self.dense = nn.Sequential(
            nn.Linear(num_hidden, num_hidden*4),
            #nn.BatchNorm1d( num_hidden*3),
            nn.GELU(),
            nn.Linear(num_hidden*4, num_hidden)
        )
        #self.ffn = PositionwiseFeedForward( num_hidden, num_hidden*4) 
        #self.ffn = FeedForward( num_hidden, num_hidden*4,  num_hidden) 

    def forward(self, h_V, h_E, E_idx ):
        center_id = E_idx[0]
        dh = self.attention(h_V, h_E, center_id )
        #dh = self.attention(h_V, h_E, center_id, batch_id)
        h_V = self.norm[0]( h_V +self.dropout( dh))
        dh = self.dense(h_V)
        h_V = self.norm[1]( h_V +self.dropout( dh))
        #h_V = self.ffn(h_V)
        # 3.36 = (2*64)^(1/4)
        #h_E = h_E + self.attention2(h_V, h_E, center_id )
        return h_V
    

class StructureEncoderV2(nn.Module):
    def __init__(self,  hidden_dim, num_encoder_layers=3, dropout=0):
        """ Graph labeling network """
        super(StructureEncoderV2, self).__init__()
        self.encoder_layers = nn.ModuleList([])
        for _ in range(num_encoder_layers):
            self.encoder_layers.append(nn.ModuleList([
                # Local_Module(hidden_dim, hidden_dim*2, is_attention=is_attention, dropout=dropout),
                GNNModuleV2(hidden_dim, hidden_dim*2, dropout=dropout),
                #GNNModuleV2(hidden_dim, hidden_dim*2, dropout=dropout), ##
                nn.LayerNorm( hidden_dim, eps=1e-6),
                #GNNModuleV2(hidden_dim, hidden_dim*2, dropout=dropout),
                nn.Sequential(
                    nn.Linear( 2*hidden_dim, hidden_dim),
                    nn.GELU(),
                    nn.BatchNorm1d(hidden_dim) )
            ]))

    def forward(self, h_V, h_E, E_idx ):
        #print(h_V.shape, h_E.shape, E_idx.shape)
        # graph encoder
        for (layer1, norm,  proj) in self.encoder_layers:
            h_EV_local = torch.cat([h_E, h_V[E_idx[1]]], dim=1)
            h_V = norm( h_V + layer1(h_V, h_EV_local, E_idx ))
            #
            #h_EV_global = torch.cat([h_E, h_V[E_idx[1]]], dim=1)
            #h_V = norm( h_V + layer2(h_V, h_EV_global, E_idx ))
            
            h_EV_local = torch.cat([h_E, h_V[E_idx[1]]], dim=1)
            h_E = h_E + proj( h_EV_local )
 
            
        return h_V



############################# Seq Decoder #########################
###################################################################
def positionalencoding1d(d_model, length):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                         -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe





class Decoder(nn.Module):
    def __init__(self,hidden_dim, input_dim, vocab=21):
        super( Decoder, self).__init__()
        #self.CNN1 = nn.Sequential(nn.Conv1d(input_dim, hidden_dim,5, padding=2),
        #                           nn.BatchNorm1d(hidden_dim),
        #                           nn.GELU(),
        #                           nn.Conv1d(hidden_dim, hidden_dim,5, padding=2))
        #self.CNN2 = nn.Sequential(nn.Conv1d(input_dim, hidden_dim, 7, padding=3),
        #                           nn.BatchNorm1d(hidden_dim),
        #                           nn.GELU(),
        #                           nn.Conv1d(hidden_dim, hidden_dim,7, padding=3))

        #self.linear1 = nn.Linear(hidden_dim*2, 128)
        #self.linear2 = nn.Linear(hidden_dim, 128)
        self.readout = nn.Sequential( 
                          #nn.Linear( hidden_dim, hidden_dim), nn.BatchNorm1d( hidden_dim), nn.GELU(),
                          nn.Linear( hidden_dim, vocab))
        self.fnn = PositionwiseFeedForward( hidden_dim, hidden_dim*4)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, h_V ):
        # pos = self.PosEnc(pos)
        # h_V = torch.cat([h_V,pos],dim=-1)
        #h_V = h_V.unsqueeze(0).permute(0,2,1)
        #print(h_V.shape)  # #residues x 128
        #hidden1 = self.CNN1(h_V).permute(0,2,1).squeeze()
        #hidden2 = self.CNN2(h_V).permute(0,2,1).squeeze()
        #print(hidden.shape)
        #hidden = torch.cat( [ hidden1, hidden2], dim=1)
        #hidden = self.linear1( hidden )
        #print(x1.shape, x2.shape, hidden.shape)
        #hidden = torch.reshape(hidden, ( hidden.shape[0], -1))
        h_V = self.fnn( h_V)
        logits = self.readout( h_V )
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs, logits



class ADesign(nn.Module):
    def __init__(self, node_features, edge_features, hidden_dim, 
        num_encoder_layers=3, num_decoder_layers=3, vocab=21, 
        k_neighbors=30, dropout=0.1, **kwargs):
        """ Graph labeling network """
        super(ADesign, self).__init__()
        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim
        self.top_k = k_neighbors
        self.num_rbf = 10
        #self.num_positional_embeddings = 16
        #node_in, edge_in = 12, 16+7
        #node_in, edge_in = 16+4+8, self.num_rbf* 25 +7
        node_in, edge_in = 12+ 12 + 10, self.num_rbf* 25 +7

        self.W_v = nn.Sequential(
            nn.Linear(node_in, node_features, bias=True),
            nn.BatchNorm1d(node_features),
            nn.GELU(),
            nn.Linear(node_features, hidden_dim, bias=True),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout( p= dropout),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
        )
        
        
        self.W_e = nn.Sequential(
            nn.Linear(edge_in, edge_features, bias=True),
            nn.BatchNorm1d(edge_features),
            nn.GELU(),
            nn.Linear(edge_features, hidden_dim, bias=True), 
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout( p= dropout),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
        )
        

        self.encoder = StructureEncoderV2(hidden_dim, num_encoder_layers, dropout)

        self.decoder = Decoder(hidden_dim, hidden_dim)
        self._init_params()
    
    def forward(self, h_V, h_E, E_idx ):
        #print(h_V.shape, h_E.shape)
        h_V = self.W_v(h_V)
        h_E = self.W_e(h_E)
        #print(h_V.shape, h_E.shape)
        
        #print(h_V.shape)  # #residues x 128
        h_V = self.encoder(h_V, h_E, E_idx )
        #print(h_V.shape)  # #residues x 128
        log_probs0, logits = self.decoder(h_V )
        return log_probs0
        
    def _init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _full_dist(self, X, mask, top_k= 30, eps=1E-6):
        top_k = self.top_k
        mask_2D = torch.unsqueeze(mask,1) * torch.unsqueeze(mask,2)
        dX = torch.unsqueeze(X,1) - torch.unsqueeze(X,2)
        D = (1. - mask_2D)*10000 + mask_2D* torch.sqrt(torch.sum(dX**2, 3) + eps)

        D_max, _ = torch.max(D, -1, keepdim=True)
        D_adjust = D + (1. - mask_2D) * (D_max+1)
        D_neighbors, E_idx = torch.topk(D_adjust, min(top_k, D_adjust.shape[-1]), dim=-1, largest=False)
        return D_neighbors, E_idx  


    def _get_features(self, S, X, mask):
        X = _add_vatom(X)
        mask_bool = (mask==1)
        num_rbf = self.num_rbf
        
        B, N, _,_ = X.shape
        X_n = X[:,:,0,:]
        X_ca= X[:,:,1,:]
        X_c = X[:,:,2,:]
        X_o = X[:,:,3,:]
        X_cb= X[:,:,4,:]
        #X_m= X[:,:,5,:]
        D_neighbors, E_idx = self._full_dist(X_ca, mask, self.top_k) # TODO: change_k
        #print(D_neighbors.shape, E_idx.shape)

        # sequence
        S = torch.masked_select(S, mask_bool)

        # node feature
        _V = _dihedrals(X) 
        #print(_V.shape)
        _V = torch.masked_select(_V, mask_bool.unsqueeze(-1)).reshape(-1,_V.shape[-1])
        #print(_V)
        #print(_V.shape)

        # edge feature
        fa = _orientations_coarse_gl(X, E_idx)
        #print(X.shape, E_idx.shape, fa.shape)
        _E = torch.cat((_rbf(D_neighbors, num_rbf), fa), -1) # [4,387,387,23]
        #print(_E.shape)
        
        # add more RBF features for edges
        RBF_all = []
        #RBF_all.append(_rbf(D_neighbors)) #Ca-Ca
        RBF_all.append(_get_rbf( X_n, X_n, E_idx)) #N-N
        RBF_all.append(_get_rbf(X_n, X_ca, E_idx)) #N-Ca
        RBF_all.append(_get_rbf(X_n, X_c, E_idx)) #N-C
        RBF_all.append(_get_rbf(X_n, X_o, E_idx)) #N-O
        RBF_all.append(_get_rbf(X_n, X_cb, E_idx)) #N-Cb
        RBF_all.append(_get_rbf(X_c, X_n, E_idx)) #C-N
        RBF_all.append(_get_rbf(X_c, X_ca, E_idx)) #C-Ca
        RBF_all.append(_get_rbf(X_c, X_c, E_idx)) #C-C
        RBF_all.append(_get_rbf(X_c, X_o, E_idx)) #C-O
        RBF_all.append(_get_rbf(X_c, X_cb, E_idx)) #C-Cb
        RBF_all.append(_get_rbf(X_ca, X_n, E_idx)) #Ca-N
        RBF_all.append(_get_rbf(X_ca, X_c, E_idx)) #Ca-C
        RBF_all.append(_get_rbf(X_ca, X_o, E_idx)) #Ca-O
        RBF_all.append(_get_rbf(X_ca, X_cb, E_idx)) #Ca-Cb
        RBF_all.append(_get_rbf(X_cb, X_n, E_idx)) #Cb-N
        RBF_all.append(_get_rbf(X_cb, X_ca, E_idx)) #Cb-Ca
        RBF_all.append(_get_rbf(X_cb, X_c, E_idx)) #Cb-C
        RBF_all.append(_get_rbf(X_cb, X_o, E_idx)) #Cb-O
        RBF_all.append(_get_rbf(X_cb, X_cb, E_idx)) #Ca-Cb
        RBF_all.append(_get_rbf(X_o, X_n, E_idx)) #O-N
        RBF_all.append(_get_rbf(X_o, X_ca, E_idx)) #O-Ca
        RBF_all.append(_get_rbf(X_o, X_c, E_idx)) #O-C
        RBF_all.append(_get_rbf(X_o, X_o, E_idx)) #O-O
        RBF_all.append(_get_rbf(X_o, X_cb, E_idx)) #O-Cb
        #RBF_all.append(_get_rbf(X_m, X_n, E_idx))
        #RBF_all.append(_get_rbf(X_m, X_ca, E_idx))
        #RBF_all.append(_get_rbf(X_m, X_c, E_idx)) 
        #RBF_all.append(_get_rbf(X_m, X_o, E_idx)) 
        #RBF_all.append(_get_rbf(X_m, X_cb, E_idx)) 
        #RBF_all.append(_get_rbf(X_m, X_m, E_idx)) 
        #RBF_all.append(_get_rbf(X_n, X_m, E_idx)) 
        #RBF_all.append(_get_rbf(X_c, X_m, E_idx)) 
        #RBF_all.append(_get_rbf(X_ca, X_m, E_idx)) 
        #RBF_all.append(_get_rbf(X_cb, X_m, E_idx)) 
        #RBF_all.append(_get_rbf(X_o, X_m, E_idx)) 
        RBF_all = torch.cat(tuple(RBF_all), dim=-1)
        #print(RBF_all.shape)
         
        _E = torch.cat(( _E, RBF_all), -1) 

        mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1) # 一阶邻居节点的mask: 1代表节点存在, 0代表节点不存在
        mask_attend = (mask.unsqueeze(-1) * mask_attend) == 1 # 自身的mask*邻居节点的mask
        _E = torch.masked_select(_E, mask_attend.unsqueeze(-1)).reshape(-1,_E.shape[-1])
        #print(_E.shape)
        
        # edge index
        shift = mask.sum(dim=1).cumsum(dim=0) - mask.sum(dim=1)
        src = shift.view(B,1,1) + E_idx
        src = torch.masked_select(src, mask_attend).view(1,-1)
        dst = shift.view(B,1,1) + torch.arange(0, N, device=src.device).view(1,-1,1).expand_as(mask_attend)
        dst = torch.masked_select(dst, mask_attend).view(1,-1)
        E_idx = torch.cat((dst, src), dim=0).long()
        
        # 3D point
        sparse_idx = mask.nonzero()
        X = X[sparse_idx[:,0],sparse_idx[:,1],:,:]
        batch_id = sparse_idx[:,0]

        #print(X.shape, S.shape, _V.shape, _E.shape, E_idx.shape, batch_id.shape)
        return X, S, _V, _E, E_idx, batch_id


def _add_vatom(X):
    B, N, k, _ = X.shape
    b = X[:,:,1,:] - X[:,:,0,:]
    c = X[:,:,2,:] - X[:,:,1,:]
    a = torch.cross(b, c, dim=-1)
    X_cb = -0.58273431*a + 0.56802827*b - 0.54067466*c + X[:,:,1,:]
    X_cb = X_cb.unsqueeze(2)
    #center = torch.mean( X, dim=2)
    #center = center.unsqueeze(2)
    #print(X.shape, X_cb.shape, center.shape)
    #X_new = torch.cat(( X, X_cb, center), 2) 
    #X_new = torch.cat(( X, center), 2) 
    X_new = torch.cat(( X, X_cb), 2) 
    return(X_new)

# Thanks for StructTrans
# https://github.com/jingraham/neurips19-graph-protein-design
def nan_to_num(tensor, nan=0.0):
    idx = torch.isnan(tensor)
    tensor[idx] = nan
    return tensor

def _normalize(tensor, dim=-1):
    return nan_to_num(
        torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))

def cal_dihedral(X, eps=1e-7):
    dX = X[:,1:,:] - X[:,:-1,:] # CA-N, C-CA, N-C, CA-N...
    U = _normalize(dX, dim=-1)
    u_0 = U[:,:-2,:] # CA-N, C-CA, N-C,...
    u_1 = U[:,1:-1,:] # C-CA, N-C, CA-N, ... 0, psi_{i}, omega_{i}, phi_{i+1} or 0, tau_{i},...
    u_2 = U[:,2:,:] # N-C, CA-N, C-CA, ...

    n_0 = _normalize(torch.cross(u_0, u_1), dim=-1)
    n_1 = _normalize(torch.cross(u_1, u_2), dim=-1)
    
    cosD = (n_0 * n_1).sum(-1)
    cosD = torch.clamp(cosD, -1+eps, 1-eps)
    
    v = _normalize(torch.cross(n_0, n_1), dim=-1)
    D = torch.sign((-v* u_1).sum(-1)) * torch.acos(cosD) # TODO: sign
    
    # D = torch.sign((u_0 * n_1).sum(-1)) * torch.acos(cosD)
    return D

def cal_nodedist(X): # X is BxNx5x3, return BxNx10
    B, N, K, _ = X.shape
    dist = []
    for i in range(K-1):
        for j in range(i+1, K):
            d = (X[:,:,i,:] - X[:,:,j,:]).pow(2).sum(2).sqrt()
            #print(d.shape)
            dist.append(d)
    dist = torch.stack( dist, 2)
    #print(dist.shape)
    return dist

def _dihedrals(X, eps=1e-7):
    B, N, _, _ = X.shape
    dist = cal_nodedist(X)
    # psi, omega, phi
    #X = X[:,:,:3,:].reshape(X.shape[0], 3*X.shape[1], 3) # ['N', 'CA', 'C', 'O']
    X = X.reshape(X.shape[0], -1, 3) # ['N', 'CA', 'C', 'O', 'CB']
    #print(X.shape)

    D = cal_dihedral(X)
    #print(D.shape)
    #print(D.shape, X.shape)
    D = F.pad(D, (1,2), 'constant', 0)
    #print(D.shape)
    D = D.view((D.size(0), N, -1 )) 
    #print(D.shape)
    
    # # tau
    CA = X[:,1::5,:]  # need to modify
    tau = cal_dihedral(CA)
    #print(D.shape, tau.shape)
    tau = F.pad(tau, (1,2), 'constant', 0).reshape(B, -1,1)
    #print(D.shape, tau.shape)
    D = torch.cat([D,tau], dim=2) # psi, omega, phi, x, y, tau

    Dihedral_Angle_features = torch.cat((torch.cos(D), torch.sin(D)), 2)
    
    
    # alpha, beta, gamma
    dX = X[:,1:,:] - X[:,:-1,:] # CA-N, C-CA, N-C, CA-N...
    U = _normalize(dX, dim=-1)
    u_0 = U[:,:-2,:] # CA-N, C-CA, N-C,...
    u_1 = U[:,1:-1,:] # C-CA, N-C, CA-N, ...
    
    cosD = (u_0*u_1).sum(-1) # alpha_{i}, gamma_{i}, beta_{i+1}
    cosD = torch.clamp(cosD, -1+eps, 1-eps)
    D = torch.acos(cosD)
    D = F.pad(D, (1,2), 'constant', 0)
    #print(D.shape)
    D = D.view((D.size(0), N, -1))
    
    # # theta
    dX = CA[:,1:,:] - CA[:,:-1,:]
    U = _normalize(dX, dim=-1)
    u_0 = U[:,:-1,:]
    u_1 = U[:,:-1,:]
    cosD = (u_0*u_1).sum(-1)
    cosD = torch.clamp(cosD, -1+eps, 1-eps)
    theta = torch.acos(cosD)
    theta = F.pad(theta, (1,1), 'constant', 0).reshape(B, -1,1)
    D = torch.cat([D,theta], dim=2)
    
    Angle_features = torch.cat((torch.cos(D), torch.sin(D)), 2)
    #print(Angle_features.shape)
    
    
    # n_0 = n_0.reshape(B,-1,3,3)[:,:,::3,:]
    # u_0 = u_0.reshape(B,-1,3,3)[:,:,::3,:]
    # u_1 = u_1.reshape(B,-1,3,3)[:,:,::3,:]
    # D_features = torch.cat((u_0, u_1, n_0), 2).reshape(B,-1,9)
    # D_features2 = F.pad(D_features, (0,0,0,1,0,0), 'constant', 0)
   
    D_features = torch.cat((dist, Dihedral_Angle_features, Angle_features), 2)
    #print(D_features.shape)

    return D_features

def _hbonds(X, E_idx, mask_neighbors, eps=1E-3):
    X_atoms = dict(zip(['N', 'CA', 'C', 'O'], torch.unbind(X, 2)))

    X_atoms['C_prev'] = F.pad(X_atoms['C'][:,1:,:], (0,0,0,1), 'constant', 0)
    X_atoms['H'] = X_atoms['N'] + _normalize(
            _normalize(X_atoms['N'] - X_atoms['C_prev'], -1)
        +  _normalize(X_atoms['N'] - X_atoms['CA'], -1)
    , -1)

    def _distance(X_a, X_b):
        return torch.norm(X_a[:,None,:,:] - X_b[:,:,None,:], dim=-1)

    def _inv_distance(X_a, X_b):
        return 1. / (_distance(X_a, X_b) + eps)

    U = (0.084 * 332) * (
            _inv_distance(X_atoms['O'], X_atoms['N'])
        + _inv_distance(X_atoms['C'], X_atoms['H'])
        - _inv_distance(X_atoms['O'], X_atoms['H'])
        - _inv_distance(X_atoms['C'], X_atoms['N'])
    )

    HB = (U < -0.5).type(torch.float32)
    neighbor_HB = mask_neighbors * gather_edges(HB.unsqueeze(-1),  E_idx)
    return neighbor_HB

def _rbf(D, num_rbf=10):
    D_min, D_max, D_count = 0., 20., num_rbf
    D_mu = torch.linspace(D_min, D_max, D_count).to(D.device)
    D_mu = D_mu.view([1,1,1,-1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)
    RBF = torch.exp(-((D_expand - D_mu) / D_sigma)**2)
    return RBF


def _get_rbf( A, B, E_idx):
    D_A_B = torch.sqrt(torch.sum((A[:,:,None,:] - B[:,None,:,:])**2,-1) + 1e-6) #[B, L, L]
    D_A_B_neighbors = gather_edges(D_A_B[:,:,:,None], E_idx)[:,:,:,0] #[B,L,K]
    RBF_A_B = _rbf(D_A_B_neighbors)
    return RBF_A_B

def _orientations_coarse(X, E_idx, eps=1e-6):
    dX = X[:,1:,:] - X[:,:-1,:]
    U = _normalize(dX, dim=-1)
    u_2 = U[:,:-2,:]
    u_1 = U[:,1:-1,:]
    u_0 = U[:,2:,:]

    n_2 = _normalize(torch.cross(u_2, u_1), dim=-1)
    n_1 = _normalize(torch.cross(u_1, u_0), dim=-1)

    cosA = -(u_1 * u_0).sum(-1)
    cosA = torch.clamp(cosA, -1+eps, 1-eps)
    A = torch.acos(cosA)

    cosD = (n_2 * n_1).sum(-1)
    cosD = torch.clamp(cosD, -1+eps, 1-eps)
    D = torch.sign((u_2 * n_1).sum(-1)) * torch.acos(cosD)

    AD_features = torch.stack((torch.cos(A), torch.sin(A) * torch.cos(D), torch.sin(A) * torch.sin(D)), 2)
    AD_features = F.pad(AD_features, (0,0,1,2), 'constant', 0)

    o_1 = _normalize(u_2 - u_1, dim=-1)
    O = torch.stack((o_1, n_2, torch.cross(o_1, n_2)), 2)
    O = O.view(list(O.shape[:2]) + [9])
    O = F.pad(O, (0,0,1,2), 'constant', 0)

    O_neighbors = gather_nodes(O, E_idx)
    X_neighbors = gather_nodes(X, E_idx)
    
    O = O.view(list(O.shape[:2]) + [3,3])
    O_neighbors = O_neighbors.view(list(O_neighbors.shape[:3]) + [3,3])

    dX = X_neighbors - X.unsqueeze(-2)
    dU = torch.matmul(O.unsqueeze(2), dX.unsqueeze(-1)).squeeze(-1)
    dU = _normalize(dU, dim=-1)
    R = torch.matmul(O.unsqueeze(2).transpose(-1,-2), O_neighbors)
    Q = _quaternions(R)

    O_features = torch.cat((dU,Q), dim=-1)
    return AD_features, O_features

def _orientations_coarse_gl(X, E_idx, eps=1e-6):
    X = X[:,:,:3,:].reshape(X.shape[0], 3*X.shape[1], 3) 
    dX = X[:,1:,:] - X[:,:-1,:] # CA-N, C-CA, N-C, CA-N...
    U = _normalize(dX, dim=-1)
    u_0, u_1 = U[:,:-2,:], U[:,1:-1,:]
    n_0 = _normalize(torch.cross(u_0, u_1), dim=-1)
    b_1 = _normalize(u_0 - u_1, dim=-1)
    
    n_0 = n_0[:,::3,:]
    b_1 = b_1[:,::3,:]
    X = X[:,::3,:]

    O = torch.stack((b_1, n_0, torch.cross(b_1, n_0)), 2)
    O = O.view(list(O.shape[:2]) + [9])
    O = F.pad(O, (0,0,0,1), 'constant', 0) # [16, 464, 9]

    O_neighbors = gather_nodes(O, E_idx) # [16, 464, 30, 9]
    X_neighbors = gather_nodes(X, E_idx) # [16, 464, 30, 3]

    O = O.view(list(O.shape[:2]) + [3,3]).unsqueeze(2) # [16, 464, 1, 3, 3]
    O_neighbors = O_neighbors.view(list(O_neighbors.shape[:3]) + [3,3]) # [16, 464, 30, 3, 3]

    dX = X_neighbors - X.unsqueeze(-2) # [16, 464, 30, 3]
    dU = torch.matmul(O, dX.unsqueeze(-1)).squeeze(-1) # [16, 464, 30, 3] 邻居的相对坐标
    R = torch.matmul(O.transpose(-1,-2), O_neighbors)
    feat = torch.cat((_normalize(dU, dim=-1), _quaternions(R)), dim=-1) # 相对方向向量+旋转四元数
    return feat


def _contacts(D_neighbors, mask_neighbors, cutoff=8):
    D_neighbors = D_neighbors.unsqueeze(-1)
    return mask_neighbors * (D_neighbors < cutoff).type(torch.float32)

def _dist(X, mask, top_k=30, eps=1E-6):
    mask_2D = torch.unsqueeze(mask,1) * torch.unsqueeze(mask,2)
    dX = torch.unsqueeze(X,1) - torch.unsqueeze(X,2)
    D = mask_2D * torch.sqrt(torch.sum(dX**2, 3) + eps)

    D_max, _ = torch.max(D, -1, keepdim=True)
    D_adjust = D + (1. - mask_2D) * D_max
    D_neighbors, E_idx = torch.topk(D_adjust, min(top_k, D_adjust.shape[-1]), dim=-1, largest=False)
    mask_neighbors = gather_edges(mask_2D.unsqueeze(-1), E_idx)
    return D_neighbors, E_idx, mask_neighbors    

def gather_edges(edges, neighbor_idx):
    neighbors = neighbor_idx.unsqueeze(-1).expand(-1, -1, -1, edges.size(-1))
    return torch.gather(edges, 2, neighbors)

def gather_nodes(nodes, neighbor_idx):
    neighbors_flat = neighbor_idx.view((neighbor_idx.shape[0], -1)) # [4, 317, 30]-->[4, 9510]
    neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1, -1, nodes.size(2)) # [4, 9510, dim]
    neighbor_features = torch.gather(nodes, 1, neighbors_flat) # [4, 9510, dim]
    return neighbor_features.view(list(neighbor_idx.shape)[:3] + [-1]) # [4, 317, 30, 128]

def gather_nodes_t(nodes, neighbor_idx):
    idx_flat = neighbor_idx.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    return torch.gather(nodes, 1, idx_flat)

def cat_neighbors_nodes(h_nodes, h_neighbors, E_idx):
    h_nodes = gather_nodes(h_nodes, E_idx)
    return torch.cat([h_neighbors, h_nodes], -1)

def _quaternions(R):
    diag = torch.diagonal(R, dim1=-2, dim2=-1)
    Rxx, Ryy, Rzz = diag.unbind(-1)
    magnitudes = 0.5 * torch.sqrt(torch.abs(1 + torch.stack([
            Rxx - Ryy - Rzz, 
        - Rxx + Ryy - Rzz, 
        - Rxx - Ryy + Rzz
    ], -1)))
    _R = lambda i,j: R[:,:,:,i,j]
    signs = torch.sign(torch.stack([
        _R(2,1) - _R(1,2),
        _R(0,2) - _R(2,0),
        _R(1,0) - _R(0,1)
    ], -1))
    xyz = signs * magnitudes
    w = torch.sqrt(F.relu(1 + diag.sum(-1, keepdim=True))) / 2.
    Q = torch.cat((xyz, w), -1)
    return _normalize(Q, dim=-1)
