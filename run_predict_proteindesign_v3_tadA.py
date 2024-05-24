#!/usr/bin/env python
# coding: utf-8

import argparse
from time import gmtime, strftime

from Bio import SeqIO

parser = argparse.ArgumentParser(description='Fixed backbone protein sequence design, given a pdb file as input to design aa sequence to fold as the pdb structure.')
parser.add_argument('-i','--input',  default='test_demo.txt', required=True,help='input file is a pdb file or  a list of pdb files, each line is one pdb file name')
parser.add_argument('-o','--outdir',  default='test_out', required=True, help='output dir')

args = parser.parse_args()
print(args.input)


import os
os.makedirs( args.outdir, exist_ok=True)


import json
import sys
import time
import pandas as pd
import numpy as np
import random

#from sklearn.model_selection import train_test_split
#from torch.utils.data import Dataset, DataLoader,TensorDataset,random_split,SubsetRandomSampler, ConcatDataset
from torch import  nn, optim
from torch.optim import lr_scheduler
import torch

from tqdm import tqdm
import _pickle as cPickle




#from sklearn.metrics import roc_curve, auc

##################
from ADesign13v3best2 import ADesign


########
from collections.abc import Mapping, Sequence
def cuda(obj, *args, **kwargs):
    """
    Transfer any nested conatiner of tensors to CUDA.
    """
    if hasattr(obj, "cuda"):
        return obj.cuda(*args, **kwargs)
    elif isinstance(obj, Mapping):
        return type(obj)({k: cuda(v, *args, **kwargs) for k, v in obj.items()})
    elif isinstance(obj, Sequence):
        return type(obj)(cuda(x, *args, **kwargs) for x in obj)
    elif isinstance(obj, np.ndarray):
        return torch.tensor(obj, *args, **kwargs)

    raise TypeError("Can't transfer object type `%s`" % type(obj))


###################################################################################
################################### NIPS19 ########################################

class DataLoader_NIPS19(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None, num_workers=0,
                 collate_fn=None, **kwargs):
        super(DataLoader_NIPS19, self).__init__(dataset, batch_size, shuffle, sampler, batch_sampler, num_workers, collate_fn,**kwargs)

def featurize_NIPS19(batch, shuffle_fraction=0.):
    """ Pack and pad batch into torch tensors """
    alphabet = 'ACDEFGHIKLMNPQRSTVWY'
    B = len(batch)
    lengths = np.array([len(b['seq']) for b in batch], dtype=np.int32)
    L_max = max([len(b['seq']) for b in batch])
    X = np.zeros([B, L_max, 4, 3])
    S = np.zeros([B, L_max], dtype=np.int32)

    def shuffle_subset(n, p):
        n_shuffle = np.random.binomial(n, p)
        ix = np.arange(n)
        ix_subset = np.random.choice(ix, size=n_shuffle, replace=False)
        ix_subset_shuffled = np.copy(ix_subset)
        np.random.shuffle(ix_subset_shuffled)
        ix[ix_subset] = ix_subset_shuffled
        return ix

    # Build the batch
    for i, b in enumerate(batch):
        x = np.stack([b[c] for c in ['N', 'CA', 'C', 'O']], 1) # [#atom, 4, 3]
        ## Replacing NaNs with interpolation of columns
        #print(x.shape)
        #x = interpolate_nans(x) 
        l = len(b['seq'])
        x_pad = np.pad(x, [[0,L_max-l], [0,0], [0,0]], 'constant', constant_values=(np.nan, )) # [#atom, 4, 3]
        X[i,:,:,:] = x_pad

        # Convert to labels
        indices = np.asarray([alphabet.index(a) for a in b['seq']], dtype=np.int32)
        if shuffle_fraction > 0.:
            idx_shuffle = shuffle_subset(l, shuffle_fraction)
            S[i, :l] = indices[idx_shuffle]
        else:
            S[i, :l] = indices

    mask = np.isfinite(np.sum(X,(2,3))).astype(np.float32) # atom mask
    numbers = np.sum(mask, axis=1).astype(np.int32)
    S_new = np.zeros_like(S)
    X_new = np.zeros_like(X)+np.nan
    for i, n in enumerate(numbers):
        X_new[i,:n,::] = X[i][mask[i]==1]
        S_new[i,:n] = S[i][mask[i]==1]

    X = X_new
    S = S_new
    isnan = np.isnan(X)
    mask = np.isfinite(np.sum(X,(2,3))).astype(np.float32)
    X[isnan] = 0.
    # Conversion
    S = torch.from_numpy(S).to(dtype=torch.long)
    X = torch.from_numpy(X).to(dtype=torch.float32)
    mask = torch.from_numpy(mask).to(dtype=torch.float32)
    return X, S, mask, lengths

################################### NIPS19 ########################################



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)

prename = 'CATH'

import os
import numpy as np
from biotite.structure.io.pdb import PDBFile


def load_pdb( pdb_file ):
    DICT_3_1 = { 'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
             'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N',
             'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W',
             'ALA': 'A', 'VAL': 'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}

    data = []
    y = {}

    y['title'] = os.path.basename(pdb_file).split('.')[0]
    source = PDBFile.read(pdb_file)
    struct = source.get_structure()[0]
    # use the largest chain
    chains = set(struct.chain_id)
    chain_id_max = 'A'
    for c in chains:
        if np.sum(struct.chain_id==c) > np.sum(struct.chain_id== chain_id_max):
            chain_id_max = c
    
    struct = struct[ (struct.chain_id == chain_id_max) & (struct.hetero==False) ]
    
    # valid residues with 4 atoms
    caa = np.intersect1d(struct[struct.atom_name=='N'].res_id , struct[struct.atom_name=='CA'].res_id)
    caa = np.intersect1d( caa, struct[struct.atom_name=='C'].res_id)
    caa = np.intersect1d( caa, struct[struct.atom_name=='O'].res_id)
    struct = struct[ np.isin(struct.res_id, caa)]

    seq = [ DICT_3_1[x] if x in DICT_3_1 else 'X' for x in struct.res_name[struct.atom_name=='CA'] ]
    seq = ''.join(seq)
    print('chain: ', chain_id_max, seq)

    y['seq'] = seq
    y['N'] = struct.coord[struct.atom_name=='N']
    y['CA'] = struct.coord[struct.atom_name=='CA']
    y['C'] = struct.coord[struct.atom_name=='C']
    y['O'] = struct.coord[struct.atom_name=='O']
    #y['score'] = np.zeros( len(seq) ,) + 100.0
    #print(y['N'].shape, y['CA'].shape,y['C'].shape,y['O'].shape)

    data.append(y)
    return data
    


def load_TS50():
    dataf = json.load(open('cath/ts50.json','r'))
    data = []
    i = 0
    for x in dataf:
        y = {}
        y['title'] = x['name']
        y['seq'] = x['seq']
        y['N'] = np.array( [z[0] for z in x['coords']])
        y['CA'] = np.array( [z[1] for z in x['coords']])
        y['C'] = np.array([ z[2] for z in x['coords']])
        y['O'] = np.array([ z[3] for z in x['coords']])
        #y['score'] = np.zeros( len(x['seq']) ,) + 100.0
        data.append(y)
    return(data)

def load_TS500():
    with open('cath/TS500.jsonl','r') as f:
        data = []
        for line in f:
            y = {}
            i = 0
            x = json.loads(line)
            y = {}
            y['title'] = x['name']
            y['seq'] = x['seq']
            y['N'] = np.array( x['coords_chain_A']['N_chain_A'])
            y['CA'] = np.array( x['coords_chain_A']['CA_chain_A'])
            y['C'] = np.array( x['coords_chain_A']['C_chain_A'])
            y['O'] = np.array( x['coords_chain_A']['O_chain_A'])
            #y['score'] = np.zeros( len(x['seq']) ,) + 100.0
            data.append(y)
        return(data)


def load_data():
    max_length = 500
    limit_length = True
    split = json.load(open('data/preprocessed/%s/split.json'%prename,'r'))  # splitF
    data_ = cPickle.load(open('data/preprocessed/%s/data_%s.pkl'%(prename, prename), 'rb'))

    data = []
    if prename.startswith('CATH'):
        #for i in range(len(data_)):
        #    data_[i]['score'] = np.zeros( len(data_[i]['seq']) ,) + 100.0
        data = data_
    else:
        #score_ = cPickle.load(open('data/preprocessed/%s/data_%s_score.pkl'%(prename, prename),'rb'))
        #for i in range(len(data_)):
        #    data_[i]['score'] = score_[i]['res_score']
        for temp in data_:
            if limit_length:
                if 30<len(temp['seq']) and len(temp['seq']) < max_length:
                    # 'title', 'seq', 'CA', 'C', 'O', 'N'
                    data.append(temp)
            else:
                data.append(temp)

    data_dict = {'train':[ data[i] for i in split['train'] ],
                     'valid':[ data[i] for i in split['valid'] ],
                     'test':[ data[i] for i in split['test'] ]}
    return data_dict



def loss_nll_flatten(S, log_probs):
    """ Negative log probabilities """
    criterion = torch.nn.NLLLoss(reduction='none')
    loss = criterion(log_probs, S)
    loss_av = loss.mean()
    return loss, loss_av


def weight_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(m.bias)

def predict_model( model, test_loader, name, outdir):
    alphabet = 'ACDEFGHIKLMNPQRSTVWY'
    counter = 0
    current_test_loss = 0
    test_weights = 0
    recovery = []
    model.eval()
    with torch.no_grad():
        for batch   in test_loader:
            X, S, mask, lengths = cuda(batch, device=device)
            #X = add_vatom(X)
            X, S, h_V, h_E, E_idx, batch_id = model._get_features(S, X=X, mask=mask)
            log_probs = model( h_V, h_E, E_idx )
            
            loss, loss_av = loss_nll_flatten(S, log_probs)
            mask = torch.ones_like(loss)
            # Accumulate
            current_test_loss += torch.sum(loss * mask).cpu().data.numpy()
            test_weights += torch.sum(mask).cpu().data.numpy()

            S_pred = torch.argmax(log_probs, dim=1)
            
            print( 'label  :', ''.join([ alphabet[x] for x in S]) )
            print( 'predict:', ''.join([ alphabet[x] for x in S_pred]) )
            #np.save('%s_predict_probs.npy'%name, log_probs.cpu().data.numpy())
            df = pd.DataFrame( nn.Softmax(dim=1)(log_probs).cpu().data.numpy() )
            #print(dict(zip( [ str(x) for x in range(len(alphabet) )] , [x for x in alphabet] )))
            df = df.rename(columns= dict(zip( [ x for x in range(len(alphabet) )] , [x for x in alphabet] )) )
            df['label'] = [ alphabet[x] for x in S]
            df['predict'] = [ alphabet[x] for x in S_pred]
            df.to_csv("%s/%s_predict_probs.csv"%(outdir,name) )
            #write fasta
            with open('%s/%s_predict_fasta.fa'%(outdir,name), 'w') as fout:
                 fout.write( '>%s_label\n'%name +''.join(df.label.values)+'\n')
                 fout.write( '>%s_design\n'%name +''.join(df.predict.values)+'\n')

            recovery_ = (S_pred == S).float().mean().cpu().numpy()
            recovery.append(recovery_)
    current_test_loss = current_test_loss / test_weights
    #test_perplexity = np.exp(current_test_loss)
    recovery = np.median(recovery)    
    print('recovery: ',  recovery)



def run_model( args ):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #print(device)

    model = ADesign(
        node_features= 128,
        edge_features= 128,
        hidden_dim= 128,
        dropout= 0.1,
        k_neighbors= 30,
        num_encoder_layers=10,
        num_decoder_layers=1
    )
    
    model.to(device)
    num_params = sum(param.numel() for param in model.parameters())
    #print('Number of parameters: %d'%(num_params) )

    model.load_state_dict( torch.load( '/share/home/liguipeng/3d21d/AlphaDesign/models/CATH_TadA_16_Wed May 22 14:07:11 2024.pth') )
    loss_fn = torch.nn.CrossEntropyLoss()
    targetlist = []
    if args.input.endswith('.pdb'):
        targetlist = [ args.input.strip()]
    else:
        with open( args.input, 'r') as fin:
            for line in fin:
                if line.strip():    
                    targetlist.append( line.strip())

    #pdb = load_pdb('test_targets/6wvs.chainA.pdb')
    for path in targetlist:
        name = os.path.basename(path)
        print( name)
        pdb = load_pdb( path)
        pdb_loader = DataLoader_NIPS19( pdb,  batch_size= 1, shuffle=False,  collate_fn= featurize_NIPS19)
        predict_model( model, pdb_loader, os.path.splitext(name)[0], args.outdir )

run_model( args )
