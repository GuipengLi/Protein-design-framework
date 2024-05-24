#!/usr/bin/env python
# coding: utf-8

#from apex import amp

from sklearn.model_selection import train_test_split

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
from tensorboardX import SummaryWriter



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
    #alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
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
    return X, S,  mask, lengths

################################### NIPS19 ########################################



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)


prename = 'CATH'


def load_CATHtest():
    with open('data/cath/chain_set_splits.json','r') as f:
        test_split = json.load(f)
    alphabet='ACDEFGHIKLMNPQRSTVWY'
    alphabet_set = set([a for a in alphabet])
    max_length = 500
    with open('data/cath/chain_set.jsonl') as f:
        lines = f.readlines()
        data_list = []
        for line in lines:
            entry = json.loads(line)
            seq = entry['seq']

            for key, val in entry['coords'].items():
                entry['coords'][key] = np.asarray(val)

            bad_chars = set([s for s in seq]).difference(alphabet_set)

            if len(bad_chars) == 0:
                if len(entry['seq']) <= max_length:
                    data_list.append({
                    'title':entry['name'],
                    'seq':entry['seq'],
                    'CA':entry['coords']['CA'],
                    'C':entry['coords']['C'],
                    'O':entry['coords']['O'],
                    'N':entry['coords']['N']
                    })


    test_full_list = []
    for data in data_list:
        if data['title'] in test_split['test']:
            test_full_list.append(data)
    return test_full_list




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
        data.append(y)
    return(data)


def load_TS500():
    dataf = json.load(open('data/ts/ts500.json','r'))
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
        data.append(y)
    return(data)



def load_TS530():
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
            data.append(y)
        return(data)


import os
import numpy as np
from biotite.structure.io.pdb import PDBFile
from pathlib import Path


def load_pdbs( pdb_dir ):
    DICT_3_1 = { 'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
             'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N',
             'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W',
             'ALA': 'A', 'VAL': 'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}

    data = []
    pdb_files = Path(pdb_dir).glob('*.pdb') 
    #print('pdb files: %d'%len([x for x in pdb_files]))
    tmpfile = pdb_dir+'_data.pkl'

    if not os.path.isfile( tmpfile) or os.path.getsize(tmpfile)==0: 
        for pdb_file in pdb_files:
            y = {}
            y['title'] = os.path.basename(pdb_file).split('.')[0]
            #print( pdb_file, y['title'])
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
            #print('chain: ', chain_id_max, seq)

            y['seq'] = seq
            y['N'] = struct.coord[struct.atom_name=='N']
            y['CA'] = struct.coord[struct.atom_name=='CA']
            y['C'] = struct.coord[struct.atom_name=='C']
            y['O'] = struct.coord[struct.atom_name=='O']
            #y['score'] = np.zeros( len(seq) ,) + 100.0
            #print(y['N'].shape, y['CA'].shape,y['C'].shape,y['O'].shape)
            data.append(y)

        with open(tmpfile, 'wb') as f:
            print ('saving cached data...')
            cPickle.dump( data, f)
    else:
        with open(tmpfile, 'rb') as f:
            print ('loading cached data...')
            data = cPickle.load(f)
        
    return data



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


def interpolate_nans_2D(X):
    """Overwrite NaNs with column value interpolations."""
    for j in range(X.shape[1]):
        mask_j = np.isnan(X[:,j])
        X[mask_j,j] = np.interp(np.flatnonzero(mask_j), np.flatnonzero(~mask_j), X[~mask_j,j])
    return X

def interpolate_nans(X):
    """Overwrite NaNs with column value interpolations."""
    for j in range(X.shape[1]):
        #mask_j = np.isnan(X[:,j,:])
        X[:,j,:] = interpolate_nans_2D( X[:,j,:] )
    return X


def loss_nll_flatten(S, log_probs):
    """ Negative log probabilities """
    criterion = torch.nn.NLLLoss(reduction='none')
    loss = criterion(log_probs, S)
    loss_av = loss.mean()
    return loss, loss_av



class SaveBestModel2:
    """
    Class to save the best model while training. If the current epoch's 
    validation loss is less than the previous least less, then save the
    model state.
    """
    def __init__( self, outname='best_model.pth', delta = 0, patience = 20): 
        self.best_loss = np.Inf
        self.outname = outname
        self.trigger_times = 0
        self.patience = patience
        self.last_loss = None
        self.early_stop = False
        self.delta = delta
        
    def __call__( self, loss, model):
        if self.last_loss is None:
            self.last_loss = loss
            torch.save(model.state_dict(), self.outname)
            self.best_loss = loss
        elif loss < self.last_loss - self.delta:
            self.last_loss = loss
            torch.save(model.state_dict(), self.outname)
            self.best_loss = loss
            self.trigger_times = 0 
        else:
            self.trigger_times += 1
            if self.trigger_times >= self.patience:
                print('Early stop with patience %d'% self.trigger_times )
                self.early_stop = True


class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's 
    validation loss is less than the previous least less, then save the
    model state.
    """
    def __init__( self, outname='best_model.pth', delta=0, patience=10):  
        self.best_loss = np.Inf
        self.outname = outname
        self.trigger_times = 0
        self.patience = patience
        self.last_loss = 1e10
        self.early_stop = False

    def __call__( self, current_loss,model):  
        if current_loss <= self.best_loss:
            self.best_loss = current_loss
            self.last_loss = current_loss
            self.trigger_times = 0
            #torch.save(model, self.outname)
            torch.save(model.state_dict(), self.outname)
        if current_loss > self.last_loss:
            self.trigger_times += 1
            self.last_loss = current_loss
            if self.trigger_times >= self.patience:
                print('Early stop with patience %d'% self.trigger_times )
                self.early_stop = True
        else:
            self.trigger_times = 0
            self.last_loss = current_loss



def weight_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(m.bias)


def test_model( model, test_loader):
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
            
            loss = loss_fn( log_probs, S)
            #mask = torch.ones_like(loss)
            # Accumulate
            current_test_loss += torch.sum(loss).cpu().data.numpy()
            #test_weights += torch.sum(mask).cpu().data.numpy()
            counter += 1

            S_pred = torch.argmax(log_probs, dim=1)
            #print( S_pred, S)
            recovery_ = (S_pred == S).float().mean().cpu().numpy()
            recovery.append(recovery_)
    current_test_loss = current_test_loss / counter
    test_perplexity = np.exp(current_test_loss)
    #print( recovery )
    m_recovery = np.median(recovery)    
    worst_recovery = np.min(recovery) 
    print('test: ', test_perplexity, m_recovery, worst_recovery)



#loss_fn = torch.nn.CrossEntropyLoss()
loss_fn = torch.nn.NLLLoss()

#@profile
def train_model( prename):
    learning_rate = 1e-6  #cath
    #learning_rate = 1e-4   #human
    #learning_rate = 1e-3   #cath4ts
    num_epochs = 5
    batch_size = 8
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device, learning_rate, batch_size)

    writer = SummaryWriter('runs/%s_k5x'%prename)
    data_dict = load_data()

    ts50 = load_TS50()
    ts500 = load_TS500()
    cathtest = load_CATHtest()  #same order as PiFold
    tadas = load_pdbs('BE/esmfold/8e2p_top4791')
 
    tadatrain, tadatest = train_test_split( tadas, test_size = 0.1, shuffle=True, random_state=6)
    
    ##auc
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    train_loader = DataLoader_NIPS19( data_dict['train'] + tadatrain , batch_size=batch_size, shuffle=True, num_workers=4, collate_fn= featurize_NIPS19)
    valid_loader = DataLoader_NIPS19( data_dict['valid'], batch_size= 1, shuffle=False,  collate_fn= featurize_NIPS19)
    #test_loader  = DataLoader_NIPS19( data_dict['test'],  batch_size= 1, shuffle=False,  collate_fn= featurize_NIPS19)
    test_loader  = DataLoader_NIPS19( cathtest,  batch_size= 1, shuffle=False,  collate_fn= featurize_NIPS19)

    ts50_loader = DataLoader_NIPS19( ts50,  batch_size= 1, shuffle=False,  collate_fn= featurize_NIPS19)
    ts500_loader = DataLoader_NIPS19( ts500,  batch_size= 1, shuffle=False,  collate_fn= featurize_NIPS19)
    
    #train_loader = DataLoader_NIPS19( tadatrain,  batch_size= batch_size, shuffle=True,  collate_fn= featurize_NIPS19)
    tadatest_loader = DataLoader_NIPS19( tadatest,  batch_size= 1, shuffle=False,  collate_fn= featurize_NIPS19)

    print( len(data_dict['train']), len(data_dict['valid']), len(data_dict['test']), len(tadatrain), len(tadatest) )
    model = ADesign(
        node_features= 128,
        edge_features= 128,
        hidden_dim= 128,
        dropout= 0.1,
        k_neighbors= 30,
        num_encoder_layers= 10,
        num_decoder_layers=1
    )

    #load pre-trained models
    model.load_state_dict( torch.load( 'models/CATH_16_Tue May 14 14:06:14 2024.pth') )

    model.to(device)
    #model.half() # convert to FP16
    #model = torch.compile(model)
    num_params = sum(param.numel() for param in model.parameters())
    outname = 'models/%s_16_%s.pth'%(prename+'_TadA', time.ctime()) 
    print(outname)
    print('Number of parameters: %d'%(num_params) )
    save_best_model = SaveBestModel(outname=outname )
    #optimizer = torch.optim.AdamW( model.parameters(), lr=learning_rate)
    optimizer = torch.optim.Adam( model.parameters(), lr=learning_rate )
    #optimizer = torch.optim.SGD( model.parameters(), lr=learning_rate, momentum=0.9)
    steps_per_epoch = len(train_loader)
    #print( steps_per_epoch )
    #model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

    scheduler = torch.optim.lr_scheduler.OneCycleLR( optimizer, max_lr= learning_rate, steps_per_epoch=steps_per_epoch, epochs= num_epochs)
    #scheduler = torch.optim.lr_scheduler.CyclicLR( optimizer, base_lr=0.0001, max_lr=0.5, step_size_up= 1000)

       

    history = {'train_loss': [], 'val_loss': [] , 'test_loss': [],'test_pcc':[]}


    for epoch in range(num_epochs):
        t0 = time.time()

        train_running_loss = 0.0
        valid_running_loss = 0.0
        counter = 0
        model.train()
        #train_pbar = tqdm(train_loader, mininterval= 20)
        #for batch   in train_pbar:
        for batch   in train_loader:
            X, S, mask, lengths = cuda(batch, device=device )
            #print( X.shape,  S.shape, mask.shape )
            X, S, h_V, h_E, E_idx, batch_id = model._get_features(S, X=X, mask=mask)
            #print( batch_id.shape,  h_V.shape, h_E.shape, E_idx.shape )
            #sys.exit()
            
            log_probs = model( h_V, h_E, E_idx )
            loss = loss_fn( log_probs , S)
            #print(log_probs.shape, S.shape)
            #sys.exit()
            #loss1 = loss_fn(log_probs, S)
            #loss = loss1 + loss
            counter += 1
            train_running_loss += loss.item()
            # to create scaled gradients
            optimizer.zero_grad()
            loss.backward()
            #with amp.scale_loss(loss, optimizer) as scaled_loss:
            #    scaled_loss.backward()

            torch.nn.utils.clip_grad_norm_( model.parameters(), 1)
            optimizer.step()
            #clr = scheduler.get_last_lr()[0]
            #print('lr: ', clr)
            scheduler.step()

            #train_pbar.set_description('train loss: {:.4f}'.format( loss.item() ))

        #clr = scheduler.get_last_lr()[0]
        #clr =0
        epoch_loss = train_running_loss / counter
        writer.add_scalar('train_loss', epoch_loss, epoch)
            
        
        # validation
        counter = 0
        current_valid_loss = 0
        validation_weights = 0
        model.eval()
        with torch.no_grad():
            for batch   in tadatest_loader:
                X, S, mask, lengths = cuda(batch, device=device )
                #X = add_vatom(X)
                X, S, h_V, h_E, E_idx, batch_id = model._get_features(S, X=X, mask=mask)
                log_probs = model( h_V, h_E, E_idx )

                loss = loss_fn(log_probs, S)
                counter += 1
                # Accumulate
                current_valid_loss += torch.sum(loss).cpu().data.numpy()
                                                                        
        current_valid_loss = current_valid_loss / counter
        validation_perplexity = np.exp( current_valid_loss)

        save_best_model(current_valid_loss, model )
        if save_best_model.early_stop:
            print( "Early stopping")
            break
        
        #if epoch%5 == 0 or epoch>60:
        if True:
            test_model( model, test_loader)
            test_model( model, ts50_loader)
            test_model( model, ts500_loader)
            test_model( model, tadatest_loader)


        print("LOG: %.4f %.4f %.4f "%( epoch_loss, current_valid_loss, validation_perplexity  ))
        #print("LOG: %.4f, %.4f, %.6f "%(recovery, recovery2, clr))

        writer.add_scalar('train_loss', epoch_loss, epoch)
        writer.add_scalar('valid_loss', current_valid_loss, epoch)

        #if epoch % 5 == 0:
        #print('{} seconds'.format(time.time() - t0), scheduler.get_last_lr() )
        print('{} seconds'.format(time.time() - t0) )
        #print('epoch [{}/{}], loss:{:.4f}, val_loss:{:.4f}, test_loss:{:.4f}, recovery:{:.4f} \n'.format(epoch + 1, num_epochs, epoch_loss, current_valid_loss, current_test_loss, recovery   ) )
        print('epoch [{}/{}], loss:{:.4f}, val_loss:{:.4f} \n'.format(epoch + 1, num_epochs, epoch_loss, current_valid_loss  ) )
        if epoch>60:
            torch.save(model.state_dict(), outname+'_epoch'+str(epoch)+'.pth')
        sys.stdout.flush()

    #writer.export_scalars_to_json('all_scalars.json')
    writer.close()
    model.load_state_dict( torch.load(outname) )
    test_model( model, test_loader)
    test_model( model, ts50_loader)
    test_model( model, ts500_loader)
    test_model( model, tadatest_loader)

train_model( prename)
