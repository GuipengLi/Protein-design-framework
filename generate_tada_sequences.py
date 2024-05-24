import pandas as pd
import numpy as np

import torch
import esm

from torch.nn.functional import softmax

def get_esm2_score( mat, idx):
    bool_mask = np.zeros_like(mat, dtype=bool)
    bool_mask[np.arange(len(mat)), idx] = True
    values = mat[bool_mask]
    score = np.mean(np.log(values))
    return score



def sample_from_matrix(mat):
    seq = ''
    n, k = mat.shape
    aas = ''.join(mat.columns.values)
    #mat = np.exp(np.array(mat))
    mat = np.array(mat)
    #mat = np.exp(mat) - 1
    rowsums = mat.sum(axis=1)
    mat = mat / rowsums[:, np.newaxis ] #normalized

    for i in range(n):
        idx = np.random.choice(k, p =mat[i])
        aa = aas[idx]
        seq += aa
    return seq

def sample_from_matrix_alpha(mat, cutoff=0.7):
    seq = ''
    n, k = mat.shape
    aas = ''.join(mat.columns.values)
    #mat = np.exp(np.array(mat))
    mat = np.array(mat)
    #mat = np.exp(mat) - 1
    rowsums = mat.sum(axis=1)
    mat = mat / rowsums[:, np.newaxis ] #normalized

    for i in range(n):
        if mat[i].max() > cutoff:
            idx = np.argmax(mat[i])
            #print(i, idx, mat[i].max())
        else:
            idx = np.random.choice(k, p =mat[i])
        aa = aas[idx]
        seq += aa
    return seq

def sample_from_matrix_alpha_cond(mat, cutoff=0.5, tseq=''):
    start = 13
    end = 15
    seq = ''
    n, k = mat.shape
    aas = ''.join(mat.columns.values)
    #mat = np.exp(np.array(mat))
    mat = np.array(mat)
    #mat = np.exp(mat) - 1
    rowsums = mat.sum(axis=1)
    mat = mat / rowsums[:, np.newaxis ] #normalized

    for i in range(n):
        if i < start or i> n-end or i in [47,50, 81, 83,105,107, 145,146]:
            aa = tseq[i]
        else:
            if mat[i].max() > cutoff:
                idx = np.argmax(mat[i])
                #print(i, idx, mat[i].max())
            else:
                idx = np.random.choice(k, p =mat[i])
            aa = aas[idx]
        seq += aa
    return seq

def get_matrix_from_seq( mat, seq):
    n, k = mat.shape
    aas = mat.columns.values
    mat = np.array(mat)
    for i in range(len(seq)):
        idx = np.where( aas == seq[i])
        mat[i,idx] += 0.25

    rowsums = mat.sum(axis=1)
    mat = mat / rowsums[:, np.newaxis ] #normalized
    mat = pd.DataFrame(mat)
    mat.columns = aas
    return mat
    


def get_matrix_from_seq_truncated( mat, seq):
    n, k = mat.shape

    aas = mat.columns.values
    mat2 = np.zeros(( len(seq), k))
    mat2[5:(5+n),:] = np.array(mat)
    for i in range(len(seq)):
        idx = np.where( aas == seq[i])
        mat2[i,idx] += 0.5

    rowsums = mat2.sum(axis=1)
    mat2 = mat2 / rowsums[:, np.newaxis ] #normalized
    mat2 = pd.DataFrame(mat2)
    mat2.columns = aas
    return mat2
    

prob_mat_file = 'finetune_tada_8e2p/8e2p.chainA_predict_probs.csv'

prob_mat = pd.read_csv(prob_mat_file, header=0, index_col=0).iloc[:,0:20]

x8e2p='MSEVEFSHEYWMRHALTLAKRARDEREVPVGAVLVLNNRVIGEGWNRAIGLHDPTAHAEIMALRQGGLVMQNYRLYDATLYSTFEPCVMCAGAMIHSRIGRVVFGVRNAKTGAAGSLMDVLHHPGMNHRVEITEGILADECAALLCRFFRMPRRVFNAQKKAQSSTD'


#mat2 =  get_matrix_from_seq(prob_mat, x8e2p)
mat2 =  get_matrix_from_seq_truncated(prob_mat, x8e2p)
print( mat2.head())


i = 0
n = 0
for x,y in zip(p1a, test):
    if x==y:
        i += 1
    n += 1

print( i/n)

def get_identity_score( xs, ys):
    i = 0
    n = 0
    for x,y in zip( xs, ys):
        if x==y:
            i += 1
        n += 1

    return( i/n)




model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()

state_dict = torch.load(fr'/share/home/liguipeng/3d21d/AlphaDesign/BE/model_TadA_4791_best.pt')
model.load_state_dict(state_dict)

batch_converter = alphabet.get_batch_converter()
model.eval()


with open('test.score.v2.txt','w') as fout:
    for k in range(10000):
        sp = sample_from_matrix_alpha_cond( mat2, cutoff=1, tseq=x8e2p)

        data = [('id', sp)]
        labels, strs, tokens = batch_converter(data)
        with torch.no_grad():
            results = model(tokens, repr_layers=[33], return_contacts=False)
        prob = softmax(results['logits'][0,1:-1, 4:24], dim=1)
        esm2_score = get_esm2_score( np.array(prob), tokens[0][1:-1]-4 )

        fout.write( '%s\t%.4f\n' %(sp, esm2_score) )


