import torch
from esm import FastaBatchedDataset, ProteinBertModel, MSATransformer
import pretrained
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm

##########################################################################################
_, alphabet = pretrained.load_model_and_alphabet_hub('esm2_t33_650M_UR50D')
truncation_seq_length = 1024
toks_per_batch =2096
train_fasta_file = './tadA_seqdump_train_4791.fasta'
train_dataset = FastaBatchedDataset.from_file(train_fasta_file)
train_batches = train_dataset.get_batch_indices(toks_per_batch, extra_toks_per_seq=1)
train_data_loader = torch.utils.data.DataLoader(
    train_dataset, collate_fn=alphabet.get_batch_converter(truncation_seq_length), batch_sampler=train_batches
)
for batch_idx, (labels, strs, Toks) in enumerate(tqdm(train_data_loader, desc='Processing')):
    file_name = str(batch_idx) + '.pt'
    save_path = './train_4791/' + file_name
    torch.save(Toks, save_path)


train_fasta_file = './tadA_seqdump_test_4791.fasta'
train_dataset = FastaBatchedDataset.from_file(train_fasta_file)
train_batches = train_dataset.get_batch_indices(toks_per_batch, extra_toks_per_seq=1)
train_data_loader = torch.utils.data.DataLoader(
    train_dataset, collate_fn=alphabet.get_batch_converter(truncation_seq_length), batch_sampler=train_batches
)

for batch_idx, (labels, strs, Toks) in enumerate(tqdm(train_data_loader, desc='Processing')):
    file_name = str(batch_idx) + '.pt'
    save_path = './test_4791/' + file_name
    torch.save(Toks, save_path)
