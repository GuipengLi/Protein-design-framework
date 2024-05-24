from torch import Tensor, nn
import torch
from torch.utils.data import Dataset, DataLoader
import os
import esm
from tqdm import tqdm
import logging

class MyDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.pt_files = [f for f in os.listdir(data_dir) if f.endswith(".pt")]

    def __len__(self):
        return len(self.pt_files)

    def __getitem__(self, idx):
        pt_file = os.path.join(self.data_dir, self.pt_files[idx])
        data = torch.load(pt_file)
        return data

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
##########################################################################################
data_dir = "./train_4791"
dataset = MyDataset(data_dir)
train_dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

data_dir = "./test_4791"
dataset = MyDataset(data_dir)
valid_dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

##########################################################################################
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)

model.to(device)
learning_rate = 1e-6
optimizer = torch.optim.Adam(model.parameters(),
                       lr=learning_rate,
                       betas=(0.9, 0.98),
                       eps=1e-8,
                       weight_decay=0.01)
criterion = nn.CrossEntropyLoss()
MODEL = 'TadA_fine_tuning'
L = []
L_train = []
L_val = []
AUC = []
PRAUC = []
min_validation_loss = 9999

logging.basicConfig(filename=f'training_{MODEL}.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
mask_idx = 32
total_epochs = 1000
train_masked_loss_o = 0
train_running_loss_o = 0
test_masked_loss_o = 0
test_running_loss_o = 0
LABEL_train = []
predict_train = []
LABEL_test = []
predict_test = []
model.eval()
with torch.no_grad():
    for Toks in tqdm(train_dataloader):
        ##################################################
        Toks = Toks.squeeze(0)
        toks = Toks.clone()
        mask = torch.zeros_like(toks, dtype=torch.bool)
        for i in range(toks.size(0)):
            not_ones_indices = ((toks[i] != 1) & (toks[i] != 0) & (toks[i] != 2)).nonzero().squeeze()
            num_not_ones = not_ones_indices.size(0)
            num_elements_to_mask_per_sequence = int(0.15 * num_not_ones)
            mask_indices = not_ones_indices[torch.randperm(num_not_ones)[:num_elements_to_mask_per_sequence]]
            mask[i, mask_indices] = True
        toks[mask] = mask_idx
        ##################################################
        out = model(toks.to(device))
        OUTPUT = out['logits']
        masked_toks = Toks[mask]
        masked_OUTPUT = OUTPUT[mask]
        predict_train.append(masked_OUTPUT.detach().cpu())
        LABEL_train.append(masked_toks.cpu())
        masked_LOSS = criterion(masked_OUTPUT.to(device), masked_toks.to(device))
        train_masked_loss_o += masked_LOSS.item()
        new_first_dim = OUTPUT.size(0) * OUTPUT.size(1)
        OUTPUT = OUTPUT.view(new_first_dim, OUTPUT.size(2))
        new_first_dim = Toks.size(0) * Toks.size(1)
        Toks = Toks.view(new_first_dim)
        non_zero_indices = torch.nonzero((Toks != 0) & (Toks != 1) & (Toks != 2)).squeeze()
        OUTPUT = OUTPUT[non_zero_indices]
        Toks = Toks[non_zero_indices]
        LOSS = criterion(OUTPUT.to(device), Toks.to(device))
        train_running_loss_o += LOSS.item()
    for Toks in tqdm(valid_dataloader):
        Toks = Toks.squeeze(0)
        toks = Toks.clone()
        ##################################################
        mask = torch.zeros_like(toks, dtype=torch.bool)
        for i in range(toks.size(0)):
            not_ones_indices = ((toks[i] != 1) & (toks[i] != 0) & (toks[i] != 2)).nonzero().squeeze()
            num_not_ones = not_ones_indices.size(0)
            num_elements_to_mask_per_sequence = int(0.15 * num_not_ones)
            mask_indices = not_ones_indices[
                torch.randperm(num_not_ones)[:num_elements_to_mask_per_sequence]]
            mask[i, mask_indices] = True
        toks[mask] = mask_idx
        ##################################################
        out = model(toks.to(device))
        OUTPUT = out['logits']
        masked_toks = Toks[mask]
        masked_OUTPUT = OUTPUT[mask]
        masked_LOSS = criterion(masked_OUTPUT.to(device), masked_toks.to(device))
        test_masked_loss_o += masked_LOSS.item()
        LABEL_test.append(masked_toks.cpu())
        predict_test.append(masked_OUTPUT.detach().cpu())
        new_first_dim = OUTPUT.size(0) * OUTPUT.size(1)
        OUTPUT = OUTPUT.view(new_first_dim, OUTPUT.size(2))
        new_first_dim = Toks.size(0) * Toks.size(1)
        Toks = Toks.view(new_first_dim)
        LOSS = criterion(OUTPUT.to(device), Toks.to(device))
        test_running_loss_o += LOSS.item()
outputs = torch.cat(predict_train, dim=0)
labels = torch.cat(LABEL_train, dim=0)
predicted = torch.argmax(outputs, dim=1)
correct = (predicted == labels).sum().item()
total = labels.size(0)
accuracy_train = correct / total
outputs = torch.cat(predict_test, dim=0)
labels = torch.cat(LABEL_test, dim=0)
predicted = torch.argmax(outputs, dim=1)
correct = (predicted == labels).sum().item()
total = labels.size(0)
accuracy_test = correct / total


logging.info(f'Epoch 00,  Train_ACC_mask: {accuracy_train}')
logging.info(f'Epoch 00,  Validation_ACC_mask: {accuracy_test}')
logging.info(f'Epoch 00,  Train_Loss: {train_running_loss_o / len(train_dataloader)}')
logging.info(f'Epoch 00,  Validation_Loss: {test_running_loss_o / len(valid_dataloader)}')
logging.info(f'Epoch 00,  Train_Loss_mask: {train_masked_loss_o / len(train_dataloader)}')
logging.info(f'Epoch 00,  Validation_Loss_mask: {test_masked_loss_o / len(valid_dataloader)}')


for epoch in range(1, total_epochs + 1):
    train_running_loss = 0.0
    train_masked_loss=0.0
    counter = 0
    LABEL_train = []
    predict_train = []
    LABEL_test = []
    predict_test = []
    for Toks in tqdm(train_dataloader):
        model.train()
        counter += 1
        ##################################################
        Toks = Toks.squeeze(0)
        toks = Toks.clone()
        mask = torch.zeros_like(toks, dtype=torch.bool)
        for i in range(toks.size(0)):
            not_ones_indices = ((toks[i] != 1) & (toks[i] != 0) & (toks[i] != 2)).nonzero().squeeze()
            num_not_ones = not_ones_indices.size(0)
            num_elements_to_mask_per_sequence = int(0.15 * num_not_ones)
            mask_indices = not_ones_indices[torch.randperm(num_not_ones)[:num_elements_to_mask_per_sequence]]
            mask[i, mask_indices] = True
        toks[mask] = mask_idx
        ##################################################
        out = model(toks.to(device))
        OUTPUT = out['logits']
        masked_toks = Toks[mask]
        masked_OUTPUT = OUTPUT[mask]
        predict_train.append(masked_OUTPUT.detach().cpu())
        LABEL_train.append(masked_toks.cpu())
        masked_LOSS = criterion(masked_OUTPUT.to(device), masked_toks.to(device))
        train_masked_loss += masked_LOSS.item()
        new_first_dim = OUTPUT.size(0) * OUTPUT.size(1)
        OUTPUT = OUTPUT.view(new_first_dim, OUTPUT.size(2))
        new_first_dim = Toks.size(0) * Toks.size(1)
        Toks = Toks.view(new_first_dim)
        non_zero_indices = torch.nonzero((Toks != 0) & (Toks != 1) & (Toks != 2)).squeeze()
        OUTPUT = OUTPUT[non_zero_indices]
        Toks = Toks[non_zero_indices]
        LOSS = criterion(OUTPUT.to(device), Toks.to(device))
        train_running_loss += LOSS.item()
        optimizer.zero_grad()
        LOSS.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        current_test_loss = 0.0
        test_masked_loss = 0.0
        for Toks in tqdm(valid_dataloader):
            Toks = Toks.squeeze(0)
            toks = Toks.clone()
            ##################################################
            mask = torch.zeros_like(toks, dtype=torch.bool)
            for i in range(toks.size(0)):
                not_ones_indices = ((toks[i] != 1) & (toks[i] != 0) & (toks[i] != 2)).nonzero().squeeze()
                num_not_ones = not_ones_indices.size(0)
                num_elements_to_mask_per_sequence = int(0.15 * num_not_ones)
                mask_indices = not_ones_indices[
                    torch.randperm(num_not_ones)[:num_elements_to_mask_per_sequence]]
                mask[i, mask_indices] = True
            toks[mask] = mask_idx
            ##################################################
            out = model(toks.to(device))
            OUTPUT = out['logits']
            masked_toks = Toks[mask]
            masked_OUTPUT = OUTPUT[mask]
            LABEL_test.append(masked_toks.cpu())
            predict_test.append(masked_OUTPUT.detach().cpu())
            masked_LOSS = criterion(masked_OUTPUT.to(device), masked_toks.to(device))
            test_masked_loss += masked_LOSS.item()
            new_first_dim = OUTPUT.size(0) * OUTPUT.size(1)
            OUTPUT = OUTPUT.view(new_first_dim, OUTPUT.size(2))
            new_first_dim = Toks.size(0) * Toks.size(1)
            Toks = Toks.view(new_first_dim)
            LOSS = criterion(OUTPUT.to(device), Toks.to(device))
            current_test_loss += LOSS.item()
        T_loss = test_masked_loss / len(valid_dataloader)
        L_val.append(T_loss)
        if min_validation_loss > T_loss:
            min_validation_loss = T_loss
            best_epoch = epoch
            print('Max pr_auc ' + str(min_validation_loss) + ' in epoch ' + str(best_epoch))
            torch.save(model.module.state_dict(), fr"./model_{MODEL}_best.pt")
    outputs = torch.cat(predict_train, dim=0)
    labels = torch.cat(LABEL_train, dim=0)
    predicted = torch.argmax(outputs, dim=1)
    correct = (predicted == labels).sum().item()
    total = labels.size(0)
    accuracy_train = correct / total
    outputs = torch.cat(predict_test, dim=0)
    labels = torch.cat(LABEL_test, dim=0)
    predicted = torch.argmax(outputs, dim=1)
    correct = (predicted == labels).sum().item()
    total = labels.size(0)
    accuracy_test = correct / total
    print("Train loss: ", train_running_loss / counter, "Val loss: ", T_loss ,"train ACC: ", accuracy_train ,"Val ACC: ", accuracy_test )
    logging.info(f'Epoch {epoch},  Train_ACC_mask: {accuracy_train}')
    logging.info(f'Epoch {epoch},  Validation_ACC_mask: {accuracy_test}')
    logging.info(f'Epoch {epoch},  Train_Loss: {train_running_loss / counter}')
    logging.info(f'Epoch {epoch},  Validation_Loss: {current_test_loss / len(valid_dataloader)}')
    logging.info(f'Epoch {epoch},  Train_Loss_mask: {train_masked_loss / counter}')
    logging.info(f'Epoch {epoch},  Validation_Loss_mask: {test_masked_loss / len(valid_dataloader)}')


# 训练结束后保存最终模型
torch.save(model.module.state_dict(), fr"./model_{MODEL}_last.pt")

