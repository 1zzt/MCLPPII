import os
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import argparse
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from dgllife.utils import EarlyStopping
from prettytable import PrettyTable
from sklearn.model_selection import KFold
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from adan import *
from dataset import *
from metrics import compute_cls_metrics, compute_reg_metrics
from model import *


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # random.seed(seed)
    torch.backends.cudnn.deterministic = True


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n

    def get_average(self):
        self.avg = self.sum / (self.count + 1e-12)
        return self.avg


data_path = './Datasets/'
K = 10


def get_dataloaders(dataset_name, data_path):
    file_path = os.path.join(data_path, task_name)

    train_df = pd.read_csv(os.path.join(file_path, dataset_name + '_train.csv'))
    test_df = pd.read_csv(os.path.join(file_path, dataset_name + '_test.csv'))

    train_smiles = train_df[train_df.columns[0]].values
    train_labels = train_df[train_df.columns[-1]].values

    test_smiles = test_df[test_df.columns[0]].values
    test_labels = test_df[test_df.columns[-1]].values

    train_loaders = []
    test_loaders = []

    train_set = PPIMModalDataset(train_smiles, train_labels)
    test_set = PPIMModalDataset(test_smiles, test_labels)

    for _ in range(K):
        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
        test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)

        train_loaders.append(train_loader)
        test_loaders.append(test_loader)

    return train_loaders, test_loaders


def run_a_train_epoch(
    device,
    epoch,
    model,
    data_loader,
    loss_criterion,
    optimizer,
    scheduler,
):
    model.train()
    tbar = tqdm(enumerate(data_loader), total=len(data_loader))

    for id, (*x, y) in tbar:
        for i in range(len(x)):
            x[i] = x[i].to(device)
        y = y.to(device)

        output, cl_loss = model(*x)
        main_loss = loss_criterion(output.view(-1), y.view(-1))
        if withcl:
            loss = main_loss + 0.1 * cl_loss
        else:
            loss = main_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad()
        # tbar.set_description(f' * Train Epoch {epoch} Loss={loss.item()  :.3f}')
        tbar.set_description(
            f' * Train Epoch {epoch} Loss={loss.item()  :.3f}  CL_loss={cl_loss.item()  :.3f}'
        )


def run_an_eval_epoch(model, data_loader, task_name, loss_criterion):
    model.eval()
    running_loss = AverageMeter()
    with torch.no_grad():
        preds = torch.Tensor()
        trues = torch.Tensor()
        for batch_id, (*x, y) in tqdm(enumerate(data_loader)):
            for i in range(len(x)):
                x[i] = x[i].to(device)
            y = y.to(device)
            logits, _ = model(*x)
            loss = loss_criterion(logits.view(-1), y.view(-1))
            if task_name == 'classification':
                logits = torch.sigmoid(logits)
            preds = torch.cat((preds, logits.cpu()), 0)
            trues = torch.cat((trues, y.view(-1, 1).cpu()), 0)
            running_loss.update(loss.item(), y.size(0))

        preds, trues = preds.numpy().flatten(), trues.numpy().flatten()

        # df = pd.DataFrame([preds, trues])
        # df.to_csv('./test-result/BHR-lfa_icam.csv')
    val_loss = running_loss.get_average()

    return preds, trues, val_loss


BATCH_SIZE = 32


def ptable_to_csv(table, filename, headers=True):
    """Save PrettyTable results to a CSV file.

    Adapted from @AdamSmith https://stackoverflow.com/questions/32128226

    :param PrettyTable table: Table object to get data from.
    :param str filename: Filepath for the output CSV.
    :param bool headers: Whether to include the header row in the CSV.
    :return: None
    """
    raw = table.get_string()
    data = [
        tuple(filter(None, map(str.strip, splitline)))
        for line in raw.splitlines()
        for splitline in [line.split('|')]
        if len(splitline) > 1
    ]
    if table.title is not None:
        data = data[1:]
    if not headers:
        data = data[1:]
    with open(filename, 'a') as f:
        for d in data:
            f.write('{}\n'.format(','.join(d)))


parser = argparse.ArgumentParser(description='Optional app description')
parser.add_argument('--task', type=str, required=False, default='regression')
parser.add_argument('--dataset', type=str, required=False, default='integrins')
parser.add_argument('--device', type=str, required=False, default='2')
parser.add_argument('--withcl', action='store_true', dest='withcl', default=True)
parser.add_argument('--seed', type=int, required=False, default='42')

args = parser.parse_args()
task_name = args.task
dataset_name = args.dataset
device_id = args.device
device = torch.device("cuda:" + device_id if torch.cuda.is_available() else "cpu")
withcl = args.withcl
seed = args.seed
# withcl = True
print('------withcl----' + str(withcl))


# setup_seed(args.seed)

lr = 5e-3

# torch.backends.cudnn.deterministic = False
# torch.backends.cudnn.benchmark = True


if task_name == 'classification':
    t_tables = PrettyTable(['epoch', 'MCC', 'F1', 'AUC'])
else:
    t_tables = PrettyTable(['epoch', 'R', 'Kendall', 'Spearman', 'RMSE', 'MAE'])



results_filename = (
'./results/regression/'
+ task_name
+ '-'
+ dataset_name
+ '-'
+ str(withcl)
+ '.csv'
)

setup_seed(seed)
train_loaders, test_loaders = get_dataloaders(dataset_name, data_path)

t_tables.float_format = '.3'

if task_name == 'classification':
    loss_criterion = nn.BCEWithLogitsLoss()
else:
    loss_criterion = nn.MSELoss()

model = PPIMMultiModalNet().to(device)


model.load_state_dict(torch.load('./models/pretrain.pth'))

num_epochs = 200
# num_epochs = 20
optimizer = Adan(
    model.parameters(),
    lr=5e-4,  # learning rate (can be much higher than Adam, up to 5-10x)
    betas=(
        0.02,
        0.08,
        0.01,
    ),  # beta 1-2-3 as described in paper - author says most sensitive to beta3 tuning
    weight_decay=0.02,  # weight decay 0.02 is optimal per author
)
scheduler = None

# stopper = EarlyStopping(
#     mode='lower',
#     patience=15,
#     filename='./models/ppim-' + task_name + '-' + dataset_name + '-' + str(withcl),
# )

for epoch in range(num_epochs):
    # print('epoch ' + str(epoch))
    run_a_train_epoch(
        device, epoch, model, train_loaders[0], loss_criterion, optimizer, scheduler
    )
    test_pred, test_y, test_loss = run_an_eval_epoch(
        model, test_loaders[0], task_name, loss_criterion
    )
    # F1, roc_auc, mcc, tn, fp, fn, tp = compute_cls_metrics(test_y, test_pred)
    # row = [epoch, mcc, F1, roc_auc]
    tau, rho, r, rmse, mae = compute_reg_metrics(test_y, test_pred)
    row = ['test', r, tau, rho, rmse, mae]


    t_tables.add_row(row)
print(t_tables)
ptable_to_csv(t_tables, results_filename)