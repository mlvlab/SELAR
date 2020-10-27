from sklearn.metrics import roc_auc_score, f1_score
from utils import accuracy
import numpy as np
import torch
import torch.nn as nn

criterion = nn.BCELoss()

def evaluate(args, dataset, loader, model, phis, device):
    
    model.eval()
    phis[0].eval()
    scores, labels = [], []
    loss_accum = 0

    for step, batch in enumerate(loader(dataset)):
        edge_index = batch.edge_index#.to(device)
        n_id = batch.n_id.to(device)
        target_items = batch.target_items.to(device)
        target_users = batch.target_users.to(device)
        label = batch.labels.to(device)

        if args.model_name == 'GTN':
            x = model(edge_index, n_id, target_users, target_items, label, n_id.shape[0], train=False)
        else:
            x = model(n_id, edge_index.to(device))
            
        item = phis[0](x[target_items])
        user = phis[0](x[target_users])
        score = item * user
        score = torch.sigmoid(torch.sum(score, 1))

        loss = criterion(score, label)
        
        loss_accum += loss.item()
        scores += score.tolist()
        labels += label.tolist()

    scores = np.array(scores)
    labels = np.array(labels)
    auc = roc_auc_score(labels, scores)
    scores[scores>=0.5] = 1
    scores[scores<0.5] = 0
    f1 = f1_score(labels, scores)
    acc = accuracy(labels, scores)

    return loss_accum/(step+1), auc, f1, acc


