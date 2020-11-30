import os
import argparse
import torch
import numpy as np

from train import *
from model import *
import sampler
from preprocess import process
from evaluate import evaluate
import pdb

def main():
    parser = argparse.ArgumentParser(description="Pytorch implementation of Adaptive Learning Meta-paths to learn on Heterogeneous graph")
    parser.add_argument('--seed', type=int, default=777)
    parser.add_argument('--dataset', type=str, default='music')
    parser.add_argument('--model_name', type=str, default='GCN')
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--wlr', type=float, default=0.005)
    parser.add_argument('--decay', type=float, default=0)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--num_hops', type=int, default=3)
    parser.add_argument('--neighbor_size',type=int, default=[8,8,8])
    parser.add_argument('--emb_dim', type=int, default=16)
    parser.add_argument('--weight_emb_dim', type=int, default=1000)
    parser.add_argument('--n_meta', type=int, default=5)
    parser.add_argument('--metapath', type=str, default=[2,8,9,10,11])
    parser.add_argument('--num_hub', type=int, default=5)
    parser.add_argument('--n_fold', type=int, default=3)
    parser.add_argument('--selar', type=str, default='True')
    parser.add_argument('--selarhint', type=str, default='False')
    parser.add_argument('--hreg', type=float, default=1)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset, train_fold, meta_fold, valid_target, test_target, kg, hub, num_nodes, n_relation, n_user, n_entity, n_item = process(args)
            
    ui = train_dataset[train_dataset[:,2]==1]
    iu = train_dataset[train_dataset[:,2]==1][:, [1,0,2]]
    ui[:, 2] = n_relation
    iu[:, 2] = n_relation+1
    train_all = torch.cat((ui, iu))
    edge_index = np.concatenate((train_all, kg)).T
    edge_index = torch.from_numpy(edge_index)

    if args.model_name == "GCN":
        model = MetaNetworkGCN
    elif args.model_name == "GAT":
        model = MetaNetworkGAT
    elif args.model_name == "GIN":
        model = MetaNetworkGIN
    elif args.model_name == "SGC":
        model = MetaNetworkSGC

    model = model(args, num_nodes).to(device)
    optimizer = torch.optim.Adam(model.params(), lr=args.lr, weight_decay=args.decay)

    ### Task-specific layers ###
    input_dim = args.n_meta + 2
    phis, opt_phi = [],[]
    for _ in range(args.n_meta+1):
        phi = Phi(args.emb_dim, 100, args.emb_dim).to(device)
        opt_phi.append(torch.optim.Adam(phi.params(), lr=args.lr))
        phis.append(phi)

    ### Weighting function ###
    vnet = Weight(input_dim, args.weight_emb_dim, 1).to(device)
    optimizer_v = torch.optim.Adam(vnet.params(), lr=args.wlr)

    ### For Hint Network ###
    if args.selarhint == 'True':
        h_nets, h_optimizers = [], []
        for _ in range(args.n_meta):
            net = MetaNetworkHUB(args).to(device)
            opt = torch.optim.Adam(net.params(), lr=args.lr, weight_decay=args.decay)
            h_nets.append(net)
            h_optimizers.append(opt)
        h_vnet = Weight(input_dim+1, args.weight_emb_dim, 1).to(device)
        h_optimizer_v = torch.optim.Adam(h_vnet.params(), lr=args.wlr, weight_decay=args.h_decay)

    best_valid_auc, best_test_auc, best_valid_auc_epoch = 0,0,0

    train_loaders = []
    for _ in range(args.n_fold):
        train_loaders.append(sampler.Meta_NeighborSampler(args=args, edge_index=edge_index, num_nodes=num_nodes, size=args.neighbor_size, num_hops=args.num_layers, batch_size=args.batch_size))                                

    for epoch in range(1, args.num_epochs + 1):
        loaders = []
        if args.selar == 'True':
            for i, train_loader in enumerate(train_loaders):
                loaders.append(train_loader(train_fold[i], meta_fold[i], None))
            train_loader = sampler.Meta_NeighborSampler(args=args, edge_index=edge_index, num_nodes=num_nodes, size=args.neighbor_size, num_hops=args.num_layers, batch_size=args.batch_size)
            loader = train_loader(train_dataset, None, None)
            train_losses, train_auc = selar_train(args, epoch, loaders, loader, model, vnet, phis, optimizer, optimizer_v, opt_phi, num_nodes, device)

        if args.selarhint == 'True':
            for i, train_loader in enumerate(train_loaders):
                loaders.append(train_loader(train_fold[i], meta_fold[i], hub))
            train_loader = sampler.Meta_NeighborSampler(args=args, edge_index=edge_index, num_nodes=num_nodes, size=args.neighbor_size, num_hops=args.num_layers, batch_size=args.batch_size)
            loader = train_loader(train_dataset, None, hub)
            train_losses, train_auc = selarhint_train(args, epoch, loaders, loader, model, h_nets, vnet, h_vnet, phis, optimizer, h_optimizers, optimizer_v, h_optimizer_v, opt_phi, num_nodes, device)                
            
        eval_loader = sampler.eval_NeighborSampler(args=args, edge_index=edge_index, num_nodes=num_nodes, size=args.neighbor_size, num_hops=args.num_layers, batch_size=args.batch_size)
        valid_losses, valid_auc, valid_f1, valid_acc = evaluate(args, valid_target, eval_loader, model, phis, device)
        test_losses, test_auc, test_f1, test_acc = evaluate(args, test_target, eval_loader, model, phis, device)
       
        print('{:2} Epoch - Train Loss: {}'.format(epoch, train_losses))
        print('{:2} Epoch - Train auc: {} '.format(epoch, train_auc))
        print('{:2} Epoch - Test auc: {} '.format(epoch, test_auc))
        
        if valid_auc > best_valid_auc:
            best_valid_auc_epoch = epoch
            best_test_auc  = test_auc
            best_valid_auc = valid_auc

    print('{:2} Best Test auc: {} '.format(epoch, best_test_auc))

if __name__ == "__main__":
    main()
