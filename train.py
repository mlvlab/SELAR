import torch
import torch.nn as nn
import numpy as np
from model import *
import sampler
from sklearn.metrics import roc_auc_score
import pdb

criterion = nn.BCELoss(reduction='none')
criterion_v = nn.BCELoss()

def selar_train(args, epoch, loaders, loader, model, vnet, phis, optimizer, optimizer_v, opt_phi, num_nodes, device):

    model.train()

    for phi in phis:
        phi.train()

    scores, labels = None, None
    loss_accum = 0

    if 'music' in args.dataset:
        num_iter = 199
    elif 'book' in args.dataset:
        num_iter = 82

    for step in range(num_iter):
        batch, tmp, metapath, metapath_label = next(loader)
        batchs, tmps, metapaths, metapath_labels= [], [], [], []
        for load in loaders:
            data = next(load)
            batchs.append(data[0])
            tmps.append(data[1])
            metapaths.append(data[2])
            metapath_labels.append(data[3])

        l_g_meta = 0
        optimizer_v.zero_grad()
        for i in range(args.n_fold):
            edge_index = batchs[i].edge_index.to(device)
            n_id = batchs[i].n_id.to(device)
            target_items = batchs[i].target_items.to(device)
            target_users = batchs[i].target_users.to(device)
            label = batchs[i].labels.to(device)

            if args.model_name == 'GCN':
                MetaNetwork = MetaNetworkGCN
            elif args.model_name == 'GAT':
                MetaNetwork = MetaNetworkGAT
            elif args.model_name == 'GIN':
                MetaNetwork = MetaNetworkGIN
            elif args.model_name == 'SGC':
                MetaNetwork = MetaNetworkSGC

            meta_model = MetaNetwork(args, num_nodes)
            meta_model.to(device)
            meta_model.load_state_dict(model.state_dict())

            meta_phis = []
            for k in range(len(phis)):
                phi = Phi(args.emb_dim, 100, args.emb_dim).to(device)
                phi.load_state_dict(phis[k].state_dict())
                meta_phis.append(phi)

            x = meta_model(n_id, edge_index)

            item = meta_phis[0](x[target_items])
            user = meta_phis[0](x[target_users])

            score_hat = item * user
            score_hat = torch.sigmoid(torch.sum(score_hat, 1))
            cost = criterion(score_hat, label)
            cost_v = torch.reshape(cost, (len(cost), 1))

            label_type = batchs[i].labels.view(-1, 1).to(device)
            target = torch.arange(args.n_meta + 1).reshape(args.n_meta + 1, 1)
            one_hot_target = (target == torch.arange(args.n_meta).reshape(1, args.n_meta)).float().to(device)
            type_v = torch.stack([one_hot_target[-1]] * len(cost_v), dim=0)
            type_v = torch.cat((type_v, label_type), dim=1)
            meta_type, meta_cost = [], []
            for n1, n2, mlabel, types, phi in zip(metapaths[i][:args.n_meta], metapaths[i][args.n_meta:], metapath_labels[i], one_hot_target[:-1], meta_phis[1:]):
                u1 = phi(x[tmps[i][n1].to(device)])
                u2 = phi(x[tmps[i][n2].to(device)])
                m_score = u1 * u2
                m_score = torch.sigmoid(torch.sum(m_score, 1))
                m_cost = criterion(m_score, mlabel.to(device))
                m_cost = torch.reshape(m_cost, (len(m_cost), 1))
                meta_cost.append(m_cost)

                mlabel_type = mlabel.view(-1, 1).to(device)
                cost_t = torch.stack([types] * len(m_cost), dim=0)
                cost_t = torch.cat((cost_t, mlabel_type), dim=1)
                meta_type.append(cost_t)

            type_m = torch.cat((meta_type))
            cost_m = torch.cat((meta_cost))
            types = torch.cat((type_v, type_m))
            costs = torch.cat((cost_v, cost_m))
            inputs = torch.cat((costs, types), 1)

            v_lambda = vnet(inputs.data)
            l_f_meta = len(cost_v)*(torch.sum(costs * v_lambda) / len(costs)) / (torch.sum(v_lambda[:len(cost_v)]))
            meta_model.update_params(lr_inner=args.lr, source_params=torch.autograd.grad(l_f_meta, (meta_model.params()), create_graph=True))
            for k in range(len(phis)):
                meta_phis[k].update_params(lr_inner=args.lr, source_params=torch.autograd.grad(l_f_meta, (meta_phis[k].params()), create_graph=True))

            v_target_items = batchs[i].m_target_items.to(device)
            v_target_users = batchs[i].m_target_users.to(device)
            v_label = batchs[i].m_labels.to(device)

            v_x = meta_model(n_id, edge_index)

            v_item = meta_phis[0](v_x[v_target_items])
            v_user = meta_phis[0](v_x[v_target_users])
            v_score = v_item * v_user
            v_score = torch.sigmoid(torch.sum(v_score, 1))

            v_cost = criterion_v(v_score, v_label)
            l_g_meta += v_cost/args.n_fold

        l_g_meta.backward()
        optimizer_v.step()

        edge_index = batch.edge_index.to(device)
        n_id = batch.n_id.to(device)
        target_items = batch.target_items.to(device)
        target_users = batch.target_users.to(device)
        label = batch.labels.to(device)

        X = model(n_id, edge_index)

        item = phis[0](X[target_items])
        user = phis[0](X[target_users])
        score = item * user
        score = torch.sigmoid(torch.sum(score, 1))
        
        optimizer.zero_grad()
        for opt in opt_phi:
            opt.zero_grad()

        cost = criterion(score, label)
        cost_v = torch.reshape(cost, (len(cost), 1))

        label_type = label.view(-1, 1).to(device)
        target = torch.arange(args.n_meta + 1).reshape(args.n_meta + 1, 1)
        one_hot_target = (target == torch.arange(args.n_meta).reshape(1, args.n_meta)).float().to(device)
        type_v = torch.stack([one_hot_target[-1]] * len(cost_v), dim=0)
        type_v = torch.cat((type_v, label_type), dim=1)
        meta_type, meta_cost = [], []
        for idx, (n1, n2, mlabel, types, phi) in enumerate(zip(metapath[:args.n_meta], metapath[args.n_meta:], metapath_label, one_hot_target[:-1], phis[1:])):
            u1 = phi(X[tmp[n1].to(device)])
            u2 = phi(X[tmp[n2].to(device)])
            m_score = u1 * u2
            m_score = torch.sigmoid(torch.sum(m_score, 1))
            m_cost = criterion(m_score, mlabel.to(device))
            m_cost = torch.reshape(m_cost, (len(m_cost), 1))
            meta_cost.append(m_cost)

            mlabel_type = mlabel.view(-1, 1).to(device)
            cost_t = torch.stack([types] * len(m_cost), dim=0)
            cost_t = torch.cat((cost_t, mlabel_type), dim=1)
            meta_type.append(cost_t)

        type_m = torch.cat((meta_type))
        cost_m = torch.cat((meta_cost))

        types = torch.cat((type_v, type_m))
        costs = torch.cat((cost_v, cost_m))
        inputs = torch.cat((costs, types), 1)

        with torch.no_grad():
            w_new = vnet(inputs.data)

        loss = len(cost_v)*(torch.sum(costs * w_new) / len(costs)) / (torch.sum(w_new[:len(cost_v)]))
        loss.backward()
        optimizer.step()
        for opt in opt_phi:
            opt.step()

        loss_accum += (torch.sum(cost_v) / len(cost_v)).item()
        if step == 0:
            scores = score.tolist()
            labels = label.tolist()
        else:
            scores += score.tolist()
            labels += label.tolist()

        if (step + 1) % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)] \t Loss: {:.6f}'.format(epoch, step + 1, num_iter, 100. * step / num_iter, loss))

    auc = roc_auc_score(np.array(labels), np.array(scores))

    return loss_accum/(step + 1), auc


def selarhint_train(args, epoch, loaders, loader, model, h_models, vnet, h_vnet, phis, optimizer, h_optimizers, optimizer_v, h_optimizer_v, opt_phi, num_nodes, device):

    model.train()
    for phi in phis:
        phi.train()
    for h_model in h_models:
        h_model.train()
    vnet.train()
    h_vnet.train()

    scores, labels = None, None
    loss_accum = 0

    if 'music' in args.dataset:
        num_iter = 199
    elif 'book' in args.dataset:
        num_iter = 82

    for step in range(num_iter):
        batch, tmp, metapath, metapath_label = next(loader)
        batchs, tmps, metapaths, metapath_labels = [], [], [], []
        for load in loaders:
            data = next(load)
            batchs.append(data[0])
            tmps.append(data[1])
            metapaths.append(data[2])
            metapath_labels.append(data[3])

        l_g_meta = 0
        optimizer_v.zero_grad()
        h_optimizer_v.zero_grad()
        for i in range(args.n_fold):
            edge_index = batchs[i].edge_index.to(device)
            h_edge_index = batchs[i].h_edge_index.to(device)
            n_id = batchs[i].n_id.to(device)
            target_items = batchs[i].target_items.to(device)
            target_users = batchs[i].target_users.to(device)
            label = batchs[i].labels.to(device)

            if args.model_name == 'GCN':
                MetaNetwork = MetaNetworkGCN
            elif args.model_name == 'GAT':
                MetaNetwork = MetaNetworkGAT
            elif args.model_name == 'GIN':
                MetaNetwork = MetaNetworkGIN
            elif args.model_name == 'SGC':
                MetaNetwork = MetaNetworkSGC

            meta_model = MetaNetwork(args, num_nodes)
            meta_model.to(device)
            meta_model.load_state_dict(model.state_dict())

            h_meta_models = []
            for k in range(len(h_models)):
                net = MetaNetworkHUB(args).to(device)
                net.load_state_dict(h_models[k].state_dict())
                h_meta_models.append(net)

            meta_phis = []
            for k in range(len(phis)):
                phi = Phi(args.emb_dim, 100, args.emb_dim).to(device)
                phi.load_state_dict(phis[k].state_dict())
                meta_phis.append(phi)

            x = meta_model(n_id, edge_index)
            
            item = meta_phis[0](x[target_items])
            user = meta_phis[0](x[target_users])

            score_hat = item * user
            score_hat = torch.sigmoid(torch.sum(score_hat, 1))
            cost = criterion(score_hat, label)
            cost_v = torch.reshape(cost, (len(cost), 1))

            label_type = batchs[i].labels.view(-1, 1).to(device)
            target = torch.arange(args.n_meta + 1).reshape(args.n_meta + 1, 1)
            one_hot_target = (target == torch.arange(args.n_meta).reshape(1, args.n_meta)).float().to(device)
            type_v = torch.stack([one_hot_target[-1]] * len(cost_v), dim=0)
            type_v = torch.cat((type_v, label_type), dim=1)

            meta_type, h_meta_type, meta_cost, h_meta_cost = [], [], [], []
            for n1, n2, mlabel, types, phi, hmodel in zip(metapaths[i][:args.n_meta], metapaths[i][args.n_meta:], metapath_labels[i], one_hot_target[:-1], meta_phis[1:], h_meta_models):
                u1 = phi(x[tmps[i][n1].to(device)])
                u2 = phi(x[tmps[i][n2].to(device)])
                m_score = u1 * u2
                m_score2 = torch.sigmoid(torch.sum(m_score, 1))
                m_cost = criterion(m_score2, mlabel.to(device)) 
                m_cost = torch.reshape(m_cost, (len(m_cost), 1))

                hx = hmodel(x, h_edge_index)
                h_u1 = hx[tmps[i][n1].to(device)]
                h_u2 = hx[tmps[i][n2].to(device)]
                h_m_score = h_u1 * h_u2
                h_m_score2 = torch.sigmoid(torch.sum(h_m_score, 1))
                h_m_cost = criterion(h_m_score2, mlabel.to(device)) 
                h_m_cost = torch.reshape(h_m_cost, (len(h_m_cost), 1))

                o_meta_cost = torch.cat((m_cost.detach(), h_m_cost.detach()), 1)
                mlabel_type = mlabel.view(-1, 1).to(device)
                h_meta_type = torch.stack([types] * len(m_cost), dim=0)
                h_meta_type = torch.cat((h_meta_type, mlabel_type), dim=1)
                h_input = torch.cat((o_meta_cost, h_meta_type), 1)
                coef = h_vnet(h_input.data)
                h_score = torch.sigmoid(torch.sum((coef * m_score + (1 - coef) * h_m_score), 1))
                
                cost = criterion(h_score, mlabel.to(device))
                meta_cost.append(torch.reshape(cost, (len(cost), 1)))                    
                meta_type.append(h_meta_type)

            type_m = torch.cat((meta_type))
            cost_m = torch.cat((meta_cost))
            types = torch.cat((type_v, type_m))
            costs = torch.cat((cost_v, cost_m))
            inputs = torch.cat((costs, types), 1)

            v_lambda = vnet(inputs.data)
            l_f_meta = len(cost_v)*(torch.sum(costs * v_lambda) / len(costs)) / (torch.sum(v_lambda[:len(cost_v)]))
            meta_model.update_params(lr_inner=args.lr, source_params=torch.autograd.grad(l_f_meta, (meta_model.params()), create_graph=True))
            for k in range(len(h_models)): 
                h_meta_models[k].update_params(lr_inner=args.lr, source_params=torch.autograd.grad(l_f_meta, h_meta_models[k].params(), create_graph=True))        
            for k in range(len(phis)):
                meta_phis[k].update_params(lr_inner=args.lr, source_params=torch.autograd.grad(l_f_meta, (meta_phis[k].params()), create_graph=True))

            v_target_items = batchs[i].m_target_items.to(device)
            v_target_users = batchs[i].m_target_users.to(device)
            v_label = batchs[i].m_labels.to(device)

            v_x = meta_model(n_id, edge_index)
            
            v_item = meta_phis[0](v_x[v_target_items])
            v_user = meta_phis[0](v_x[v_target_users])
            v_score = v_item * v_user
            v_score = torch.sigmoid(torch.sum(v_score, 1))

            v_cost = criterion_v(v_score, v_label)
            l_g_meta += v_cost/args.n_fold

        l_g_meta.backward()
        optimizer_v.step()
        h_optimizer_v.step()

        edge_index = batch.edge_index.to(device)
        h_edge_index = batch.h_edge_index.to(device)
        n_id = batch.n_id.to(device)
        target_items = batch.target_items.to(device)
        target_users = batch.target_users.to(device)
        label = batch.labels.to(device)

        X = model(n_id, edge_index)

        item = phis[0](X[target_items])
        user = phis[0](X[target_users])
        score = item * user
        score = torch.sigmoid(torch.sum(score, 1))

        optimizer.zero_grad()
        for h_opt in h_optimizers:
            h_opt.zero_grad()
        for opt in opt_phi:
            opt.zero_grad()

        cost = criterion(score, label)
        cost_v = torch.reshape(cost, (len(cost), 1))

        label_type = label.view(-1, 1).to(device)
        target = torch.arange(args.n_meta + 1).reshape(args.n_meta + 1, 1)
        one_hot_target = (target == torch.arange(args.n_meta).reshape(1, args.n_meta)).float().to(device)
        type_v = torch.stack([one_hot_target[-1]] * len(cost_v), dim=0)
        type_v = torch.cat((type_v, label_type), dim=1)

        meta_type, h_meta_type, meta_cost, h_meta_cost = [], [], [], []
        for idx, (n1, n2, mlabel, types, phi, hmodel) in enumerate(zip(metapath[:args.n_meta], metapath[args.n_meta:], metapath_label, one_hot_target[:-1], phis[1:], h_models)):
            u1 = phi(X[tmp[n1].to(device)])
            u2 = phi(X[tmp[n2].to(device)])
            m_score = u1 * u2
            m_score2 = torch.sigmoid(torch.sum(m_score, 1))
            m_cost = criterion(m_score2, mlabel.to(device)) 
            m_cost = torch.reshape(m_cost, (len(m_cost), 1))

            hx = hmodel(X, h_edge_index)
            h_u1 = hx[tmp[n1].to(device)]
            h_u2 = hx[tmp[n2].to(device)]
            h_m_score = h_u1 * h_u2
            h_m_score2 = torch.sigmoid(torch.sum(h_m_score, 1))
            h_m_cost = criterion(h_m_score2, mlabel.to(device)) 
            h_m_cost = torch.reshape(h_m_cost, (len(h_m_cost), 1))
            o_meta_cost = torch.cat((m_cost.detach(), h_m_cost.detach()), 1)
            h_meta_type = torch.stack([types] * len(m_cost), dim=0)
            mlabel_type = mlabel.view(-1, 1).to(device)
            h_meta_type = torch.cat((h_meta_type, mlabel_type), dim=1)
            h_input = torch.cat((o_meta_cost, h_meta_type), 1)
            coef = h_vnet(h_input.data)**args.hreg
            #h_score = torch.sigmoid(torch.sum(coef * m_score, 1) + torch.sum((1 - coef) * h_m_score,1))
            h_score = torch.sigmoid(torch.sum((coef * m_score + (1 - coef) * h_m_score), 1))

            cost = criterion(h_score, mlabel.to(device))
            meta_cost.append(torch.reshape(cost, (len(cost), 1)))                    
            meta_type.append(h_meta_type)

        type_m = torch.cat((meta_type))
        cost_m = torch.cat((meta_cost))
        types = torch.cat((type_v, type_m))
        costs = torch.cat((cost_v, cost_m))
        inputs = torch.cat((costs, types), 1)
        
        with torch.no_grad():
            w_new = vnet(inputs.data)

        loss = len(cost_v)*(torch.sum(costs * w_new) / len(costs)) / (torch.sum(w_new[:len(cost_v)]))
        loss.backward()
        optimizer.step() 
        for h_opt in h_optimizers:
            h_opt.step()
        for opt in opt_phi:
            opt.step()

        loss_accum += (torch.sum(cost_v) / len(cost_v)).item()
        if step == 0:
            scores = score.tolist()
            labels = label.tolist()
        else:
            scores += score.tolist()
            labels += label.tolist()

        if (step + 1) % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)] \t Loss: {:.6f}'.format(epoch, step + 1, num_iter, 100. * step / num_iter, loss))
        
    auc = roc_auc_score(np.array(labels), np.array(scores))
    return loss_accum / (step + 1), auc
