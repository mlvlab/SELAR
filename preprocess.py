import os
from collections import Counter
from sklearn.model_selection import KFold
import numpy as np
import torch
import pdb

def process(args):

    def map_data(data):
        uniq = list(set(data))
        id_dict = {old: new for new, old in enumerate(sorted(uniq))}
        data = list(map(lambda x: id_dict[x], data))
        n = len(uniq)
        return data, n

    rating_file = '../data/' + args.dataset + '/ratings_final'
    if os.path.exists(rating_file + '.npy'):
        rating_np = np.load(rating_file + '.npy')
    else:
        rating_np = np.loadtxt(rating_file + '.txt', dtype=np.int64)
        np.save(rating_file + '.npy', rating_np)
    users, n_user = map_data(rating_np[:, 0])
    items, n_item = map_data(rating_np[:, 1])

    rating_np[:, 0] = users
    rating_np[:, 1] = items

    n_entity, n_relation, kg, edge_lists = load_kg(args)
    num_nodes = n_user + n_entity
    rating_np[:, 0] += n_entity
    
    train_all, train_fold, meta_fold, valid_target, test_target = dataset_split(args, rating_np)
    rating = np.asarray(train_all)
    item_cnt = Counter(rating[:, 1]).most_common(args.num_hub)
    hub = [i[0] for i in item_cnt]

    return train_all, train_fold, meta_fold, valid_target, test_target, kg, hub, num_nodes, n_relation, n_user, n_entity, n_item
   
def load_kg(args):
    print('reading kg file ...')

    kg_file = '../data/{}/kg_final'.format(args.dataset)
    if os.path.exists(kg_file + '.npy'):
        kg_np = np.load(kg_file + '.npy')
    else:
        kg_np = np.loadtxt(kg_file + '.txt', dtype=np.int64)
        np.save(kg_file + '.npy', kg_np)

    kg = np.concatenate((kg_np[:,[0,2]], kg_np[:,[2,0]]))
    kg_type = np.concatenate((2*kg_np[:,1], 2*kg_np[:,1]+1))
    
    n_entity = len(set(kg_np[:, 0]) | set(kg_np[:, 2]))
    n_relation = 2*len(set(kg_np[:, 1]))
    
    kg_type = kg_type.reshape(-1,1)
    kgg = np.concatenate((kg, kg_type), 1)
    edges = construct_adj(kg_np, n_relation)

    return n_entity, n_relation, kgg, edges

def construct_adj(kg, relation_num):
    edges_lists = [[] for i in range(relation_num)]
    for triple in kg:
        head = triple[0]
        relation = triple[1]
        tail = triple[2]
        # construct edge list for each relation type (head->tail, tail->head)
        edges_lists[2*relation].append([head,tail])
        edges_lists[2*relation+1].append([tail,head])
    for i in range(len(edges_lists)):
        edges_lists[i] = np.array(edges_lists[i]).transpose()
    return edges_lists

def dataset_split(args, rating_np):

    test_ratio = 0.2
    valid_ratio = 0.2

    n_ratings = rating_np.shape[0]
    test_indices = np.random.choice(list(range(n_ratings)), size=int(n_ratings * test_ratio), replace=False)
    left = set(range(n_ratings)) - set(test_indices)
    valid_indices = np.random.choice(list(left), size=int(n_ratings * valid_ratio), replace=False)
    train_indices = list(left - set(valid_indices))
    X = rating_np[train_indices] # train_data
    test_data = torch.from_numpy(rating_np[test_indices])
    valid_data = torch.from_numpy(rating_np[valid_indices])

    train_fold, meta_fold = [], []
    kf = KFold(n_splits=args.n_fold, shuffle=True)
    for train_idx, meta_idx in kf.split(X):
        train = X[train_idx]
        meta = X[meta_idx]
        train_fold.append(torch.from_numpy(train))
        meta_fold.append(torch.from_numpy(meta))
    train_all = torch.from_numpy(X)
    return train_all, train_fold, meta_fold, valid_data, test_data
