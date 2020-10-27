from __future__ import division
import warnings
import torch
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.utils import degree, segregate_self_loops
from torch_cluster import neighbor_sampler
import numpy as np
import pickle
import pdb

def size_repr(value):
    if torch.is_tensor(value):
        return list(value.size())
    elif isinstance(value, int) or isinstance(value, float):
        return [1]
    else:
        return value

class Block(object):
    def __init__(self, n_id, res_n_id, e_id, edge_index, size):
        self.n_id = n_id
        self.res_n_id = res_n_id
        self.e_id = e_id
        self.edge_index = edge_index
        self.size = size

    def __repr__(self):
        info = [(key, getattr(self, key)) for key in self.__dict__]
        info = ['{}={}'.format(key, size_repr(item)) for key, item in info]
        return '{}({})'.format(self.__class__.__name__, ', '.join(info))

class DataFlow(object):
    def __init__(self, n_id, flow='source_to_target'):
        self.n_id = n_id
        self.flow = flow
        self.__last_n_id__ = n_id
        self.blocks = []

    @property
    def batch_size(self):
        return self.n_id.size(0)

    def append(self, n_id, res_n_id, e_id, edge_index):
        i, j = (0, 1) if self.flow == 'target_to_source' else (1, 0)
        size = [None, None]
        size[i] = self.__last_n_id__.size(0)
        size[j] = n_id.size(0)
        block = Block(n_id, res_n_id, e_id, edge_index, tuple(size))
        self.blocks.append(block)
        self.__last_n_id__ = n_id

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, idx):
        return self.blocks[::-1][idx]

    def __iter__(self):
        for block in self.blocks[::-1]:
            yield block

    def to(self, device):
        for block in self.blocks:
            block.edge_index = block.edge_index.to(device)
        return self

    def __repr__(self):
        n_ids = [self.n_id] + [block.n_id for block in self.blocks]
        csep = '<-' if self.flow == 'source_to_target' else '->'
        info = sep.join([str(n_id.size(0)) for n_id in n_ids])
        return '{}({})'.format(self.__class__.__name__, info)

class NeighborSampler(object):
    def __init__(self, args, edge_index, num_nodes, size, num_hops, batch_size=1, shuffle=False,
                 drop_last=False, bipartite=True, add_self_loops=False,
                 flow='source_to_target'):
        self.model_name = args.model_name
        self.edge_index = edge_index[:2, :]
        self.edge_type = edge_index[2,:]
        self.use_hint = args.selarhint
        self.num_nodes = num_nodes
        self.size = size
        self.num_hops = num_hops
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.bipartite = bipartite
        self.add_self_loops = add_self_loops
        self.flow = flow
        self.e_id = torch.arange(self.edge_index.size(1))

        if bipartite and add_self_loops:
            tmp = segregate_self_loops(self.edge_index, self.e_id)
            self.edge_index, self.e_id, self.edge_index_loop = tmp[:3]
            self.e_id_loop = self.e_id.new_full((self.num_nodes,), -1)
            self.e_id_loop[tmp[2][0]] = tmp[3]

        assert flow in ['source_to_target', 'target_to_source']
        self.i, self.j = (0, 1) if flow == 'target_to_source' else (1, 0)

        edge_index_i, self.e_assoc = self.edge_index[self.i].sort()
        self.edge_index_j = self.edge_index[self.j, self.e_assoc]
        deg = degree(edge_index_i, self.num_nodes, dtype=torch.long)
        self.cumdeg = torch.cat([deg.new_zeros(1), deg.cumsum(0)])
        self.tmp = torch.empty(self.num_nodes, dtype=torch.long)

    def __get_batches__(self, subset, users, labels):
        r"""Returns a list of mini-batches from the initial nodes in
        :obj:`subset`."""
        if subset is None and not self.shuffle:
            subset = torch.arange(self.data.num_nodes, dtype=torch.long)
        elif subset is None and self.shuffle:
            subset = torch.randperm(self.data.num_nodes)
        else:
            if subset.dtype == torch.bool or subset.dtype == torch.uint8:
                subset = subset.nonzero().view(-1)

        subsets = torch.split(subset, self.batch_size)
        users = torch.split(users, self.batch_size)
        labels = torch.split(labels, self.batch_size)
        if self.drop_last and subsets[-1].size(0) < self.batch_size:
            subsets = subsets[:-1]
        assert len(subsets) > 0
        return subsets, users, labels

    def __produce_subgraph__(self, data, h_item):
        r"""Produces a :obj:`Data` object holding the subgraph data for a given
        mini-batch :obj:`b_id`."""

        b_id, u_id = data[:2]
        labels = data[2]

        n_ids = [torch.cat((b_id, u_id))]
        e_ids = []
        edge_indices = []
        edge_type_indices = []

        for l in range(self.num_hops):
            e_id = neighbor_sampler(n_ids[-1], self.cumdeg, self.size[l])
            n_id = self.edge_index_j.index_select(0, e_id)
            n_id = n_id.unique(sorted=False)
            n_ids.append(n_id)
            e_ids.append(self.e_assoc.index_select(0, e_id))

            edge_index = self.edge_index.index_select(1, e_ids[-1])
            edge_indices.append(edge_index[:2, :])
            edge_type = self.edge_type.index_select(0, e_ids[-1])
            edge_type_indices.append(edge_type)

        if self.use_hint == 'True':
            h_edge_list=[]
            h_n_id = []
            for hub in h_item:
                users = self.edge_index[0, self.edge_index[1,:] == hub]
                for urs in users:
                    if urs in n_id:
                        h_edge_list.append([urs.item(), hub.item()])
                        h_n_id.append(urs.item())
            n_ids.append(torch.tensor(h_n_id))
            n_ids.append(torch.tensor(h_item))

        n_id = torch.unique(torch.cat(n_ids, dim=0), sorted=False)
        self.tmp[n_id] = torch.arange(n_id.size(0)) 
        e_id = torch.cat(e_ids, dim=0)
        edge_index = self.tmp[torch.cat(edge_indices, dim=1)]
        
        num_nodes = n_id.size(0)
        idx = edge_index[0] * num_nodes + edge_index[1]
        idx, inv = idx.unique(sorted=False, return_inverse=True)
        edge_index = torch.stack([idx // num_nodes, idx % num_nodes], dim=0)

        if self.use_hint == 'True':
            h_edge_index = torch.cat((edge_index, self.tmp[torch.tensor(h_edge_list).T]), 1)
            idx = h_edge_index[0] * num_nodes + h_edge_index[1]
            idx, inv = idx.unique(sorted=False, return_inverse=True)
            h_edge_index = torch.stack([idx // num_nodes, idx % num_nodes], dim=0)
            return Data(edge_index=edge_index, h_edge_index=h_edge_index, n_id=n_id, target_items=self.tmp[b_id], target_users=self.tmp[u_id], labels=labels)
        else:
            return Data(edge_index=edge_index, n_id=n_id, target_items=self.tmp[b_id], target_users=self.tmp[u_id], labels=labels)

    def __call__(self, subset, hub):
        users = subset[:, 0]
        items = subset[:, 1]
        labels = subset[:, 2].float()
        items, users, labels = self.__get_batches__(items, users, labels)
        if self.use_hint == 'True':
            for data in zip(items, users, labels):
                yield self.__produce_subgraph__(data, hub)
        else:
            for data in zip(items, users, labels):
                yield self.__produce_subgraph__(data, None)

class eval_NeighborSampler(object):
    def __init__(self, args, edge_index, num_nodes, size, num_hops, batch_size=1, shuffle=False,
                 drop_last=False, bipartite=True, add_self_loops=False,
                 flow='source_to_target'):

        self.edge_index = edge_index[:2, :]
        self.edge_type = edge_index[2,:]
        self.model_name = args.model_name
        self.num_nodes = num_nodes
        self.size = size
        self.num_hops = num_hops
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.bipartite = bipartite
        self.add_self_loops = add_self_loops
        self.flow = flow
        self.e_id = torch.arange(self.edge_index.size(1))

        if bipartite and add_self_loops:
            tmp = segregate_self_loops(self.edge_index, self.e_id)
            self.edge_index, self.e_id, self.edge_index_loop = tmp[:3]
            self.e_id_loop = self.e_id.new_full((self.num_nodes,), -1)
            self.e_id_loop[tmp[2][0]] = tmp[3]

        assert flow in ['source_to_target', 'target_to_source']
        self.i, self.j = (0, 1) if flow == 'target_to_source' else (1, 0)

        edge_index_i, self.e_assoc = self.edge_index[self.i].sort()
        self.edge_index_j = self.edge_index[self.j, self.e_assoc]
        deg = degree(edge_index_i, self.num_nodes, dtype=torch.long)
        self.cumdeg = torch.cat([deg.new_zeros(1), deg.cumsum(0)])
        self.tmp = torch.empty(self.num_nodes, dtype=torch.long)

    def __get_batches__(self, subset, users, labels):
        r"""Returns a list of mini-batches from the initial nodes in
        :obj:`subset`."""
        if subset is None and not self.shuffle:
            subset = torch.arange(self.data.num_nodes, dtype=torch.long)
        elif subset is None and self.shuffle:
            subset = torch.randperm(self.data.num_nodes)
        else:
            if subset.dtype == torch.bool or subset.dtype == torch.uint8:
                subset = subset.nonzero().view(-1)

        subsets = torch.split(subset, self.batch_size)
        users = torch.split(users, self.batch_size)
        labels = torch.split(labels, self.batch_size)

        if self.drop_last and subsets[-1].size(0) < self.batch_size:
            subsets = subsets[:-1]
        assert len(subsets) > 0

        return subsets, users, labels

    def __produce_subgraph__(self, data):
        r"""Produces a :obj:`Data` object holding the subgraph data for a given
        mini-batch :obj:`b_id`."""

        b_id, u_id = data[:2]
        labels = data[2]

        n_ids = [torch.cat((b_id, u_id))]
        e_ids = []
        edge_indices = []
        edge_type_indices = []

        for l in range(self.num_hops):
            e_id = neighbor_sampler(n_ids[-1], self.cumdeg, self.size[l])
            n_id = self.edge_index_j.index_select(0, e_id)
            n_id = n_id.unique(sorted=False)
            n_ids.append(n_id)
            e_ids.append(self.e_assoc.index_select(0, e_id))

            edge_index = self.edge_index.index_select(1, e_ids[-1])
            edge_indices.append(edge_index[:2, :])
            edge_type = self.edge_type.index_select(0, e_ids[-1])
            edge_type_indices.append(edge_type)

        n_id = torch.unique(torch.cat(n_ids, dim=0), sorted=False)  # selected node = subgraph
        self.tmp[n_id] = torch.arange(n_id.size(0))  # renamed
        e_id = torch.cat(e_ids, dim=0)
        edge_index = self.tmp[torch.cat(edge_indices, dim=1)]  # re-indexing edge_index
        num_nodes = n_id.size(0)  # selected node size
        return Data(edge_index=edge_index, n_id=n_id,
                    target_items=self.tmp[b_id], target_users=self.tmp[u_id], labels=labels)

    def __call__(self, subset):
        users = subset[:, 0]
        items = subset[:, 1]
        labels = subset[:, 2].float()

        items, users, labels = self.__get_batches__(items, users, labels)
        for data in zip(items, users, labels):
            yield self.__produce_subgraph__(data)

class NoMeta_NeighborSampler(object):
    def __init__(self, args, edge_index, num_nodes, size, num_hops, batch_size=1, shuffle=False,
                 drop_last=False, bipartite=True, add_self_loops=False,
                 flow='source_to_target'):
        self.edge_index = edge_index[:2, :]
        self.edge_type = edge_index[2,:]
        self.model_name = args.model_name
        self.use_hint = args.selarhint
        self.num_nodes = num_nodes
        self.size = size
        self.num_hops = num_hops
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.bipartite = bipartite
        self.add_self_loops = add_self_loops
        self.flow = flow
        self.e_id = torch.arange(self.edge_index.size(1))

        if bipartite and add_self_loops:
            tmp = segregate_self_loops(self.edge_index, self.e_id)
            self.edge_index, self.e_id, self.edge_index_loop = tmp[:3]
            self.e_id_loop = self.e_id.new_full((self.num_nodes,), -1)
            self.e_id_loop[tmp[2][0]] = tmp[3]

        assert flow in ['source_to_target', 'target_to_source']
        self.i, self.j = (0, 1) if flow == 'target_to_source' else (1, 0)

        edge_index_i, self.e_assoc = self.edge_index[self.i].sort()
        self.edge_index_j = self.edge_index[self.j, self.e_assoc]
        deg = degree(edge_index_i, self.num_nodes, dtype=torch.long)
        self.cumdeg = torch.cat([deg.new_zeros(1), deg.cumsum(0)])
        self.tmp = torch.empty(self.num_nodes, dtype=torch.long)
    
    def __get_batches__(self, subset, users, labels, m_subset, m_user, m_label):
        r"""Returns a list of mini-batches from the initial nodes in
        :obj:`subset`."""
        if subset is None and not self.shuffle:
            subset = torch.arange(self.data.num_nodes, dtype=torch.long)
        elif subset is None and self.shuffle:
            subset = torch.randperm(self.data.num_nodes)
        else:
            if subset.dtype == torch.bool or subset.dtype == torch.uint8:
                subset = subset.nonzero().view(-1)

        subsets = torch.split(subset, self.batch_size)
        users = torch.split(users, self.batch_size)
        labels = torch.split(labels, self.batch_size)

        if self.drop_last and subsets[-1].size(0) < self.batch_size:
            subsets = subsets[:-1]
        assert len(subsets) > 0

        if m_subset is not None:
            temp = m_subset.shape[0] // len(subsets)
            mi = torch.split(m_subset, temp)
            mu = torch.split(m_user, temp)
            ml = torch.split(m_label, temp)
            return subsets, users, labels, mi, mu, ml

        return subsets, users, labels
    
    def __produce_subgraph__(self, data, h_item):
        r"""Produces a :obj:`Data` object holding the subgraph data for a given
        mini-batch :obj:`b_id`."""
        
        b_id, u_id = data[:2]
        labels = data[2]
        
        if len(data) == 6:
            mb_id, mu_id = data[3:5]
            m_labels = data[5]
            n_ids = [torch.cat((b_id, u_id, mb_id, mu_id))]
        else:
            n_ids = [torch.cat((b_id, u_id))]

        n_ids = [torch.from_numpy(np.unique(np.asarray(n_ids[0])))]
        e_ids = []
        edge_indices = []
        edge_type_indices = []

        for l in range(self.num_hops):
            e_id = neighbor_sampler(n_ids[-1], self.cumdeg, self.size[l])
            n_id = self.edge_index_j.index_select(0, e_id)
            n_id = n_id.unique(sorted=False)
            n_ids.append(n_id)
            e_ids.append(self.e_assoc.index_select(0, e_id))
            edge_index = self.edge_index.index_select(1, e_ids[-1])
            edge_indices.append(edge_index[:2, :])
            edge_type = self.edge_type.index_select(0, e_ids[-1])
            edge_type_indices.append(edge_type)

        if self.use_hint == 'True':
            h_edge_list, h_n_id = [], []
            for hub in h_item:
                users = self.edge_index[0, self.edge_index[1,:] == hub]
                for urs in users:
                    if urs in n_id:
                        h_edge_list.append([urs.item(), hub.item()])
                        h_n_id.append(urs.item())
            n_ids.append(torch.tensor(h_n_id))
            n_ids.append(torch.tensor(h_item))
        
        n_id = torch.unique(torch.cat(n_ids, dim=0), sorted=False)
        self.tmp[n_id] = torch.arange(n_id.size(0)) 
        e_id = torch.cat(e_ids, dim=0)
        edge_index = self.tmp[torch.cat(edge_indices, dim=1)]
        
        num_nodes = n_id.size(0)
        idx = edge_index[0] * num_nodes + edge_index[1]
        idx, inv = idx.unique(sorted=False, return_inverse=True)
        edge_index = torch.stack([idx // num_nodes, idx % num_nodes], dim=0)

        if self.use_hint == 'True':
            h_edge_index = torch.cat((edge_index, self.tmp[torch.tensor(h_edge_list).T]), 1)
            idx = h_edge_index[0] * num_nodes + h_edge_index[1]
            idx, inv = idx.unique(sorted=False, return_inverse=True)
            h_edge_index = torch.stack([idx // num_nodes, idx % num_nodes], dim=0)
            if len(data) == 6:   
                return Data(edge_index=edge_index, h_edge_index=h_edge_index, n_id=n_id,
                        target_items=self.tmp[b_id], target_users=self.tmp[u_id], labels=labels, m_target_items=self.tmp[mb_id], m_target_users=self.tmp[mu_id], m_labels=m_labels)
            else:
                return Data(edge_index=edge_index, h_edge_index=h_edge_index, n_id=n_id,
                        target_items=self.tmp[b_id], target_users=self.tmp[u_id], labels=labels)
        else:
            if len(data) == 6:
                return Data(edge_index=edge_index, n_id=n_id, target_items=self.tmp[b_id], target_users=self.tmp[u_id], labels=labels, m_target_items=self.tmp[mb_id], m_target_users=self.tmp[mu_id], m_labels=m_labels)
            else:
                return Data(edge_index=edge_index, n_id=n_id, target_items=self.tmp[b_id], target_users=self.tmp[u_id], labels=labels)
    
    def __call__(self, subset, m_subset, hub):
        users = subset[:, 0]
        items = subset[:, 1]
        labels = subset[:, 2].float()

        if m_subset is not None:
            m_users = m_subset[:, 0]
            m_items = m_subset[:, 1]
            m_labels = m_subset[:, 2].float()
            items, users, labels, m_items, m_users, m_labels= self.__get_batches__(items, users, labels, m_items, m_users, m_labels)
            Zip = [items] + [users] + [labels] + [m_items] + [m_users] + [m_labels]
        else:
            m_users, m_items, m_labels = None, None, None
            items, users, labels = self.__get_batches__(items, users, labels, m_items, m_users, m_labels)
            Zip = [items] + [users] + [labels]
        
        if self.use_hint == 'True':
            for data in zip(*Zip):
                yield self.__produce_subgraph__(data, hub)
        else:
            for data in zip(*Zip):
                yield self.__produce_subgraph__(data, None)


class Meta_NeighborSampler(object):
    def __init__(self, args, edge_index, num_nodes, size, num_hops, batch_size=1, shuffle=False,
                 drop_last=False, bipartite=True, add_self_loops=False,
                 flow='source_to_target'):
        self.model_name = args.model_name
        self.use_hint = args.selarhint
        self.n_meta = args.n_meta
        self.metapath = args.metapath
        self.dataset = args.dataset
        self.edge_index = edge_index[:2, :]
        self.edge_type = edge_index[2,:]
        self.num_nodes = num_nodes
        self.size = size
        self.num_hops = num_hops
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.bipartite = bipartite
        self.add_self_loops = add_self_loops
        self.flow = flow
        self.e_id = torch.arange(self.edge_index.size(1))

        if bipartite and add_self_loops:
            tmp = segregate_self_loops(self.edge_index, self.e_id)
            self.edge_index, self.e_id, self.edge_index_loop = tmp[:3]
            self.e_id_loop = self.e_id.new_full((self.num_nodes,), -1)
            self.e_id_loop[tmp[2][0]] = tmp[3]

        assert flow in ['source_to_target', 'target_to_source']
        self.i, self.j = (0, 1) if flow == 'target_to_source' else (1, 0)

        edge_index_i, self.e_assoc = self.edge_index[self.i].sort()
        self.edge_index_j = self.edge_index[self.j, self.e_assoc]
        self.edge_type = self.edge_type[self.e_assoc]
        deg = degree(edge_index_i, self.num_nodes, dtype=torch.long)
        self.cumdeg = torch.cat([deg.new_zeros(1), deg.cumsum(0)])
        self.tmp = torch.empty(self.num_nodes, dtype=torch.long)

        if 'music' in args.dataset:
            num_iter = 199
        elif 'book' in args.dataset:
            num_iter = 82

        self.meta1, self.meta2, self.n_meta1, self.n_meta2 = [],[],[],[]
        for i in self.metapath:
            idx = list(range(10000))
            for n, j in enumerate(idx):
                with open('../data/{}/meta_labels/pos_meta{}_{}.pickle'.format(self.dataset, i, j), 'rb') as f:
                    meta = pickle.load(f).T
                if n == 0:
                    metapath = meta
                else:
                    metapath = torch.cat((metapath, meta), 1)
                if (metapath.size(1) >= self.batch_size*num_iter) or (j == 9):
                    break
            self.meta1.append(metapath[0,:])
            self.meta2.append(metapath[1,:])                

        if args.dataset != 'book':
            for i in self.metapath:
                idx = list(range(10000))
                for n, j in enumerate(idx):
                    with open('../data/{}/meta_labels/neg_meta{}_{}.pickle'.format(self.dataset, i, j), 'rb') as f:
                        meta = pickle.load(f).T
                    if n == 0:
                        metapath = meta
                    else:
                        metapath = torch.cat((metapath, meta), 1)
                    if (metapath.size(1) >= self.batch_size*num_iter) or (j ==9):
                        break
                self.n_meta1.append(metapath[0,:])
                self.n_meta2.append(metapath[1,:])
        else:
            neg = []
            for i in range(self.n_meta):
                u_unique = torch.unique(self.meta1[i])
                i_unique = torch.unique(self.meta2[i])

                edge_set = set([str(self.meta1[i][j].item()) + "," + str(self.meta2[i][j].item()) for j in range(len(self.meta1[i]))])
                u_sample = np.random.choice(u_unique.tolist(), size=len(self.meta1[i])*10, replace=True)
                i_sample = np.random.choice(i_unique.tolist(), size=len(self.meta1[i])*10, replace=True)
                
                sampled_edge_set = set([])
                sampled_ind = []
                for k in range(len(self.meta1[i])*10):
                    node1 = u_sample[k].item()
                    node2 = i_sample[k].item()
                    edge_str = str(node1) +","+ str(node2)
                    if not edge_str in edge_set and not edge_str in sampled_edge_set and not node1 == node2:
                        sampled_edge_set.add(edge_str)
                        sampled_ind.append(k)
                    if len(sampled_ind) == len(self.meta1[i]):
                        break
                self.n_meta1.append(torch.from_numpy(u_sample[sampled_ind]))
                self.n_meta2.append(torch.from_numpy(i_sample[sampled_ind]))
                del sampled_edge_set
        del metapath
        del meta

    def __get_batches__(self, subset, users, labels, m_subset, m_user, m_label):
        r"""Returns a list of mini-batches from the initial nodes in
        :obj:`subset`."""

        subsets = torch.split(subset, self.batch_size)
        users = torch.split(users, self.batch_size)
        labels = torch.split(labels, self.batch_size)
        
        meta1 =self.meta1
        meta2 =self.meta2
        n_meta1 =self.n_meta1
        n_meta2 =self.n_meta2
        m1, m2, metalabel = [], [], []
        Zip1 = [self.meta1] + [self.meta2] + [self.n_meta1] + [self.n_meta2]
        for mi, mu, nmi, nmu in zip(*Zip1):
            if mi.size(0) > self.batch_size*len(subsets)//2:
                rand_idx = np.random.choice(list(range(mi.size(0))), size=self.batch_size*len(subsets)//2, replace=False)
                mi, mu = mi[rand_idx], mu[rand_idx]

            if 3*mi.size(0) < nmi.size(0):        
                neg_idx = np.random.choice(list(range(len(nmi))), size=3*mi.size(0), replace=False)
                nmi, nmu = nmi[neg_idx], nmu[neg_idx]

            ml = torch.ones(mi.size(0)).float()
            nml = torch.zeros(nmi.size(0)).float()

            item = torch.cat((mi, nmi))                          
            user = torch.cat((mu, nmu))                          
            label = torch.cat((ml, nml))                          
            perm = torch.randperm(len(item))

            temp = item.size(0) // len(subsets)
            mis = torch.split(item[perm], temp)
            mus = torch.split(user[perm], temp)
            mls = torch.split(label[perm], temp)
                                    
            if np.abs(len(mis)-len(subsets))!= 0:
                mis = mis[:-np.abs(len(mis) - len(subsets))]
                mus = mus[:-np.abs(len(mus) - len(subsets))]
                mls = mls[:-np.abs(len(mls) - len(subsets))]
            m1.append(mis)
            m2.append(mus)
            metalabel.append(mls)

        if m_subset is not None:
            temp = m_subset.shape[0] // len(subsets)
            m_subsets = torch.split(m_subset, temp)
            m_users = torch.split(m_user, temp)
            m_labels = torch.split(m_label, temp)
            return subsets, users, labels, m_subsets, m_users, m_labels, m1, m2, metalabel
        else:
            return subsets, users, labels, m1, m2, metalabel

    def __produce_subgraph__(self, data, h_item):
        r"""Produces a :obj:`Data` object holding the subgraph data for a given
        mini-batch :obj:`b_id`."""

        b_id, u_id = data[:2]
        labels = data[2]
        if len(data) == 6+3*self.n_meta:
            mb_id, mu_id = data[3:5]
            m_labels = data[5]
            meta_id = torch.cat(data[6:6+2*self.n_meta])
            meta_labels = data[6+2*self.n_meta:]
            n_ids = [torch.cat((b_id, u_id, mb_id, mu_id, meta_id))]

        else:
            meta_id = torch.cat(data[3:3+2*self.n_meta])
            meta_labels = data[3+2*self.n_meta:]
            n_ids = [torch.cat((b_id, u_id, meta_id))]

        n_ids = [torch.from_numpy(np.unique(np.asarray(n_ids[0])))]
        e_ids = []
        edge_indices = []
        edge_type_indices = []
        for l in range(self.num_hops):
            e_id = neighbor_sampler(n_ids[-1], self.cumdeg, self.size[l])
            n_id = self.edge_index_j.index_select(0, e_id)
            n_id = n_id.unique(sorted=False)
            n_ids.append(n_id)
            e_ids.append(self.e_assoc.index_select(0, e_id))

            edge_index = self.edge_index.index_select(1, e_ids[-1])
            edge_type = self.edge_type.index_select(0, e_ids[-1])
            edge_type_indices.append(edge_type)
            edge_indices.append(edge_index[:2, :])

        if self.use_hint == 'True':
            h_edge_list=[]
            h_n_id = []
            for hub in h_item:
                users = self.edge_index[0, self.edge_index[1,:] == hub]
                for urs in users:
                    if urs in n_id:
                        h_edge_list.append([urs.item(), hub.item()])
                        h_n_id.append(urs.item())
            n_ids.append(torch.tensor(h_n_id))
            n_ids.append(torch.tensor(h_item))

        n_id = torch.unique(torch.cat(n_ids, dim=0), sorted=False)
        self.tmp[n_id] = torch.arange(n_id.size(0))
        e_id = torch.cat(e_ids, dim=0)
        edge_index = self.tmp[torch.cat(edge_indices, dim=1)]
        
        num_nodes = n_id.size(0)
        idx = edge_index[0] * num_nodes + edge_index[1]
        idx, inv = idx.unique(sorted=False, return_inverse=True)
        edge_index = torch.stack([idx / num_nodes, idx % num_nodes], dim=0)

        if self.use_hint == 'True':
            h_edge_index = torch.cat((edge_index, self.tmp[torch.tensor(h_edge_list).T]), 1)
            idx = h_edge_index[0] * num_nodes + h_edge_index[1]
            idx, inv = idx.unique(sorted=False, return_inverse=True)
            h_edge_index = torch.stack([idx / num_nodes, idx % num_nodes], dim=0)
            if len(data) == 6+3*self.n_meta:
                return Data(edge_index=edge_index, h_edge_index=h_edge_index, n_id=n_id, target_items=self.tmp[b_id], target_users=self.tmp[u_id], labels=labels, m_target_items=self.tmp[mb_id], m_target_users=self.tmp[mu_id], m_labels=m_labels), self.tmp, data[6:6+2*self.n_meta], meta_labels
            else:
                return Data(edge_index=edge_index, h_edge_index=h_edge_index, n_id=n_id,
                    target_items=self.tmp[b_id], target_users=self.tmp[u_id], labels=labels), self.tmp, data[3:3+2*self.n_meta], meta_labels            
        else:
            if len(data) == 6+3*self.n_meta:
                return Data(edge_index=edge_index, n_id=n_id, target_items=self.tmp[b_id], target_users=self.tmp[u_id], labels=labels, m_target_items=self.tmp[mb_id], m_target_users=self.tmp[mu_id], m_labels=m_labels), self.tmp, data[6:6+2*self.n_meta], meta_labels
            else:
                return Data(edge_index=edge_index, n_id=n_id, target_items=self.tmp[b_id], target_users=self.tmp[u_id], labels=labels), self.tmp, data[3:3+2*self.n_meta], meta_labels            

    def __call__(self, subset, m_subset, hub):

        users = subset[:, 0]
        items = subset[:, 1]
        labels = subset[:, 2].float()

        if m_subset is not None:
            m_users = m_subset[:, 0]
            m_items = m_subset[:, 1]
            m_labels = m_subset[:, 2].float()

            items, users, labels, m_items, m_users, m_labels, meta_users1, meta_users2, meta_labels = self.__get_batches__(items, users, labels, m_items, m_users, m_labels)
            Zip = [items] + [users] + [labels] +  [m_items] + [m_users] + [m_labels] + meta_users1 + meta_users2 + meta_labels
        else:
            m_users, m_items, m_labels = None, None, None
            items, users, labels, meta_users1, meta_users2, meta_labels = self.__get_batches__(items, users, labels, m_items, m_users, m_labels)
            Zip = [items] + [users] + [labels] + meta_users1 + meta_users2 + meta_labels
        
        if self.use_hint == 'True':
            for data in zip(*Zip):
                yield self.__produce_subgraph__(data, hub)
        else:
            for data in zip(*Zip):
                yield self.__produce_subgraph__(data, None)