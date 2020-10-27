import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GINConv
from torch.autograd import Variable
import numpy as np
from base_model import *
from inits import glorot, zeros
import pdb

# adopted from: https://github.com/xjtushujun/meta-weight-net 
def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)

class MetaModule(nn.Module):
    # adopted from: Adrien Ecoffet https://github.com/AdrienLE
    def params(self):
        for name, param in self.named_params(self):
            yield param

    def named_leaves(self):
        return []

    def named_submodules(self):
        return []

    def named_params(self, curr_module=None, memo=None, prefix=''):
        if memo is None:
            memo = set()

        if hasattr(curr_module, 'named_leaves'):
            for name, p in curr_module.named_leaves():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p
        else:
            for name, p in curr_module._parameters.items():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p

        for mname, module in curr_module.named_children():
            submodule_prefix = prefix + ('.' if prefix else '') + mname
            for name, p in self.named_params(module, memo, submodule_prefix):
                yield name, p

    def update_params(self, lr_inner, first_order=False, source_params=None, detach=False):
        if source_params is not None:
            for tgt, src in zip(self.named_params(self), source_params):
                name_t, param_t = tgt
                grad = src
                if first_order:
                    grad = to_var(grad.detach().data)
                tmp = param_t - lr_inner * grad
                self.set_param(self, name_t, tmp)
        else:

            for name, param in self.named_params(self):
                if not detach:
                    grad = param.grad
                    if first_order:
                        grad = to_var(grad.detach().data)
                    tmp = param - lr_inner * grad
                    self.set_param(self, name, tmp)
                else:
                    param = param.detach_()  # https://blog.csdn.net/qq_39709535/article/details/81866686
                    self.set_param(self, name, param)

    def set_param(self, curr_mod, name, param):
        if '.' in name:
            n = name.split('.')
            module_name = n[0]
            rest = '.'.join(n[1:])
            for name, mod in curr_mod.named_children():
                if module_name == name:
                    self.set_param(mod, rest, param)
                    break
        else:
            setattr(curr_mod, name, param)

    def detach_params(self):
        for name, param in self.named_params(self):
            self.set_param(self, name, param.detach())

    def copy(self, other, same_var=False):
        for name, param in other.named_params():
            if not same_var:
                param = to_var(param.data.clone(), requires_grad=True)
            self.set_param(name, param)


class MetaEmb(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.Embedding(*args, **kwargs)
        nn.init.xavier_normal_(ignore.weight)
        self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))

    def forward(self, x):
        return F.embedding(x, self.weight)

    def named_leaves(self):
        return [('weight', self.weight)]

class MetaGCN(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = GCNConv(*args, **kwargs)
        self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))
        self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))
        self.gcn = [GCN(self.weight.shape[0], self.weight.shape[1], self.weight, self.bias)]

    def forward(self, n_id, edge_index):
        return self.gcn[0](n_id, edge_index)

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]

class MetaGAT(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = GATConv(*args, **kwargs)
        self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))
        self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))
        self.register_buffer('att', to_var(ignore.att.data, requires_grad=True))

        self.gat= [GAT(ignore.weight.shape[0], ignore.weight.shape[1], self.weight, self.att, self.bias)]

    def forward(self, n_id, edge_index):
        return self.gat[0](n_id, edge_index)

    def named_leaves(self):
        return [('weight', self.weight), ('att', self.att), ('bias', self.bias)]

class MetaGIN(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.Linear(*args, **kwargs)
        self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))
        self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))

        ignore2 = nn.Linear(*args, **kwargs)
        self.register_buffer('weight2', to_var(ignore2.weight.data, requires_grad=True))
        self.register_buffer('bias2', to_var(ignore2.bias.data, requires_grad=True))
        
        self.gin = [GIN(self.weight, self.bias, self.weight2, self.bias2)]

    def forward(self, n_id, edge_index):
        return self.gin[0](n_id, edge_index)

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias), ('weight2', self.weight2), ('bias2', self.bias2)]

class MetaSGConv(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.Linear(*args, **kwargs)
        self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))
        self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))
        self.sgc = [SGC(self.weight, self.bias, K=2)]

    def forward(self, n_id, edge_index):
        return self.sgc[0](n_id, edge_index)

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]

class MetaNetworkGCN(MetaModule):
    def __init__(self, args, num_node):
        super(MetaNetworkGCN, self).__init__()
        self.node_emb = MetaEmb(num_node, args.emb_dim)

        conv_layers = []
        for _ in range(args.num_layers):
            conv_layers.append(MetaGCN(args.emb_dim, args.emb_dim))
        self.conv_layers = nn.ModuleList(conv_layers)

    def forward(self, n_id, edge_index):

        for i, conv in enumerate(self.conv_layers):
            if i == 0:
                x = conv(self.node_emb(n_id), edge_index.detach())
            else:
                x = F.relu(x)
                x = conv(x, edge_index.detach())
        return x

class MetaNetworkGAT(MetaModule):
    def __init__(self, args, num_node):
        super(MetaNetworkGAT, self).__init__()
        self.node_emb = MetaEmb(num_node, args.emb_dim)

        conv_layers = []
        for _ in range(args.num_layers):
            conv_layers.append(MetaGAT(args.emb_dim, args.emb_dim))
        self.conv_layers = nn.ModuleList(conv_layers)

    def forward(self, n_id, edge_index):

        for i, conv in enumerate(self.conv_layers):
            if i == 0:
                x = conv(self.node_emb(n_id), edge_index)
            else:
                x = F.relu(x)
                x = conv(x, edge_index)
        return x

class MetaNetworkGIN(MetaModule):
    def __init__(self, args, num_node):
        super(MetaNetworkGIN, self).__init__()
        self.node_emb = MetaEmb(num_node, args.emb_dim)

        conv_layers = []
        for _ in range(args.num_layers):
            conv_layers.append(MetaGIN(args.emb_dim, args.emb_dim))
        self.conv_layers = nn.ModuleList(conv_layers)

    def forward(self, n_id, edge_index):
        for i, conv in enumerate(self.conv_layers):
            if i == 0:
                x = conv(self.node_emb(n_id), edge_index)
            else:
                x = F.relu(x)
                x = conv(x, edge_index)
        return x

class MetaNetworkSGC(MetaModule):
    def __init__(self, args, num_node):
        super(MetaNetworkSGC, self).__init__()
        self.node_emb = MetaEmb(num_node, args.emb_dim)
        conv_layers = []
        for _ in range(args.num_layers):
            conv_layers.append(MetaSGConv(args.emb_dim, args.emb_dim))
        self.conv_layers = nn.ModuleList(conv_layers)

    def forward(self, n_id, edge_index):
        for i, conv in enumerate(self.conv_layers):
            if i == 0:
                x = conv(self.node_emb(n_id), edge_index)
            else:
                x = F.relu(x)
                x = conv(x, edge_index)
        return x

class MetaLinear(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.Linear(*args, **kwargs)

        self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))
        self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]

class Weight(MetaModule):
    def __init__(self, in_channel, hidden, out_channel):
        super(Weight, self).__init__()
        self.linear1 = MetaLinear(in_channel, hidden)
        self.linear2 = MetaLinear(hidden, out_channel)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.sigmoid(x)
        return x

class Phi(MetaModule):
    def __init__(self, in_channel, hidden, out_channel):
        super(Phi, self).__init__()
        
        self.linear1 = MetaLinear(in_channel, hidden)
        self.linear2 = MetaLinear(hidden, out_channel)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        return x

class MetaNetworkHUB(MetaModule):
    def __init__(self, args):
        super(MetaNetworkHUB, self).__init__()
        conv_layers = []
        for _ in range(args.num_layers):
            conv_layers.append(MetaGAT(args.emb_dim, args.emb_dim))
        self.conv_layers = nn.ModuleList(conv_layers)

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.conv_layers):
            if i == 0:
                x = conv(x, edge_index)
            else:
                x = F.relu(x)
                x = conv(x, edge_index)
        return x
