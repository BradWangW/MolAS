import torch
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Linear, ReLU, Dropout, Sequential, BatchNorm1d
from torch_geometric.nn import GCNConv, GINConv, GATv2Conv, TransformerConv, GATConv
from torch_geometric.nn.norm import GraphNorm, LayerNorm, BatchNorm, PairNorm, DiffGroupNorm
from torch_geometric.nn.pool import global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.nn.aggr import MultiAggregation, DeepSetsAggregation, GraphMultisetTransformer, AttentionalAggregation

class EGNN(nn.Module):
    def __init__(self, num_node_features=960, num_layers=3, hidden_dim=64, num_attn_heads=1,
                 norm_type='graph', aggr_type='attentional'):
        super(EGNN, self).__init__()
        self.num_layers = num_layers

        self.layers = nn.ModuleDict()
        
        if num_node_features != hidden_dim:
            self.layers['embedding_in'] = Linear(num_node_features, hidden_dim)

        for i in range(num_layers):
            self.layers[f'gl{i+1}'] = E_GCL(hidden_dim, hidden_dim, hidden_dim)
            self.layers[f'n{i+1}'] = construct_norm(norm_type, hidden_dim)

        self.layers['aggr'] = MultiAttentionAggr(hidden_dim, num_heads=num_attn_heads)
        self.out_dim = hidden_dim * num_attn_heads

    def forward(self, data):
        x, edge_index, pos, batch = data.x, data.edge_index, data.pos, data.batch

        if 'embedding_in' in self.layers:
            x = self.layers['embedding_in'](x)

        for i in range(self.num_layers):
            x, pos, _ = self.layers[f'gl{i+1}'](x=x, edge_index=edge_index, pos=pos)
            x = self.layers[f'n{i+1}'](x)
            x = F.relu(x)

        return self.layers['aggr'](x, batch)
    
class GCN(nn.Module):
    def __init__(self, num_node_features=960, num_layers=3, hidden_dim=128, num_attn_heads=1,
                 norm_type='graph', aggr_type='attentional'):
        super(GCN, self).__init__()
        self.num_layers = num_layers

        self.layers = nn.ModuleDict()
        
        if num_node_features != hidden_dim:
            self.layers['embedding_in'] = Linear(num_node_features, hidden_dim)

        for i in range(num_layers):
            self.layers[f'gl{i+1}'] = GCNConv(hidden_dim, hidden_dim)
            self.layers[f'n{i+1}'] = construct_norm(norm_type, hidden_dim)

        self.layers['aggr'] = MultiAttentionAggr(hidden_dim, num_heads=num_attn_heads)
        self.out_dim = hidden_dim * num_attn_heads

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        if 'embedding_in' in self.layers:
            x = self.layers['embedding_in'](x)

        for i in range(self.num_layers):
            x = self.layers[f'gl{i+1}'](x=x, edge_index=edge_index)
            x = self.layers[f'n{i+1}'](x)
            x = F.relu(x)

        return self.layers['aggr'](x, batch)
    
class GAT(nn.Module):
    def __init__(self, num_node_features=960, num_layers=3, hidden_dims=[128, 64, 32], num_attn_heads=1,
                 heads=4, norm_type='graph', aggr_type='attentional'):
        super(GAT, self).__init__()
        assert len(hidden_dims) == num_layers, "Length of hidden_dims must match num_layers"
        self.num_layers = num_layers

        self.layers = nn.ModuleDict()

        dims = [num_node_features] + [dim * heads for dim in hidden_dims]
        for i in range(num_layers):
            self.layers[f'gl{i+1}'] = GATv2Conv(dims[i], hidden_dims[i], heads=heads)
            self.layers[f'n{i+1}'] = construct_norm(norm_type, dims[i+1])

        self.layers['aggr'] = MultiAttentionAggr(dims[-1], num_heads=num_attn_heads)
        self.out_dim = dims[-1] * num_attn_heads

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        for i in range(self.num_layers):
            x = self.layers[f'gl{i+1}'](x=x, edge_index=edge_index)
            x = self.layers[f'n{i+1}'](x)
            x = F.relu(x)

        return self.layers['aggr'](x, batch)
    
class GIN(nn.Module):
    def __init__(self, num_node_features=960, num_layers=3, hidden_dims=[128, 64, 32], num_attn_heads=1,
                 norm_type='graph', aggr_type='attentional'):
        super(GIN, self).__init__()
        assert len(hidden_dims) == num_layers, "Length of hidden_dims must match num_layers"
        self.num_layers = num_layers

        self.layers = nn.ModuleDict()

        dims = [num_node_features] + hidden_dims
        for i in range(num_layers):
            nn_conv = Sequential(
                Linear(dims[i], dims[i+1]),
                ReLU(),
                Linear(dims[i+1], dims[i+1])
            )
            self.layers[f'gl{i+1}'] = GINConv(nn_conv)
            self.layers[f'n{i+1}'] = construct_norm(norm_type, dims[i+1])

        self.layers['aggr'] = MultiAttentionAggr(hidden_dims[-1], num_heads=num_attn_heads)
        self.out_dim = hidden_dims[-1] * num_attn_heads

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        for i in range(self.num_layers):
            x = self.layers[f'gl{i+1}'](x=x, edge_index=edge_index)
            x = self.layers[f'n{i+1}'](x)
            x = F.relu(x)

        return self.layers['aggr'](x, batch)
    
class GT(nn.Module):
    def __init__(self, num_node_features=960, num_layers=3, hidden_dims=[128, 64, 32], num_attn_heads=1,
                 heads=4, norm_type='graph', aggr_type='attentional'):
        super(GT, self).__init__()
        assert len(hidden_dims) == num_layers, "Length of hidden_dims must match num_layers"
        self.num_layers = num_layers

        self.layers = nn.ModuleDict()

        dims = [num_node_features] + [dim * heads for dim in hidden_dims]
        for i in range(num_layers):
            self.layers[f'gl{i+1}'] = TransformerConv(dims[i], hidden_dims[i], heads=heads)
            self.layers[f'n{i+1}'] = construct_norm(norm_type, dims[i+1])

        self.layers['aggr'] = MultiAttentionAggr(dims[-1], num_heads=num_attn_heads)
        self.out_dim = dims[-1] * num_attn_heads

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        for i in range(self.num_layers):
            x = self.layers[f'gl{i+1}'](x=x, edge_index=edge_index)
            x = self.layers[f'n{i+1}'](x)
            x = F.relu(x)

        return self.layers['aggr'](x, batch)

# Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_rate=0.3):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)  # Output same size as input for residual
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        residual = x
        
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.fc2(out)
        
        return self.relu(out + residual)  # Apply activation after residual addition
    
class MultiAttentionAggr(nn.Module):
    def __init__(self, in_dim, num_heads=4):
        super().__init__()
        self.heads = nn.ModuleList([construct_aggr('attentional', in_dim) for _ in range(num_heads)])
    def forward(self, x, batch):
        outs = [h(x, batch) for h in self.heads]
        return torch.cat(outs, dim=-1)
    
def construct_norm(norm_type, dim):
    norm_map = {
        'batch': BatchNorm,
        'layer': LayerNorm,
        'graph': GraphNorm
    }
    if norm_type in norm_map:
        return norm_map[norm_type](dim)
    elif norm_type == 'pair':
        return PairNorm()
    elif norm_type == 'diffgroup':
        return DiffGroupNorm(dim, 30)
    else:
        raise Exception('Unsupported norm type: %s' % norm_type)

def construct_aggr(aggr_type, dim):

    aggr_map = {
        'add': global_add_pool,
        'mean': global_mean_pool,
        'max': global_max_pool
    }

    if aggr_type in ['add', 'mean', 'max']:
        return aggr_map[aggr_type]
    elif aggr_type == 'multi':
        return MultiAggregation(['mean', 'max', 'min', 'std'])
    elif aggr_type == 'deepset':
        return DeepSetsAggregation(Linear(dim, dim), Linear(dim, dim))
    elif aggr_type == 'gmt':
        return GraphMultisetTransformer(channels=dim, k=4, num_encoder_blocks=1, heads=4, layer_norm=True, dropout=0.1)
    elif aggr_type == 'attentional':
        return AttentionalAggregation(gate_nn=Sequential(Linear(dim, min(128, dim)), ReLU(), Linear(min(128, dim), 1)))
    else:
        raise Exception('Unsupported aggr type: %s' % aggr_type)

class GCN_GAT_GINE_FIXED_OUT_DIM(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(GCN_GAT_GINE_FIXED_OUT_DIM, self).__init__()
        self.num_classes = num_classes
        # GCN-representation
        self.conv1 = GCNConv(num_node_features, 128, cached=False)
        self.bn01 = BatchNorm1d(128, track_running_stats=False)
        self.conv2 = GCNConv(128, 64, cached=False)
        self.bn02 = BatchNorm1d(64, track_running_stats=False)
        self.conv3 = GCNConv(64, 32, cached=False)
        self.bn03 = BatchNorm1d(32, track_running_stats=False)
        # GAT-representation
        self.gat1 = GATConv(num_node_features, 128, heads=8)
        self.bn11 = BatchNorm1d(128 * 8, track_running_stats=False)
        self.gat2 = GATConv(128 * 8, 64, heads=8)
        self.bn12 = BatchNorm1d(64 * 8, track_running_stats=False)
        self.gat3 = GATConv(64 * 8, 32, heads=8)
        self.bn13 = BatchNorm1d(32 * 8, track_running_stats=False)
        # GIN-representation
        fc_gin1 = Sequential(Linear(num_node_features, 128), ReLU(), Linear(128, 128))
        self.gin1 = GINConv(fc_gin1)
        self.bn21 = BatchNorm1d(128, track_running_stats=False)
        fc_gin2 = Sequential(Linear(128, 64), ReLU(), Linear(64, 64))
        self.gin2 = GINConv(fc_gin2)
        self.bn22 = BatchNorm1d(64, track_running_stats=False)
        fc_gin3 = Sequential(Linear(64, 32), ReLU(), Linear(32, 32))
        self.gin3 = GINConv(fc_gin3)
        self.bn23 = BatchNorm1d(32, track_running_stats=False)
        # Fully connected layers for concatenating outputs
        self.fc1 = Linear(32 + 32 * 8 + 32, 128)  # Adjusted input size to match concatenated size
        self.dropout1 = Dropout(p=0.2)
        self.res_block10 = ResidualBlock(128, 256, dropout_rate=0.3)
        self.res_block20 = ResidualBlock(128, 256, dropout_rate=0.3)
        self.res_block30 = ResidualBlock(128, 256, dropout_rate=0.3)
        self.res_block11 = ResidualBlock(128, 256, dropout_rate=0.3)
        self.res_block21 = ResidualBlock(128, 256, dropout_rate=0.3)
        self.res_block31 = ResidualBlock(128, 256, dropout_rate=0.3)
        # Dense connection: 128 + 128 + 128 + 128 + 128 + 128 + 128 = 896
        self.fc2 = Linear(896, 96)  # Updated from 128 to 896

    def forward(self, data):
        x, edge_index, edge_w, batch = data.x, data.edge_index, data.edge_attr, data.batch
        y = x
        z = x
        # GCN-representation
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn01(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn02(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.bn03(x)
        x = global_add_pool(x, batch)
        # GAT-representation
        y = F.relu(self.gat1(y, edge_index))
        y = self.bn11(y)
        y = F.relu(self.gat2(y, edge_index))
        y = self.bn12(y)
        y = F.relu(self.gat3(y, edge_index))
        y = self.bn13(y)
        y = global_add_pool(y, batch)
        # GIN-representation
        z = F.relu(self.gin1(z, edge_index))
        z = self.bn21(z)
        z = F.relu(self.gin2(z, edge_index))
        z = self.bn22(z)
        z = F.relu(self.gin3(z, edge_index))
        z = self.bn23(z)
        z = global_add_pool(z, batch)
        # Concatenating representations
        cr = torch.cat((x, y, z), 1)
        cr = F.relu(self.fc1(cr))
        cr = self.dropout1(cr)
        
        # Apply residual blocks and collect outputs for dense connections
        x1 = self.res_block10(cr)
        x2 = self.res_block11(x1)
        x3 = self.res_block20(x2)
        x4 = self.res_block21(x3)
        x5 = self.res_block30(x4)
        x6 = self.res_block31(x5)
        
        # Dense connection: concatenate all residual block outputs
        dense_features = torch.cat((cr, x1, x2, x3, x4, x5, x6), dim=1)
        
        cr = F.relu(self.fc2(dense_features))

        return cr.view(-1, 96)
     
class GAT_L_NO_FIXED_OUTPUT_DIM(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(GAT_L_NO_FIXED_OUTPUT_DIM, self).__init__()
        self.num_classes = num_classes
        # GAT-representation
        self.gat1 = GATConv(num_node_features, 128, heads=3)
        self.bn11 = BatchNorm1d(128 * 3, track_running_stats=False)
        self.gat2 = GATConv(128 * 3, 64, heads=3)
        self.bn12 = BatchNorm1d(64 * 3, track_running_stats=False)
        self.gat3 = GATConv(64 * 3, 32, heads=3)
        self.bn13 = BatchNorm1d(32 * 3, track_running_stats=False)
        # Fully connected layers for concatenating outputs
        self.fc1 = Linear(32 * 3, 64)  # Adjusted input size to match concatenated size
        self.dropout1 = Dropout(p=0.2)
        self.res_block1 = ResidualBlock(64, 128, dropout_rate=0.3)
        self.res_block2 = ResidualBlock(64, 128, dropout_rate=0.3)
        self.res_block3 = ResidualBlock(64, 128, dropout_rate=0.3)
        # Dense connection: 64 + 64 + 64 + 64 = 256
        self.fc2 = Linear(256, 32)  # Updated from 64 to 256

    def forward(self, data):
        x, edge_index, edge_w, batch = data.x, data.edge_index, data.edge_attr, data.batch
        y = x
        z = x
        # GAT-representation
        y = F.relu(self.gat1(y, edge_index, edge_w))
        y = self.bn11(y)
        y = F.relu(self.gat2(y, edge_index, edge_w))
        y = self.bn12(y)
        y = F.relu(self.gat3(y, edge_index, edge_w))
        y = self.bn13(y)
        y = global_add_pool(y, batch)
        # Concatenating representations
        cr = y
        cr = F.relu(self.fc1(cr))
        cr = self.dropout1(cr)
        
        # Apply residual blocks and collect outputs for dense connections
        x1 = self.res_block1(cr)
        x2 = self.res_block2(x1)
        x3 = self.res_block3(x2)
        
        # Dense connection: concatenate all residual block outputs
        dense_features = torch.cat((cr, x1, x2, x3), dim=1)
        
        cr = F.relu(self.fc2(dense_features))
        return cr.view(-1, 32)

class E_GCL(nn.Module):
    """
    E(n) Equivariant Convolutional Layer
    re
    """

    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0, act_fn=nn.SiLU(), residual=True, attention=True, normalize=True, coords_agg='mean', tanh=True):
        super(E_GCL, self).__init__()
        input_edge = input_nf * 2
        self.residual = residual
        self.attention = attention
        self.normalize = normalize
        self.coords_agg = coords_agg
        self.tanh = tanh
        self.epsilon = 1e-8
        edge_coords_nf = 1

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edge_coords_nf + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn)

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf))

        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

        coord_mlp = []
        coord_mlp.append(nn.Linear(hidden_nf, hidden_nf))
        coord_mlp.append(act_fn)
        coord_mlp.append(layer)
        if self.tanh:
            coord_mlp.append(nn.Tanh())
        self.coord_mlp = nn.Sequential(*coord_mlp)

        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid())

    def edge_model(self, source, target, radial, edge_attr):
        if edge_attr is None:  # Unused.
            out = torch.cat([source, target, radial], dim=1)
        else:
            out = torch.cat([source, target, radial, edge_attr], dim=1)
        out = self.edge_mlp(out)
        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val
        return out

    def node_model(self, x, edge_index, edge_attr, node_attr):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        out = self.node_mlp(agg)
        if self.residual:
            out = x + out
        return out, agg

    def coord_model(self, pos, edge_index, coord_diff, edge_feat):
        row, col = edge_index
        trans = coord_diff * self.coord_mlp(edge_feat)
        if self.coords_agg == 'sum':
            agg = unsorted_segment_sum(trans, row, num_segments=pos.size(0))
        elif self.coords_agg == 'mean':
            agg = unsorted_segment_mean(trans, row, num_segments=pos.size(0))
        else:
            raise Exception('Wrong coords_agg parameter' % self.coords_agg)
        pos += agg
        return pos

    def coord2radial(self, edge_index, pos):
        row, col = edge_index
        coord_diff = pos[row] - pos[col]
        radial = torch.sum(coord_diff**2, 1).unsqueeze(1)

        if self.normalize:
            norm = torch.sqrt(radial).detach() + self.epsilon
            coord_diff = coord_diff / norm

        return radial, coord_diff

    def forward(self, x, edge_index, pos, edge_attr=None, node_attr=None):
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, pos)

        edge_feat = self.edge_model(x[row], x[col], radial, edge_attr)
        pos = self.coord_model(pos, edge_index, coord_diff, edge_feat)
        x, agg = self.node_model(x, edge_index, edge_feat, node_attr)

        return x, pos, edge_attr
    
def unsorted_segment_sum(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    return result

def unsorted_segment_mean(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    count = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)

def get_edges(n_nodes):
    rows, cols = [], []
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                rows.append(i)
                cols.append(j)

    edges = [rows, cols]
    return edges

def get_edges_batch(n_nodes, batch_size):
    edges = get_edges(n_nodes)
    edge_attr = torch.ones(len(edges[0]) * batch_size, 1)
    edges = [torch.LongTensor(edges[0]), torch.LongTensor(edges[1])]
    if batch_size == 1:
        return edges, edge_attr
    elif batch_size > 1:
        rows, cols = [], []
        for i in range(batch_size):
            rows.append(edges[0] + n_nodes * i)
            cols.append(edges[1] + n_nodes * i)
        edges = [torch.cat(rows), torch.cat(cols)]
    return edges, edge_attr