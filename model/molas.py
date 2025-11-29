import torch
import torch.nn as nn
import torch.nn.functional as F

from model.molas_blocks import *
from torch.nn import Linear, ReLU, Dropout, Sequential, BatchNorm1d


class MolAS(nn.Module):
    def __init__(self, num_node_features=960, num_ligand_features=384, num_classes=8, 
                 num_res_blocks=4, res_dim=128, hidden_res_dim=256, dropout_rate=0.3,
                 norm_type='graph', aggr_type='attentional', num_heads_protein=1, num_heads_ligand=2):
        super(MolAS, self).__init__()
        self.num_res_blocks = num_res_blocks
        self.num_ligand_features = num_ligand_features

        self.pool1 = MultiAttentionAggr(num_node_features, num_heads=num_heads_protein)

        # Residual decoder
        self.Res_decoder = nn.ModuleDict()

        self.Res_decoder['res_projection'] = nn.Linear(num_node_features * num_heads_protein + num_ligand_features, res_dim)
            
        for i in range(num_res_blocks):
            self.Res_decoder[f'res_block{i+1}'] = ResidualBlock(res_dim, hidden_res_dim, dropout_rate=dropout_rate)

        self.regressor = nn.Linear(res_dim * (1 + num_res_blocks), num_classes)

    def forward(self, ligand_data, protein_data, return_features=False):

        # Graph embeddings
        x1 = self.pool1(protein_data.x, protein_data.batch)
        # x2 = self.pool2(ligand_data.x, ligand_data.batch)
        x2 = ligand_data.x.view(-1, self.num_ligand_features)

        if return_features:
            return torch.cat((x1, x2), dim=1)
        
        x = self.Res_decoder['res_projection'](torch.cat((x1, x2), dim=1))
            
        # Apply residual blocks (no checkpointing to avoid determinism issues)
        res_outputs = [x]
        for i in range(self.num_res_blocks):
            res_out = self.Res_decoder[f'res_block{i+1}'](res_outputs[-1])
            res_outputs.append(res_out)

        x = torch.cat(res_outputs, dim=1)

        return self.regressor(x)
    
class MolAS_GCN_GAT_GINE(nn.Module):
    def __init__(self, num_node_features=960, num_ligand_features=384, num_classes=8, 
                 num_res_blocks=4, res_dim=128, hidden_res_dim=256, dropout_rate=0.3,
                 norm_type='graph', aggr_type='attentional', num_heads_protein=1, num_heads_ligand=2):
        super(MolAS_GCN_GAT_GINE, self).__init__()
        self.num_res_blocks = num_res_blocks
        self.num_ligand_features = num_ligand_features

        # GNN encoders
        self.GCN_encoder = GCN(num_node_features=num_node_features, num_layers=3, hidden_dim=128, 
                               norm_type=norm_type, aggr_type=aggr_type)
        self.GAT_encoder = GAT(num_node_features=num_node_features, num_layers=3, hidden_dims=[128, 64, 32], heads=4,
                             norm_type=norm_type, aggr_type=aggr_type)
        self.GIN_encoder = GIN(num_node_features=num_node_features, num_layers=3, hidden_dims=[128, 64, 32], 
                             norm_type=norm_type, aggr_type=aggr_type)

        out_dim = self.GCN_encoder.out_dim + self.GAT_encoder.out_dim + self.GIN_encoder.out_dim + self.num_ligand_features

        # Residual decoder
        self.Res_decoder = nn.ModuleDict()

        self.Res_decoder['res_projection'] = nn.Linear(out_dim, res_dim)
            
        for i in range(num_res_blocks):
            self.Res_decoder[f'res_block{i+1}'] = ResidualBlock(res_dim, hidden_res_dim, dropout_rate=dropout_rate)

        self.regressor = nn.Linear(res_dim * (1 + num_res_blocks), num_classes)

    def forward(self, ligand_data, protein_data, return_features=False):

        # Graph embeddings
        x_gcn = F.normalize(self.GCN_encoder(protein_data), dim=-1)
        x_gat = F.normalize(self.GAT_encoder(protein_data), dim=-1)
        x_gin = F.normalize(self.GIN_encoder(protein_data), dim=-1)
        # x2 = self.pool2(ligand_data.x, ligand_data.batch)
        x_ligand = ligand_data.x.view(-1, self.num_ligand_features)

        if return_features:
            return torch.cat((x_gcn, x_gat, x_gin, x_ligand), dim=1)
        
        x = self.Res_decoder['res_projection'](torch.cat((x_gcn, x_gat, x_gin, x_ligand), dim=1))
            
        # Apply residual blocks (no checkpointing to avoid determinism issues)
        res_outputs = [x]
        for i in range(self.num_res_blocks):
            res_out = self.Res_decoder[f'res_block{i+1}'](res_outputs[-1])
            res_outputs.append(res_out)

        x = torch.cat(res_outputs, dim=1)

        return self.regressor(x)
    

class MolAS_EGNN_GAT_GINE(nn.Module):
    def __init__(self, num_node_features=960, num_ligand_features=384, num_classes=8, 
                 num_res_blocks=4, res_dim=128, hidden_res_dim=256, dropout_rate=0.3,
                 norm_type='graph', aggr_type='attentional', num_heads_protein=1, num_heads_ligand=2):
        super(MolAS_EGNN_GAT_GINE, self).__init__()
        self.num_res_blocks = num_res_blocks
        self.num_ligand_features = num_ligand_features

        # GNN encoders
        self.EGNN_encoder = EGNN(num_node_features=num_node_features, num_layers=3, hidden_dim=128, 
                               norm_type=norm_type, aggr_type=aggr_type)
        self.GAT_encoder = GAT(num_node_features=num_node_features, num_layers=3, hidden_dims=[128, 64, 32], heads=4,
                             norm_type=norm_type, aggr_type=aggr_type)
        self.GIN_encoder = GIN(num_node_features=num_node_features, num_layers=3, hidden_dims=[128, 64, 32], 
                             norm_type=norm_type, aggr_type=aggr_type)

        out_dim = self.EGNN_encoder.out_dim + self.GAT_encoder.out_dim + self.GIN_encoder.out_dim + self.num_ligand_features

        # Residual decoder
        self.Res_decoder = nn.ModuleDict()

        self.Res_decoder['res_projection'] = nn.Linear(out_dim, res_dim)
            
        for i in range(num_res_blocks):
            self.Res_decoder[f'res_block{i+1}'] = ResidualBlock(res_dim, hidden_res_dim, dropout_rate=dropout_rate)

        self.regressor = nn.Linear(res_dim * (1 + num_res_blocks), num_classes)

    def forward(self, ligand_data, protein_data, return_features=False):

        # Graph embeddings
        x_egnn = F.normalize(self.EGNN_encoder(protein_data), dim=-1)
        x_gat = F.normalize(self.GAT_encoder(protein_data), dim=-1)
        x_gin = F.normalize(self.GIN_encoder(protein_data), dim=-1)
        # x2 = self.pool2(ligand_data.x, ligand_data.batch)
        x_ligand = ligand_data.x.view(-1, self.num_ligand_features)

        if return_features:
            return torch.cat((x_egnn, x_gat, x_gin, x_ligand), dim=1)
        
        x = self.Res_decoder['res_projection'](torch.cat((x_egnn, x_gat, x_gin, x_ligand), dim=1))
            
        # Apply residual blocks (no checkpointing to avoid determinism issues)
        res_outputs = [x]
        for i in range(self.num_res_blocks):
            res_out = self.Res_decoder[f'res_block{i+1}'](res_outputs[-1])
            res_outputs.append(res_out)

        x = torch.cat(res_outputs, dim=1)

        return self.regressor(x)

class MolASGT(nn.Module):
    def __init__(self, num_node_features=960, num_ligand_features=384, num_classes=8, 
                 num_res_blocks=4, res_dim=128, hidden_res_dim=256, dropout_rate=0.3,
                 norm_type='graph', aggr_type='attentional', num_heads_protein=1, num_heads_ligand=2):
        super(MolASGT, self).__init__()
        self.num_res_blocks = num_res_blocks
        self.num_ligand_features = num_ligand_features

        self.gt = GT(num_node_features=num_node_features, num_layers=3, hidden_dims=[128, 64, 32],
                     norm_type=norm_type, aggr_type=aggr_type)
        
        out_dim = self.gt.out_dim + self.num_ligand_features

        # Residual decoder
        self.Res_decoder = nn.ModuleDict()

        self.Res_decoder['res_projection'] = nn.Linear(out_dim, res_dim)
            
        for i in range(num_res_blocks):
            self.Res_decoder[f'res_block{i+1}'] = ResidualBlock(res_dim, hidden_res_dim, dropout_rate=dropout_rate)

        self.regressor = nn.Linear(res_dim * (1 + num_res_blocks), num_classes)

    def forward(self, ligand_data, protein_data, return_features=False):

        # Graph embeddings
        x_gt = self.gt(protein_data)
        # x2 = self.pool2(ligand_data.x, ligand_data.batch)
        x2 = ligand_data.x.view(-1, self.num_ligand_features)

        if return_features:
            return torch.cat((x_gt, x2), dim=1)
        
        x = self.Res_decoder['res_projection'](torch.cat((x_gt, x2), dim=1))
            
        # Apply residual blocks (no checkpointing to avoid determinism issues)
        res_outputs = [x]
        for i in range(self.num_res_blocks):
            res_out = self.Res_decoder[f'res_block{i+1}'](res_outputs[-1])
            res_outputs.append(res_out)

        x = torch.cat(res_outputs, dim=1)

        return self.regressor(x)
    
class MolASMinimal(nn.Module):
    def __init__(self, num_node_features=1152, num_classes=8, num_ligand_features=384, 
                 num_res_blocks=3, res_dim=128, hidden_res_dim=256, dropout_rate=0.3, num_heads_protein=1, 
                 aggr_type='attentional'):
        super(MolASMinimal, self).__init__()
        self.num_res_blocks = num_res_blocks
        self.num_res_blocks = num_res_blocks
        self.num_ligand_features = num_ligand_features

        self.pool1 = MultiAttentionAggr(num_node_features, num_heads=num_heads_protein)

        self.processor = Sequential(
            Linear(num_node_features + num_ligand_features, res_dim),
            ReLU(),
            Dropout(dropout_rate),
            nn.Linear(res_dim, num_classes)
        )

    def forward(self, ligand_data, protein_data, return_features=False):

        x1 = self.pool1(protein_data.x, protein_data.batch)
        x2 = ligand_data.x.view(-1, self.num_ligand_features)

        if return_features:
            return torch.cat((x1, x2), dim=1)
        
        x = self.processor(torch.cat((x1, x2), dim=1))

        return x

class MCGNNASDock(nn.Module):
    def __init__(self, num_node_features=1152, num_node_features_ligand=25, num_classes=8, dropout_rate=0.3, input_dropout_rate=0.1):
        super(MCGNNASDock, self).__init__()
        self.model1 = GAT_L_NO_FIXED_OUTPUT_DIM(num_node_features=num_node_features_ligand, num_classes=num_classes)
        self.model2 = GCN_GAT_GINE_FIXED_OUT_DIM(num_node_features=num_node_features, num_classes=num_classes)

        """ No Residual connections
        self.combiner = nn.Sequential(
            # We move the concatenation to the combiner
            # 3 linear layers with dropout and ReLU activation
            nn.Linear(320 + 96, 256), # 320 + 96 -> 128 + 128
            nn.ReLU(), ## nn.ReLU,
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128), 
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64), 
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32), 
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 16), 
            nn.ReLU(), ## nn.Sigmoid
            nn.Dropout(dropout_rate),
            nn.Linear(16, num_classes),
            nn.Sigmoid()
        )
        """

        # MLP with residual connections
        # self.input_projection = nn.Linear(320 + 96, 256)  # Project to initial hidden size
            
        # Residual blocks
        # self.res_block1 = ResidualBlock(256, 512, dropout_rate)
        # self.fc1 = nn.Linear(256, 128)
        self.res_block20 = ResidualBlock(128, 256, dropout_rate=0.3)
        self.res_block21 = ResidualBlock(128, 256, dropout_rate=0.3)
        self.res_block22 = ResidualBlock(128, 256, dropout_rate=0.3)
        # Calculate dense connection input sizes
        # After 3 res blocks: 128 + 128 + 128 + 128 = 512
        self.fc2 = nn.Linear(512, 64)  # Updated from 128 to 512
        self.res_block30 = ResidualBlock(64, 128, dropout_rate=0.3)
        self.res_block31 = ResidualBlock(64, 128, dropout_rate=0.3)
        self.res_block32 = ResidualBlock(64, 128, dropout_rate=0.3)
        # After next 3 res blocks: 64 + 64 + 64 + 64 = 256  
        self.fc3 = nn.Linear(256, 32)  # Updated from 64 to 256

        # Add batch normalization
        # self.bn1 = nn.BatchNorm1d(256)
        # self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(32)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        # self.input_dropout = nn.Dropout(input_dropout_rate)
            
        # Final classifier
        self.classifier = nn.Linear(32, num_classes)

    """No Residual connections
    def forward(self, data1, data2):
        # Get predictions from both models
        output1 = self.model1(data1)
        output2 = self.model2(data2)

        # Concatenate the outputs from both models
        combined_output = torch.cat((output1, output2), dim=1)

        # Apply the combiner to the concatenated outputs
        final_output = self.combiner(combined_output)

        return final_output
    """
    def forward(self, data1, data2):
        # Get predictions from both models
        output1 = self.model1(data1)
        output2 = self.model2(data2)

        # Concatenate the outputs from both models
        combined_output = torch.cat((output1, output2), dim=1)
            
        # Apply residual blocks with dense connections
        x1 = self.res_block20(combined_output)
        x2 = self.res_block21(x1)
        x3 = self.res_block22(x2)
    
        # Dense connection: concatenate all previous block outputs
        dense_input = torch.cat((combined_output, x1, x2, x3), dim=1)
    
        x = self.fc2(dense_input)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout(x)
    
        x4 = self.res_block30(x)
        x5 = self.res_block31(x4)
        x6 = self.res_block32(x5)
    
        # Another dense connection
        dense_input2 = torch.cat((x, x4, x5, x6), dim=1)
    
        x = self.fc3(dense_input2)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Final classification
        final_output = self.classifier(x)
            
        return final_output