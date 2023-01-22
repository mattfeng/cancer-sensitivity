#!/usr/bin/env python
# coding: utf-8

import pickle
from datetime import datetime

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch

# dataset
class SensitivityDataset(Dataset):
    def __init__(self, sensitivity_data, cell_line_to_gene_expr, drug_to_features):
        # cell_line_to_gene_expr: dict<cosmic_id -> gene expr>
        # sensitivity_data: DataFrame<cosmic_id, drug_id, ln_ic50>
        # drug_to_features: dict<drug_id -> molecular features>
        
        self.sensitivity_data = sensitivity_data
        self.cell_line_to_gene_expr = cell_line_to_gene_expr
        self.drug_to_features = drug_to_features
        
    @staticmethod
    def collate(batch_list):
        gexpr_batch = torch.tensor([i[0][0] for i in batch_list])
        molgraphs = [i[0][1] for i in batch_list]
        molgraph_batch = Batch.from_data_list(molgraphs)
        targets = torch.tensor([i[1] for i in batch_list]).float()
        
        return (gexpr_batch, molgraph_batch), targets
        
    
    def __len__(self):
        return len(self.sensitivity_data)
    
    def __getitem__(self, idx):
        row = self.sensitivity_data.iloc[idx]
        cell_line = row["cosmic_id"]
        # make gene expr
        gexpr = self.cell_line_to_gene_expr[cell_line]
        
        # make drug
        drug_id = row["drug_id"]
        atoms = torch.tensor(self.drug_to_features[drug_id]["atoms"]).float()
        bonds = torch.tensor(self.drug_to_features[drug_id]["bonds"])
        mol_graph = Data(x=atoms, edge_index=bonds)
        
        
        ln_ic50 = row["ln_IC50"]
        return (gexpr, mol_graph), ln_ic50

# molecule encoder
class ResGCNConv(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        out = F.relu(self.conv1(x, edge_index))
        out = self.conv2(out, edge_index)
        out += x
        
        return F.relu(out)

class MoleculeEncoder(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        
        self.conv1 = ResGCNConv(in_features, 32, 32)
        self.conv2 = ResGCNConv(32, 32, 32)
        self.conv3 = ResGCNConv(32, 32, 32)
        self.lin = nn.Linear(32, out_features)
    
    def forward(self, x, edge_index, batch):
        # x is node features
        # edge_index is connectivity
        # batch assigns each node to its graph index
        
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.conv3(x, edge_index))
        
        # takes the average over all node embeddings
        # if we implement this ourselves, we need to account for batch
        # it's not automatic
        x = global_mean_pool(x, batch)
        x = F.dropout(x, training=self.training)
        x = self.lin(x)
        
        return x

# sensitivity model
class SensitivityPredictor(nn.Module):
    def __init__(self, mol_dim, num_genes):
        super().__init__()
        self.mol_dim = mol_dim
        self.num_genes = num_genes
        
        self.mol_encoder = MoleculeEncoder(1, mol_dim)
        self.lin1 = nn.Linear(mol_dim + num_genes, 1024)
        self.lin2 = nn.Linear(1024, 1024)
        self.lin3 = nn.Linear(1024, 1024)
        self.lin4 = nn.Linear(1024, 1)
    
    def forward(self, gexprs, molgraphs):
        molembed = self.mol_encoder(molgraphs.x, molgraphs.edge_index, molgraphs.batch)
        
        inputs = torch.cat((gexprs, molembed), dim=1)
        
        out = F.relu(self.lin1(inputs))
        out = F.relu(self.lin2(out))
        out = F.relu(self.lin3(out))
        out = self.lin4(out)
        
        return out

def training_loop(model, optimizer, num_epochs):
    for epoch in range(1, num_epochs + 1):
        total_loss = 0
        num_batches = 0
        
        for (gexprs, molgraphs), target in tqdm(train_loader):
            pred = model(gexprs, molgraphs).squeeze()
            
            optimizer.zero_grad()
            loss = F.mse_loss(pred, target)
            loss.backward()
            optimizer.step()
            
            num_batches += 1
            total_loss += loss.detach().item()
        
        avg_loss = total_loss / num_batches
        
        print(f"Epoch {epoch}, Loss {avg_loss}")
        

with open("./drug_response_with_cell_line.pkl", "rb") as fin:
    sensitivity_data = pickle.load(fin)
    
with open("./cell_line_gexpr.pkl", "rb") as fin:
    cell_line_to_gene_expr = pickle.load(fin)
    
with open("./drugid_to_molecular_graphs.pkl", "rb") as fin:
    drug_to_features = pickle.load(fin)
    
dataset = SensitivityDataset(sensitivity_data, cell_line_to_gene_expr, drug_to_features)
train_loader = DataLoader(dataset, batch_size=64, collate_fn=SensitivityDataset.collate, shuffle=True)
model = SensitivityPredictor(32, 17419)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
# 3e-5: 4.8
training_loop(model, optimizer, 100)
