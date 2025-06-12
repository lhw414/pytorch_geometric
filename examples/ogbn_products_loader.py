import argparse
import os.path as osp
import time

import torch
import torch.nn.functional as F
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

from torch_geometric import device
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import GCNConv
from torch_geometric.utils import spmm_scatter, spmm

parser = argparse.ArgumentParser()
parser.add_argument('--hidden_channels', type=int, default=64)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--spmm', type=str, default='default', choices=['default', 'scatter'])
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = PygNodePropPredDataset('ogbn-products', root=osp.join('data', 'OGB'))
data = dataset[0]
split_idx = dataset.get_idx_split()

train_loader = NeighborLoader(
    data,
    num_neighbors=[10, 10],
    input_nodes=split_idx['train'],
    batch_size=args.batch_size,
    shuffle=True,
)


eval_loader = NeighborLoader(
    data,
    num_neighbors=[10, 10],
    input_nodes=torch.cat([split_idx['valid'], split_idx['test']]),
    batch_size=args.batch_size,
    shuffle=False,
)


class CustomGCNConv(GCNConv):
    def message_and_aggregate(self, adj_t, x):
        if args.spmm == 'scatter':
            return spmm_scatter(adj_t, x, reduce=self.aggr)
        return spmm(adj_t, x, reduce=self.aggr)


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        Conv = CustomGCNConv
        self.conv1 = Conv(in_channels, hidden_channels)
        self.conv2 = Conv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x


def train(loader):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index)
        loss = F.cross_entropy(out, batch.y.squeeze())
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * batch.num_nodes
    return total_loss / len(loader.dataset)


def test(loader):
    model.eval()
    correct = 0
    evaluator = Evaluator('ogbn-products')
    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index)
        pred = out.argmax(dim=-1, keepdim=True)
        correct += evaluator.eval({'y_true': batch.y, 'y_pred': pred})['acc'] * batch.num_nodes
    return correct / len(loader.dataset)


model = GCN(dataset.num_features, args.hidden_channels, dataset.num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

for epoch in range(1, args.epochs + 1):
    start = time.time()
    loss = train(train_loader)
    acc = test(eval_loader)
    print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Acc: {acc:.4f}, Time: {time.time()-start:.2f}s')
