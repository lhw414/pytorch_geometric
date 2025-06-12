import argparse
import os.path as osp
import time

import torch
import torch.nn.functional as F
import dgl
from dgl.nn import GraphConv

from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.logging import init_wandb, log

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Cora')
parser.add_argument('--hidden_channels', type=int, default=16)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--wandb', action='store_true')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
init_wandb(name=f'DGL-GCN-{args.dataset}', hidden_channels=args.hidden_channels,
           lr=args.lr, epochs=args.epochs, device=device)

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Planetoid')
dataset = Planetoid(path, args.dataset, transform=T.NormalizeFeatures())
data = dataset[0]

g = dgl.graph((data.edge_index[0], data.edge_index[1]), num_nodes=data.num_nodes)
features = data.x
labels = data.y
train_mask = data.train_mask
test_mask = data.test_mask
val_mask = data.val_mask

g = g.to(device)
features = features.to(device)
labels = labels.to(device)


class DGLGCN(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = GraphConv(in_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, out_dim)

    def forward(self, g, x):
        x = F.relu(self.conv1(g, x))
        x = self.conv2(g, x)
        return x


model = DGLGCN(dataset.num_features, args.hidden_channels, dataset.num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


def train():
    model.train()
    optimizer.zero_grad()
    out = model(g, features)
    loss = F.cross_entropy(out[train_mask], labels[train_mask])
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def test():
    model.eval()
    out = model(g, features)
    pred = out.argmax(dim=-1)
    accs = []
    for mask in [train_mask, val_mask, test_mask]:
        accs.append(int((pred[mask] == labels[mask]).sum()) / int(mask.sum()))
    return accs


times = []
best_val = test_acc = 0
for epoch in range(1, args.epochs + 1):
    start = time.time()
    loss = train()
    train_acc, val_acc, tmp_test = test()
    if val_acc > best_val:
        best_val = val_acc
        test_acc = tmp_test
    log(Epoch=epoch, Loss=loss, Train=train_acc, Val=val_acc, Test=test_acc)
    times.append(time.time() - start)

print(f'Median time per epoch: {torch.tensor(times).median():.4f}s')
