import argparse
import time
import torch

from torch_geometric import EdgeIndex
from torch_geometric.utils import spmm, spmm_scatter

parser = argparse.ArgumentParser()
parser.add_argument('--num_nodes', type=int, default=1000)
parser.add_argument('--num_edges', type=int, default=5000)
parser.add_argument('--hidden', type=int, default=64)
parser.add_argument('--runs', type=int, default=10)
args = parser.parse_args()

row = torch.randint(args.num_nodes, (args.num_edges,))
col = torch.randint(args.num_nodes, (args.num_edges,))
edge_index = EdgeIndex(torch.stack([row, col]))
x = torch.randn(args.num_nodes, args.hidden)

for name, func in [('default', spmm), ('scatter', spmm_scatter)]:
    times = []
    for _ in range(args.runs):
        start = time.time()
        out = func(edge_index, x)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        times.append(time.time() - start)
    print(f'{name}: {sum(times) / len(times):.6f}s')
