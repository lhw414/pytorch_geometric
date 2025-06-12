import torch

from torch_geometric import EdgeIndex
from torch_geometric.testing import withCUDA
from torch_geometric.utils import spmm, spmm_scatter


@withCUDA
def test_spmm_scatter_equivalence(device):
    src = EdgeIndex(
        [[0, 1, 1, 2], [1, 0, 2, 1]],
        sparse_size=(4, 3),
        sort_order='row',
        device=device,
    )
    other = torch.randn(3, 4, device=device)
    out1 = spmm(src.flip(0), other)
    out2 = spmm_scatter(src.flip(0), other)
    assert torch.allclose(out1, out2, atol=1e-6)
