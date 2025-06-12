import torch
from torch import Tensor

from torch_geometric import EdgeIndex
from torch_geometric.typing import Adj
from torch_geometric.edge_index import _scatter_spmm


def spmm_scatter(src: Adj, other: Tensor, reduce: str = 'sum') -> Tensor:
    """Sparse-dense matrix multiplication via scatter reduce.

    This utilizes a lightweight scatter-based implementation and can be used
    as an alternative to :func:`torch_geometric.utils.spmm` for benchmarking
    purposes.
    Currently, only :class:`~torch_geometric.EdgeIndex` inputs are supported.
    """
    if isinstance(src, EdgeIndex):
        return _scatter_spmm(src, other, None, reduce, False)
    raise NotImplementedError("Only 'EdgeIndex' inputs are supported")
