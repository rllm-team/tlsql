"""Utility functions for graph construction

This module contains utility functions for building graphs from relational data.
"""

import pandas as pd
import torch

from rllm.data import GraphData


def build_homo_graph(
    relation_df: pd.DataFrame,
    n_all: int,
    x: torch.Tensor = None,
    y: torch.Tensor = None,
):
    """Build a simple undirected and unweighted homogeneous graph"""
    src_nodes, tgt_nodes = torch.from_numpy(relation_df.iloc[:, :2].values).t()
    indices = torch.cat(
        [
            torch.stack([src_nodes, tgt_nodes], dim=0),
            torch.stack([tgt_nodes, src_nodes], dim=0),
        ],
        dim=1,
    )

    values = torch.ones((indices.shape[1],), dtype=torch.float32)
    adj = torch.sparse_coo_tensor(indices, values, (n_all, n_all))

    graph = GraphData(x=x, y=y, adj=adj)
    return graph
