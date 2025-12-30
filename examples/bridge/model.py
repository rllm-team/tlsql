"""BRIDGE Model Construction and Graph Utilities
"""

import pandas as pd
import torch

from rllm.data import GraphData
from rllm.nn.conv.graph_conv import GCNConv
from rllm.nn.conv.table_conv import TabTransformerConv
from rllm.nn.models import BRIDGE, TableEncoder, GraphEncoder


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


def build_bridge_model(num_classes, metadata, emb_size):
    """Build BRIDGE model
    Args:
        num_classes: Number of output classes
        metadata: Table metadata for TabTransformer
        emb_size: Embedding size
    Returns:
        BRIDGE model instance
    """
    t_encoder = TableEncoder(
        in_dim=emb_size,
        out_dim=emb_size,
        table_conv=TabTransformerConv,
        metadata=metadata,
    )
    g_encoder = GraphEncoder(
        in_dim=emb_size,
        out_dim=num_classes,
        graph_conv=GCNConv,
    )
    model = BRIDGE(
        table_encoder=t_encoder,
        graph_encoder=g_encoder,
    )
    return model
