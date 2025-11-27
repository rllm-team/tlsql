"""Utility functions for graph construction 

This module contains utility functions for building graphs from relational data.
"""

from typing import Optional

import pandas as pd
import numpy as np
import torch
from torch import Tensor

from rllm.data import GraphData


def reorder_ids(
    relation_df: pd.DataFrame,
    src_col_name: str,
    tgt_col_name: str,
    n_src: int,
) -> pd.DataFrame:
    """Reorder IDs in the relationship DataFrame 
    
    Reorders the IDs in the relationship DataFrame by adjusting the
    original source IDs and target column IDs.
    
    Args:
        relation_df: DataFrame containing the relationships 
        src_col_name: Name of the source column in the DataFrame 
        tgt_col_name: Name of the target column in the DataFrame 
        n_src: Number of source nodes 
        
    Returns:
        DataFrame with reordered IDs 
    """
    # Making relationship 
    ordered_rating = relation_df.assign(
        **{
            src_col_name: relation_df[src_col_name] - 1,
            tgt_col_name: relation_df[tgt_col_name] + n_src - 1,
        }
    )

    return ordered_rating


def build_homo_graph(
    relation_df: pd.DataFrame,
    n_all: int,
    x: Tensor = None,
    y: Tensor = None,
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





