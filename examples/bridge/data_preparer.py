"""BRIDGE pipeline utilities for preparing data and training models."""

from typing import Dict, Optional

import torch
import pandas as pd

from tlsql.examples.bridge.model import build_homo_graph
from rllm.transforms.graph_transforms import GCNTransform
from rllm.transforms.table_transforms import TabTransformerTransform
from rllm.data import TableData
from rllm.types import ColType


def prepare_bridge_data(
    train_data: Dict[str, pd.DataFrame],
    validate_data: Optional[Dict[str, pd.DataFrame]],
    test_data: Optional[pd.DataFrame],
    target_column: str,
    device: torch.device = None,
):
    """Prepare data for BRIDGE model"""
    emb_size = 128  # Embedding dimension
    target_pkey = 'UserID'  # primary key

    table_name, col_name = target_column.split('.')

    train_df = train_data[table_name]
    validate_df = validate_data[table_name] if validate_data else None
    test_df = test_data if test_data is not None else None

    all_dfs = [df for df in [train_df, validate_df, test_df] if df is not None]
    common_cols = set.intersection(*[set(df.columns) for df in all_dfs])
    target_df = pd.concat([df[list(common_cols)] for df in all_dfs], ignore_index=True).set_index(target_pkey)

    col_types = {col: ColType.CATEGORICAL for col in target_df.columns}
    target_table = TableData(df=target_df, col_types=col_types, target_col=col_name, pkey=target_pkey)

    n = len(target_table)
    train_len, val_len = len(train_df), len(validate_df) if validate_df is not None else 0
    test_len = len(test_df) if test_df is not None else 0

    train_mask = torch.cat([
        torch.ones(train_len, dtype=torch.bool),
        torch.zeros(val_len + test_len, dtype=torch.bool)
    ])
    val_mask = torch.cat([
        torch.zeros(train_len, dtype=torch.bool),
        torch.ones(val_len, dtype=torch.bool),
        torch.zeros(test_len, dtype=torch.bool)
    ]) if val_len > 0 else torch.zeros(n, dtype=torch.bool)
    test_mask = torch.cat([
        torch.zeros(train_len + val_len, dtype=torch.bool),
        torch.ones(test_len, dtype=torch.bool)
    ]) if test_len > 0 else torch.zeros(n, dtype=torch.bool)

    encoded, _ = pd.factorize(target_df[col_name].values)
    target_table.y = torch.from_numpy(encoded).long()

    relation_df = train_data['ratings']
    src_col = 'userID' if 'userID' in relation_df.columns else [col for col in relation_df.columns if 'user' in col.lower()][0]
    tgt_col = 'movieID' if 'movieID' in relation_df.columns else [col for col in relation_df.columns if 'movie' in col.lower()][0]

    user_id_map = {uid: i + 1 for i, uid in enumerate(target_table.df.index.tolist())}
    unique_movies = sorted(relation_df[tgt_col].unique())
    movie_id_map = {mid: i + 1 for i, mid in enumerate(unique_movies)}

    relation_df = relation_df.copy()
    relation_df['UserID'] = relation_df[src_col].map(user_id_map)
    relation_df['MovieID'] = relation_df[tgt_col].map(movie_id_map)
    relation_df = relation_df[relation_df['UserID'].notna() & relation_df['MovieID'].notna()][['UserID', 'MovieID']]

    movie_embeddings = torch.randn(len(unique_movies), emb_size)
    graph = build_homo_graph(relation_df, n_all=len(target_table) + len(unique_movies))

    adj = GCNTransform()(graph).adj.to_sparse_coo()
    target_table = TabTransformerTransform(out_dim=emb_size, metadata=target_table.metadata)(data=target_table)

    target_table.y = target_table.y.to(device)
    target_table.train_mask = train_mask.to(device)
    target_table.val_mask = val_mask.to(device)
    target_table.test_mask = test_mask.to(device)
    movie_embeddings = movie_embeddings.to(device)
    adj = adj.to(device)

    return target_table, movie_embeddings, adj
