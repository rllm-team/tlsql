"""BRIDGE pipeline utilities for preparing data and training models."""

from typing import Dict

import torch
import pandas as pd

from tlsql.examples.bridge.model import build_homo_graph
from rllm.transforms.graph_transforms import GCNTransform
from rllm.transforms.table_transforms import TabTransformerTransform
from rllm.data import TableData
from rllm.types import ColType


def reorder_ids(relation_df: pd.DataFrame, src_col_name: str, tgt_col_name: str, n_src: int):
    """Reorders IDs by adjusting source IDs and target column IDs."""
    return relation_df.assign(**{src_col_name: relation_df[src_col_name] - 1, tgt_col_name: relation_df[tgt_col_name] + n_src - 1})


def prepare_bridge_data(
    train_data: Dict[str, pd.DataFrame],
    validate_data: Dict[str, pd.DataFrame],
    test_data: pd.DataFrame,
    target_column: str,
    device: torch.device = None,
):
    """Prepare data for BRIDGE model

    Args:
        train_data: Dictionary mapping table names to training DataFrames
        validate_data: Dictionary mapping table names to validation DataFrames
        test_data: Dictionary mapping table names to test DataFrames
        target_column: Target column name in format 'table.column'
        device: PyTorch device
    """
    target_pkey, emb_size = 'UserID', 384
    table_name, col_name = target_column.split('.')
    train_df, validate_df, test_df = train_data[table_name], validate_data[table_name], test_data
    all_dfs = [train_df, validate_df, test_df]

    for df in all_dfs:
        df[target_pkey] = pd.to_numeric(df[target_pkey], errors='coerce')
        df.sort_values(by=target_pkey, inplace=True)

    common_cols = list(set.intersection(*[set(df.columns) for df in all_dfs]))
    target_df = pd.concat([df[common_cols] for df in all_dfs], ignore_index=True).set_index(target_pkey)

    target_table = TableData(df=target_df, col_types={col: ColType.CATEGORICAL for col in target_df.columns},target_col=col_name, pkey=target_pkey)
    train_len, val_len, test_len = len(train_df), len(validate_df), len(test_df)
    train_mask = torch.cat([torch.ones(train_len, dtype=torch.bool), torch.zeros(val_len + test_len, dtype=torch.bool)])
    val_mask = torch.cat([torch.zeros(train_len, dtype=torch.bool), torch.ones(val_len, dtype=torch.bool),
                          torch.zeros(test_len, dtype=torch.bool)])
    test_mask = torch.cat([torch.zeros(train_len + val_len, dtype=torch.bool), torch.ones(test_len, dtype=torch.bool)])

    target_table.y = torch.from_numpy(pd.factorize(target_df[col_name].values)[0]).long()

    relation_df = train_data['ratings'].copy()
    relation_df[['UserID', 'MovieID']] = relation_df[['UserID', 'MovieID']].apply(pd.to_numeric, errors='coerce')
    user_id_map = {uid: i + 1 for i, uid in enumerate(target_table.df.index)}
    relation_df['UserID'] = relation_df['UserID'].map(user_id_map)
    unique_movies = sorted(relation_df['MovieID'].unique())
    relation_df['MovieID'] = relation_df['MovieID'].map({mid: i + 1 for i, mid in enumerate(unique_movies)})

    num_movies = len(unique_movies)
    # For convenience, random parameters are used here.
    # In fact, the embeddings are obtained by processing the movie table with the model all-MiniLM-L6-v2.
    movie_embeddings = torch.randn(num_movies, emb_size)
    ordered_rating = reorder_ids(relation_df[['UserID', 'MovieID']], "UserID", "MovieID", len(target_table))
    graph = build_homo_graph(ordered_rating, n_all=len(target_table) + num_movies)

    adj = GCNTransform()(graph).adj.to_sparse_coo()
    target_table = TabTransformerTransform(out_dim=emb_size, metadata=target_table.metadata)(data=target_table)

    target_table.y = target_table.y.to(device)
    target_table.train_mask = train_mask.to(device)
    target_table.val_mask = val_mask.to(device)
    target_table.test_mask = test_mask.to(device)
    movie_embeddings = movie_embeddings.to(device)
    adj = adj.to(device)

    return target_table, movie_embeddings, adj
