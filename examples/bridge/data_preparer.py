"""BRIDGE pipeline utilities for preparing data and training models."""

from typing import Dict, Any, Optional

import torch
import pandas as pd

from tlsql.examples.bridge.graph_builder import build_homo_graph
from rllm.transforms.graph_transforms import GCNTransform
from rllm.transforms.table_transforms import TabTransformerTransform
from rllm.data import TableData
from rllm.types import ColType


def prepare_bridge_data(
    train_data: Dict[str, pd.DataFrame],
    validate_data: Optional[Dict[str, pd.DataFrame]],
    test_data: Optional[pd.DataFrame],
    target_column: str,
    task_type: str,
    device: torch.device,
    db_config: Dict[str, Any]
):
    """Prepare data for BRIDGE model"""
    table_name, col_name = target_column.split('.')

    train_df = train_data[table_name]
    validate_df = validate_data[table_name] if validate_data else None
    test_df = test_data if test_data is not None else None

    from tlsql.examples.executor.db_executor import DatabaseExecutor, DatabaseConfig
    with DatabaseExecutor(DatabaseConfig(**db_config)) as executor:
        pkeys = executor.get_primary_keys(table_name)
        target_pkey = pkeys[0]

    all_dfs = [train_df]
    data_sources = [('train', len(train_df))]
    if validate_df is not None:
        all_dfs.append(validate_df)
        data_sources.append(('validate', len(validate_df)))
    if test_df is not None:
        all_dfs.append(test_df)
        data_sources.append(('test', len(test_df)))

    common_cols = set.intersection(*[set(df.columns) for df in all_dfs])
    target_df = pd.concat([df[list(common_cols)] for df in all_dfs], ignore_index=True)
    target_df = target_df.set_index(target_pkey)

    col_types = {}
    for col in target_df.columns:
        if col == target_pkey:
            continue
        if col == col_name and task_type.upper() == 'CLF':
            col_types[col] = ColType.CATEGORICAL
        elif target_df[col].dtype in ['int64', 'float64']:
            unique_count = target_df[col].nunique()
            total_rows = len(target_df)
            if unique_count <= 20 or (unique_count < total_rows * 0.1 and unique_count <= 100):
                col_types[col] = ColType.CATEGORICAL
            else:
                col_types[col] = ColType.NUMERICAL
        else:
            col_types[col] = ColType.CATEGORICAL

    target_table = TableData(
        df=target_df,
        col_types=col_types,
        target_col=col_name,
        pkey=target_pkey
    )

    if task_type.upper() == 'CLF':
        encoded, _ = pd.factorize(target_df[col_name].values)
        target_table.y = torch.from_numpy(encoded).long()
    else:
        target_table.y = torch.from_numpy(target_df[col_name].values).float()

    n = len(target_table)
    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)

    idx = 0
    for source_type, count in data_sources:
        mask = {'train': train_mask, 'validate': val_mask, 'test': test_mask}[source_type]
        mask[idx:idx + count] = True
        idx += count

    relation_df = train_data['ratings']
    src_col = [col for col in relation_df.columns if 'user' in col.lower()][0]
    tgt_col = [col for col in relation_df.columns if 'movie' in col.lower()][0]

    user_ids = target_table.df.index.tolist()
    n_users = len(target_table)
    user_id_map = {uid: i + 1 for i, uid in enumerate(user_ids)}

    unique_movies = sorted(relation_df[tgt_col].unique())
    n_movies = len(unique_movies)
    movie_id_map = {mid: i + 1 for i, mid in enumerate(unique_movies)}

    relation_df = relation_df.copy()
    relation_df['UserID'] = relation_df[src_col].map(user_id_map)
    relation_df['MovieID'] = relation_df[tgt_col].map(movie_id_map)
    relation_df = relation_df[relation_df['UserID'].notna() & relation_df['MovieID'].notna()]

    emb_size = 128
    movie_embeddings = torch.randn(n_movies, emb_size).to(device)

    graph = build_homo_graph(
        relation_df=relation_df[['UserID', 'MovieID']],
        n_all=n_users + n_movies
    ).to(device)

    graph_transform = GCNTransform()
    adj = graph_transform(graph).adj

    table_transform = TabTransformerTransform(out_dim=emb_size, metadata=target_table.metadata)
    target_table = table_transform(data=target_table).to(device)

    if not adj.is_sparse:
        adj = adj.to_sparse_coo()

    if task_type.upper() == 'CLF':
        target_table.y = target_table.y.long().to(device)
    else:
        target_table.y = target_table.y.float().to(device)

    target_table.train_mask = train_mask.to(device)
    target_table.val_mask = val_mask.to(device)
    target_table.test_mask = test_mask.to(device)

    return (
        target_table,
        target_table.y,
        movie_embeddings,
        adj,
        target_table.train_mask,
        target_table.val_mask,
        target_table.test_mask
    )
