
import time
from typing import Dict, Any, Optional

import torch
import torch.nn.functional as F
import pandas as pd

# Import from tl_sql package 
from tl_sql.examples.bridge.utils.data_preparer import (
    convert_dataframe_to_tabledata,
    remove_overlap_rows,
)
from tl_sql.examples.bridge.utils.utils import build_homo_graph, reorder_ids

# Import from rllm (external dependency) 
try:
    from rllm.transforms.graph_transforms import GCNTransform
    from rllm.transforms.table_transforms import TabTransformerTransform
    from rllm.nn.conv.graph_conv import GCNConv
    from rllm.nn.conv.table_conv import TabTransformerConv
    from rllm.nn.models import BRIDGE, TableEncoder, GraphEncoder
    from rllm.data import TableData
    from rllm.types import ColType
    RLLM_AVAILABLE = True
except ImportError:
    RLLM_AVAILABLE = False
    # Provide error message if rllm is not available 
    GCNTransform = None
    TabTransformerTransform = None
    GCNConv = None
    TabTransformerConv = None
    BRIDGE = None
    TableEncoder = None
    GraphEncoder = None
    TableData = None
    ColType = None


def prepare_bridge_data(
    train_data: Dict[str, pd.DataFrame],
    validate_data: Optional[Dict[str, pd.DataFrame]],
    test_data: Optional[pd.DataFrame],
    target_column: str,
    task_type: str,
    device: torch.device,
    db_config: Dict[str, Any]
):
    """Prepare data for BRIDGE model 
    
    Args:
        train_data: Training data dictionary 
        validate_data: Validation data dictionary 
        test_data: Test data DataFrame 
        target_column: Target column name
        task_type: Task type (CLF or REG) 
        device: Device (CPU/GPU) 
        db_config: Database configuration 
        
    Returns:
        Tuple containing prepared data 
        
    Raises:
        ImportError: If rllm is not installed 
    """
    # Check if rllm is available 
    if not RLLM_AVAILABLE:
        raise ImportError(
            "rllm is required for BRIDGE model preparation. "
            "Please install rllm: pip install rllm. "
        )
    
    # Target column is specified by PREDICT 
    # Parse target column 
    if '.' not in target_column:
        raise ValueError(f"Target column format error, should be 'table.column'")
    
    table_name, col_name = target_column.split('.', 1)
    
    train_df = train_data[table_name].copy()
    validate_df = validate_data[table_name].copy() if validate_data and table_name in validate_data else None
    test_df = test_data.copy() if test_data is not None else None
    
    from tl_sql.executor import DatabaseExecutor, DatabaseConfig
    table_primary_keys = {}
    with DatabaseExecutor(DatabaseConfig(**db_config)) as executor:
        all_tables = set(train_data.keys())
        if validate_data:
            all_tables.update(validate_data.keys())
        all_tables.add(table_name)
        for tbl_name in all_tables:
            pkeys = executor.get_primary_keys(tbl_name)
            table_primary_keys[tbl_name] = pkeys[0]
    target_pkey_for_features = table_primary_keys[table_name]
    
    all_data_frames = [train_df]
    data_sources = [('train', len(train_df))]
    
    if validate_df is not None and len(validate_df) > 0:
        all_data_frames.append(validate_df)
        data_sources.append(('validate', len(validate_df)))
    
    if test_df is not None and len(test_df) > 0:
        all_data_frames.append(test_df)
        data_sources.append(('test', len(test_df)))
    
    if len(all_data_frames) > 1:
        common_columns = set(all_data_frames[0].columns)
        for df in all_data_frames[1:]:
            common_columns = common_columns.intersection(set(df.columns))
        common_columns_list = sorted(list(common_columns))
        all_data_frames = [df[common_columns_list] for df in all_data_frames]
        target_df = pd.concat(all_data_frames, ignore_index=True)
    else:
        target_df = all_data_frames[0]
    
    
    col_types = {}
    
    discrete_numerical_keywords = [
        'age', 'year', 'years',
    ]
    
    # Define categorical column keywords 
    categorical_keywords = [
        'gender', 'occupation', 'zip', 'code',
    ]
    
    # Define continuous numerical column keywords (true continuous numerical features) 
    continuous_numerical_keywords = [
        'rating', 'score', 'price', 'amount', 'value',
    ]
    
    for col in target_df.columns:
        if col == target_pkey_for_features:
            continue
        
        dtype = target_df[col].dtype
        unique_count = target_df[col].nunique()
        total_rows = len(target_df)
        col_lower = col.lower()
        
        # If it's the target column and it's a classification task, force it to be categorical 
        if col == col_name and task_type.upper() == 'CLF':
            col_types[col] = ColType.CATEGORICAL
            continue
        
        # Discrete numerical features as categorical features 
        is_discrete_numerical = any(keyword in col_lower for keyword in discrete_numerical_keywords)
        is_categorical_by_name = any(keyword in col_lower for keyword in categorical_keywords)
        is_continuous_numerical = any(keyword in col_lower for keyword in continuous_numerical_keywords)
        
        if is_discrete_numerical:
            # Age, Year, etc. are usually discrete, more suitable as categorical features 
            col_types[col] = ColType.CATEGORICAL
        elif is_categorical_by_name:
            col_types[col] = ColType.CATEGORICAL
        elif is_continuous_numerical:
            # True continuous numerical features, keep as NUMERICAL 
            col_types[col] = ColType.NUMERICAL
        elif dtype in ['int64', 'float64']:
            if unique_count <= 20:
                # Very few unique values, definitely categorical feature 
                col_types[col] = ColType.CATEGORICAL
            elif unique_count < total_rows * 0.1 and unique_count <= 100:
                # Small proportion of unique values and moderate quantity, may be ID class or category encoding 
                col_types[col] = ColType.CATEGORICAL
            else:
                # Many unique values or large proportion, as continuous numerical feature 
                col_types[col] = ColType.NUMERICAL
        else:
            # Non-numerical type, default as categorical feature
            col_types[col] = ColType.CATEGORICAL
    
    categorical_cols = [col for col, col_type in col_types.items() if col_type == ColType.CATEGORICAL]
    
    target_df = target_df.set_index(target_pkey_for_features)
    pkey_for_table = target_pkey_for_features
    
    # Create TableData 
    target_table = TableData(
        df=target_df,
        col_types=col_types,  # Primary key column already excluded from col_types 
        target_col=col_name,
        pkey=pkey_for_table  # Specify primary key, TableData will handle correctly
    )
    
    
    if task_type.upper() == 'CLF':
        target_values = target_df[col_name].values
        encoded, unique = pd.factorize(target_values)
        y = torch.from_numpy(encoded).long()
        target_table.y = y
    else:
        target_values = target_df[col_name].values
        y = torch.from_numpy(target_values).float()
        target_table.y = y
    

    # Create mask 
    n = len(target_table)
    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)
    
    # Create mask normally 
    current_idx = 0
    for source_type, count in data_sources:
        if source_type == 'train':
            train_mask[current_idx:current_idx + count] = True
        elif source_type == 'validate':
            val_mask[current_idx:current_idx + count] = True
        elif source_type == 'test':
            test_mask[current_idx:current_idx + count] = True
        current_idx += count
    
    # Determine graph structure based on database
    # tacm12k: papers -> citations (homogeneous graph)
    # tml1m: users -> ratings -> movies (heterogeneous graph)
    database_name = db_config.get('database', '').lower()
    
    if 'tacm12k' in database_name:
        relation_table_name = 'citations'
        is_homogeneous = True
    else:  # tml1m
        relation_table_name = 'ratings'
        is_homogeneous = False
    
    relation_df = train_data[relation_table_name].copy()
    target_pkey = table_primary_keys[table_name]
    
    if is_homogeneous:
        # tacm12k: citations table has two paper ID columns
        pkey_cols = [col for col in relation_df.columns 
                     if target_pkey.lower() in col.lower() or col.lower() in target_pkey.lower()]
        src_col = pkey_cols[0]
        tgt_col = pkey_cols[1]
    else:
        # tml1m: ratings table has userID and movieID columns
        src_col = [col for col in relation_df.columns if 'user' in col.lower()][0]
        tgt_col = [col for col in relation_df.columns if 'movie' in col.lower()][0]
    
    if is_homogeneous:
        target_table_pkeys = target_table.df.index.tolist()
        
        node_id_map = {pkey: idx for idx, pkey in enumerate(target_table_pkeys)}
        n_nodes = len(target_table)
        
        relation_df_mapped = relation_df[[src_col, tgt_col]].copy()
        relation_df_mapped['src_mapped'] = relation_df_mapped[src_col].map(node_id_map)
        relation_df_mapped['tgt_mapped'] = relation_df_mapped[tgt_col].map(node_id_map)
        
        relation_df_mapped = relation_df_mapped[
            relation_df_mapped['src_mapped'].notna() & 
            relation_df_mapped['tgt_mapped'].notna()
        ].copy()
        
        relation_for_graph = relation_df_mapped[['src_mapped', 'tgt_mapped']].copy()
        relation_for_graph.columns = [0, 1]
        
        graph = build_homo_graph(
            relation_df=relation_for_graph,
            n_all=n_nodes
        ).to(device)
        
        graph_transform = GCNTransform()
        adj = graph_transform(graph).adj
        movie_embeddings = None
    else:
        user_id_col = src_col
        movie_id_col = tgt_col
        
        target_user_ids = target_table.df.index.tolist()
        actual_n_users = len(target_table)
        
        user_id_map = {uid: idx + 1 for idx, uid in enumerate(target_user_ids)}
        
        unique_movies = sorted(relation_df[movie_id_col].unique())
        n_unique_movies = len(unique_movies)
        
        if 'movies' in train_data:
            movies_df = train_data['movies']
            n_movies = max(len(movies_df), n_unique_movies)
        else:
            n_movies = n_unique_movies
        
        movie_id_map = {mid: idx + 1 for idx, mid in enumerate(unique_movies)}
        
        relation_df['UserID'] = relation_df[user_id_col].map(user_id_map)
        relation_df['MovieID'] = relation_df[movie_id_col].map(movie_id_map)
        
        relation_df_filtered = relation_df[
            relation_df['UserID'].notna() & relation_df['MovieID'].notna()
        ].copy()
        
        ordered_rating = reorder_ids(
            relation_df=relation_df_filtered[['UserID', 'MovieID']],
            src_col_name="UserID",
            tgt_col_name="MovieID",
            n_src=actual_n_users
        )
        
        emb_size = 128
        movie_embeddings = torch.randn(n_movies, emb_size).to(device)
        
        relation_for_graph = ordered_rating[['UserID', 'MovieID']].copy()
        n_all = actual_n_users + n_movies
        
        graph = build_homo_graph(
            relation_df=relation_for_graph,
            n_all=n_all
        ).to(device)
        
        graph_transform = GCNTransform()
        adj = graph_transform(graph).adj
    
    emb_size = movie_embeddings.size(1) if movie_embeddings is not None else 128
    table_transform = TabTransformerTransform(
        out_dim=emb_size,
        metadata=target_table.metadata
    )
    target_table = table_transform(data=target_table)
    
    from rllm.data import GraphData
    graph = GraphData(adj=adj)
    graph_transform = GCNTransform()
    graph = graph_transform(data=graph)
    adj = graph.adj
    
    target_table = target_table.to(device)
    
    if task_type.upper() == 'CLF':
        target_table.y = target_table.y.long().to(device)
    else:
        target_table.y = target_table.y.float().to(device)
    
    target_table.train_mask = train_mask.to(device)
    target_table.val_mask = val_mask.to(device)
    target_table.test_mask = test_mask.to(device)
    
    if not adj.is_sparse:
        adj = adj.to_sparse_coo()
    
    
    return (
        target_table,
        target_table.y, 
        movie_embeddings,
        adj,
        target_table.train_mask,  
        target_table.val_mask,
        target_table.test_mask
    )


def train_bridge_model(
    target_table,
    y,
    movie_embeddings,
    adj,
    train_mask,
    val_mask,
    test_mask,
    task_type: str,
    num_classes: int = None,
    max_epochs: int = 100,
    lr: float = 0.001,
    wd: float = 1e-4,
    device: torch.device = None
):
    """Train BRIDGE model 
    
    Args:
        target_table: Target table data 
        y: Labels 
        movie_embeddings: Movie embeddings 
        adj: Adjacency matrix 
        train_mask: Training set mask 
        val_mask: Validation set mask 
        test_mask: Test set mask 
        task_type: Task type 
        num_classes: Number of classes 
        max_epochs: Maximum training epochs 
        lr: Learning rate 
        wd: Weight decay 
        device: Device 
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Determine output dimension based on task type 
    task_type_upper = task_type.upper()
    if task_type_upper == 'CLF':
        if num_classes is None:
            num_classes = len(torch.unique(y))
        out_dim = num_classes
    else:  # REG
        out_dim = 1  # Regression task outputs single value 
    
    emb_size = movie_embeddings.size(1) if movie_embeddings is not None else 128
    
    # Set up model
    t_encoder = TableEncoder(
        in_dim=emb_size,
        out_dim=emb_size,
        table_conv=TabTransformerConv,
        metadata=target_table.metadata,
    )
    g_encoder = GraphEncoder(
        in_dim=emb_size,
        out_dim=out_dim,
        graph_conv=GCNConv,
    )
    model = BRIDGE(
        table_encoder=t_encoder,
        graph_encoder=g_encoder,
    ).to(device)
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=wd,
    )
    
    def train_epoch() -> float:
        model.train()
        optimizer.zero_grad()
        output = model(
            table=target_table,
            non_table=movie_embeddings,
            adj=adj,
        )
        
        if task_type_upper == 'CLF':
            # Classification task: use cross-entropy loss
            output = output.squeeze() if output.dim() > 1 else output
            loss = F.cross_entropy(output[train_mask], y[train_mask])
        else:
            # Regression task: use MSE loss 
            output = output.squeeze()
            loss = F.mse_loss(output[train_mask], y[train_mask].float())
        
        loss.backward()
        optimizer.step()
        return loss.item()
    
    @torch.no_grad()
    def evaluate():
        model.eval()
        output = model(
            table=target_table,
            non_table=movie_embeddings,
            adj=adj,
        )
        
        if task_type_upper == 'CLF':
            # Classification task uses accuracy 
            output = output.squeeze() if output.dim() > 1 else output
            preds = output.argmax(dim=1)
            metrics = []
            for mask in [train_mask, val_mask, test_mask]:
                correct = float(preds[mask].eq(y[mask]).sum().item())
                metrics.append(correct / int(mask.sum()))
        else:
            # Regression task uses RMSE
            output = output.squeeze()
            metrics = []
            for mask in [train_mask, val_mask, test_mask]:
                rmse = torch.sqrt(F.mse_loss(output[mask], y[mask].float())).item()
                metrics.append(rmse)
        
        return metrics
    
    # Training loop 
    if task_type_upper == 'CLF':
        metric = "Acc"
        best_val_metric = test_metric = 0
        compare_func = lambda x, y: x > y  # Classification task: higher accuracy is better 
    else:
        metric = "RMSE"
        best_val_metric = test_metric = float('inf')
        compare_func = lambda x, y: x < y  # Regression task: lower RMSE is better 
    
    times = []
    
    for epoch in range(1, max_epochs + 1):
        start = time.time()
        
        train_loss = train_epoch()
        train_metric, val_metric, tmp_test_metric = evaluate()
        
        if compare_func(val_metric, best_val_metric):
            best_val_metric = val_metric
            test_metric = tmp_test_metric
        
        times.append(time.time() - start)
        print(
            f"Epoch: [{epoch}/{max_epochs}] "
            f"Train Loss: {train_loss:.4f} Train {metric}: {train_metric:.4f} "
            f"Val {metric}: {val_metric:.4f}, Test {metric}: {tmp_test_metric:.4f} "
        )
    
    return model, test_metric

