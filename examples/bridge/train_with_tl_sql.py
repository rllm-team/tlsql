"""BRIDGE Model Training Utilities with TL-SQL 

This module provides utilities for training BRIDGE models using TL-SQL statements.

Note: This module requires the rllm package to be available.


"""

import time
import argparse
import numpy as np
import sys
import os
from sklearn.model_selection import KFold
from typing import Optional, Dict, List

import torch
import torch.nn.functional as F
import pandas as pd
import torch.nn as nn

current_dir = os.path.dirname(os.path.abspath(__file__))
bridge_dir = os.path.dirname(current_dir)
tl_sql_dir = os.path.dirname(bridge_dir)
project_root = os.path.dirname(tl_sql_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from rllm.nn.conv.graph_conv import GCNConv
from rllm.nn.conv.table_conv import TabTransformerConv
from rllm.nn.models import BRIDGE, TableEncoder, GraphEncoder

from tl_sql.config import load_dataset_config
from tl_sql.examples.bridge.utils import prepare_bridge_data


def build_bridge_model(num_classes, metadata, emb_size):
    """Build BRIDGE model """
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


class MLPBaseline(nn.Module):
    """MLP model"""
    def __init__(self, input_dim, num_classes):
        super(MLPBaseline, self).__init__()
        # Single linear layer 
        self.linear = nn.Linear(input_dim, num_classes)
    
    def forward(self, x):
        return self.linear(x)


class RandomBaseline:
    """Random guessing baseline """
    def __init__(self, num_classes):
        self.num_classes = num_classes
    
    def __call__(self, y, mask):
        """Random prediction """
        n = int(mask.sum())
        preds = torch.randint(0, self.num_classes, (n,), device=y.device)
        return preds
    
    def eval(self):
        pass
    
    def train(self):
        pass


def extract_features_for_mlp(target_table, non_table_embeddings, device):
    """Extract features from TableData for MLP model """
    target_table.lazy_materialize()
    
    feature_list = []
    for feat_tensor in target_table.feat_dict.values():
        if isinstance(feat_tensor, tuple):
            feat_tensor = feat_tensor[0]
        if feat_tensor.dim() > 2:
            feat_tensor = feat_tensor.flatten(start_dim=1)
        elif feat_tensor.dim() == 1:
            feat_tensor = feat_tensor.unsqueeze(1)
        feature_list.append(feat_tensor)
    
    table_features = torch.cat(feature_list, dim=1).float()
    return torch.cat([table_features, non_table_embeddings.to(device).float()], dim=1).to(device)


def train_mlp(model, features, y, train_mask, val_mask, test_mask, epochs, lr, wd, device):
    """Train MLP baseline model """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.CrossEntropyLoss()
    
    metric = "Acc"
    best_val_acc = test_acc = 0
    times = []
    
    for epoch in range(1, epochs + 1):
        start = time.time()
        model.train()
        optimizer.zero_grad()
        
        # Training 
        train_features = features[train_mask]
        train_y = y[train_mask]
        logits = model(train_features)
        loss = criterion(logits, train_y)
        loss.backward()
        optimizer.step()
        
        # Evaluation 
        model.eval()
        with torch.no_grad():
            logits_all = model(features)
            preds = logits_all.argmax(dim=1)
            
            train_acc = float(preds[train_mask].eq(y[train_mask]).sum().item()) / int(train_mask.sum())
            test_acc_curr = float(preds[test_mask].eq(y[test_mask]).sum().item()) / int(test_mask.sum())
            val_acc = float(preds[val_mask].eq(y[val_mask]).sum().item()) / int(val_mask.sum())
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = test_acc_curr
        
        times.append(time.time() - start)
        print(
            f"Epoch: [{epoch}/{epochs}] "
            f"Train Loss: {loss.item():.4f} Train {metric}: {train_acc:.4f} "
            f"Val {metric}: {val_acc:.4f}, Test {metric}: {test_acc_curr:.4f} "
        )
    
    print(f"Mean time per epoch: {torch.tensor(times).mean():.4f}s")
    print(f"Total time: {sum(times):.4f}s")
    
    return best_val_acc, test_acc


def evaluate_random_baseline(model, y, train_mask, val_mask, test_mask):
    """Evaluate random guessing baseline model """
    metric = "Acc"
    results = {}
    
    for mask_name, mask in [("Train", train_mask), ("Val", val_mask), ("Test", test_mask)]:
        preds = model(y, mask)
        acc = float(preds.eq(y[mask]).sum().item()) / int(mask.sum())
        results[mask_name] = acc
        print(f"{mask_name} {metric}: {acc:.4f}")
    
    return results["Val"], results["Test"]


def train(model, optimizer, target_table, non_table_embeddings, adj, y, train_mask):
    """Train for one epoch """
    model.train()
    optimizer.zero_grad()
    logits = model(table=target_table, non_table=non_table_embeddings, adj=adj)
    
    if logits.dim() > 2:
        logits = logits.view(logits.size(0), -1)
    elif logits.dim() == 1:
        raise ValueError(f"Logits dimension error: {logits.shape}")
    
    loss = F.cross_entropy(logits[train_mask], y[train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def test(model, target_table, non_table_embeddings, adj, y, masks):
    """Test"""
    model.eval()
    logits = model(table=target_table, non_table=non_table_embeddings, adj=adj)
    preds = logits.argmax(dim=1)
    accs = []
    for mask in masks:
        correct = float(preds[mask].eq(y[mask]).sum().item())
        accs.append(correct / int(mask.sum()))
    return accs


def train_bridge_model(model, target_table, non_table_embeddings, adj, epochs, lr, wd):
    """Train """
    y = target_table.y
    train_mask, val_mask, test_mask = (
        target_table.train_mask,
        target_table.val_mask,
        target_table.test_mask,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    metric = "Acc"
    best_val_acc = test_acc = 0
    times = []
    for epoch in range(1, epochs + 1):
        start = time.time()
        train_loss = train(model, optimizer, target_table, non_table_embeddings, adj, y, train_mask)
        train_acc, val_acc, tmp_test_acc = test(model, target_table, non_table_embeddings, adj, y, [train_mask, val_mask, test_mask])

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = tmp_test_acc

        times.append(time.time() - start)
        print(
            f"Epoch: [{epoch}/{epochs}] "
            f"Train Loss: {train_loss:.4f} Train {metric}: {train_acc:.4f} "
            f"Val {metric}: {val_acc:.4f}, Test {metric}: {tmp_test_acc:.4f} "
        )

    print(f"Mean time per epoch: {torch.tensor(times).mean():.4f}s")
    print(f"Total time: {sum(times):.4f}s")
    return model, best_val_acc, test_acc


def train_bridge_model_with_kfold(model, target_table, non_table_embeddings, adj, epochs, lr, wd, k_folds=5):
    """Use k-fold cross-validation 
    
    1. Perform k-fold cross-validation on current training data 
    """
    y = target_table.y
    test_mask = target_table.test_mask  
    
    # Get test set indices 
    test_indices = torch.where(test_mask)[0].cpu().numpy()
    
    # Remaining data = total data - test set 
    total_indices = torch.arange(len(target_table)).cpu().numpy()
    remaining_indices = np.setdiff1d(total_indices, test_indices)
    
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    fold_results = []
    metric = "Acc"
    
    
    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(remaining_indices), 1):
        print(f"Fold {fold_idx}/{k_folds}")

        # Create mask for current fold 
        fold_train_mask = torch.zeros_like(test_mask)
        fold_val_mask = torch.zeros_like(test_mask)
        fold_test_mask = test_mask.clone()  # Test set remains fixed 
        
        # Get training set indices for current fold from remaining data 
        fold_train_indices = remaining_indices[train_idx]
        fold_train_mask[fold_train_indices] = True
        
        # Get validation set indices for current fold from remaining data 
        fold_val_indices = remaining_indices[val_idx]
        fold_val_mask[fold_val_indices] = True
        
        # Each fold uses an independent model 
        fold_model = build_bridge_model(
            target_table.num_classes, 
            target_table.metadata, 
            non_table_embeddings.size(1) if non_table_embeddings is not None else 128
        ).to(target_table.y.device)
        
        # Train current fold
        optimizer = torch.optim.Adam(fold_model.parameters(), lr=lr, weight_decay=wd)
        
        best_val_acc = test_acc = 0
        times = []
        
        for epoch in range(1, epochs + 1):
            start = time.time()
            train_loss = train(fold_model, optimizer, target_table, non_table_embeddings, adj, y, fold_train_mask)
            train_acc, val_acc, tmp_test_acc = test(
                fold_model, target_table, non_table_embeddings, adj, y, 
                [fold_train_mask, fold_val_mask, fold_test_mask]
            )
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                test_acc = tmp_test_acc
            
            times.append(time.time() - start)
            if epoch % 10 == 0 or epoch == epochs: 
                print(
                    f"Fold {fold_idx} Epoch: [{epoch}/{epochs}] "
                    f"Train Loss: {train_loss:.4f} Train {metric}: {train_acc:.4f} "
                    f"Val {metric}: {val_acc:.4f}, Test {metric}: {tmp_test_acc:.4f} "
                )
        
        fold_results.append({
            'fold': fold_idx,
            'best_val_acc': best_val_acc,
            'test_acc': test_acc,
            'mean_time': np.mean(times),
            'total_time': sum(times)
        })
        
        print(f"Fold {fold_idx} - Mean time per epoch: {np.mean(times):.4f}s")
        print(f"Fold {fold_idx} - Total time: {sum(times):.4f}s")
        print(f"Fold {fold_idx} - Best Val {metric}: {best_val_acc:.4f}")
        print(f"Fold {fold_idx} - Test {metric} at best Val: {test_acc:.4f}")
    
    
    val_accs = [r['best_val_acc'] for r in fold_results]
    test_accs = [r['test_acc'] for r in fold_results]
    
    print(f"Validation accuracy - Mean: {np.mean(val_accs):.4f}, Std: {np.std(val_accs):.4f}")
    print(f"Test accuracy - Mean: {np.mean(test_accs):.4f}, Std: {np.std(test_accs):.4f}")
    print(f"\nDetailed results for each fold:")
    for r in fold_results:
        print(f"  Fold {r['fold']}: Val {metric}={r['best_val_acc']:.4f}, Test {metric}={r['test_acc']:.4f}")
    
    return fold_results


def _execute_sql_list(executor, sql_list, error_prefix="load failed"):
    """Execute SQL list and return data dictionary 
    Args:
        executor: Database executor 
        sql_list: SQL list generated by SQL generator 
        error_prefix: Error message prefix 
        
    Returns:
        dict: {table_name: DataFrame}
    """
    data_dict = {}
    for gen_sql in sql_list:
        result = executor.execute(gen_sql.sql)
        if result.success:
            data_dict[gen_sql.table] = result.data
        else:
            raise Exception(f"{error_prefix}: {gen_sql.table}: {result.error}")
    return data_dict


def _load_test_data(executor, target_table_name, predict_where_condition):
    """Load test data based on PREDICT condition"""
    if predict_where_condition:
        sql = f"SELECT * FROM {target_table_name} WHERE {predict_where_condition}"
    else:
        sql = f"SELECT * FROM {target_table_name}"
    # print(sql)
    result = executor.execute(sql)
    if result.success:
        return result.data
    raise Exception(f"Failed to get test data: {result.error}")


def _load_train_data_dict(executor, train_ast, sql_generator, target_table_name,
                          target_pkey, test_df):
    """Load training data dictionary based on SQL conditions 
    
    If TRAIN statement is not specified, use remaining data from target table by default 
    """
    if train_ast:
        sql_list = sql_generator.generate_train_sql(train_ast)
        train_data = _execute_sql_list(executor, sql_list, "Failed to load training data")
        return train_data
    
    result = executor.execute(f"SELECT * FROM {target_table_name}")
    target_df_all = result.data
    train_df = _compute_train_data_from_remaining(
        target_df_all=target_df_all,
        test_df_filtered=test_df,
        target_pkey=target_pkey,
        table_name=target_table_name
    )
    
    train_data = {target_table_name: train_df}
    
    database_name = executor.config.database
    config_tables = _load_tables_from_config(database_name, target_table_name)
    
    if config_tables:
        for rel_table in config_tables.get('relation_tables', []):
            result = executor.execute(f"SELECT * FROM {rel_table}")
            train_data[rel_table] = result.data
        
        for aux_table in config_tables.get('aux_tables', []):
            if aux_table not in train_data:
                result = executor.execute(f"SELECT * FROM {aux_table}")
                train_data[aux_table] = result.data
    else:
        all_tables_in_db = executor.list_tables()
        other_tables = [tbl for tbl in all_tables_in_db if tbl != target_table_name]
        
        for tbl in other_tables:
            result = executor.execute(f"SELECT * FROM {tbl}")
            train_data[tbl] = result.data
    
    return train_data


def _load_tables_from_config(database_name: str, target_table_name: str) -> Optional[Dict[str, List[str]]]:
    """Load table information from config file"""
    config = load_dataset_config()
    
    dataset_config = config.get(database_name)
    if dataset_config and dataset_config.get('target_table') == target_table_name:
        return {
            'relation_tables': dataset_config.get('relation_tables', []),
            'aux_tables': dataset_config.get('aux_tables', [])
        }

    for dataset_name, dataset_config in config.items():
        if dataset_config.get('target_table') == target_table_name:
            return {
                'relation_tables': dataset_config.get('relation_tables', []),
                'aux_tables': dataset_config.get('aux_tables', [])
            }
    
    return None


def _load_validate_data_dict(executor, validate_ast, sql_generator):
    """Load validation data dictionary based on SQL conditions """
    if not validate_ast:
        return None
    sql_list = sql_generator.generate_validate_sql(validate_ast)
    validate_data = _execute_sql_list(executor, sql_list, "Failed to load validation data")
    return validate_data


def _parse_tl_sql_statements(train_sql, validate_sql, predict_sql):
    """Parse all ML-SQL statements and extract key information 
    
    Args:
        train_sql: TRAIN SQL statement 
        validate_sql: VALIDATE SQL statement 
        predict_sql: PREDICT SQL statement 
        
    Returns:
        tuple: (predict_ast, train_ast, validate_ast, sql_generator, target_column, 
                task_type, table_name, col_name, target_table_name, predict_where_condition, all_tables)
    """
    from tl_sql.core.parser import Parser
    from tl_sql.executor import SQLGenerator
    
    # 1. PREDICT must be specified 
    if not predict_sql:
        raise ValueError("PREDICT SQL statement must be specified")
    
    sql_generator = SQLGenerator()
    
    # Parse PREDICT statement 
    parser = Parser(predict_sql)
    predict_ast = parser.parse().predict
    
    # Parse TRAIN statement 
    train_ast = None
    if train_sql:
        parser = Parser(train_sql)
        train_ast = parser.parse().train
    
    # Parse VALIDATE statement
    validate_ast = None
    if validate_sql:
        parser = Parser(validate_sql)
        validate_ast = parser.parse().validate
    
    # Extract target column and task type
    target = predict_ast.value.target
    target_table_name = predict_ast.from_table.table
    
    # Build target_column: if target.table is empty, use from_table.table
    if target.table:
        target_column = f"{target.table}.{target.column}"
    else:
        target_column = f"{target_table_name}.{target.column}"
    
    task_type = predict_ast.value.predict_type.type_name
    table_name, col_name = target_column.split('.', 1)
    
    # Extract WHERE condition from PREDICT
    predict_where_condition = None
    if predict_ast.where:
        predict_where_condition = sql_generator._expr_to_sql(
            predict_ast.where.condition,
            include_table_prefix=False
        )
    
    # Determine list of tables to fetch
    if train_ast:
        all_tables = train_ast.tables.tables
        if target_table_name not in all_tables:
            all_tables.append(target_table_name)
    else:
        all_tables = [target_table_name]
    
    return (predict_ast, train_ast, validate_ast, sql_generator, target_column, 
            task_type, table_name, col_name, target_table_name, predict_where_condition, all_tables)




def _compute_train_data_from_remaining(target_df_all, test_df_filtered, target_pkey, table_name):
    """Exclude test data from full data to get training data"""
    if target_pkey and target_pkey in target_df_all.columns and target_pkey in test_df_filtered.columns:
        test_ids = set(test_df_filtered[target_pkey].values)
        mask = ~target_df_all[target_pkey].isin(test_ids).values
        train_df_filtered = target_df_all[mask].copy()
    else:
        train_df_filtered = target_df_all.copy()
    return train_df_filtered


def prepare_data_from_tl_sql(
    train_sql: Optional[str],
    validate_sql: Optional[str],
    predict_sql: str,
    db_config: dict,
    device: torch.device
):
    """Get data and prepare in format required by bridge model 
    
    Priority logic:
    Level I: PREDICT
    Level II: TRAIN WITH (if not specified, use all data except "data to be predicted" )
    Level III: VALIDATE WITH (if not specified, perform k=5 fold cross-validation on "training data" )
    
    Args:
        train_sql: TRAIN SQL statement 
        validate_sql: VALIDATE SQL statement 
        predict_sql: PREDICT SQL statement 
        db_config: Database configuration dictionary 
        device: Device (CPU/GPU) 
        
    Returns:
        tuple: (target_table, non_table_embeddings, adj, emb_size, use_kfold)
        where target_table contains train_mask, val_mask, test_mask and y attributes 
        use_kfold: Whether to use k-fold cross-validation 
    """
    from tl_sql.executor import DatabaseExecutor, DatabaseConfig
    
    # First parse all SQL statements 
    (predict_ast, train_ast, validate_ast, sql_generator, target_column, 
     task_type, table_name, col_name, target_table_name, predict_where_condition, all_tables) = \
        _parse_tl_sql_statements(train_sql, validate_sql, predict_sql)
    
    use_kfold = (validate_sql is None)
    
    # Load required data based on WHERE conditions 
    executor = DatabaseExecutor(DatabaseConfig(**db_config))
    
    with executor:
        pkeys = executor.get_primary_keys(target_table_name)
        target_pkey = pkeys[0] if pkeys else None
        
        test_df = _load_test_data(executor, target_table_name, predict_where_condition)
        
        train_data = _load_train_data_dict(
            executor=executor,
            train_ast=train_ast,
            sql_generator=sql_generator,
            target_table_name=target_table_name,
            target_pkey=target_pkey,
            test_df=test_df
        )
        
        validate_data = _load_validate_data_dict(
            executor=executor,
            validate_ast=validate_ast,
            sql_generator=sql_generator
        )
    
    
    if not train_data:
        raise ValueError("Unable to load training data")
    
    # Build TableData based on filtering results 
    target_table, y, non_table_embeddings, adj, train_mask, val_mask, test_mask = prepare_bridge_data(
        train_data=train_data,
        validate_data=validate_data,
        test_data=test_df,
        target_column=target_column,
        task_type=task_type,
        device=device,
        db_config=db_config
    )
    
    target_table.train_mask = train_mask
    target_table.val_mask = val_mask
    target_table.test_mask = test_mask
    target_table.y = y
    
    # If validation set is missing and test set exists, enable k-fold cross-validation 
    if not use_kfold and val_mask.sum() == 0 and test_mask.sum() > 0:
        use_kfold = True
    
    emb_size = non_table_embeddings.size(1) if non_table_embeddings is not None else 128
    return target_table, non_table_embeddings, adj, emb_size, use_kfold


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get data using tl_sql ")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--wd", type=float, default=1e-4, help="Weight decay")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    db_config = {
        'db_type': 'mysql',
        'host': 'localhost',
        'port': 3306,
        'database': 'tml1m',
        'username': 'root',
        'password': 'cfy1007'  
    }
    


    
    # PREDICT:
    predict_sql = """
    PREDICT VALUE(Age, CLF)
    FROM users
    WHERE users.Gender='F' or users.userID in(1,2,3,4,5,6,7)
    """
    
    #  """
    # PREDICT VALUE(papers.conference, CLF)
    # FROM papers
    # WHERE  
    # year=2001
    # """
    # train_sql=None
    
    train_sql = """
    TRAIN WITH (users.*, movies.*, ratings.*)
    FROM Tables(users, movies, ratings)
    WHERE users.Gender='M' and movies.Year >=2000  and ratings.rating >4
    """

    # validate_sql = None
    validate_sql = """
    VALIDATE WITH (users.*,movies.*,ratings.userID)
    FROM Tables(users,movies,ratings)
    WHERE users.Gender='M' and movies.Year < 2000 and ratings.rating<4
    """
    
    target_table, non_table_embeddings, adj, emb_size, use_kfold = prepare_data_from_tl_sql(
        train_sql=train_sql,
        validate_sql=validate_sql,
        predict_sql=predict_sql,
        db_config=db_config,
        device=device
    )
    
    results = {}
    
    print("1. Random guessing")
    random_model = RandomBaseline(target_table.num_classes)
    val_acc_random, test_acc_random = evaluate_random_baseline(
        random_model, target_table.y, 
        target_table.train_mask, target_table.val_mask, target_table.test_mask
    )
    results['Random'] = {'val_acc': val_acc_random, 'test_acc': test_acc_random}
    
    print("2. BRIDGE model")
    bridge_model = build_bridge_model(target_table.num_classes, target_table.metadata, emb_size).to(device)
    
    if use_kfold:
        print("Train with k-fold cross-validation")
        fold_results = train_bridge_model_with_kfold(
            bridge_model, target_table, non_table_embeddings, adj, 
            args.epochs, args.lr, args.wd, k_folds=5
        )
        val_accs = [r['best_val_acc'] for r in fold_results]
        test_accs = [r['test_acc'] for r in fold_results]
        val_acc_bridge = np.mean(val_accs)
        test_acc_bridge = np.mean(test_accs)
    else:
        _, val_acc_bridge, test_acc_bridge = train_bridge_model(
            bridge_model, target_table, non_table_embeddings, adj, 
            args.epochs, args.lr, args.wd
        )
    results['BRIDGE'] = {'val_acc': val_acc_bridge, 'test_acc': test_acc_bridge}

