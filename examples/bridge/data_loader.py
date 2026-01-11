"""Data Loading Utilities for TLSQL Statements"""

import tlsql
from tlsql.examples.bridge.data_preparer import prepare_bridge_data
from tlsql.examples.executor.db_executor import DatabaseExecutor, DatabaseConfig


def _load_data(executor, sqls):
    """Load data using SQL from convert result."""
    if not sqls or not sqls.sql_list:
        return {}
    data_dict = {}
    for gen_sql in sqls.sql_list:
        result = executor.execute(gen_sql.sql)
        if result.success:
            data_dict[gen_sql.table] = result.data
    return data_dict


def prepare_data_from_tlsql(train_tlsql, validate_tlsql, predict_tlsql, db_config, device):
    """Get data and prepare in format required by bridge model.

    Args:
        train_tlsql: TRAIN TLSQL statement
        validate_tlsql: VALIDATE TLSQL statement
        predict_tlsql: PREDICT TLSQL statement
        db_config: Database configuration dictionary
        device: Device (CPU/GPU)

    Returns:
        tuple: (target_table, non_table_embeddings, adj, emb_size)
    """
    predict_sqls = tlsql.convert(predict_tlsql)
    train_sqls = tlsql.convert(train_tlsql)
    validate_sqls = tlsql.convert(validate_tlsql)

    executor = DatabaseExecutor(DatabaseConfig(**db_config))
    with executor:
        train_data = _load_data(executor, train_sqls)
        validate_data = _load_data(executor, validate_sqls)
        test_data = _load_data(executor, predict_sqls)

    test_df = list(test_data.values())[0] if test_data else None
    target_table, non_table_embeddings, adj = prepare_bridge_data(
        train_data, validate_data, test_df, predict_sqls.target_column, device
    )

    return target_table, non_table_embeddings, adj, non_table_embeddings.size(1)
