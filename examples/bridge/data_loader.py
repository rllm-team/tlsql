"""Data Loading Utilities for TLSQL Statements"""

import tlsql
from tlsql.examples.bridge.data_preparer import prepare_bridge_data
from tlsql.examples.executor.db_executor import DatabaseExecutor, DatabaseConfig


def _load_data(executor, sqls):
    """Load data using SQL from convert_workflow_queries result."""
    if not sqls or not sqls.sql_list:
        return {}
    data_dict = {}
    for gen_sql in sqls.sql_list:
        result = executor.execute(gen_sql.sql)
        if result.success:
            data_dict[gen_sql.table] = result.data
    return data_dict


def prepare_data_from_tlsql(predict_query, train_query, validate_query, db_config, device, table_list=None):
    """Get data and prepare in format required by bridge model.

    Args:
        predict_query: PREDICT TLSQL statement
        train_query: TRAIN TLSQL statement
        validate_query: VALIDATE TLSQL statement
        db_config: Database configuration dictionary
        device: Device (CPU/GPU)
        table_list: List of table names (required when train_query is None)

    Returns:
        tuple: (target_table, non_table_embeddings, adj, emb_size)
    """
    result = tlsql.convert_workflow_queries(
        query_list=[predict_query, train_query, validate_query],
        table_list=table_list
    )

    executor = DatabaseExecutor(DatabaseConfig(**db_config))
    with executor:
        train_data = _load_data(executor, result.train_result)
        validate_data = _load_data(executor, result.validate_result)
        test_data = _load_data(executor, result.predict_result)

    test_df = list(test_data.values())[0]
    target_table, non_table_embeddings, adj = prepare_bridge_data(
        train_data, validate_data, test_df, result.predict_result.target_column, device
    )

    return target_table, non_table_embeddings, adj, non_table_embeddings.size(1)
