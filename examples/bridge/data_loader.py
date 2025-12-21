"""Data Loading Utilities for TLSQL Statements

"""

import torch

from tlsql.examples.bridge.data_preparer import prepare_bridge_data


def _parse_tlsql_statements(train_tlsql, validate_tlsql, predict_tlsql):
    """Parse all TLSQL statements and return conversion results.
    
    Returns:
        tuple: (predict_sqls, train_sqls, validate_sqls).
    """
    import tlsql

    predict_sqls = tlsql.convert(predict_tlsql) if predict_tlsql else None
    train_sqls = tlsql.convert(train_tlsql) if train_tlsql else None
    validate_sqls = tlsql.convert(validate_tlsql) if validate_tlsql else None

    return predict_sqls, train_sqls, validate_sqls


def _execute_sql_list(executor, sql_list, error_prefix="load failed"):
    """Execute SQL list and return data dictionary."""
    data_dict = {}
    if not sql_list:
        return data_dict
    for gen_sql in sql_list:
        result = executor.execute(gen_sql.sql)
        if result.success:
            data_dict[gen_sql.table] = result.data
    return data_dict


def _load_test_data(executor, predict_sqls):
    """Load test data using SQL from convert result.
    
    Returns:
        dict: Dictionary mapping table names to DataFrames, e.g., {tablename: df, ...}.
    """
    if not predict_sqls or not predict_sqls.sql_list:
        return {}
    test_data = _execute_sql_list(executor, predict_sqls.sql_list, "Failed to load test data")
    return test_data


def _load_train_data(executor, train_sqls):
    """Load training data using SQL from convert result.
    
    Returns:
        dict: Dictionary mapping table names to DataFrames, e.g., {tablename: df, ...}.
    """
    if not train_sqls or not train_sqls.sql_list:
        return {}
    train_data = _execute_sql_list(executor, train_sqls.sql_list, "Failed to load training data")
    return train_data


def _load_validate_data(executor, validate_sqls):
    """Load validation data using SQL from convert result
    
    Returns:
        dict: Dictionary mapping table names to DataFrames, e.g., {tablename: df, ...}.
    """
    if not validate_sqls or not validate_sqls.sql_list:
        return {}
    validate_data = _execute_sql_list(executor, validate_sqls.sql_list, "Failed to load validation data")
    return validate_data


def prepare_data_from_tlsql(
    train_tlsql: str,
    validate_tlsql: str,
    predict_tlsql: str,
    db_config: dict,
    device: torch.device
):
    """Get data and prepare in format required by bridge model.

    Args:
        train_tlsql: TRAIN TLSQL statement (required).
        validate_tlsql: VALIDATE TLSQL statement (required).
        predict_tlsql: PREDICT TLSQL statement (required).
        db_config: Database configuration dictionary.
        device: Device (CPU/GPU).

    Returns:
        tuple: (target_table, non_table_embeddings, adj, emb_size),
        where target_table contains train_mask, val_mask, test_mask and y attributes.
    """
    from tlsql.examples.executor.db_executor import DatabaseExecutor, DatabaseConfig

    predict_sqls, train_sqls, validate_sqls = _parse_tlsql_statements(train_tlsql, validate_tlsql, predict_tlsql)
    target_column = predict_sqls.target_column if predict_sqls else None
    task_type = predict_sqls.task_type if predict_sqls else None

    executor = DatabaseExecutor(DatabaseConfig(**db_config))

    with executor:
        test_data = _load_test_data(executor, predict_sqls)

        train_data = _load_train_data(
            executor=executor,
            train_sqls=train_sqls
        )

        validate_data = _load_validate_data(
            executor=executor,
            validate_sqls=validate_sqls
        )

    test_df = list(test_data.values())[0] if test_data else None

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

    emb_size = non_table_embeddings.size(1) if non_table_embeddings is not None else 128
    return target_table, non_table_embeddings, adj, emb_size
