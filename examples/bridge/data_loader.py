"""Data Loading Utilities for TLSQL Statements

"""

import torch

from tlsql.examples.bridge.data_preparer import prepare_bridge_data


def _parse_tlsql_statements(train_sql, validate_sql, predict_sql):
    """Parse all TLSQL statements and extract key information using convert API

    """
    import tlsql

    if not predict_sql:
        raise ValueError("PREDICT SQL statement must be specified")
    if not train_sql:
        raise ValueError("TRAIN SQL statement must be specified")
    if not validate_sql:
        raise ValueError("VALIDATE SQL statement must be specified")

    predict_result = tlsql.convert(predict_sql)
    train_result = tlsql.convert(train_sql)
    validate_result = tlsql.convert(validate_sql)

    if not predict_result.is_predict:
        raise ValueError("predict_sql must be a PREDICT statement")
    if not train_result.is_train:
        raise ValueError("train_sql must be a TRAIN statement")
    if not validate_result.is_validate:
        raise ValueError("validate_sql must be a VALIDATE statement")

    target_column = predict_result.target_column
    task_type = predict_result.task_type
    target_table_name = predict_result.target_table
    predict_where_condition = predict_result.where_condition

    table_name, col_name = target_column.split('.', 1)

    all_tables = train_result.tables.copy()
    if target_table_name and target_table_name not in all_tables:
        all_tables.append(target_table_name)

    return (predict_result, train_result, validate_result, target_column,
            task_type, table_name, col_name, target_table_name,
            predict_where_condition, all_tables)


def _execute_sql_list(executor, sql_list, error_prefix="load failed"):
    """Execute SQL list and return data dictionary"""
    data_dict = {}
    for gen_sql in sql_list:
        result = executor.execute(gen_sql.sql)
        if result.success:
            data_dict[gen_sql.table] = result.data
        else:
            raise Exception(f"{error_prefix}: {gen_sql.table}: {result.error}")
    return data_dict


def _load_test_data(executor, target_table_name, predict_where_condition):
    """Load test data"""
    if predict_where_condition:
        sql = f"SELECT * FROM {target_table_name} WHERE {predict_where_condition}"
    else:
        sql = f"SELECT * FROM {target_table_name}"
    result = executor.execute(sql)
    if result.success:
        return result.data
    raise Exception(f"Failed to get test data: {result.error}")


def _load_train_data_dict(executor, train_result):
    """Load training data"""
    if not train_result.sql_list:
        raise ValueError("TRAIN statement did not generate SQL list")
    train_data = _execute_sql_list(executor, train_result.sql_list, "Failed to load training data")
    return train_data


def _load_validate_data_dict(executor, validate_result):
    """Load validation data dictionary based on SQL conditions"""
    if not validate_result.sql_list:
        raise ValueError("VALIDATE statement did not generate SQL list")
    validate_data = _execute_sql_list(executor, validate_result.sql_list, "Failed to load validation data")
    return validate_data


def prepare_data_from_tlsql(
    train_sql: str,
    validate_sql: str,
    predict_sql: str,
    db_config: dict,
    device: torch.device
):
    """Get data and prepare in format required by bridge model

    Args:
        train_sql: TRAIN SQL statement (required)
        validate_sql: VALIDATE SQL statement (required)
        predict_sql: PREDICT SQL statement (required)
        db_config: Database configuration dictionary
        device: Device (CPU/GPU)

    Returns:
        tuple: (target_table, non_table_embeddings, adj, emb_size)
        where target_table contains train_mask, val_mask, test_mask and y attributes
    """
    from tlsql.examples.executor.db_executor import DatabaseExecutor, DatabaseConfig

    (predict_result, train_result, validate_result, target_column,
     task_type, table_name, col_name, target_table_name,
     predict_where_condition, all_tables) = \
        _parse_tlsql_statements(train_sql, validate_sql, predict_sql)

    executor = DatabaseExecutor(DatabaseConfig(**db_config))

    with executor:
        test_df = _load_test_data(executor, target_table_name, predict_where_condition)

        train_data = _load_train_data_dict(
            executor=executor,
            train_result=train_result
        )

        validate_data = _load_validate_data_dict(
            executor=executor,
            validate_result=validate_result
        )

    if not train_data:
        raise ValueError("Unable to load training data")

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
