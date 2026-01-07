SQL Generator
=============

The SQL Generator module contains the core SQL generation functionality for TLSQL.
The main :func:`tlsql.convert` API function uses these classes internally to convert
TLSQL statements into standard SQL with full metadata.

Example
-------

.. code-block:: python

    import tlsql

    result = tlsql.convert("PREDICT VALUE(users.Age, CLF) FROM users")
    
    print(f"Statement Type: {result.statement_type}")
    print(f"Target Column: {result.target_column}")
    
    if result.sql_list:
        for sql_obj in result.sql_list:
            print(f"SQL: {sql_obj.sql}")

Main API
~~~~~~~~

The main conversion functionality is provided through the global :func:`tlsql.convert` function.

.. autofunction:: tlsql.convert

SQL Generator Class
~~~~~~~~~~~~~~~~~~~

The :class:`SQLGenerator` class provides the core SQL generation functionality:

.. autoclass:: tlsql.tlsql.sql_generator.SQLGenerator
   :members: convert
   :exclude-members: __init__, generate, generate_with_metadata, _generate_train_result, _generate_validate_result, _generate_predict_result, generate_train_sql, generate_validate_sql, _group_columns_by_table, _split_where_by_table, _extract_and_conditions, _extract_table_from_expr, _build_select_sql, generate_predict_sql, _expr_to_sql
   :no-inherited-members:
   :show-inheritance:

Result Classes
~~~~~~~~~~~~~~

.. autoclass:: tlsql.tlsql.sql_generator.ConversionResult
   :no-members:
   :no-inherited-members:
   :show-inheritance:
   :noindex:

.. autoclass:: tlsql.tlsql.sql_generator.GeneratedSQL
   :no-members:
   :no-inherited-members:
   :show-inheritance:
   :noindex:
