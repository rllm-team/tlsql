SQL Generator
=============

The SQL Generator module contains the core SQL generation functionality for TLSQL.
The main :func:`tlsql.convert` API function uses these classes internally to convert
TLSQL statements into standard SQL with full metadata.

Example
-------

.. code-block:: python

    import tlsql

    
    result = tlsql.convert(
        predict_query="PREDICT VALUE(users.Age, CLF) FROM users WHERE users.Gender='F'"
    )
    
    print(f"PREDICT Statement Type: {result.predict.statement_type}")
    print(f"Target Column: {result.predict.target_column}")
    print(f"PREDICT SQL: {result.predict.sql}")
    

Main API
~~~~~~~~

The main conversion functionality is provided through the global :func:`tlsql.convert` function.

.. autofunction:: tlsql.convert

SQL Generator Class
~~~~~~~~~~~~~~~~~~~

The :class:`SQLGenerator` class provides the core SQL generation functionality:

.. autoclass:: tlsql.tlsql.sql_generator.SQLGenerator
   :members: convert_query
   :exclude-members: __init__, generate, build, generate_train_sql, generate_validate_sql, generate_predict_sql, auto_generate_train, _build_select_sql, _expr_to_sql, _group_columns_by_table, _split_where_by_table, _extract_and_conditions, _extract_table_from_expr
   :no-inherited-members:
   :show-inheritance:

Result Classes
~~~~~~~~~~~~~~

.. autoclass:: tlsql.tlsql.sql_generator.ConversionResult
   :members: predict, train, validate
   :no-inherited-members:
   :show-inheritance:
   :noindex:

.. autoclass:: tlsql.tlsql.sql_generator.StatementResult
   :members: sql, get_sql, format_sql_list, is_predict, is_train, is_validate
   :no-inherited-members:
   :show-inheritance:
   :noindex:

.. autoclass:: tlsql.tlsql.sql_generator.GeneratedSQL
   :no-members:
   :no-inherited-members:
   :show-inheritance:
   :noindex:
