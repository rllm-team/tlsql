SQL Generator
=============

The SQL Generator module contains the core SQL generation functionality for TLSQL.
:func:`tlsql.convert` and :func:`tlsql.convert_workflow_queries` use these classes internally.

Example
-------

.. code-block:: python

    import tlsql
    
    result = tlsql.convert("PREDICT VALUE(users.Age, CLF) FROM users WHERE users.Gender='F'")
    print(result.statement_type)
    print(result.target_column)
    print(result.sql)
    

Main API
~~~~~~~~

Single-statement conversion: :func:`tlsql.convert`. Workflow conversion: :func:`tlsql.convert_workflow_queries`.

.. autofunction:: tlsql.convert

Source code for **convert**:

.. literalinclude:: ../../../../__init__.py
   :lines: 21-31
   :language: python

.. autofunction:: tlsql.convert_workflow_queries

SQL Generator Class
~~~~~~~~~~~~~~~~~~~

The :class:`SQLGenerator` class provides the core SQL generation functionality:

.. autoclass:: tlsql.tlsql.sql_generator.SQLGenerator
   :members: convert
   :exclude-members: __init__, generate, build, generate_train_sql, generate_validate_sql, generate_predict_sql, auto_generate_train, _build_select_sql, _expr_to_sql, _group_columns_by_table, _split_where_by_table, _extract_and_conditions, _extract_table_from_expr
   :no-inherited-members:
   :show-inheritance:

Result Classes
~~~~~~~~~~~~~~

.. autoclass:: tlsql.tlsql.sql_generator.ConversionResult
   :no-members:
   :no-inherited-members:
   :show-inheritance:
   :noindex:

.. autoclass:: tlsql.tlsql.sql_generator.StatementResult
   :no-members:
   :no-inherited-members:
   :show-inheritance:
   :noindex:

.. autoclass:: tlsql.tlsql.sql_generator.GeneratedSQL
   :no-members:
   :no-inherited-members:
   :show-inheritance:
   :noindex:
