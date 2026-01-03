SQL Generator
=============

The SQL Generator converts parsed TLSQL AST into standard SQL statements. It provides the main
conversion function :func:`tlsql.convert` and supporting classes for representing results.

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

Main Function
~~~~~~~~~~~~~

.. autofunction:: tlsql.convert

Core Classes
~~~~~~~~~~~~

.. autoclass:: tlsql.tlsql.sql_generator.SQLGenerator
   :no-members:
   :no-inherited-members:
   :show-inheritance:

.. autoclass:: tlsql.tlsql.sql_generator.ConversionResult
   :no-members:
   :no-inherited-members:
   :show-inheritance:

.. autoclass:: tlsql.tlsql.sql_generator.GeneratedSQL
   :no-members:
   :no-inherited-members:
   :show-inheritance:
