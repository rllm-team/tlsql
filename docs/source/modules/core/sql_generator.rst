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
   :no-members:
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
