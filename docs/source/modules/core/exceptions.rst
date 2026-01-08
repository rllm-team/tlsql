Exceptions
===========

TLSQL uses a hierarchy of custom exceptions for error handling. All exceptions inherit from
:class:`TLSQLError` and include line/column information when available.

Exception Hierarchy
-------------------

.. code-block:: text

    TLSQLError (base class)
    ├── LexerError (lexical analysis errors)
    ├── ParseError (parsing errors)
    └── GenerationError (SQL generation errors)

.. autoclass:: tlsql.tlsql.exceptions.TLSQLError
   :no-members:
   :no-inherited-members:
   :show-inheritance:

.. autoclass:: tlsql.tlsql.exceptions.LexerError
   :no-members:
   :no-inherited-members:
   :show-inheritance:

.. autoclass:: tlsql.tlsql.exceptions.ParseError
   :no-members:
   :no-inherited-members:
   :show-inheritance:

.. autoclass:: tlsql.tlsql.exceptions.GenerationError
   :no-members:
   :no-inherited-members:
   :show-inheritance:
