TL-SQL Documentation
====================

TL-SQL is a library for converting custom SQL statements into standard SQL.

The library supports three types of custom SQL statements:
1. **TRAIN WITH** - Training data queries
2. **PREDICT VALUE** - Prediction target queries
3. **VALIDATE WITH** - Validation data queries

Quick Start
-----------

.. code-block:: python

    from tl_sql import Parser, SQLGenerator
    from tl_sql.executor import TlSqlPipeline, DatabaseExecutor, DatabaseConfig
    
    # Parse TL-SQL
    parser = Parser("PREDICT VALUE(users.Age, CLF) FROM users")
    ast = parser.parse()
    
    # Generate SQL
    generator = SQLGenerator()
    result = generator.generate(ast)
    
    # Use Pipeline
    pipeline = TlSqlPipeline(db_config)
    result = pipeline.run(train_sql, predict_sql, validate_sql)

Core Concepts
-------------

TL-SQL provides the following core functionality:

- **Lexer**: Tokenizes SQL text into tokens
- **Parser**: Parses tokens into Abstract Syntax Tree (AST)
- **SQL Generator (SQLGenerator)**: Converts AST to standard SQL
- **Database Executor (DatabaseExecutor)**: Executes SQL and returns results
- **Pipeline**: High-level interface that integrates all functionality

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Modules

   modules/core
   modules/executor

Indices and Tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

