TLSQL Documentation
===================

TLSQL is a Python library that converts custom SQL-like statements into standard SQL queries for machine learning workflows. It provides a declarative syntax for specifying training, validation, and test datasets in machine learning pipelines.

Overview
--------

TLSQL uses three types of statements to divide your dataset:

- **PREDICT VALUE**: Specifies the test set, target column and task type (REQUIRED)
- **TRAIN WITH**: Specifies the training set (OPTIONAL - defaults to all data except PREDICT data)
- **VALIDATE WITH**: Specifies the validation set (OPTIONAL - defaults to k=5 fold cross validation)

Quick Start
-----------

.. code-block:: python

    import tlsql
    
    result = tlsql.convert("PREDICT VALUE(users.Age, CLF) FROM users")
    print(result.statement_type)  # 'PREDICT'
    print(result.target_column)   # 'users.Age'
    print(result.task_type)       # 'CLF'

Components
----------

TLSQL consists of the following main components:

**Core Components:**

- :doc:`modules/core/lexer` - Tokenizes TLSQL text into tokens
- :doc:`modules/core/parser` - Parses tokens into Abstract Syntax Tree (AST)
- :doc:`modules/core/sql_generator` - Generates standard SQL from AST
- :doc:`modules/core/ast_nodes` - AST node definitions
- :doc:`modules/core/tokens` - Token type definitions
- :doc:`modules/core/exceptions` - Exception classes for error handling

**API:**

- :py:func:`tlsql.convert` - Main conversion function

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   modules/core

Indices and Tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
