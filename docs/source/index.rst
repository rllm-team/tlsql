TLSQL Documentation
===================

TLSQL is a system designed to simplify machine learning workflows on structured tabular data. It translates SQL-like statements into standard SQL queries and structured learning task descriptions, enabling data scientists and engineers to focus on model development instead of writing complex SQL or manually managing datasets.

TLSQL works seamlessly with **relational databases, data warehouses, and data lakes**, enabling end-to-end table-based ML workflows.



Overview
--------

TLSQL supports three types of statements that map directly to ML workflows:

- **PREDICT VALUE**: Test set.
- **TRAIN WITH**: Training set.
- **VALIDATE WITH**: Validation set.

.. image:: _static/workflow.svg
   :alt: TLSQL Workflow
   :width: 700px
   :align: center

.. centered:: **The TLSQL Workflow**

Quick Start
-----------

.. code-block:: python

    import tlsql

    # Single statement: convert one TLSQL to SQL
    result = tlsql.convert("PREDICT VALUE(users.Age, CLF) FROM users WHERE users.Gender='F'")
    print(result.statement_type)   # 'PREDICT'
    print(result.target_column)    # 'users.Age'
    print(result.sql)              # generated SQL

Components
----------

**API:**

- :py:func:`convert <tlsql.convert>` — Convert a single TLSQL statement to SQL.
- :py:func:`convert_workflow_queries <tlsql.convert_workflow_queries>` — Convert a workflow (query_list of PREDICT, TRAIN, VALIDATE) to SQL.

**Core Components:**

- :doc:`modules/core/lexer` - Tokenizes TLSQL text into tokens.
- :doc:`modules/core/parser` - Parses tokens into Abstract Syntax Tree (AST).
- :doc:`modules/core/sql_generator` - Generates standard SQL from AST.
- :doc:`modules/core/ast_nodes` - AST node definitions.
- :doc:`modules/core/tokens` - Token type definitions.
- :doc:`modules/core/exceptions` - Exception classes for error handling.

Contents
--------

.. toctree::
   :maxdepth: 2

   modules/core

Indices and Tables
------------------

* :ref:`genindex`
* :ref:`search`
