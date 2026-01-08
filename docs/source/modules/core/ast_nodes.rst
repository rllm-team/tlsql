AST Nodes
=========

AST (Abstract Syntax Tree) nodes represent parsed TLSQL statements. The AST is organized hierarchically:

- **Statements**: Top-level constructs (TRAIN, PREDICT, VALIDATE).
- **Clauses**: Statement components (WITH, FROM, WHERE, VALUE).
- **Expressions**: Conditional and logical expressions.
- **References**: Column and table references.

Base Classes
------------

.. autoclass:: tlsql.tlsql.ast_nodes.ASTNode
   :no-members:
   :no-inherited-members:
   :show-inheritance:

.. autoclass:: tlsql.tlsql.ast_nodes.Expr
   :no-members:
   :no-inherited-members:
   :show-inheritance:

Column and Reference Classes
----------------------------

.. autoclass:: tlsql.tlsql.ast_nodes.ColumnReference
   :no-members:
   :no-inherited-members:
   :show-inheritance:

.. autoclass:: tlsql.tlsql.ast_nodes.ColumnSelector
   :no-members:
   :no-inherited-members:
   :show-inheritance:

Expression Classes
------------------

.. autoclass:: tlsql.tlsql.ast_nodes.LiteralExpr
   :no-members:
   :no-inherited-members:
   :show-inheritance:

.. autoclass:: tlsql.tlsql.ast_nodes.ColumnExpr
   :no-members:
   :no-inherited-members:
   :show-inheritance:

.. autoclass:: tlsql.tlsql.ast_nodes.BinaryExpr
   :no-members:
   :no-inherited-members:
   :show-inheritance:

.. autoclass:: tlsql.tlsql.ast_nodes.UnaryExpr
   :no-members:
   :no-inherited-members:
   :show-inheritance:

.. autoclass:: tlsql.tlsql.ast_nodes.BetweenExpr
   :no-members:
   :no-inherited-members:
   :show-inheritance:

.. autoclass:: tlsql.tlsql.ast_nodes.InExpr
   :no-members:
   :no-inherited-members:
   :show-inheritance:

Statement Classes
-----------------

.. autoclass:: tlsql.tlsql.ast_nodes.Statement
   :no-members:
   :no-inherited-members:
   :show-inheritance:

.. autoclass:: tlsql.tlsql.ast_nodes.TrainStatement
   :no-members:
   :no-inherited-members:
   :show-inheritance:

.. autoclass:: tlsql.tlsql.ast_nodes.PredictStatement
   :no-members:
   :no-inherited-members:
   :show-inheritance:

.. autoclass:: tlsql.tlsql.ast_nodes.ValidateStatement
   :no-members:
   :no-inherited-members:
   :show-inheritance:

Clause Classes
--------------

.. autoclass:: tlsql.tlsql.ast_nodes.WithClause
   :no-members:
   :no-inherited-members:
   :show-inheritance:

.. autoclass:: tlsql.tlsql.ast_nodes.TablesClause
   :no-members:
   :no-inherited-members:
   :show-inheritance:

.. autoclass:: tlsql.tlsql.ast_nodes.ValueClause
   :no-members:
   :no-inherited-members:
   :show-inheritance:

.. autoclass:: tlsql.tlsql.ast_nodes.FromClause
   :no-members:
   :no-inherited-members:
   :show-inheritance:

.. autoclass:: tlsql.tlsql.ast_nodes.WhereClause
   :no-members:
   :no-inherited-members:
   :show-inheritance:

.. autoclass:: tlsql.tlsql.ast_nodes.PredictType
   :no-members:
   :no-inherited-members:
   :show-inheritance:
