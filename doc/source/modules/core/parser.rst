Parser
======

The :class:`Parser` converts tokens into an Abstract Syntax Tree (AST). It supports three statement types:

- **TRAIN WITH**: Training data selection.
- **PREDICT VALUE**: Prediction targets and task types.
- **VALIDATE WITH**: Validation data selection.

Example
-------

.. code-block:: python

    from tlsql.tlsql.parser import Parser

    parser = Parser("PREDICT VALUE(users.Age, CLF) FROM users WHERE users.Gender = 'F'")
    ast = parser.parse()
    
    if ast.predict:
        print(f"Target: {ast.predict.value.target}")
        print(f"Task Type: {ast.predict.value.predict_type.type_name}")

.. autoclass:: tlsql.tlsql.parser.Parser
   :members: parse
   :exclude-members: __init__, advance, peek, expect, match, _parse_train_or_validate_statement, parse_train_statement, parse_validate_statement, parse_with_clause, parse_column_selector, parse_tables_clause, parse_predict_statement, parse_value_clause, parse_from_clause, parse_column_reference, parse_where_clause, parse_where_expression, parse_or_expr, parse_and_expr, parse_not_expr, parse_comparison_expr, parse_primary_expr, parse_column_expr, tokens, pos, current_token
   :special-members:
   :no-inherited-members:
   :show-inheritance:
