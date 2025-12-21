Parser
======

The :class:`Parser` converts tokens into an Abstract Syntax Tree (AST). It supports three statement types:

- **TRAIN WITH**: Training data selection
- **PREDICT VALUE**: Prediction targets and task types
- **VALIDATE WITH**: Validation data selection

Example
-------

.. code-block:: python

    from tlsql.tlsql.parser import Parser

    parser = Parser("PREDICT VALUE(users.Age, CLF) FROM users WHERE users.Gender = 'F'")
    ast = parser.parse()
    
    if ast.predict:
        print(f"Target: {ast.predict.value.target}")
        print(f"Task Type: {ast.predict.value.predict_type.type_name}")

API Reference
-------------

.. autoclass:: tlsql.tlsql.parser.Parser
   :no-members:
   :no-inherited-members:
   :show-inheritance:
