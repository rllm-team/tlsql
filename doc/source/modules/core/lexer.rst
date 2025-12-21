Lexer
=====

The :class:`Lexer` converts TLSQL text into a stream of tokens. It recognizes keywords, identifiers,
literals, operators, and punctuation.

Supported Elements
------------------

- **Keywords**: TRAIN, PREDICT, VALIDATE, WITH, FROM, WHERE
- **Identifiers**: Table and column names
- **Literals**: Strings (with escape sequences) and numbers
- **Operators**: Comparison (>, <, >=, <=, =, !=) and logical (AND, OR, NOT)
- **Comments**: Single-line (``--``) and multi-line (``/* */``)

Example
-------

.. code-block:: python

    from tlsql.tlsql.lexer import Lexer

    lexer = Lexer("PREDICT VALUE(users.Age, CLF) FROM users")
    tokens = lexer.tokenize()
    
    for token in tokens:
        print(f"{token.type.name}: {token.value}")

API Reference
-------------

.. autoclass:: tlsql.tlsql.lexer.Lexer
   :no-members:
   :no-inherited-members:
   :show-inheritance:
