Tokens
======

Token definitions categorize lexical elements in TLSQL statements. The system consists of:

- **TokenType**: Enumeration of all token types.
- **Token**: Data class representing a token with type, value, and position.

Token Categories
----------------

- **Keywords**: TRAIN, PREDICT, VALIDATE, WITH, FROM, WHERE.
- **Operators**: Comparison (>, <, >=, <=, =, !=) and logical (AND, OR, NOT).
- **Literals**: IDENTIFIER, STRING, NUMBER.
- **Punctuation**: Parentheses, commas, semicolons, dots, asterisks.
- **Special**: EOF marker.

.. autoclass:: tlsql.tlsql.tokens.TokenType
   :no-members:
   :no-inherited-members:
   :show-inheritance:

.. autoclass:: tlsql.tlsql.tokens.Token
   :no-members:
   :no-inherited-members:
   :show-inheritance:
   :no-index:
