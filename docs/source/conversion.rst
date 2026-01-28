Conversion
==========

This section shows the conversion process of a TLSQL statement to standard SQL, using a PREDICT VALUE statement as an example.

Example: PREDICT VALUE Statement
---------------------------------

**TLSQL:**

.. code-block:: sql

    PREDICT VALUE(users.Age, CLF)
    FROM users
    WHERE users.Gender = 'F'

Conversion Process
------------------

The conversion process consists of four main steps:

1. **Lexical Analysis (Tokenization)**
   
   **Used Class:** :class:`Lexer`
   
   The TLSQL text is tokenized into a stream of tokens using the ``Lexer`` class.

   **Tokens Stream:**
   
   - ``PREDICT`` (keyword)
   - ``VALUE`` (keyword)
   - ``(`` (left parenthesis)
   - ``users`` (identifier)
   - ``.`` (dot)
   - ``Age`` (identifier)
   - ``,`` (comma)
   - ``CLF`` (keyword)
   - ``)`` (right parenthesis)
   - ``FROM`` (keyword)
   - ``users`` (identifier)
   - ``WHERE`` (keyword)
   - ``users`` (identifier)
   - ``.`` (dot)
   - ``Gender`` (identifier)
   - ``=`` (equals)
   - ``'F'`` (string literal)

2. **Syntax Analysis (Parsing)**
   
   **Used Class:** :class:`Parser`
   
   The token stream is parsed into an Abstract Syntax Tree (AST) using the ``Parser`` class.

   **AST Structure:**
   
   .. code-block:: python

       PredictStatement(
         value=ValueClause(
           target=ColumnReference(table='users', column='Age'),
           predict_type=PredictType(type_name='CLF')
         ),
         from_table=FromClause(table='users'),
         where=WhereClause(
           condition=BinaryExpr(
               op='=',
               left=ColumnExpr(column=ColumnReference(table='users', column='Gender')),
             right=LiteralExpr(value='F')
           )
         )
       )

3. **SQL Generation**
   
   **Used Class:** :class:`SQLGenerator`
   
   The AST is traversed to generate standard SQL components using the ``SQLGenerator`` class.

   **Extracted Information:**
   
   - Target Table: ``users``.
   - Target Column: ``users.Age``.
   - Task Type: ``CLF`` (Classification).
   - WHERE Condition: ``Gender = 'F'``.

4. **Result Assembly**
   
   **Used Classes:** :class:`ConversionResult` and :class:`StatementResult`
   
   The final standard SQL statement is constructed and wrapped in a ``StatementResult`` object, which contains all extracted metadata for a single statement. For workflow conversions, multiple ``StatementResult`` objects are wrapped in a ``ConversionResult`` object.

**Standard SQL:**

.. code-block:: sql

    SELECT * FROM users WHERE Gender = 'F'

**Conversion Result:**

The conversion returns a ``ConversionResult`` object containing:

- ``predict_result``: A ``StatementResult`` object with:
  - ``statement_type``: ``'PREDICT'``
  - ``target_column``: ``'users.Age'``
  - ``task_type``: ``'CLF'``
  - ``target_table``: ``'users'``
  - ``where_condition``: ``"Gender = 'F'"``
  - ``sql_list``: List of ``GeneratedSQL`` objects
- ``train_result``: A ``StatementResult`` object for TRAIN statement
- ``validate_result``: An optional ``StatementResult`` object for VALIDATE statement (None if not provided)

You can access these results using the shortcut properties:

- ``result.predict.sql`` - Get PREDICT SQL string
- ``result.train.sql`` - Get TRAIN SQL string  
- ``result.validate.sql`` - Get VALIDATE SQL string (returns None if not provided)

