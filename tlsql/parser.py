"""Parser for TLSQL syntax.

Supports three statement types:
1. TRAIN WITH - training.
2. PREDICT VALUE - prediction.
3. VALIDATE WITH - validation.
"""

from typing import Optional
from .lexer import Lexer
from .tokens import Token, TokenType
from .ast_nodes import (
    Statement,
    TrainStatement,
    PredictStatement,
    ValidateStatement,
    ColumnReference,
    ColumnSelector,
    WithClause,
    TablesClause,
    ValueClause,
    FromClause,
    PredictType,
    WhereClause,
    Expr,
    LiteralExpr,
    ColumnExpr,
    BinaryExpr,
    UnaryExpr,
    BetweenExpr,
    InExpr,
)
from .exceptions import ParseError


class Parser:
    """Parser for TLSQL syntax.

    Uses a recursive descent parser that supports TRAIN, PREDICT, and VALIDATE statements.

    Attributes:
        tokens: Token list.
        pos: Current token index.
        current_token: Token currently being processed.
    """

    def __init__(self, text: str):
        """Initialize parser.

        Args:
            text: Input text to parse.
        """
        lexer = Lexer(text)
        self.tokens = lexer.tokenize()
        self.pos = 0
        self.current_token = self.tokens[0] if self.tokens else None

    def advance(self) -> None:
        """Advance to next token.

        Moves to the next token in the stream.
        """
        self.pos += 1
        self.current_token = self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def peek(self, offset: int = 1) -> Optional[Token]:
        """Look ahead without consuming token.

        Args:
            offset: Offset for lookahead.

        Returns:
            Token at the specified position or None if out-of-range.
        """
        peek_pos = self.pos + offset
        return self.tokens[peek_pos] if peek_pos < len(self.tokens) else None

    def expect(self, token_type: TokenType) -> Token:
        """Expect and consume a token type.

        Consumes and returns the token if it matches.

        Args:
            token_type: Expected token type.

        Returns:
            Matching token object.

        Raises:
            ParseError: Raised when token type mismatches expectation or EOF reached.
        """
        if self.current_token is None:
            raise ParseError(
                f"Expected {token_type.name}, but reached end of input",
                0,
                0
            )
        if self.current_token.type != token_type:
            raise ParseError(
                f"Expected {token_type.name}, got {self.current_token.type.name}",
                self.current_token.line,
                self.current_token.column
            )
        token = self.current_token
        self.advance()
        return token

    def match(self, *token_types: TokenType) -> bool:
        """Check whether current token matches any given types.

        Args:
            *token_types: Token types to match.

        Returns:
            True if current token matches, else False.
        """
        if self.current_token is None:
            return False
        return self.current_token.type in token_types

    def parse(self) -> Statement:
        """Parse a complete TLSQL statement.

        Determines statement type based on first keyword:
        - TRAIN WITH: training statement.
        - PREDICT VALUE: prediction statement.
        - VALIDATE WITH: validation statement.

        Returns:
            Statement AST root node.
        """
        if self.current_token is None or self.current_token.type == TokenType.EOF:
            raise ParseError(
                "Empty input: expected TRAIN, PREDICT or VALIDATE statement",
                0,
                0
            )

        if self.match(TokenType.TRAIN):
            statement = Statement(train=self.parse_train_statement())
        elif self.match(TokenType.PREDICT):
            statement = Statement(predict=self.parse_predict_statement())
        elif self.match(TokenType.VALIDATE):
            statement = Statement(validate=self.parse_validate_statement())
        else:
            raise ParseError(
                f"Expected TRAIN, PREDICT or VALIDATE, got {self.current_token.type.name if self.current_token else 'EOF'}",
                self.current_token.line if self.current_token else 0,
                self.current_token.column if self.current_token else 0
            )

        if self.current_token and self.current_token.type != TokenType.EOF:
            if self.current_token.type != TokenType.SEMICOLON:
                raise ParseError(
                    f"Unexpected token after statement: {self.current_token.type.name}",
                    self.current_token.line,
                    self.current_token.column
                )
            self.advance()

        return statement

    def _parse_train_or_validate_statement(self, statement_type: str) -> tuple:
        """Parse TRAIN or VALIDATE statement.

        Both statements share the same structure with WITH clause.

        Args:
            statement_type: 'TRAIN' or 'VALIDATE'.

        Returns:
            Tuple (with_clause, tables, where).
        """
        with_clause = self.parse_with_clause()

        tables = self.parse_tables_clause()

        where = None
        if self.match(TokenType.WHERE):
            where = self.parse_where_clause()

        if self.match(TokenType.SEMICOLON):
            self.advance()

        if self.current_token and self.current_token.type != TokenType.EOF:
            raise ParseError(
                f"Unexpected token after {statement_type} statement: {self.current_token.type.name}",
                self.current_token.line,
                self.current_token.column
            )

        return with_clause, tables, where

    def parse_train_statement(self) -> TrainStatement:
        """Parse TRAIN statement.

        Returns:
            TrainStatement node.
        """
        self.expect(TokenType.TRAIN)
        with_clause, tables, where = self._parse_train_or_validate_statement('TRAIN')
        return TrainStatement(with_clause=with_clause, tables=tables, where=where)

    def parse_validate_statement(self) -> ValidateStatement:
        """Parse VALIDATE statement.

        Returns:
            ValidateStatement node.
        """
        self.expect(TokenType.VALIDATE)
        with_clause, tables, where = self._parse_train_or_validate_statement('VALIDATE')
        return ValidateStatement(with_clause=with_clause, tables=tables, where=where)

    def parse_with_clause(self) -> WithClause:
        """Parse WITH clause.

        Syntax: WITH (selector1, selector2, ...).
        Selector format: table.column or table.*.

        Returns:
            WithClause node.
        """
        self.expect(TokenType.WITH)
        self.expect(TokenType.LPAREN)

        selectors = []

        selectors.append(self.parse_column_selector())

        while self.match(TokenType.COMMA):
            self.advance()
            selectors.append(self.parse_column_selector())

        self.expect(TokenType.RPAREN)

        return WithClause(selectors=selectors)

    def parse_column_selector(self) -> ColumnSelector:
        """Parse column selector.

        Format: table.column or table.*.

        Returns:
            ColumnSelector node.
        """
        table_token = self.expect(TokenType.IDENTIFIER)

        self.expect(TokenType.DOT)

        if self.match(TokenType.ASTERISK):
            column = '*'
            self.advance()
        else:
            column_token = self.expect(TokenType.IDENTIFIER)
            column = column_token.value

        return ColumnSelector(table=table_token.value, column=column)

    def parse_tables_clause(self) -> TablesClause:
        """Parse FROM clause for multiple tables.

        Syntax: FROM table1, table2, ...

        Returns:
            TablesClause node.
        """
        self.expect(TokenType.FROM)

        tables = []

        table_token = self.expect(TokenType.IDENTIFIER)
        tables.append(table_token.value)

        while self.match(TokenType.COMMA):
            self.advance()
            table_token = self.expect(TokenType.IDENTIFIER)
            tables.append(table_token.value)

        return TablesClause(tables=tables)

    def parse_predict_statement(self) -> PredictStatement:
        """Parse PREDICT statement.

        Returns:
            PredictStatement node.
        """
        self.expect(TokenType.PREDICT)

        value = self.parse_value_clause()

        from_table = self.parse_from_clause()

        where = None
        if self.match(TokenType.WHERE):
            where = self.parse_where_clause()

        if self.match(TokenType.SEMICOLON):
            self.advance()

        if self.current_token and self.current_token.type != TokenType.EOF:
            raise ParseError(
                f"Unexpected token after PREDICT statement: {self.current_token.type.name}",
                self.current_token.line,
                self.current_token.column
            )

        return PredictStatement(value=value, from_table=from_table, where=where)

    def parse_value_clause(self) -> ValueClause:
        """Parse VALUE clause.

        Returns:
            ValueClause node.
        """
        self.expect(TokenType.VALUE)
        self.expect(TokenType.LPAREN)

        target = self.parse_column_reference()
        self.expect(TokenType.COMMA)

        if self.match(TokenType.CLF):
            predict_type = PredictType(type_name='CLF')
            self.advance()
        elif self.match(TokenType.REG):
            predict_type = PredictType(type_name='REG')
            self.advance()
        else:
            raise ParseError(
                f"Expected CLF or REG, got {self.current_token.type.name if self.current_token else 'EOF'}",
                self.current_token.line if self.current_token else 0,
                self.current_token.column if self.current_token else 0
            )

        self.expect(TokenType.RPAREN)

        return ValueClause(target=target, predict_type=predict_type)

    def parse_from_clause(self) -> FromClause:
        """Parse FROM clause (single table).

        Returns:
            FromClause node.
        """
        self.expect(TokenType.FROM)
        table_token = self.expect(TokenType.IDENTIFIER)
        return FromClause(table=table_token.value)

    def parse_column_reference(self) -> ColumnReference:
        """Parse column reference.

        Format: column_name or table.column_name.

        Returns:
            ColumnReference node.
        """
        first_token = self.expect(TokenType.IDENTIFIER)

        if self.match(TokenType.DOT):
            self.advance()
            column_token = self.expect(TokenType.IDENTIFIER)
            return ColumnReference(table=first_token.value, column=column_token.value)
        else:
            return ColumnReference(column=first_token.value)

    def parse_where_clause(self) -> WhereClause:
        """Parse WHERE clause.

        Returns:
            WhereClause node.
        """
        self.expect(TokenType.WHERE)
        condition_expr = self.parse_where_expression()
        return WhereClause(condition=condition_expr)

    def parse_where_expression(self) -> Expr:
        """Parse WHERE expression (entry point).

        Returns:
            Root of expression tree.
        """
        return self.parse_or_expr()

    def parse_or_expr(self) -> Expr:
        """Parse OR expression.

        Grammar: or_expr = and_expr (OR and_expr)*.
        """
        left = self.parse_and_expr()  # A OR B AND C parsed as A OR (B AND C)

        while self.match(TokenType.OR):
            self.advance()
            right = self.parse_and_expr()
            left = BinaryExpr(left=left, operator='OR', right=right)

        return left

    def parse_and_expr(self) -> Expr:
        """Parse AND expression.

        Grammar: and_expr = not_expr (AND not_expr)*.
        """
        left = self.parse_not_expr()

        while self.match(TokenType.AND):
            self.advance()
            right = self.parse_not_expr()
            left = BinaryExpr(left=left, operator='AND', right=right)

        return left

    def parse_not_expr(self) -> Expr:
        """Parse NOT expression.

        Grammar: not_expr = NOT not_expr | comparison.
        """
        if self.match(TokenType.NOT):
            self.advance()
            operand = self.parse_not_expr()
            return UnaryExpr(operator='NOT', operand=operand)

        return self.parse_comparison_expr()

    def parse_comparison_expr(self) -> Expr:
        """Parse comparison expression.

        Grammar: comparison = primary (comp_op primary | BETWEEN primary AND primary | IN (primary, ...) ).
        """
        left = self.parse_primary_expr()

        if self.match(TokenType.BETWEEN):
            self.advance()
            lower = self.parse_primary_expr()
            self.expect(TokenType.AND)
            upper = self.parse_primary_expr()
            return BetweenExpr(column=left, lower=lower, upper=upper)

        if self.match(TokenType.IN):
            self.advance()
            self.expect(TokenType.LPAREN)
            values = []
            if not self.match(TokenType.RPAREN):
                while True:
                    values.append(self.parse_primary_expr())
                    if self.match(TokenType.COMMA):
                        self.advance()
                    else:
                        break
            self.expect(TokenType.RPAREN)
            return InExpr(column=left, values=values)

        if self.match(TokenType.GT, TokenType.LT, TokenType.GTE,
                      TokenType.LTE, TokenType.EQ, TokenType.NEQ, TokenType.EQUALS):
            operator = self.current_token.value
            self.advance()
            right = self.parse_primary_expr()
            return BinaryExpr(left=left, operator=operator, right=right)

        return left

    def parse_primary_expr(self) -> Expr:
        """Parse primary expression.

        Grammar: primary = literal | column_ref | (expression).
        """
        if self.match(TokenType.LPAREN):
            self.advance()
            expr = self.parse_where_expression()
            self.expect(TokenType.RPAREN)
            return expr

        if self.match(TokenType.NUMBER):
            token = self.current_token
            self.advance()
            try:
                value = int(token.value)
            except ValueError:
                value = float(token.value)
            return LiteralExpr(value=value, value_type='number')

        if self.match(TokenType.STRING):
            token = self.current_token
            self.advance()
            return LiteralExpr(value=token.value, value_type='string')

        if self.match(TokenType.IDENTIFIER):
            return self.parse_column_expr()

        raise ParseError(
            f"Unexpected token in expression: {self.current_token.type.name if self.current_token else 'EOF'}",
            self.current_token.line if self.current_token else 0,
            self.current_token.column if self.current_token else 0
        )

    def parse_column_expr(self) -> ColumnExpr:
        """Parse column expression.

        Returns:
            ColumnExpr node.
        """
        col_ref = self.parse_column_reference()
        return ColumnExpr(column=col_ref)
