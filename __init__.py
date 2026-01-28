"""TLSQL: A SQL-like language designed for relational table learning tasks and workflows.

This package converts TLSQL workflow statements into standard SQL:
- PREDICT VALUE: Prediction target queries (required)
- TRAIN WITH: Training data queries (optional, auto-generated if not provided)
- VALIDATE WITH: Validation data queries (optional)

Usage:
    >>> from tlsql import convert
    >>> result = convert(
    ...     predict_query="PREDICT VALUE(users.Age, CLF) FROM users WHERE users.Gender='F'"
    ... )
    >>> print(result.predict.sql)
    >>> print(result.train.sql)
"""

from typing import Optional
from tlsql.tlsql.sql_generator import SQLGenerator, ConversionResult
__version__ = "0.1.0"
__author__ = "TLSQL Team"


def convert(
    predict_query: str,
    train_query: Optional[str] = None,
    validate_query: Optional[str] = None
) -> ConversionResult:
    """Convert TLSQL workflow statements to standard SQL.

    Converts PREDICT, TRAIN, and VALIDATE TLSQL statements into standard SQL.
    PREDICT is required. If TRAIN is not provided, it will be auto-generated
    by excluding PREDICT data from the same table.

    Args:
        predict_query: PREDICT TLSQL statement (required).
        train_query: TRAIN TLSQL statement (optional, auto-generated if not provided).
        validate_query: VALIDATE TLSQL statement (optional).

    Returns:
        ConversionResult: Contains predict_result, train_result, and validate_result.

    """

    if not predict_query or not predict_query.strip():
        raise ValueError("predict_query is required.")

    generator = SQLGenerator()

    # Process PREDICT
    predict_result = SQLGenerator.convert_query(predict_query)

    # Process TRAIN
    train_result = (
        SQLGenerator.convert_query(train_query)
        if train_query and train_query.strip()
        else generator.auto_generate_train(predict_result)
    )

    # Process VALIDATE
    validate_result = (
        SQLGenerator.convert_query(validate_query)
        if validate_query and validate_query.strip()
        else None
    )

    return ConversionResult(
        predict_result=predict_result,
        train_result=train_result,
        validate_result=validate_result
    )


# Tokens
from tlsql.tlsql.tokens import Token, TokenType

# Core classes
from tlsql.tlsql.lexer import Lexer
from tlsql.tlsql.parser import Parser
from tlsql.tlsql.sql_generator import (
    GeneratedSQL,
    ConversionResult,
)

# AST nodes (all AST components)
from tlsql.tlsql.ast_nodes import (
    ASTNode,
    Statement,
    TrainStatement,
    ValidateStatement,
    PredictStatement,
    ValueClause,
    FromClause,
    WhereClause,
    ColumnSelector,
    WithClause,
    TablesClause,
    BinaryExpr,
    UnaryExpr,
    ColumnExpr,
    LiteralExpr,
    BetweenExpr,
    InExpr,
    ColumnReference,
    PredictType,
)

# Exceptions
from tlsql.tlsql.exceptions import (
    TLSQLError,
    LexerError,
    ParseError,
    GenerationError,
)

__all__ = [
    # Top-level API
    "convert",
    # Tokens
    "Token",
    "TokenType",
    # Core classes
    "Lexer",
    "Parser",
    "SQLGenerator",
    "GeneratedSQL",
    "ConversionResult",
    # AST nodes
    "ASTNode",
    "Statement",
    "TrainStatement",
    "ValidateStatement",
    "PredictStatement",
    "ValueClause",
    "FromClause",
    "WhereClause",
    "ColumnSelector",
    "WithClause",
    "TablesClause",
    "BinaryExpr",
    "UnaryExpr",
    "ColumnExpr",
    "LiteralExpr",
    "BetweenExpr",
    "InExpr",
    "ColumnReference",
    "PredictType",
    # Exceptions
    "TLSQLError",
    "LexerError",
    "ParseError",
    "GenerationError",
]
