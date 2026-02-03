"""TLSQL: A SQL-like language designed for relational table learning tasks and workflows.

This package converts TLSQL workflow statements into standard SQL:
- PREDICT VALUE: Prediction target queries (required)
- TRAIN WITH: Training data queries (optional, auto-generated if not provided)
- VALIDATE WITH: Validation data queries (optional)

Usage:
    >>> from tlsql import convert
    >>> result = convert("PREDICT VALUE(users.Age, CLF) FROM users WHERE users.Gender='F'")
    >>> print(result.statement_type)
    >>> print(result.sql)
"""

from typing import Optional, List
from tlsql.tlsql.sql_generator import SQLGenerator, ConversionResult, StatementResult
__version__ = "0.1.0"
__author__ = "TLSQL Team"


def convert(tlsql: str) -> StatementResult:
    """Convert a single TLSQL statement to standard SQL.

    Args:
        tlsql: One TLSQL statement (PREDICT, TRAIN, or VALIDATE).

    Returns:
        StatementResult with sql_list, target_table, where_condition, etc.
    """
    return SQLGenerator.convert(tlsql)


def convert_workflow_queries(
    query_list: List[Optional[str]],
    table_list: Optional[List[str]] = None
) -> ConversionResult:
    """Workflow-level API built on convert().

    It takes three statements at once and returns a ConversionResult. 
    PREDICT is required; TRAIN and VALIDATEare optional. 
    When TRAIN is omitted, it is auto-generated from the PREDICT
    statement and table_list (PREDICT table: NOT(WHERE); other tables: SELECT *).

    Args:
        query_list: List of three TLSQL strings [PREDICT, TRAIN, VALIDATE].
            Only the first (PREDICT) is required; the others may be None or "".
        table_list: List of table names used when auto-generating TRAIN.
            Required when query_list[1] (TRAIN) is not provided.

    Returns:
        ConversionResult containing predict_result, train_result, and
        validate_result (None if VALIDATE not provided). Use result.predict,
        result.train, and result.validate for access.
    """
    if not query_list or len(query_list) != 3:
        raise ValueError("query_list must contain exactly three elements [PREDICT, TRAIN, VALIDATE].")

    predict_query, train_query, validate_query = query_list[0], query_list[1], query_list[2]

    if not predict_query or not str(predict_query).strip():
        raise ValueError("PREDICT statement (query_list[0]) is required.")

    generator = SQLGenerator()
    predict_result = SQLGenerator.convert(predict_query)

    has_train = train_query and str(train_query).strip()
    if not has_train and not table_list:
        raise ValueError("table_list is required when TRAIN statement (query_list[1]) is not provided.")

    train_result = (
        SQLGenerator.convert(train_query)
        if has_train
        else generator.auto_generate_train(predict_result, table_list=table_list)
    )

    validate_result = (
        SQLGenerator.convert(validate_query)
        if validate_query and str(validate_query).strip()
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
    StatementResult,
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
    "convert_workflow_queries",
    # Tokens
    "Token",
    "TokenType",
    # Core classes
    "Lexer",
    "Parser",
    "SQLGenerator",
    "GeneratedSQL",
    "ConversionResult",
    "StatementResult",
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
