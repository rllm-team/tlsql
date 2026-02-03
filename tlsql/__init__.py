"""TLSQL Core components.
This module provides the core components for parsing and converting TLSQL statements
to standard SQL.

Main components:
    - Lexer: Tokenizes TLSQL input text
    - Parser: Parses tokens into Abstract Syntax Tree (AST)
    - SQLGenerator: Converts AST to standard SQL statements
"""

from .tokens import Token, TokenType
from .lexer import Lexer
from .parser import Parser
from .ast_nodes import (
    ASTNode,
    ColumnReference,
    Expr,
    LiteralExpr,
    ColumnExpr,
    BinaryExpr,
    UnaryExpr,
    BetweenExpr,
    InExpr,
    WhereClause,
    ColumnSelector,
    WithClause,
    TablesClause,
    TrainStatement,
    ValidateStatement,
    PredictType,
    ValueClause,
    FromClause,
    PredictStatement,
    Statement,
)
from .exceptions import TLSQLError, LexerError, ParseError, GenerationError
from .sql_generator import SQLGenerator, GeneratedSQL, ConversionResult, StatementResult
from typing import Optional, List


def convert_workflow_queries(
    query_list: List[Optional[str]],
    table_list: Optional[List[str]] = None
) -> ConversionResult:
    """Convert TLSQL workflow (PREDICT, TRAIN, VALIDATE) to standard SQL.

    Workflow-level API built on convert(). PREDICT is required; TRAIN and 
    VALIDATE are optional. If TRAIN is notprovided, it is auto-generated: for the PREDICT table, train SQL uses
    NOT(predict WHERE); for other tables in table_list, train SQL is SELECT *.
    table_list is required when TRAIN is not provided.

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
    import sys
    parent_module = sys.modules.get('tlsql')
    if parent_module is None:
        import importlib
        parent_module = importlib.import_module('tlsql')
    return parent_module.convert_workflow_queries(query_list=query_list, table_list=table_list)

__all__ = [
    # Tokens
    "Token",
    "TokenType",
    # Core classes
    "Lexer",
    "Parser",
    "SQLGenerator",
    # AST nodes
    "ASTNode",
    "ColumnReference",
    "Expr",
    "LiteralExpr",
    "ColumnExpr",
    "BinaryExpr",
    "UnaryExpr",
    "BetweenExpr",
    "InExpr",
    "WhereClause",
    "ColumnSelector",
    "WithClause",
    "TablesClause",
    "TrainStatement",
    "ValidateStatement",
    "PredictType",
    "ValueClause",
    "FromClause",
    "PredictStatement",
    "Statement",
    # Exceptions
    "TLSQLError",
    "LexerError",
    "ParseError",
    "GenerationError",
    # SQL generator results
    "GeneratedSQL",
    "ConversionResult",
    "StatementResult",
    # Top-level API
    "convert_workflow_queries",
]
