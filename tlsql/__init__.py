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
from typing import Optional


def convert(predict_query: str, train_query: Optional[str] = None, validate_query: Optional[str] = None) -> ConversionResult:
    """Convert TLSQL statements to standard SQL.

    Args:
        predict_query: PREDICT TLSQL statement (required).
        train_query: TRAIN TLSQL statement (optional).
        validate_query: VALIDATE TLSQL statement (optional).

    Returns:
        ConversionResult: Contains predict_result (StatementResult), train_result (StatementResult), 
        and validate_result (Optional[StatementResult]). Use shortcut properties result.predict, 
        result.train, and result.validate to access individual statement results.
    """
    import sys
    parent_module = sys.modules.get('tlsql')
    if parent_module is None:
        import importlib
        parent_module = importlib.import_module('tlsql')
    return parent_module.convert(
        predict_query=predict_query,
        train_query=train_query,
        validate_query=validate_query
    )

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
    "convert",
]
