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
from .sql_generator import SQLGenerator, GeneratedSQL, ConversionResult

def convert(tlsql: str):
    """Convert TLSQL statement to standard SQL.
    
    This function is re-exported from the parent tlsql module.
    Uses lazy import to avoid circular dependency.
    """
    import sys
   
    parent_module = sys.modules.get('tlsql')
    if parent_module is None:
        import importlib
        parent_module = importlib.import_module('tlsql')
    return parent_module.convert(tlsql)

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
    # Top-level API
    "convert",
]
