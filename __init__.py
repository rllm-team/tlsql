"""TLSQL: A SQL-like language designed for relational table learning tasks and workflows.

This package converts three types of custom SQL statements into standard SQL:
1. TRAIN WITH - Training data queries
2. PREDICT VALUE - Prediction target queries
3. VALIDATE WITH - Validation data queries

Usage:
    >>> from tlsql import convert
    >>> result = convert("PREDICT VALUE(users.Age, CLF) FROM users")
    >>> print(result.sql_list)  # List of GeneratedSQL objects
"""

__version__ = "0.1.0"
__author__ = "TLSQL Team"


def convert(tlsql: str):
    """Convert TLSQL statement to standard SQL.

    This is the main entry point for TLSQL conversion.

    Args:
        tlsql: TLSQL statement string.

    Returns:
        ConversionResult: Unified result containing statement type and all metadata.

    """
    from tlsql.tlsql.parser import Parser
    from tlsql.tlsql.sql_generator import SQLGenerator

    # Parse the TLSQL statement
    parser = Parser(tlsql)
    ast = parser.parse()

    # Generate SQL with metadata
    generator = SQLGenerator()
    return generator.build(ast)


# Tokens
from tlsql.tlsql.tokens import Token, TokenType

# Core classes (re-exported for convenience)
from tlsql.tlsql.lexer import Lexer
from tlsql.tlsql.parser import Parser
from tlsql.tlsql.sql_generator import (
    SQLGenerator,
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
