"""Token definitions for TLSQL lexer.
"""

from dataclasses import dataclass
from enum import Enum, auto


class TokenType(Enum):
    """Token types for TLSQL language.
    """
    # Core Keywords
    TRAIN = auto()
    PREDICT = auto()
    VALIDATE = auto()

    # SQL Keywords
    WITH = auto()
    USING = auto()
    VALUE = auto()
    TABLES = auto()
    FROM = auto()
    WHERE = auto()

    # Task Types
    CLF = auto()
    REG = auto()

    # Logical Operators
    AND = auto()
    OR = auto()
    NOT = auto()
    BETWEEN = auto()
    IN = auto()

    # Comparison Operators
    GT = auto()
    LT = auto()
    GTE = auto()
    LTE = auto()
    EQ = auto()
    NEQ = auto()
    EQUALS = auto()

    # Literals
    IDENTIFIER = auto()
    STRING = auto()
    NUMBER = auto()

    # Punctuation
    LPAREN = auto()      # (
    RPAREN = auto()      # )
    COMMA = auto()       # ,
    SEMICOLON = auto()   # ;
    DOT = auto()         # .
    ASTERISK = auto()    # *

    # Special Tokens
    EOF = auto()
    NEWLINE = auto()


# Keyword mapping: string -> TokenType
KEYWORDS = {
    # Core TL Operation Keywords
    'TRAIN': TokenType.TRAIN,
    'PREDICT': TokenType.PREDICT,
    'VALIDATE': TokenType.VALIDATE,

    # SQL Keywords
    'WITH': TokenType.WITH,
    'USING': TokenType.USING,
    'VALUE': TokenType.VALUE,
    'TABLES': TokenType.TABLES,
    'FROM': TokenType.FROM,
    'WHERE': TokenType.WHERE,

    # Task Types
    'CLF': TokenType.CLF,
    'REG': TokenType.REG,

    # Logical Operators
    'AND': TokenType.AND,
    'OR': TokenType.OR,
    'NOT': TokenType.NOT,
    'BETWEEN': TokenType.BETWEEN,
    'IN': TokenType.IN,
}


@dataclass
class Token:
    """Represents a single token in the input.

    Attributes:
        type: Token type.
        value: String value of token.
        line: Line number where token is located.
        column: Column number where token is located.
    """

    type: TokenType
    value: str
    line: int
    column: int

    def __repr__(self) -> str:
        """Return string representation of token."""
        return f"Token({self.type.name}, '{self.value}', {self.line}:{self.column})"
