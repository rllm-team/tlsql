"""Custom exception classes for TLSQL."""

from typing import Optional


class TLSQLError(Exception):
    """Base class for all TLSQL errors."""

    def __init__(self, message: str, line_num: Optional[int] = None, col_num: Optional[int] = None):
        """Initialize error with optional location info."""
        self.message = message
        self.line_num = line_num
        self.col_num = col_num
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        """Format error message with optional location information."""
        if self.line_num is not None and self.col_num is not None:
            return f"Line {self.line_num}, Column {self.col_num}: {self.message}"
        elif self.line_num is not None:
            return f"Line {self.line_num}: {self.message}"
        return self.message


class LexerError(TLSQLError):
    """Error thrown during lexing."""

    pass


class ParseError(TLSQLError):
    """Error thrown during parsing."""

    pass


class GenerationError(TLSQLError):
    """Error thrown during SQL generation."""

    pass
