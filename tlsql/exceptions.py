"""Custom exception classes for TLSQL."""


class TLSQLError(Exception):
    """Base class for all TLSQL errors."""

    def __init__(self, message: str, line: int = None, column: int = None):
        """Initialize error with optional location info."""
        self.message = message
        self.line = line
        self.column = column
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        """Format message."""
        if self.line is not None and self.column is not None:
            return f"Line {self.line}, Column {self.column}: {self.message}"
        elif self.line is not None:
            return f"Line {self.line}: {self.message}"
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
