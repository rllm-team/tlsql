"""Custom exception classes for ML-SQL """


class MLSQLError(Exception):
    """Base class for all ML-SQL errors"""

    def __init__(self, message: str, line: int = None, column: int = None):
        """Initialize error with optional location info"""
        self.message = message
        self.line = line
        self.column = column
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        """Format message including line/column info"""
        if self.line is not None and self.column is not None:
            return f"Line {self.line}, Column {self.column}: {self.message}"
        elif self.line is not None:
            return f"Line {self.line}: {self.message}"
        return self.message


class LexerError(MLSQLError):
    """Error thrown during lexing """

    pass


class ParseError(MLSQLError):
    """Error thrown during parsing """

    pass


class ValidationError(MLSQLError):
    """Error thrown during semantic validation """

    pass
