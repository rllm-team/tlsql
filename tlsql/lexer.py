"""Lexer that converts TLSQL text to a token stream."""

from typing import List
from .tokens import Token, TokenType, KEYWORDS
from .exceptions import LexerError


class Lexer:
    """Convert TLSQL input into a token stream.

    Attributes:
        text: Input text.
        pos: Current character index.
        line: Current line number.
        column: Current column number.
        current_char: Current character.
    """

    def __init__(self, text: str):
        """Initialize lexer.

        Args:
            text: Input text.
        """
        self.text = text
        self.pos = 0
        self.line = 1
        self.column = 1
        self.current_char = self.text[0] if text else None

    def advance(self) -> None:
        """Advance to next character.

        Moves to the next character, updating line/column counters.
        """
        if self.current_char == '\n':
            self.line += 1
            self.column = 1
        else:
            self.column += 1

        self.pos += 1
        self.current_char = self.text[self.pos] if self.pos < len(self.text) else None

    def peek(self, offset: int = 1) -> str:
        """Look ahead without consuming.

        Args:
            offset: Lookahead offset.

        Returns:
            Character at position or None if out-of-range.
        """
        peek_pos = self.pos + offset
        return self.text[peek_pos] if peek_pos < len(self.text) else None

    def skip_whitespace(self) -> None:
        """Skip whitespace characters.

        Skips spaces, tabs, carriage returns, and newlines.
        """
        while self.current_char and self.current_char in ' \t\r\n':
            self.advance()

    def skip_comment(self) -> None:
        """Skip SQL comments."""
        if self.current_char == '-' and self.peek() == '-':
            while self.current_char and self.current_char != '\n':
                self.advance()
            if self.current_char == '\n':
                self.advance()

        elif self.current_char == '/' and self.peek() == '*':
            self.advance()
            self.advance()
            while self.current_char:
                if self.current_char == '*' and self.peek() == '/':
                    self.advance()
                    self.advance()
                    break
                self.advance()

    def read_string(self) -> str:
        """Read string literal.

        Supports single/double quotes and escape sequences (\\, \', \", \\n, \\t).

        Returns:
            String content without quotes.

        Raises:
            LexerError: If string is unterminated.
        """
        quote_char = self.current_char
        value = ''
        self.advance()
        while self.current_char and self.current_char != quote_char:
            if self.current_char == '\\':
                self.advance()
                if self.current_char in (quote_char, '\\', 'n', 't'):
                    escape_map = {'n': '\n', 't': '\t'}
                    value += escape_map.get(self.current_char, self.current_char)
                    self.advance()
                else:
                    value += self.current_char
                    self.advance()
            else:
                value += self.current_char
                self.advance()

        if self.current_char != quote_char:
            raise LexerError("Unterminated string literal", self.line, self.column)

        self.advance()
        return value

    def read_number(self) -> str:
        """Read numeric literal.

        Returns:
            String representation of number.
        """
        value = ''
        has_dot = False

        while self.current_char and (self.current_char.isdigit() or self.current_char == '.'):
            if self.current_char == '.':
                if has_dot:
                    break  # second dot -> stop
                has_dot = True
            value += self.current_char
            self.advance()

        return value

    def read_identifier(self) -> str:
        """Read identifier or keyword.

        Identifiers consist of letters, digits, underscore.

        Returns:
            Identifier string.
        """
        value = ''

        while self.current_char and (self.current_char.isalnum() or self.current_char == '_'):
            value += self.current_char
            self.advance()

        return value

    def tokenize(self) -> List[Token]:
        """Tokenize entire input.

        Steps: Skip whitespace, recognize strings and numbers literals,
        recognize identifiers & keywords, recognize operators append EOF token.

        Returns:
            List of tokens ending with EOF.

        Raises:
            LexerError: Raised for unknown characters.
        """
        tokens = []

        while self.current_char:
            if self.current_char in ' \t\r\n':
                self.skip_whitespace()
                continue

            if self.current_char == '-' and self.peek() == '-':
                self.skip_comment()
                continue

            if self.current_char == '/' and self.peek() == '*':
                self.skip_comment()
                continue

            line, column = self.line, self.column

            if self.current_char in ('"', "'"):
                value = self.read_string()
                tokens.append(Token(TokenType.STRING, value, line, column))
                continue

            if self.current_char.isdigit():
                value = self.read_number()
                tokens.append(Token(TokenType.NUMBER, value, line, column))
                continue

            if self.current_char.isalpha() or self.current_char == '_':
                value = self.read_identifier()
                token_type = KEYWORDS.get(value.upper(), TokenType.IDENTIFIER)
                tokens.append(Token(token_type, value, line, column))
                continue

            if self.current_char == '>':
                if self.peek() == '=':
                    tokens.append(Token(TokenType.GTE, '>=', line, column))
                    self.advance()
                    self.advance()
                else:
                    tokens.append(Token(TokenType.GT, '>', line, column))
                    self.advance()
                continue

            if self.current_char == '<':
                if self.peek() == '=':
                    tokens.append(Token(TokenType.LTE, '<=', line, column))
                    self.advance()
                    self.advance()
                elif self.peek() == '>':
                    tokens.append(Token(TokenType.NEQ, '<>', line, column))
                    self.advance()
                    self.advance()
                else:
                    tokens.append(Token(TokenType.LT, '<', line, column))
                    self.advance()
                continue

            if self.current_char == '!':
                if self.peek() == '=':
                    tokens.append(Token(TokenType.NEQ, '!=', line, column))
                    self.advance()
                    self.advance()
                else:
                    raise LexerError(
                        "Unexpected character '!', did you mean '!='?",
                        self.line,
                        self.column
                    )
                continue

            if self.current_char == '=':
                if self.peek() == '=':
                    tokens.append(Token(TokenType.EQ, '==', line, column))
                    self.advance()
                    self.advance()
                else:
                    tokens.append(Token(TokenType.EQUALS, '=', line, column))
                    self.advance()
                continue

            char_tokens = {
                '(': TokenType.LPAREN,
                ')': TokenType.RPAREN,
                ',': TokenType.COMMA,
                ';': TokenType.SEMICOLON,
                '.': TokenType.DOT,
                '*': TokenType.ASTERISK,
            }

            if self.current_char in char_tokens:
                token_type = char_tokens[self.current_char]
                tokens.append(Token(token_type, self.current_char, line, column))
                self.advance()
                continue

            raise LexerError(
                f"Unexpected character '{self.current_char}'",
                self.line,
                self.column
            )

        tokens.append(Token(TokenType.EOF, '', self.line, self.column))
        return tokens
