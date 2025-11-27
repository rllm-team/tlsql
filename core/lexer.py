"""Lexer that converts ML-SQL text to a token stream """

from typing import List
from tl_sql.core.tokens import Token, TokenType, KEYWORDS
from tl_sql.core.exceptions import LexerError


class Lexer:
    """Convert ML-SQL input into a token stream 
    
    Attributes:
        text: Input text 
        pos: Current character index
        line: Current line number 
        column: Current column number 
        current_char: Current character 
    """

    def __init__(self, text: str):
        """Initialize lexer 
        
        Args:
            text: Input text to lex 
        """
        self.text = text                                      
        self.pos = 0                                          
        self.line = 1                                         
        self.column = 1                                       
        self.current_char = self.text[0] if text else None    

    def advance(self) -> None:
        """Advance to next character 
        
        Moves to the next character, updating line/column counters
        """
        if self.current_char == '\n':
            self.line += 1
            self.column = 1
        else:
            self.column += 1

        self.pos += 1
        self.current_char = self.text[self.pos] if self.pos < len(self.text) else None

    def peek(self, offset: int = 1) -> str:
        """Look ahead without consuming 
        
        Args:
            offset: Lookahead offset /
            
        Returns:
            Character at position or None if out-of-range 
        """
        peek_pos = self.pos + offset
        return self.text[peek_pos] if peek_pos < len(self.text) else None

    def skip_whitespace(self) -> None:
        """Skip whitespace characters /
        
        Skips spaces, tabs, carriage returns, and newlines
        """
        while self.current_char and self.current_char in ' \t\r\n':
            self.advance()

    def skip_comment(self) -> None:
        """Skip SQL comments """
        # Single-line comment: -- 
        if self.current_char == '-' and self.peek() == '-':
            while self.current_char and self.current_char != '\n':
                self.advance()
            if self.current_char == '\n':
                self.advance()

        # Multi-line comment: /* ... */ 
        elif self.current_char == '/' and self.peek() == '*':
            self.advance()  # skip '/' 
            self.advance()  # skip '*' 
            while self.current_char:
                if self.current_char == '*' and self.peek() == '/':
                    self.advance()  # skip '*' 
                    self.advance()  # skip '/'
                    break
                self.advance()

    def read_string(self) -> str:
        """Read string literal 
        
        Supports single/double quotes and escape sequences (\\, \', \", \\n, \\t). 
        
        Returns:
            String content without quotes 
            
        Raises:
            LexerError: If string is unterminated 
        """
        quote_char = self.current_char
        value = ''
        self.advance()  # skip starting quote 
        while self.current_char and self.current_char != quote_char:
            if self.current_char == '\\':
                self.advance()
                # Handle escape sequences 
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

        self.advance()  # skip closing quote 
        return value

    def read_number(self) -> str:
        """Read numeric literal 
        
        Returns:
            String representation of number 
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
        """Read identifier or keyword 
        
        Identifiers consist of letters, digits, underscore; must start with letter/underscore
        
        Returns:
            Identifier string 
        """
        value = ''

        while self.current_char and (self.current_char.isalnum() or self.current_char == '_'):
            value += self.current_char
            self.advance()

        return value

    def tokenize(self) -> List[Token]:
        """Tokenize entire input 
        
        Steps:
        1. Skip whitespace/comments 
        2. Recognize string literals 
        3. Recognize numeric literals 
        4. Recognize identifiers & keywords
        5. Recognize multi-character operators 
        6. Recognize single-character operators & punctuation 
        7. Append EOF token 
        
        Returns:
            List of tokens ending with EOF 
            
        Raises:
            LexerError: Raised for unknown characters 
        """
        tokens = []

        while self.current_char:
            # Whitespace characters 
            if self.current_char in ' \t\r\n':
                self.skip_whitespace()
                continue

            # Comments 
            if self.current_char == '-' and self.peek() == '-':
                self.skip_comment()
                continue

            if self.current_char == '/' and self.peek() == '*':
                self.skip_comment()
                continue

            line, column = self.line, self.column

            # Quoted strings 
            if self.current_char in ('"', "'"):
                value = self.read_string()
                tokens.append(Token(TokenType.STRING, value, line, column))
                continue

            # Numbers 
            if self.current_char.isdigit():
                value = self.read_number()
                tokens.append(Token(TokenType.NUMBER, value, line, column))
                continue

            # Identifiers and keywords
            if self.current_char.isalpha() or self.current_char == '_':
                value = self.read_identifier()
                token_type = KEYWORDS.get(value.upper(), TokenType.IDENTIFIER)
                tokens.append(Token(token_type, value, line, column))
                continue

            # >= operator 
            if self.current_char == '>':
                if self.peek() == '=':
                    tokens.append(Token(TokenType.GTE, '>=', line, column))
                    self.advance()
                    self.advance()
                else:
                    tokens.append(Token(TokenType.GT, '>', line, column))
                    self.advance()
                continue

            # <= and <> operators
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

            # != operator 
            if self.current_char == '!':
                if self.peek() == '=':
                    tokens.append(Token(TokenType.NEQ, '!=', line, column))
                    self.advance()
                    self.advance()
                else:
                    raise LexerError(
                        f"Unexpected character '!', did you mean '!='?",
                        self.line,
                        self.column
                    )
                continue

            # = and == operators 
            if self.current_char == '=':
                if self.peek() == '=':
                    tokens.append(Token(TokenType.EQ, '==', line, column))
                    self.advance()
                    self.advance()
                else:
                    tokens.append(Token(TokenType.EQUALS, '=', line, column))
                    self.advance()
                continue

            # Single-character tokens 
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
            
            # Unknown character 
            raise LexerError(
                f"Unexpected character '{self.current_char}'",
                self.line,
                self.column
            )

        # Append EOF token 
        tokens.append(Token(TokenType.EOF, '', self.line, self.column))
        return tokens

