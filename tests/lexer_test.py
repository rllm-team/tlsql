"""Test lexer functionality
"""

import sys

sys.path.append("./")
sys.path.append("../")
sys.path.append("../../")

from tlsql import Lexer


query = "PREDICT VALUE(users.Age, CLF) FROM users WHERE users.Gender='F' "
lexer = Lexer(query)

tokens = lexer.tokenize()

for token in tokens:
    print(f"{token.type.name:15s} '{token.value}' at {token.line_num}:{token.col_num}")
