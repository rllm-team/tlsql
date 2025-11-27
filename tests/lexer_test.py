try:
    from ..core.lexer import Lexer
except ImportError:
    from tl_sql.core.lexer import Lexer



query = "validate with (user.*, movie.title, rating.*) FROM Tables(user, movie, rating) WHERE user.loc = 'BJ' AND moive.year > 1990 AND rating.xxx > xx"
lexer = Lexer(query)


tokens = lexer.tokenize()


for token in tokens:
    print(f"{token.type.name:15s} '{token.value}' at {token.line}:{token.column}")
    
    
    
    
