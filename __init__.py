"""TL-SQL: A SQL conversion library for custom SQL statements

This package converts three types of custom SQL statements into standard SQL:
1. TRAIN WITH - Training data queries
2. PREDICT VALUE - Prediction target queries 
3. VALIDATE WITH - Validation data queries 

Core API:
    from tl_sql import Parser, SQLGenerator
    from tl_sql.executor import TlSqlPipeline, DatabaseExecutor, DatabaseConfig
    
    # Parse TL-SQL 
    parser = Parser("PREDICT VALUE(users.Age, CLF) FROM users")
    ast = parser.parse()
    
    # Generate SQL 
    generator = SQLGenerator()
    result = generator.generate(ast)
    pipeline = TlSqlPipeline(db_config)
    result = pipeline.run(train_sql, predict_sql, validate_sql)


"""

__version__ = "0.1.0"
__author__ = "TL-SQL Team"

# Core language components
from tl_sql.core.lexer import Lexer
from tl_sql.core.parser import Parser
from tl_sql.core.exceptions import (
    MLSQLError,
    LexerError,
    ParseError,
    ValidationError,
)

# Executor components 
from tl_sql.executor import (
    SQLGenerator,
    DatabaseExecutor,
    DatabaseConfig,
    ExecutionResult,
    TlSqlPipeline,  
    PipelineResult,
)


__all__ = [
    # Core Language 
    "Lexer",
    "Parser",
    # Exceptions 
    "MLSQLError",
    "LexerError",
    "ParseError",
    "ValidationError",
    # Executor 
    "SQLGenerator",
    "DatabaseExecutor",
    "DatabaseConfig",
    "ExecutionResult",
    "TlSqlPipeline",  
    "PipelineResult",
]


