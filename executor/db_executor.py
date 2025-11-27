"""Database executor that runs SQL statements and returns data"""

import logging
from typing import List, Dict, Optional, Any, Union
from dataclasses import dataclass
import pandas as pd


try:
    import pymysql
    MYSQL_AVAILABLE = True
except ImportError:
    MYSQL_AVAILABLE = False

try:
    from sqlalchemy import create_engine
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False


# Configure logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)


@dataclass
class DatabaseConfig:
    """Database configuration"""
    db_type: str                            # Database type
    database: str                           # Database name/path
    host: Optional[str] = None              # Host
    port: Optional[int] = None              # Port
    username: Optional[str] = None          # Username
    password: Optional[str] = None          # Password
    charset: str = 'utf8'                   # Charset
    
    def __post_init__(self):
        """Validate configuration"""
        valid_types = [ 'mysql']
        if self.db_type not in valid_types:
            raise ValueError(f"Unsupported database type: {self.db_type}. "
                           f"Supported types: {', '.join(valid_types)}")


@dataclass
class ExecutionResult:
    """Execution result wrapper
    
    Encapsulates SQL execution outcome and metadata.
    """
    
    success: bool                           # Success flag
    data: Optional[pd.DataFrame] = None     # Result DataFrame
    row_count: int = 0                      # Row count
    execution_time: float = 0.0             # Execution time
    error: Optional[str] = None             # Error message
    sql: Optional[str] = None               # SQL string


class DatabaseExecutor:
    """Database executor that manages connections and queries"""
    
    def __init__(self, config: DatabaseConfig):
        """Initialize executor
        
        Args:
            config: DatabaseConfig object
            
        Raises:
            ValueError: Unsupported db type
            ImportError: Missing driver
        """
        self.config = config
        self.engine = None  # SQLAlchemy engine
        self._validate_dependencies()
        logger.info(f"DatabaseExecutor initialized for {config.db_type}")
    
    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        """Context manager exit"""
        self.disconnect()
        # Propagate exceptions
        return False
    
    def _validate_dependencies(self):
        """Validate required drivers"""
        if self.config.db_type == 'mysql':
            if not MYSQL_AVAILABLE:
                raise ImportError("pymysql is not installed.")
            if not SQLALCHEMY_AVAILABLE:
                raise ImportError("SQLAlchemy is not installed.")

    
    def connect(self) -> None:
        """Establish database connection"""
        try:
            if self.config.db_type == 'mysql':
                # Use SQLAlchemy for MySQL
                connection_string = (
                    f"mysql+pymysql://{self.config.username}:{self.config.password}"
                    f"@{self.config.host}:{self.config.port or 3306}/{self.config.database}"
                    f"?charset={self.config.charset}"
                )
                self.engine = create_engine(connection_string)
                logger.info(f"Connected to MySQL database: {self.config.database}")
            else:
                raise ValueError(f"Unsupported database type: {self.config.db_type}")
        
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise
    
    def disconnect(self) -> None:
        """Disconnect from database"""
        if self.engine is not None:
            self.engine.dispose()
            self.engine = None
            logger.info("Database connection closed")
    
    def is_connected(self) -> bool:
        """Check if engine is connected"""
        return self.engine is not None
    
    def execute(self, sql: str, params: Optional[tuple] = None) -> ExecutionResult:
        """Execute a single SQL query and return DataFrame"""
        import time
        
        # Ensure connection
        if not self.is_connected():
            self.connect()
        
        start_time = time.time()
        
        try:
            # Execute query
            logger.info(f"Executing SQL: {sql[:100]}...")
            if self.engine is None:
                raise RuntimeError("Database engine is not initialized. Please call connect() first.")
            
            df = pd.read_sql(sql, self.engine, params=params)
            
            execution_time = time.time() - start_time
            row_count = len(df)
            
            logger.info(f"Query executed successfully. Rows: {row_count}, Time: {execution_time:.3f}s")
            
            return ExecutionResult(
                success=True,
                data=df,
                row_count=row_count,
                execution_time=execution_time,
                sql=sql
            )
        
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = str(e)
            
            logger.error(f"Query failed: {error_msg}")
            
            return ExecutionResult(
                success=False,
                error=error_msg,
                execution_time=execution_time,
                sql=sql
            )
    
    def execute_batch(self, sql_list: List[str]) -> List[ExecutionResult]:
        """Execute multiple SQL statements sequentially"""
        logger.info(f"Executing batch of {len(sql_list)} SQL statements")
        
        results = []
        for i, sql in enumerate(sql_list, 1):
            logger.info(f"Executing SQL {i}/{len(sql_list)}")
            result = self.execute(sql)
            results.append(result)
        
        # Summaries
        success_count = sum(1 for r in results if r.success)
        fail_count = len(results) - success_count
        
        logger.info(f"Batch execution completed. Success: {success_count}, Failed: {fail_count}")
        
        return results
    
    def execute_with_dict(self, sql_dict: Dict[str, str]) -> Dict[str, ExecutionResult]:
        """Execute labeled SQL dictionary"""
        logger.info(f"Executing {len(sql_dict)} labeled SQL statements")
        
        results = {}
        for label, sql in sql_dict.items():
            logger.info(f"Executing SQL for: {label}")
            result = self.execute(sql)
            results[label] = result
        
        return results
    
    def get_table_info(self, table_name: str) -> pd.DataFrame:
        """Get column info for a table"""

        if self.config.db_type == 'mysql':
            sql = f"DESCRIBE {table_name}"
        else:
            raise ValueError(f"Unsupported database type: {self.config.db_type}")
        
        result = self.execute(sql)
        if result.success:
            return result.data
        else:
            raise Exception(f"Failed to get table info: {result.error}")
    
    def list_tables(self) -> List[str]:
        """List all tables in database"""
        if self.config.db_type == 'mysql':
            sql = "SHOW TABLES"
        else:
            raise ValueError(f"Unsupported database type: {self.config.db_type}")
        
        result = self.execute(sql)
        if result.success:
            return result.data.iloc[:, 0].tolist()
        else:
            raise Exception(f"Failed to list tables: {result.error}")
    
    def get_primary_keys(self, table_name: str) -> List[str]:
        """Retrieve primary keys for table"""
        if self.config.db_type == 'mysql':
            sql = f"""
                SELECT column_name 
                FROM information_schema.key_column_usage
                WHERE table_schema = '{self.config.database}' 
                AND table_name = '{table_name}' 
                AND constraint_name = 'PRIMARY'
            """
        else:
            raise ValueError(f"Unsupported database type: {self.config.db_type}")
        
        result = self.execute(sql)
        if not result.success:
            logger.warning(f"Failed to get primary keys for {table_name}: {result.error}")
            return []
        

        return result.data.iloc[:, 0].tolist()
    
    def get_foreign_keys(self, table_name: str) -> Dict[str, tuple]:
        """Retrieve foreign keys"""
        if self.config.db_type == 'mysql':
            sql = f"""
                SELECT column_name, referenced_table_name, referenced_column_name
                FROM information_schema.key_column_usage
                WHERE table_schema = '{self.config.database}' 
                AND table_name = '{table_name}' 
                AND referenced_table_name IS NOT NULL
            """
        else:
            raise ValueError(f"Unsupported database type: {self.config.db_type}")
        
        result = self.execute(sql)
        if not result.success:
            logger.warning(f"Failed to get foreign keys for {table_name}: {result.error}")
            return {}
        
        fk_dict = {}

        # Access by column names when available
        for _, row in result.data.iterrows():
            # Use column names (case-insensitive)
            if 'column_name' in result.data.columns:
                fk_col = row['column_name']
                ref_table = row.get('referenced_table_name', None)
                ref_col = row.get('referenced_column_name', None)
            else:
                # Fallback to positional indexing
                fk_col = row.iloc[0]
                ref_table = row.iloc[1] if len(row) > 1 else None
                ref_col = row.iloc[2] if len(row) > 2 else None
            
            if fk_col and ref_table and ref_col:
                fk_dict[fk_col] = (ref_table, ref_col)
        
        return fk_dict
    
    def get_table_schema(self, table_names: List[str]) -> Dict[str, Dict]:
        """Get schema info for multiple tables"""
        schema = {}
        for table_name in table_names:
            # Column info
            table_info = self.get_table_info(table_name)
            if table_info is not None and not table_info.empty:
                if self.config.db_type == 'mysql':
                    columns = table_info['Field'].tolist()
            else:
                columns = []
            
            # Primary keys
            primary_keys = self.get_primary_keys(table_name)
            
            # Foreign keys
            foreign_keys = self.get_foreign_keys(table_name)
            
            schema[table_name] = {
                'columns': columns,
                'primary_keys': primary_keys,
                'foreign_keys': foreign_keys
            }
        
        return schema
    
    def __enter__(self):
        """Context manager entry: open connection"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit: close connection"""
        self.disconnect()
    
    def __del__(self):
        """Ensure connection closed on destruction"""
        self.disconnect()

