"""TL-SQL Query Pipeline

This module provides a high-level pipeline for executing TL-SQL statements.

The pipeline processes three types of statements:
1. TRAIN - Specify training data
2. PREDICT - Specify target column and task type
3. VALIDATE - Specify validation data

The pipeline converts TL-SQL statements to standard SQL and executes them,
returning DataFrames for further processing.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import pandas as pd

from tl_sql.core.parser import Parser
from tl_sql.core.ast_nodes import TrainStatement, PredictStatement, ValidateStatement
from tl_sql.executor import SQLGenerator, DatabaseExecutor, DatabaseConfig


@dataclass
class PipelineResult:
    """Pipeline execution result
    
    Attributes:
        train_data: Training data dictionary
        test_data: Test data DataFrame
        validate_data: Validation data dictionary
        target_column: Target column name
        task_type: Task type (CLF/REG)
        feature_columns: Feature column list
        train_tables: Training table list
        metadata: Additional metadata
    """
    train_data: Dict[str, pd.DataFrame]      # Training data
    test_data: Optional[pd.DataFrame] = None  # Test data
    validate_data: Optional[Dict[str, pd.DataFrame]] = None  # Validation data
    target_column: Optional[str] = None       # Target column
    task_type: Optional[str] = None           # Task type
    feature_columns: List[str] = None         # Feature columns
    train_tables: List[str] = None            # Training tables
    metadata: Dict[str, Any] = None           # Metadata


class TlSqlPipeline:
    """TL-SQL Query Pipeline
    
    This class provides an interface for executing TL-SQL statements.
    It parses TL-SQL statements, converts them to standard SQL, executes them,
    and returns the results as DataFrames.
    
    Example:
        pipeline = TlSqlPipeline(db_config)
        result = pipeline.run(train_sql, predict_sql, validate_sql)
        # result.train_data contains training DataFrames
    """
    
    def __init__(self, db_config: Dict[str, Any]):
        """Initialize
        
        Args:
            db_config: Database configuration dictionary, containing:
                - db_type: 'mysql', 'sqlite'
                - host: Host address
                - port: Port
                - database: Database name
                - username: Username
                - password: Password
        """
        self.db_config = DatabaseConfig(**db_config)
        self.sql_generator = SQLGenerator()
    
    def run(self, train_sql: str, predict_sql: str, validate_sql: str) -> PipelineResult:
        """Execute Pipeline
        
        Process:
        1. Parse PREDICT statement → get target column and task type
        2. Parse TRAIN statement → get training data and features
        3. Parse VALIDATE statement → get validation data
        4. Execute queries from database → get data
        5. Assemble results
        
        Args:
            train_sql: TRAIN statement
            predict_sql: PREDICT statement
            validate_sql: VALIDATE statement
            
        Returns:
            PipelineResult: Contains all prepared data
        """
        # Parse SQL
        predict_ast = self._parse_predict(predict_sql)
        train_ast = self._parse_train(train_sql)
        validate_ast = self._parse_validate(validate_sql)
        
        # Extract information
        train_info = self._extract_train_info(train_ast)
        predict_info = self._extract_predict_info(predict_ast)
        validate_info = self._extract_validate_info(validate_ast)
        
        # Generate SQL and execute
        train_data = self._load_train_data(train_ast)
        test_data = self._load_test_data(predict_ast)
        validate_data = self._load_validate_data(validate_ast)
        
        # Assemble results
        result = PipelineResult(
            train_data=train_data,
            test_data=test_data,
            validate_data=validate_data,
            target_column=predict_info['target_column'],
            task_type=predict_info['task_type'],
            feature_columns=train_info['features'],
            train_tables=train_info['tables'],
            metadata={
                'train_where': train_info.get('where_condition'),
                'validate_where': validate_info.get('where_condition'),
                'test_where': predict_info.get('where_condition'),
            }
        )
        
        return result
    
    def _parse_train(self, train_sql: str) -> TrainStatement:
        """Parse TRAIN statement
        
        Args:
            train_sql: TRAIN SQL statement
            
        Returns:
            TrainStatement: Parsed AST
        """
        parser = Parser(train_sql)
        ast = parser.parse()
        
        if not ast.train:
            raise ValueError("Input is not a valid TRAIN statement")
        
        return ast.train
    
    def _parse_predict(self, predict_sql: str) -> PredictStatement:
        """Parse PREDICT statement
        
        Args:
            predict_sql: PREDICT SQL statement
            
        Returns:
            PredictStatement: Parsed AST
        """
        parser = Parser(predict_sql)
        ast = parser.parse()
        
        if not ast.predict:
            raise ValueError("Input is not a valid PREDICT statement")
        
        return ast.predict
    
    def _parse_validate(self, validate_sql: str) -> ValidateStatement:
        """Parse VALIDATE statement
        
        Args:
            validate_sql: VALIDATE SQL statement
            
        Returns:
            ValidateStatement: Parsed AST
        """
        parser = Parser(validate_sql)
        ast = parser.parse()
        
        if not ast.validate:
            raise ValueError("Input is not a valid VALIDATE statement")
        
        return ast.validate
    
    def _extract_train_info(self, train_ast: TrainStatement) -> Dict[str, Any]:
        """Extract training information from TRAIN AST
        
        Args:
            train_ast: TRAIN statement AST
            
        Returns:
            Dictionary containing:
            - tables: Table list
            - features: Feature column list
            - where_condition: WHERE condition
        """
        # Extract tables
        tables = train_ast.tables.tables
        
        # Extract features
        features = []
        for selector in train_ast.with_clause.selectors:
            if selector.is_wildcard:
                # Handle table.* 
                features.append(f"{selector.table}.*")
            else:
                # table.column
                features.append(f"{selector.table}.{selector.column}")
        
        # Extract WHERE condition
        where_condition = None
        if train_ast.where:
            where_condition = self.sql_generator._expr_to_sql(
                train_ast.where.condition, 
                include_table_prefix=True
            )
        
        return {
            'tables': tables,
            'features': features,
            'where_condition': where_condition
        }
        
    def _extract_validate_info(self, validate_ast: ValidateStatement) -> Dict[str, Any]:
        """Extract validation information from VALIDATE AST
        
        Args:
            validate_ast: VALIDATE statement AST
            
        Returns:
            Dictionary containing:
            - tables: Table list
            - features: Feature column list
            - where_condition: WHERE condition
        """
        # Extract tables
        tables = validate_ast.tables.tables
        
        # Extract features
        features = []
        for selector in validate_ast.with_clause.selectors:
            if selector.is_wildcard:
                # Handle table.*
                features.append(f"{selector.table}.*")
            else:
                # table.column
                features.append(f"{selector.table}.{selector.column}")
        
        # Extract WHERE condition
        where_condition = None
        if validate_ast.where:
            where_condition = self.sql_generator._expr_to_sql(
                validate_ast.where.condition, 
                include_table_prefix=True
            )
        
        return {
            'tables': tables,
            'features': features,
            'where_condition': where_condition
        }
    
    def _extract_predict_info(self, predict_ast: PredictStatement) -> Dict[str, Any]:
        """Extract prediction information from PREDICT AST
        
        Args:
            predict_ast: PREDICT statement AST
            
        Returns:
            Dictionary containing:
            - target_column: Target column name (table.column format)
            - task_type: 'CLF' or 'REG'
            - where_condition: WHERE condition
        """
        # Extract target column
        target = predict_ast.value.target
        target_column = f"{target.table}.{target.column}" if target.table else target.column
        
        # Extract task type
        task_type = predict_ast.value.predict_type.type_name
        
        # Extract WHERE condition
        where_condition = None
        if predict_ast.where:
            where_condition = self.sql_generator._expr_to_sql(
                predict_ast.where.condition,
                include_table_prefix=True
            )
        
        return {
            'target_column': target_column,
            'task_type': task_type,
            'where_condition': where_condition
        }
    
    def _load_train_data(self, train_ast: TrainStatement) -> Dict[str, pd.DataFrame]:
        """Load training data
        
        Generate SQL from TRAIN statement and execute
        
        Args:
            train_ast: TRAIN statement AST
            
        Returns:
            Dict[str, pd.DataFrame]: {table_name: DataFrame}
        """
        # Generate SQL
        sql_list = self.sql_generator.generate_train_sql(train_ast)
        
        train_data = {}
        # Execute SQL with managed connection
        with DatabaseExecutor(self.db_config) as executor:
            for gen_sql in sql_list:
                result = executor.execute(gen_sql.sql)
               
                if result.success:
                    train_data[gen_sql.table] = result.data
                else:
                    raise Exception(
                        f"Failed to load training data: {gen_sql.table}: {result.error}"
                    )
        
        return train_data
    
    def _load_validate_data(self, validate_ast: ValidateStatement) -> Dict[str, pd.DataFrame]:
        """Load validation data
        
        Generate SQL from VALIDATE statement and execute to get data for each table
        
        Args:
            validate_ast: VALIDATE statement AST
            
        Returns:
            Dict[str, pd.DataFrame]: {table_name: DataFrame}
        """
        # Generate SQL
        sql_list = self.sql_generator.generate_train_sql(validate_ast)
        
        validate_data = {}
        with DatabaseExecutor(self.db_config) as executor:
            for gen_sql in sql_list:
                result = executor.execute(gen_sql.sql)
                
                if result.success:
                    validate_data[gen_sql.table] = result.data
                else:
                    raise Exception(
                        f"Failed to load validation data: {gen_sql.table}: {result.error}"
                    )
        
        return validate_data
    
    def _load_test_data(self, predict_ast: PredictStatement) -> pd.DataFrame:
        """Load test data
        
        Generate SQL from PREDICT statement and execute to get test data
        
        Args:
            predict_ast: PREDICT statement AST
            
        Returns:
            pd.DataFrame: Test data
        """
        # Get table name and filter condition
        table = predict_ast.from_table.table
        filter_cond = self.sql_generator.generate_predict_filter(predict_ast)
        
        # Build SQL
        sql = f"SELECT * FROM {table}"
        if filter_cond.condition:
            sql += f" WHERE {filter_cond.condition}"
        
        # Execute SQL
        with DatabaseExecutor(self.db_config) as executor:
            result = executor.execute(sql)
        
        if result.success:
            return result.data
        else:
            raise Exception(f"Failed to load test data: {result.error}")

