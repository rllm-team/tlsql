"""SQL Generator

Converts ML-SQL AST into standard SQL statements or filtering conditions.

Main functions:
1. TRAIN/VALIDATE: multiple SELECT statements separated by table
2. PREDICT: filtering condition
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from tl_sql.core.ast_nodes import (
    Statement,
    TrainStatement,
    ValidateStatement,
    PredictStatement,
    ColumnSelector,
    WhereClause,
    Expr,
    BinaryExpr,
    UnaryExpr,
    LiteralExpr,
    ColumnExpr,
    ColumnReference,
    BetweenExpr,
    InExpr,
)


@dataclass
class GeneratedSQL:
    """Generated SQL statement
    
    Represents a complete SQL statement for a given table.
    
    Attributes:
        table: Table name
        sql: Complete SQL string
        columns: Selected column list
    """
    table: str                          # Table name
    sql: str                            # Full SQL
    columns: List[str] = field(default_factory=list)  # Columns


@dataclass
class FilterCondition:
    """Filter condition for PREDICT statements
    
    Attributes:
        table: Table name
        condition: SQL expression, e.g. user.loc = 'Florida'
    """
    table: str                          # Table name
    condition: str                      # SQL condition

class SQLGenerator:
    """SQL generator for ML-SQL statements"""
    
    def __init__(self):
        """Initialize SQL generator"""
        pass
    
    def generate(self, statement: Statement):
        """Generate SQL statements or filters
        
        Args:
            statement: Parsed Statement node
            
        Returns:
            TRAIN/VALIDATE: List[GeneratedSQL] per table
            PREDICT: FilterCondition object
            
        Raises:
            ValueError: Unknown statement type
        """
        if statement.train:
            return self.generate_train_sql(statement.train)
        elif statement.validate:
            return self.generate_validate_sql(statement.validate)
        elif statement.predict:
            return self.generate_predict_filter(statement.predict)
        else:
            raise ValueError("Unknown statement type")
    
    
    def generate_train_sql(self, train: TrainStatement) -> List[GeneratedSQL]:
        """Generate SQL statements for TRAIN
        
        Steps:
        1. Group selectors by table
        2. Split WHERE conditions by table
        3. Build SELECT per table
        """
        # Group columns by table
        table_columns = self._group_columns_by_table(train.with_clause.selectors)
        
        # Split WHERE by table
        table_conditions = {}
        if train.where:
            table_conditions = self._split_where_by_table(train.where)
        
        # Generate SQL for each table
        result = []
        for table in train.tables.tables:
            columns = table_columns.get(table, [])
            condition = table_conditions.get(table, None)
            
            sql = self._build_select_sql(table, columns, condition)
            result.append(GeneratedSQL(
                table=table,
                sql=sql,
                columns=columns
            ))
        
        return result
    
    def generate_validate_sql(self, validate: ValidateStatement) -> List[GeneratedSQL]:
        """Generate SQL for VALIDATE (reuse TRAIN logic)"""
        train_stmt = TrainStatement(
            with_clause=validate.with_clause,
            tables=validate.tables,
            where=validate.where
        )
        return self.generate_train_sql(train_stmt)
    
    def _group_columns_by_table(self, selectors: List[ColumnSelector]) -> Dict[str, List[str]]:
        """Group column selectors by table"""
        table_columns = {}
        for selector in selectors:
            if selector.table not in table_columns:
                table_columns[selector.table] = []
            table_columns[selector.table].append(selector.column)
        return table_columns
    
    def _split_where_by_table(self, where: WhereClause) -> Dict[str, str]:
        """Split WHERE conditions per table"""
        # Extract conditions
        conditions = self._extract_and_conditions(where.condition)
        
        # Group by table
        table_conditions = {}
        for cond in conditions:
            table = self._extract_table_from_expr(cond)
            if table:
                cond_str = self._expr_to_sql(cond, include_table_prefix=False)
                if table not in table_conditions:
                    table_conditions[table] = []
                table_conditions[table].append(cond_str)
        
        # Combine conditions per table
        result = {}
        for table, conds in table_conditions.items():
            result[table] = ' AND '.join(conds)
        
        return result
    
    def _extract_and_conditions(self, expr: Expr) -> List[Expr]:
        """Recursively extract AND-connected subconditions"""
        if isinstance(expr, BinaryExpr) and expr.operator.upper() == 'AND':
            # Extract left and right
            left_conds = self._extract_and_conditions(expr.left)
            right_conds = self._extract_and_conditions(expr.right)
            return left_conds + right_conds
        else:
            return [expr]

    
    def _extract_table_from_expr(self, expr: Expr) -> Optional[str]:
        """Extract table name from expression"""
        if isinstance(expr, ColumnExpr):
            return expr.column.table
        elif isinstance(expr, BinaryExpr):
            # Search left first
            left_table = self._extract_table_from_expr(expr.left)
            if left_table:
                return left_table
            # Then search right
            return self._extract_table_from_expr(expr.right)
        elif isinstance(expr, UnaryExpr):
            return self._extract_table_from_expr(expr.operand)
        elif isinstance(expr, BetweenExpr):
            # BETWEEN expression: extract table from column
            return self._extract_table_from_expr(expr.column)
        elif isinstance(expr, InExpr):
            # IN expression: extract table from column
            return self._extract_table_from_expr(expr.column)
        return None
    
    def _build_select_sql(self, table: str, columns: List[str], condition: Optional[str]) -> str:
        """Build SELECT statement"""
        # SELECT clause
        if not columns or '*' in columns:
            select_clause = '*'
        else:
            select_clause = ', '.join(columns)
        
        sql = f"SELECT {select_clause} FROM {table}"
        
        # Append WHERE
        if condition:
            sql += f" WHERE {condition}"
        
        return sql
    
    
    def generate_predict_filter(self, predict: PredictStatement) -> FilterCondition:
        """Convert PREDICT WHERE clause to filter"""
        table = predict.from_table.table
        
        if not predict.where:
            return FilterCondition(
                table=table,
                condition=""
            )
        
        # SQL condition
        sql_condition = self._expr_to_sql(predict.where.condition, include_table_prefix=False)
        
        return FilterCondition(
            table=table,
            condition=sql_condition
        )
    

    
    def _expr_to_sql(self, expr: Expr, include_table_prefix: bool = True) -> str:
        """Convert expression tree to SQL string"""
        if isinstance(expr, LiteralExpr):
            # Literal
            if expr.value_type == 'string':
                return f"'{expr.value}'"
            return str(expr.value)
        
        elif isinstance(expr, ColumnExpr):
            # Column reference
            if include_table_prefix and expr.column.table:
                return f"{expr.column.table}.{expr.column.column}"
            return expr.column.column
        
        elif isinstance(expr, BinaryExpr):
            # Binary expression
            left = self._expr_to_sql(expr.left, include_table_prefix)
            right = self._expr_to_sql(expr.right, include_table_prefix)
            
            # Normalize operator
            op = expr.operator.upper()
            if op == 'EQUALS' or op == '=':
                op = '='
            elif op == 'EQ' or op == '==':
                op = '='
            elif op == 'NEQ' or op == '!=':
                op = '!='
            elif op in ['GT', 'LT', 'GTE', 'LTE']:
                op_map = {'GT': '>', 'LT': '<', 'GTE': '>=', 'LTE': '<='}
                op = op_map.get(op, op)
            
            return f"{left} {op} {right}"
        
        elif isinstance(expr, UnaryExpr):
            # Unary expression
            operand = self._expr_to_sql(expr.operand, include_table_prefix)
            return f"{expr.operator.upper()} {operand}"
        
        elif isinstance(expr, BetweenExpr):
            # BETWEEN expression
            column = self._expr_to_sql(expr.column, include_table_prefix)
            lower = self._expr_to_sql(expr.lower, include_table_prefix)
            upper = self._expr_to_sql(expr.upper, include_table_prefix)
            return f"{column} BETWEEN {lower} AND {upper}"
        
        elif isinstance(expr, InExpr):
            # IN expression
            column = self._expr_to_sql(expr.column, include_table_prefix)
            values = [self._expr_to_sql(val, include_table_prefix) for val in expr.values]
            values_str = ', '.join(values)
            return f"{column} IN ({values_str})"
        
        return ""

