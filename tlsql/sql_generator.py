"""SQL Generator.

Converts TLSQL AST into standard SQL statements or filtering conditions.

"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from .ast_nodes import (
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
    BetweenExpr,
    InExpr,
)
from .exceptions import GenerationError
from .parser import Parser


@dataclass
class GeneratedSQL:
    """Generated SQL statement.

    Attributes:
        table: Table name.
        sql: SQL string.
        columns: Selected column list.
    """
    table: str
    sql: str
    columns: List[str] = field(default_factory=list)


@dataclass
class ConversionResult:
    """Result from TLSQL conversion.

    Attributes:
        statement_type: Type of statement.
        sql_list: List of GeneratedSQL objects per table.
        target_column: Target column reference.
        task_type: Task type.
        target_table: Target table name.
        tables: List of all tables involved in the statement.
        where_condition: WHERE condition as SQL string.
    """
    statement_type: str
    sql_list: Optional[List[GeneratedSQL]] = None
    target_column: Optional[str] = None
    task_type: Optional[str] = None
    target_table: Optional[str] = None
    tables: List[str] = field(default_factory=list)
    where_condition: Optional[str] = None

    @property
    def is_train(self) -> bool:
        return self.statement_type == 'TRAIN'

    @property
    def is_validate(self) -> bool:
        return self.statement_type == 'VALIDATE'

    @property
    def is_predict(self) -> bool:
        return self.statement_type == 'PREDICT'


class SQLGenerator:
    """SQL generator for TLSQL statements."""

    def __init__(self):
        pass

    @classmethod
    def convert(cls, tlsql: str) -> ConversionResult:
        """Convert TLSQL statement to standard SQL.

        Args:
            tlsql: TLSQL statement string.

        Returns:
            ConversionResult: Contains statement type, generated SQL statements (sql_list),
            involved tables, WHERE condition, and for PREDICT statements: target column,
            task type, and target table.
        """
        parser = Parser(tlsql)
        ast = parser.parse()
        generator = cls()
        return generator.build(ast)

    def generate(self, statement: Statement):
        """Generate SQL statements or filters.

        Args:
            statement: Parsed Statement node.

        Returns:
            TRAIN/VALIDATE/PREDICT: List[GeneratedSQL] per table.

        Raises:
            GenerationError: Unknown statement type.
        """
        if statement.train:
            return self.generate_train_sql(statement.train)
        elif statement.validate:
            return self.generate_validate_sql(statement.validate)
        elif statement.predict:
            return self.generate_predict_sql(statement.predict)
        else:
            raise GenerationError("Unknown statement type")

    def build(self, statement: Statement) -> ConversionResult:
        """Generate SQL with full metadata.

        Args:
            statement: Parsed Statement node.

        Returns:
            ConversionResult: Unified result with all metadata.

        Raises:
            GenerationError: Unknown statement type.
        """
        if statement.train:
            return self._generate_train_result(statement.train)
        elif statement.validate:
            return self._generate_validate_result(statement.validate)
        elif statement.predict:
            return self._generate_predict_result(statement.predict)
        else:
            raise GenerationError("Unknown statement type")

    def _generate_train_result(self, train: TrainStatement) -> ConversionResult:
        """Generate ConversionResult for TRAIN statement."""
        sql_list = self.generate_train_sql(train)
        tables = train.tables.tables
        where_condition = None
        if train.where:
            where_condition = self._expr_to_sql(
                train.where.condition,
                include_table_prefix=False
            )

        return ConversionResult(
            statement_type='TRAIN',
            sql_list=sql_list,
            tables=tables,
            where_condition=where_condition
        )

    def _generate_validate_result(self, validate: ValidateStatement) -> ConversionResult:
        """Generate ConversionResult for VALIDATE statement."""
        sql_list = self.generate_validate_sql(validate)
        tables = validate.tables.tables
        where_condition = None
        if validate.where:
            where_condition = self._expr_to_sql(
                validate.where.condition,
                include_table_prefix=False
            )

        return ConversionResult(
            statement_type='VALIDATE',
            sql_list=sql_list,
            tables=tables,
            where_condition=where_condition
        )

    def _generate_predict_result(self, predict: PredictStatement) -> ConversionResult:
        """Generate ConversionResult for PREDICT statement.
        Returns a ConversionResult with sql_list containing the generated SQL
        for test data loading.
        """
        # Generate SQL list for PREDICT statement
        sql_list = self.generate_predict_sql(predict)
        
        if not sql_list:
            raise GenerationError("PREDICT statement failed to generate SQL list")

        target = predict.value.target
        target_table_name = predict.from_table.table
        if target.table:
            target_column = f"{target.table}.{target.column}"
        else:
            target_column = f"{target_table_name}.{target.column}"

        task_type = predict.value.predict_type.type_name.upper()

        where_condition = None
        if predict.where:
            where_condition = self._expr_to_sql(
                predict.where.condition,
                include_table_prefix=False
            )

        return ConversionResult(
            statement_type='PREDICT',
            sql_list=sql_list,  # Contains GeneratedSQL objects for direct execution
            target_column=target_column,
            task_type=task_type,
            target_table=target_table_name,
            tables=[target_table_name],
            where_condition=where_condition
        )

    def generate_train_sql(self, train: TrainStatement) -> List[GeneratedSQL]:
        """Generate SQL statements for TRAIN."""
        table_columns = self._group_columns_by_table(train.with_clause.selectors)

        table_conditions = {}
        if train.where:
            table_conditions = self._split_where_by_table(train.where)

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
        """Generate SQL for VALIDATE."""
        train_stmt = TrainStatement(
            with_clause=validate.with_clause,
            tables=validate.tables,
            where=validate.where
        )
        return self.generate_train_sql(train_stmt)

    def _group_columns_by_table(self, selectors: List[ColumnSelector]) -> Dict[str, List[str]]:
        """Group column selectors by table."""
        table_columns = {}
        for selector in selectors:
            if selector.table not in table_columns:
                table_columns[selector.table] = []
            table_columns[selector.table].append(selector.column)
        return table_columns

    def _split_where_by_table(self, where: WhereClause) -> Dict[str, str]:
        """Split WHERE conditions per table."""
        conditions = self._extract_and_conditions(where.condition)

        table_conditions = {}
        for cond in conditions:
            table = self._extract_table_from_expr(cond)
            if table:
                cond_str = self._expr_to_sql(cond, include_table_prefix=False)
                if table not in table_conditions:
                    table_conditions[table] = []
                table_conditions[table].append(cond_str)

        result = {}
        for table, conds in table_conditions.items():
            result[table] = ' AND '.join(conds)

        return result

    def _extract_and_conditions(self, expr: Expr) -> List[Expr]:
        """Recursively extract AND-connected subconditions."""
        if isinstance(expr, BinaryExpr) and expr.operator.upper() == 'AND':
            left_conds = self._extract_and_conditions(expr.left)
            right_conds = self._extract_and_conditions(expr.right)
            return left_conds + right_conds
        else:
            return [expr]

    def _extract_table_from_expr(self, expr: Expr) -> Optional[str]:
        """Extract table name from expression."""
        if isinstance(expr, ColumnExpr):
            return expr.column.table
        elif isinstance(expr, BinaryExpr):
            left_table = self._extract_table_from_expr(expr.left)
            if left_table:
                return left_table
            return self._extract_table_from_expr(expr.right)
        elif isinstance(expr, UnaryExpr):
            return self._extract_table_from_expr(expr.operand)
        elif isinstance(expr, BetweenExpr):
            return self._extract_table_from_expr(expr.column)
        elif isinstance(expr, InExpr):
            return self._extract_table_from_expr(expr.column)
        return None

    def _build_select_sql(self, table: str, columns: List[str], condition: Optional[str]) -> str:
        """Build SELECT statement."""
        if not columns or '*' in columns:
            select_clause = '*'
        else:
            select_clause = ', '.join(columns)

        sql = f"SELECT {select_clause} FROM {table}"
        if condition:
            sql += f" WHERE {condition}"

        return sql

    def generate_predict_sql(self, predict: PredictStatement) -> List[GeneratedSQL]:
        """Generate SQL statements for PREDICT statement.

        Generates a complete SQL statement for loading test data, consistent with
        TRAIN and VALIDATE statements. The returned sql_list can be directly used
        for database execution.

        Args:
            predict: Parsed PREDICT statement.

        Returns:
            List[GeneratedSQL]: A list containing a single GeneratedSQL object
            with the complete SELECT statement for test data loading.
            Format: SELECT * FROM table [WHERE condition].

        Example:
            PREDICT VALUE users.Age AS CLF FROM users WHERE users.Gender='F'
            Returns: [GeneratedSQL(table='users', sql='SELECT * FROM users WHERE Gender = \'F\'', columns=['*'])].
        """
        table = predict.from_table.table

        where_condition = None
        if predict.where:
            where_condition = self._expr_to_sql(
                predict.where.condition,
                include_table_prefix=False
            )

        sql = self._build_select_sql(table, ['*'], where_condition)
        return [GeneratedSQL(
            table=table,
            sql=sql,
            columns=['*']
        )]

    def _expr_to_sql(self, expr: Expr, include_table_prefix: bool = True) -> str:
        """Convert expression tree to SQL string."""
        if isinstance(expr, LiteralExpr):
            if expr.value_type == 'string':
                return f"'{expr.value}'"
            return str(expr.value)

        elif isinstance(expr, ColumnExpr):
            if include_table_prefix and expr.column.table:
                return f"{expr.column.table}.{expr.column.column}"
            return expr.column.column

        elif isinstance(expr, BinaryExpr):
            left = self._expr_to_sql(expr.left, include_table_prefix)
            right = self._expr_to_sql(expr.right, include_table_prefix)

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
            operand = self._expr_to_sql(expr.operand, include_table_prefix)
            return f"{expr.operator.upper()} {operand}"

        elif isinstance(expr, BetweenExpr):
            column = self._expr_to_sql(expr.column, include_table_prefix)
            lower = self._expr_to_sql(expr.lower, include_table_prefix)
            upper = self._expr_to_sql(expr.upper, include_table_prefix)
            return f"{column} BETWEEN {lower} AND {upper}"

        elif isinstance(expr, InExpr):
            column = self._expr_to_sql(expr.column, include_table_prefix)
            values = [self._expr_to_sql(val, include_table_prefix) for val in expr.values]
            values_str = ', '.join(values)
            return f"{column} IN ({values_str})"

        return ""
