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


class _NoneResult:
    """Special object that returns None for all attribute access.

    Used as a placeholder when validate_result is None.
    """
    def __getattr__(self, name):
        return None

    def __bool__(self):
        return False

    def __repr__(self):
        return "None"


_NONE_RESULT = _NoneResult()


@dataclass
class ConversionResult:
    """Result from TLSQL conversion.

    This class represents both individual statement results and workflow results.
    For individual statements (PREDICT, TRAIN, VALIDATE), use statement_type and sql_list.
    For workflow results, use predict_result, train_result, and validate_result.

    Attributes:
        statement_type: Type of statement ('PREDICT', 'TRAIN', or 'VALIDATE').
        sql_list: List of GeneratedSQL objects per table.
        target_column: Target column reference (for PREDICT statements).
        task_type: Task type (for PREDICT statements).
        target_table: Target table name.
        tables: List of all tables involved in the statement.
        where_condition: WHERE condition as SQL string.
        predict_result: ConversionResult for PREDICT statement (workflow mode).
        train_result: ConversionResult for TRAIN statement (workflow mode).
        validate_result: ConversionResult for VALIDATE statement (workflow mode).
    """
    statement_type: str = ""
    sql_list: Optional[List[GeneratedSQL]] = None
    target_column: Optional[str] = None
    task_type: Optional[str] = None
    target_table: Optional[str] = None
    tables: List[str] = field(default_factory=list)
    where_condition: Optional[str] = None
    predict_result: Optional['ConversionResult'] = None  # type: ignore
    train_result: Optional['ConversionResult'] = None  # type: ignore
    validate_result: Optional['ConversionResult'] = None  # type: ignore

    @property
    def is_train(self) -> bool:
        """Check if this is a TRAIN statement result."""
        return self.statement_type == 'TRAIN'

    @property
    def is_validate(self) -> bool:
        """Check if this is a VALIDATE statement result."""
        return self.statement_type == 'VALIDATE'

    @property
    def is_predict(self) -> bool:
        """Check if this is a PREDICT statement result."""
        return self.statement_type == 'PREDICT'

    @property
    def is_workflow(self) -> bool:
        """Check if this is a workflow result (contains multiple statement results)."""
        return self.predict_result is not None

    @property
    def predict(self) -> 'ConversionResult':
        """Shortcut to predict_result for workflow results."""
        return self.predict_result if self.predict_result else _NONE_RESULT

    @property
    def train(self) -> 'ConversionResult':
        """Shortcut to train_result for workflow results."""
        return self.train_result if self.train_result else _NONE_RESULT

    @property
    def validate(self) -> 'ConversionResult':
        """Shortcut to validate_result for workflow results.

        Returns _NONE_RESULT if validate_result is None, allowing
        result.validate.sql to return None instead of raising AttributeError.
        """
        return self.validate_result if self.validate_result else _NONE_RESULT

    @property
    def sql(self) -> Optional[str]:
        """Get the first SQL string from sql_list.

        Returns:
            First SQL string if sql_list exists and is not empty, None otherwise.
        """
        if self.sql_list and len(self.sql_list) > 0:
            return self.sql_list[0].sql
        return None

    def get_sql(self, table: Optional[str] = None) -> Optional[str]:
        """Get SQL string for a specific table.

        Args:
            table: Table name. If None, returns the first SQL string.

        Returns:
            SQL string for the specified table, or first SQL string if table is None.
            Returns None if table not found or sql_list is empty.
        """
        if not self.sql_list or len(self.sql_list) == 0:
            return None

        if table is None:
            return self.sql_list[0].sql

        for gen_sql in self.sql_list:
            if gen_sql.table == table:
                return gen_sql.sql

        return None

    def format_sql_list(self, indent: str = "    ") -> str:
        """Format sql_list as a multi-line string.

        Args:
            indent: Indentation string for each line.

        Returns:
            Formatted string with all SQL statements, one per table.
        """
        if not self.sql_list or len(self.sql_list) == 0:
            return f"{indent}None"

        lines = []
        for i, gen_sql in enumerate(self.sql_list, 1):
            lines.append(f"{indent}{i}. Table: {gen_sql.table}")
            lines.append(f"{indent}   SQL: {gen_sql.sql}")

        return "\n".join(lines)


class SQLGenerator:
    """SQL generator that converts TLSQL AST nodes to standard SQL statements."""

    def __init__(self):
        """Initialize SQL generator."""

    @classmethod
    def convert_query(cls, tlsql: str) -> ConversionResult:
        """Convert a single TLSQL query to standard SQL.

        This is an internal method for converting individual TLSQL queries.
        For workflow conversion, use the top-level tlsql.convert() function.

        Args:
            tlsql: TLSQL query string.

        Returns:
            ConversionResult object containing SQL and other meta information.
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
            List of GeneratedSQL objects, one per table. Each object contains:
                - table: Table name
                - sql: Complete SELECT statement
                - columns: List of selected columns (or ['*'] for all columns)

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

        result = ConversionResult(
            statement_type='PREDICT',
            sql_list=sql_list,  # Contains GeneratedSQL objects for direct execution
            target_column=target_column,
            task_type=task_type,
            target_table=target_table_name,
            tables=[target_table_name],
            where_condition=where_condition
        )
        # Store source AST for later use in auto_generate_train
        result._source_ast = predict
        return result

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
        """Split WHERE conditions per table.
        Splits WHERE clause conditions into table-specific conditions by extracting
        AND-connected subconditions and grouping them by table.
        """
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

    def auto_generate_train(
        self,
        predict_result: ConversionResult
    ) -> ConversionResult:
        """Auto-generate TRAIN SQL from PREDICT SQL.

        Generates TRAIN SQL that excludes PREDICT data from the same table.
        Uses WHERE condition negation: NOT (WHERE condition).

        Args:
            predict_result: ConversionResult from PREDICT statement.

        Returns:
            ConversionResult for TRAIN statement.
        """
        table = predict_result.target_table
        predict_where = predict_result.where_condition

        if not predict_where:
            condition = None
        else:
            # Use WHERE condition negation
            condition = f"NOT ({predict_where})"

        sql = self._build_select_sql(table, ['*'], condition)

        return ConversionResult(
            statement_type='TRAIN',
            sql_list=[GeneratedSQL(
                table=table,
                sql=sql,
                columns=['*']
            )],
            target_table=table,
            tables=[table],
            where_condition=condition
        )
