"""Abstract Syntax Tree node definitions.

Supports 3 statement types:
1. TRAIN WITH.
2. PREDICT VALUE.
3. VALIDATE WITH.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Any


@dataclass
class ASTNode:
    """Base class for all AST nodes.All AST nodes inherit from this class, used for type identification and unified interface.
    """
    pass


@dataclass
class ColumnReference(ASTNode):
    """Column reference, format is 'table.column' or 'column'.

    Attributes:
        table: Table name.
        column: Column name.
    """

    table: Optional[str] = None
    column: str = ""

    def __str__(self) -> str:
        """Return string representation of column reference."""
        if self.table:
            return f"{self.table}.{self.column}"
        return self.column


@dataclass
class Expr(ASTNode):
    """Base class for all expressions.
    """
    pass


@dataclass
class LiteralExpr(Expr):
    """Literal value.

    Attributes:
        value: Value of literal.
        value_type: Type of value, 'number' and 'string'.
    """

    value: Any
    value_type: str


@dataclass
class ColumnExpr(Expr):
    """Column reference in expression.

    Attributes:
        column: Column reference object.
    """

    column: ColumnReference


@dataclass
class BinaryExpr(Expr):
    """Binary expression.

    Supported operators:
    - Comparison operators: >, <, >=, <=, ==, !=, =.
    - Logical operators: AND, OR.

    Attributes:
        left: Left operand expression.
        operator: Operator.
        right: Right operand expression.
    """

    left: Expr
    operator: str
    right: Expr


@dataclass
class UnaryExpr(Expr):
    """Unary expression.

    Attributes:
        operator: Operator.
        operand: Operand expression.
    """

    operator: str
    operand: Expr


@dataclass
class BetweenExpr(Expr):
    """BETWEEN expression.

    Syntax: column BETWEEN value1 AND value2.

    Attributes:
        column: Column reference expression.
        lower: Lower bound value expression.
        upper: Upper bound value expression.
    """

    column: Expr
    lower: Expr
    upper: Expr


@dataclass
class InExpr(Expr):
    """IN expression.

    Syntax: column IN (value1, value2, ...).

    Attributes:
        column: Column reference expression.
        values: Value list.
    """
    column: Expr
    values: List[Expr]


@dataclass
class WhereClause(ASTNode):
    """WHERE clause.

    Attributes:
        condition: Condition expression tree.
    """

    condition: Expr


@dataclass
class ColumnSelector(ASTNode):
    """Column selector in USING clause.

    Attributes:
        table: Table name.
        column: Column name, '*' means all columns.
    """

    table: str
    column: str

    def __str__(self) -> str:
        return f"{self.table}.{self.column}"

    @property
    def is_wildcard(self) -> bool:
        """Determine if it's a wildcard selector (table.*)."""
        return self.column == '*'


@dataclass
class WithClause(ASTNode):
    """WITH clause in TRAIN/VALIDATE statement.

    Attributes:
        selectors: Column selector list.
    """

    selectors: List[ColumnSelector] = field(default_factory=list)


@dataclass
class TablesClause(ASTNode):
    """FROM clause for multiple tables.

    Syntax: FROM table1, table2, ...

    Attributes:
        tables: Table name list.
    """

    tables: List[str] = field(default_factory=list)


@dataclass
class TrainStatement(ASTNode):
    """TRAIN statement.

    Complete syntax:
    TRAIN WITH (column_selectors)
    FROM table1, table2, ...
    [WHERE conditions]

    Attributes:
        with_clause: WITH clause.
        tables: Tables clause.
        where: WHERE clause.
    """

    with_clause: WithClause
    tables: TablesClause
    where: Optional[WhereClause] = None

    def __repr__(self) -> str:
        parts = ["TrainStatement("]
        parts.append(f"  with={len(self.with_clause.selectors)} selectors")
        parts.append(f"  tables={', '.join(self.tables.tables)}")
        if self.where:
            parts.append("  where=<expression>")
        parts.append(")")
        return "\n".join(parts)


@dataclass
class ValidateStatement(ASTNode):
    """VALIDATE statement.

    VALIDATE WITH (column_selectors)
    FROM table1, table2, ...
    [WHERE conditions]

    Attributes:
        with_clause: WITH clause.
        tables: Tables clause.
        where: WHERE clause.
    """

    with_clause: WithClause
    tables: TablesClause
    where: Optional[WhereClause] = None

    def __repr__(self) -> str:
        parts = ["ValidateStatement("]
        parts.append(f"  with={len(self.with_clause.selectors)} selectors")
        parts.append(f"  tables={', '.join(self.tables.tables)}")
        if self.where:
            parts.append("  where=<expression>")
        parts.append(")")
        return "\n".join(parts)


@dataclass
class PredictType(ASTNode):
    """Prediction type, CLF/REG.

    Attributes:
        type_name: Prediction type.
    """

    type_name: str

    @property
    def is_classifier(self) -> bool:
        return self.type_name.upper() == 'CLF'

    @property
    def is_regressor(self) -> bool:
        return self.type_name.upper() == 'REG'


@dataclass
class ValueClause(ASTNode):
    """VALUE clause in PREDICT statement.

    Attributes:
        target: Prediction target column.
        predict_type: Prediction type.
    """

    target: ColumnReference
    predict_type: PredictType


@dataclass
class FromClause(ASTNode):
    """FROM clause.

    Attributes:
        table: Table name.
    """

    table: str


@dataclass
class PredictStatement(ASTNode):
    """PREDICT statement.

    PREDICT VALUE(target_column, predict_type)
    FROM table
    [WHERE conditions]

    Attributes:
        value: VALUE clause.
        from_table: FROM clause.
        where: WHERE clause.
    """

    value: ValueClause
    from_table: FromClause
    where: Optional[WhereClause] = None

    def __repr__(self) -> str:
        parts = ["PredictStatement("]
        parts.append(f"  target={self.value.target}")
        parts.append(f"  type={self.value.predict_type.type_name}")
        parts.append(f"  from={self.from_table.table}")
        if self.where:
            parts.append("  where=<expression>")
        parts.append(")")
        return "\n".join(parts)


@dataclass
class Statement(ASTNode):
    """Contains TRAIN/PREDICT/validate statements.

    Attributes:
        train: TRAIN statement.
        predict: PREDICT statement.
        validate: VALIDATE statement.
    """

    train: Optional[TrainStatement] = None
    predict: Optional[PredictStatement] = None
    validate: Optional[ValidateStatement] = None

    @property
    def statement_type(self) -> str:
        """Return statement type."""
        if self.train:
            return "TRAIN"
        elif self.predict:
            return "PREDICT"
        elif self.validate:
            return "VALIDATE"
        return "UNKNOWN"

    def __repr__(self) -> str:
        if self.train:
            return repr(self.train)
        elif self.predict:
            return repr(self.predict)
        elif self.validate:
            return repr(self.validate)
        return "Statement(empty)"
