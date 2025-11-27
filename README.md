# TL-SQL: Design SQL-like API for Table learning

TL-SQL is a Python library that converts custom SQL-like statements into standard SQL queries for machine learning workflows. TL-SQL supports three types of statements for machine learning workflows. Each statement has a specific syntax and purpose.

## Dataset Splitting

TL-SQL uses three statements to divide your dataset into training, validation, and test sets:

- **`TRAIN WITH`**: Specifies the **training set** - data used to train the model
- **`PREDICT VALUE`**: Specifies the **test set** - data used for final evaluation (prediction targets)
- **`VALIDATE WITH`**: Specifies the **validation set** - data used for model selection and hyperparameter tuning

While both `train_sql` and `validate_sql` can be `None`:

- **`train_sql = None`**: Uses all data from target table (excluding PREDICT data) as training data
- **`validate_sql = None`**: Uses k-fold cross-validation on training data

## TL-SQL Syntax

### 1. TRAIN WITH Statement

The `TRAIN WITH` statement specifies which columns and tables to use for training data, along with optional filtering conditions. This statement defines the dataset used to train your machine learning model.

#### Syntax

```sql
TRAIN WITH (column_selectors)
FROM Tables(table1, table2, ...)
[WHERE conditions]
```

#### Examples

```sql
TRAIN WITH (users.*, movies.*, ratings.*)
FROM Tables(users, movies, ratings)
WHERE users.Gender='M' AND movies.Year >= 2000
```

### 2. PREDICT VALUE Statement

The `PREDICT VALUE` statement specifies the target column for prediction and the task type (classification or regression). This statement defines the test set - the data for which you want to make predictions. The `WHERE` clause in this statement filters which rows are included in the test set.

#### Syntax

```sql
PREDICT VALUE(table.column, TASK_TYPE)
FROM table
[WHERE conditions]
```

#### Task Types

- **`CLF`**: Classification task - predicts discrete categories
- **`REG`**: Regression task - predicts continuous values

#### Examples

```sql
PREDICT VALUE(users.Age, CLF)
FROM users
WHERE users.Gender='F' OR users.userID IN (1,2,3,4,5)
```

### 3. VALIDATE WITH Statement

The `VALIDATE WITH` statement specifies validation data with the same syntax as `TRAIN WITH`. This statement defines the validation set used for model selection. If omitted, the pipeline will use k-fold cross-validation on the training data.

#### Syntax

```sql
VALIDATE WITH (column_selectors)
FROM Tables(table1, table2, ...)
[WHERE conditions]
```

#### Examples

```sql
VALIDATE WITH (users.*, movies.*, ratings.*)
FROM Tables(users, movies, ratings)
WHERE users.Gender='M' AND movies.Year < 2000
```

## Supported Operators

### Comparison Operators
- `=`, `!=`, `<>`, `<`, `<=`, `>`, `>=`

### Logical Operators
- `AND`, `OR`, `NOT`

### Special Operators
- `IN (value1, value2, ...)`: Check if value is in list
- `BETWEEN value1 AND value2`: Range check


## Project Structure

```
tl_sql/
├── core/                    # Core language components
│   ├── lexer.py            # Tokenizer for TL-SQL
│   ├── parser.py            # Parser for TL-SQL syntax
│   ├── ast_nodes.py         # Abstract Syntax Tree nodes
│   ├── tokens.py            # Token definitions
│   └── exceptions.py        # Custom exceptions
│
├── executor/                # SQL execution components
│   ├── sql_generator.py     # Converts AST to SQL
│   ├── db_executor.py       # Database connection and execution
│   └── pipeline.py          #  pipeline interface
│
├── config/                  # Configuration utilities
│   ├── __init__.py         # Config loading functions
│   └── dataset_config.json # Dataset configuration
│
├── examples/                # Example scripts
│   ├── sql_conversion.py    # SQL conversion demo
│   ├── bridge_demo.py       # BRIDGE model demo
│   ├── train_with_tl_sql.py # Training utilities
│   └── bridge/             # BRIDGE model utilities
│       └── utils/          # Helper functions

```
