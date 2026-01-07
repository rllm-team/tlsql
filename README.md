<div align="center">

# TLSQL: SQL-like API for Table Learning 

*A Python library that converts custom SQL-like statements into standard SQL queries for machine learning workflows on tables in modern data management systems.*

<br/>

[![Python](https://img.shields.io/badge/Python-3.8+-3776ab?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Documentation](https://img.shields.io/badge/Documentation-latest-blue?style=flat-square)](https://tlsql.readthedocs.io/en/latest/index.html)


---

</div>

## About

TLSQL simplifies machine learning on structured tables by converting SQL-like statements into standard SQL queries. This allows data scientists and engineers to focus on modeling rather than writing complex SQL or managing datasets manually.

TLSQL supports three types of statements that map directly to ML workflows:

- **`TRAIN WITH`**: Specifies the training set
- **`PREDICT VALUE`**: Specifies the test set
- **`VALIDATE WITH`**: Specifies the validation set

TLSQL works seamlessly with **RDBs, data warehouses, and data lakes**, enabling end-to-end table-based ML workflows.


## TLSQL Syntax

### 1. TRAIN WITH Statement

The `TRAIN WITH` statement specifies which columns and tables to use for training data, along with optional filtering conditions. This statement defines the dataset used to train your machine learning model.

#### Syntax

```sql
TRAIN WITH (column_selectors)
FROM table1, table2, ...
[WHERE conditions]
```

#### Examples

```sql
TRAIN WITH (users.*, movies.*, ratings.*)
FROM users, movies, ratings
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
FROM table1, table2, ...
[WHERE conditions]
```

#### Examples

```sql
VALIDATE WITH (users.*, movies.*, ratings.*)
FROM users, movies, ratings
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

## How to Try

### Installation

Install TLSQL from the `tlsql` root directory:

```bash
cd tlsql
pip install -e .
```

This will install TLSQL in development mode, allowing you to make changes to the code without reinstalling.

### Quick Start

```python
import tlsql


result = tlsql.convert("PREDICT VALUE(users.Age, CLF) FROM users")
print(result.statement_type)  # 'PREDICT'
print(result.target_column)   # 'users.Age'
print(result.task_type)       # 'CLF'
```

### Examples

Check out the examples directory for more usage examples:

- **`examples/sql_conversion.py`**: Basic TLSQL to SQL conversion examples
- **`examples/bridge_demo.py`**: BRIDGE model training with TLSQL
- **`examples/tl_workflow.py`**: TLSQL workflow with three-level logic






