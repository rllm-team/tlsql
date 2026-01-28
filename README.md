<div align="center">

# TLSQL: SQL-like API for Table Learning

 

*A Python library that converts custom SQL-like statements into standard SQL queries for machine learning workflows on tables in modern data management systems.*

<br/>

[![Python](https://img.shields.io/badge/Python-3.8+-3776ab?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Documentation](https://img.shields.io/badge/Documentation-latest-blue?style=flat-square)](https://tlsql.readthedocs.io/en/latest/index.html)
[![Paper](https://img.shields.io/badge/Paper-arXiv:2601.14109-b31b1b?style=flat-square)](https://arxiv.org/abs/2601.14109)


---

</div>

## About
TLSQL is a system designed to simplify machine learning workflows on structured tabular data. It translates SQL-like statements into standard SQL queries and structured learning task descriptions, enabling data scientists and engineers to focus on model development instead of writing complex SQL or manually managing datasets.

TLSQL works seamlessly with **RDBs, data warehouses, and data lakes**, enabling end-to-end table-based ML workflows.

- **`PREDICT VALUE`**: Specifies the test set.
- **`TRAIN WITH`**: Specifies the training set.
- **`VALIDATE WITH`**: Specifies the validation set.

<div align="center">
  <img src="docs/source/_static/workflow.jpg" alt="TLSQL Workflow" width="450"/>
  <br/>
  <small><strong>The TLSQL Workflow</strong></small>
</div>

## TLSQL Syntax

### 1. PREDICT VALUE Statement

The `PREDICT` statement specifies the target column for prediction and the task type (classification or regression). This statement defines the test set - the data for which you want to make predictions. The `WHERE` clause in this statement filters which rows are included in the test set.

#### Syntax

```sql
PREDICT VALUE(column_selector, TASK_TYPE)
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
WHERE users.Gender='F' 
```

### 2. TRAIN WITH Statement

The `TRAIN` statement specifies which columns and tables to use for training data, along with optional filtering conditions. This statement defines the dataset used to train your machine learning model.

#### Syntax

```sql
TRAIN WITH column_selector
FROM table1, table2, ...
[WHERE conditions]
```

#### Examples

```sql
TRAIN WITH (users.*, movies.*, ratings.*)
FROM users, movies, ratings
WHERE users.Gender='M' AND users.userID<3000
```

### 3. VALIDATE WITH Statement

The `VALIDATE` statement is used to specify the validation data, and its syntax is similar to that of `PREDICT`, with only slight differences in the beginning part.

#### Syntax

```sql
VALIDATE WITH column_selector
FROM table
[WHERE conditions]
```

#### Examples

```sql
VALIDATE WITH (users.Age)
FROM users
WHERE users.Gender='M' and users.userID>3000
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


result = tlsql.convert(
    predict_query="PREDICT VALUE(users.Age, CLF) FROM users WHERE users.Gender='F'"
)
print(result.predict.statement_type)  # 'PREDICT'
print(result.predict.target_column)   # 'users.Age'
print(result.predict.task_type)       # 'CLF'
```

### Examples

Check out the examples directory for more usage examples:

- **`examples/sql_conversion.py`**: Basic TLSQL to SQL conversion examples
- **`examples/bridge_demo.py`**: BRIDGE model training with TLSQL
- **`examples/tl_workflow.py`**: TLSQL workflow with three-level logic






### Citation

```
@article{chen2026tlsql,
  title={TLSQL: Table Learning Structured Query Language},
  author={Chen, Feiyang and Zhong, Ken and Zhang, Aoqian and Wang, Zheng and Pan, Li and Li, Jianhua},
  journal={arXiv preprint arXiv:2601.14109},
  year={2026}
}
```
**Paper Link**: [arXiv:2601.14109](https://arxiv.org/abs/2601.14109)