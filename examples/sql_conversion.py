"""TLSQL to Standard SQL Conversion Example

This example demonstrates how TLSQL statements are converted to standard SQL.
"""

import tlsql


def train():
    """Demonstrate TRAIN statement conversion"""
    print("TRAIN Statement")

    train_tlsql = """
    TRAIN WITH (users.*, movies.*, ratings.*)
    FROM users, movies, ratings
    WHERE users.Gender='M' and users.userID BETWEEN 1 AND 3000
    """

    print("TLSQL:")
    print(f"    {train_tlsql.strip()}")

    train_sqls = tlsql.convert(train_tlsql)
    print(f"Type: {train_sqls.statement_type}")
    print("Generated Standard SQL:")
    for i, gen_sql in enumerate(train_sqls.sql_list, 1):
        print(f"{i}. Table: {gen_sql.table}")
        print(f"   SQL: {gen_sql.sql}")


def test():
    """Demonstrate PREDICT statement conversion"""
    print("\nPREDICT Statement")

    predict_tlsql = """
    PREDICT VALUE(users.Age, CLF)
    FROM users
    WHERE users.Gender='F'
    """

    print("TLSQL:")
    print(f"    {predict_tlsql.strip()}")

    predict_sqls = tlsql.convert(predict_tlsql)

    print(f"Target Column: {predict_sqls.target_column}")
    print(f"Task Type: {predict_sqls.task_type}")
    print(f"Target Table: {predict_sqls.target_table}")
    print(f"WHERE Condition: {predict_sqls.where_condition}")
    print(f"SQL Condition: {predict_sqls.filter_condition.condition}")
    print("\nGenerated Standard SQL:")
    print(f"SELECT * FROM {predict_sqls.filter_condition.table} WHERE {predict_sqls.filter_condition.condition}")


def validate():
    """Demonstrate VALIDATE statement conversion"""
    print("\nVALIDATE Statement")

    validate_tlsql = """
    VALIDATE WITH (users.*)
    FROM users
    WHERE users.Gender='M' and users.userID>3000
    """

    print("TLSQL:")
    print(f"    {validate_tlsql.strip()}")

    validate_sqls = tlsql.convert(validate_tlsql)

    print(f"Type: {validate_sqls.statement_type}")

    print("\nGenerated Standard SQL:")
    for i, gen_sql in enumerate(validate_sqls.sql_list, 1):
        print(f"{i}. Table: {gen_sql.table}")
        print(f"   SQL: {gen_sql.sql}")


if __name__ == "__main__":
    train()
    test()
    validate()
