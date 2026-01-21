"""TLSQL to Standard SQL Conversion Example

The demo uses the TML1M dataset with three relational tables: users, movies, and ratings.
"""

import tlsql


def train_query_conversion():
    """Demonstrate TRAIN statement conversion"""
    print("TRAIN Statement")

    train_query = """
    TRAIN WITH (users.*, movies.*, ratings.*)
    FROM users, movies, ratings
    WHERE users.Gender='M' and users.userID BETWEEN 1 AND 3000
    """

    print("TLSQL:")
    print(f"    {train_query.strip()}")

    train_sqls = tlsql.convert(train_query)
    print(f"Type: {train_sqls.statement_type}")
    print("Generated Standard SQL:")
    for i, gen_sql in enumerate(train_sqls.sql_list, 1):
        print(f"{i}. Table: {gen_sql.table}")
        print(f"   SQL: {gen_sql.sql}")


def test_query_conversion():
    """Demonstrate PREDICT statement conversion"""
    print("\nPREDICT Statement")

    predict_query = """
    PREDICT VALUE(users.Age, CLF)
    FROM users
    WHERE users.Gender='F'
    """

    print("TLSQL:")
    print(f"    {predict_query.strip()}")

    predict_sqls = tlsql.convert(predict_query)

    print(f"Target Column: {predict_sqls.target_column}")
    print(f"Task Type: {predict_sqls.task_type}")
    print(f"Target Table: {predict_sqls.target_table}")
    print(f"WHERE Condition: {predict_sqls.where_condition}")
    print("\nGenerated Standard SQL:")
    for i, gen_sql in enumerate(predict_sqls.sql_list, 1):
        print(f"{i}. Table: {gen_sql.table}")
        print(f"   SQL: {gen_sql.sql}")


def validate_query_conversion():
    """Demonstrate VALIDATE statement conversion"""
    print("\nVALIDATE Statement")

    validate_query = """
    VALIDATE WITH (users.*)
    FROM users
    WHERE users.Gender='M' and users.userID>3000
    """

    print("TLSQL:")
    print(f"    {validate_query.strip()}")

    validate_sqls = tlsql.convert(validate_query)

    print(f"Type: {validate_sqls.statement_type}")

    print("\nGenerated Standard SQL:")
    for i, gen_sql in enumerate(validate_sqls.sql_list, 1):
        print(f"{i}. Table: {gen_sql.table}")
        print(f"   SQL: {gen_sql.sql}")


if __name__ == "__main__":
    train_query_conversion()
    test_query_conversion()
    validate_query_conversion()
