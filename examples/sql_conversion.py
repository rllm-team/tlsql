"""TLSQL to Standard SQL Conversion Example

This example demonstrates how TLSQL statements are converted to standard SQL.
"""

import tlsql


def train():
    """Demonstrate TRAIN statement conversion"""
    print("TRAIN Statement")

    train_sql = """
    TRAIN WITH (users.*, movies.title, ratings.*)
    FROM users, movies, ratings
    WHERE users.Gender='M' and users.userID BETWEEN 1 AND 3000
    """

    print("TLSQL:")
    print(f"\t{train_sql.strip()}")

    result = tlsql.convert(train_sql)
    print(f"Type: {result.statement_type}")
    print("Generated Standard SQL:")
    for i, gen_sql in enumerate(result.sql_list, 1):
        print(f"{i}. Table: {gen_sql.table}")
        print(f"   SQL: {gen_sql.sql}")


def test():
    """Demonstrate PREDICT statement conversion"""
    print("\nPREDICT Statement")

    predict_sql = """
    PREDICT VALUE(users.Age, CLF)
    FROM users
    WHERE users.Gender='F'
    """

    print("TLSQL:")
    print(f"\t{predict_sql.strip()}")

    result = tlsql.convert(predict_sql)

    print(f"Target Column: {result.target_column}")
    print(f"Task Type: {result.task_type}")
    print(f"Target Table: {result.target_table}")
    print(f"WHERE Condition: {result.where_condition}")
    print(f"SQL Condition: {result.filter_condition.condition}")
    print("\nGenerated Standard SQL:")
    print(f"SELECT * FROM {result.filter_condition.table} WHERE {result.filter_condition.condition}")


def validate():
    """Demonstrate VALIDATE statement conversion"""
    print("\nVALIDATE Statement")

    validate_sql = """
    VALIDATE WITH (users.*, movies.*)
    FROM users, movies
    WHERE users.Gender='M' and users.userID>3000
    """

    print("TLSQL:")
    print(f"\t{validate_sql.strip()}")

    result = tlsql.convert(validate_sql)

    print(f"Type: {result.statement_type}")

    print("\nGenerated Standard SQL:")
    for i, gen_sql in enumerate(result.sql_list, 1):
        print(f"{i}. Table: {gen_sql.table}")
        print(f"   SQL: {gen_sql.sql}")


if __name__ == "__main__":
    train()
    test()
    validate()
