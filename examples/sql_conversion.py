"""TLSQL to Standard SQL Conversion Example

The demo uses the TML1M dataset with three relational tables: users, movies, and ratings.
Demonstrates single statement conversion using SQLGenerator.convert_query().
"""

from tlsql.tlsql.sql_generator import SQLGenerator

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

    predict_sqls = SQLGenerator.convert_query(predict_query)

    print(f"Target Column: {predict_sqls.target_column}")
    print(f"Task Type: {predict_sqls.task_type}")
    print(f"Target Table: {predict_sqls.target_table}")
    print(f"WHERE Condition: {predict_sqls.where_condition}")
    print("\nGenerated Standard SQL:")
    print(predict_sqls.format_sql_list())

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

    train_sqls = SQLGenerator.convert_query(train_query)
    print(f"Type: {train_sqls.statement_type}")
    print("Generated Standard SQL:")
    print(train_sqls.format_sql_list())




def validate_query_conversion():
    """Demonstrate VALIDATE statement conversion"""
    print("\nVALIDATE Statement")

    validate_query = """
    VALIDATE WITH (users.Age)
    FROM users
    WHERE users.Gender='M' and users.userID>3000
    """

    print("TLSQL:")
    print(f"    {validate_query.strip()}")

    validate_sqls = SQLGenerator.convert_query(validate_query)

    print(f"Type: {validate_sqls.statement_type}")

    print("\nGenerated Standard SQL:")
    print(validate_sqls.format_sql_list())


if __name__ == "__main__":
    test_query_conversion()
    train_query_conversion()
    validate_query_conversion()
