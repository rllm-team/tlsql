"""TLSQL Workflow Demo

Demonstrates three-level logic for TLSQL statements:
- Level I: PREDICT only (required, TRAIN auto-generated)
- Level II: PREDICT + TRAIN (TRAIN optional)
- Level III: PREDICT + TRAIN + VALIDATE (both TRAIN and VALIDATE optional)

The demo uses the TML1M dataset with three relational tables: users, movies, and ratings.
"""

import tlsql


def level_I():
    """Level I: PREDICT - REQUIRED, TRAIN auto-generated"""
    print("Level I: Only PREDICT")
    predict_query = """
    PREDICT VALUE(users.Age, CLF)
    FROM users
    WHERE users.Gender='F'
    """
    result = tlsql.convert(predict_query=predict_query)
    print("PREDICT Query:")
    print(f"    {predict_query.strip()}\n")
    print("PREDICT SQL:")
    print(f"    {result.predict.sql}")
    print("TRAIN SQL:")
    print(result.train.format_sql_list())
    print("(Not specified, default to using all data except PREDICT data)")
    print("VALIDATE SQL:")
    print(f"    {result.validate.sql}")
    print("(Not specified)")


def level_II():
    """Level II: TRAIN - OPTIONAL, defaults to all data except PREDICT"""
    print("\nLevel II: PREDICT and TRAIN")

    predict_query = """
    PREDICT VALUE(users.Age, CLF)
    FROM users
    WHERE users.Gender='F'
    """

    train_query = """
    TRAIN WITH (users.*, movies.*, ratings.*)
    FROM users, movies, ratings
    WHERE users.Gender='M' and users.userID BETWEEN 1 AND 3000
    """

    result = tlsql.convert(predict_query=predict_query, train_query=train_query)
    print("PREDICT Query:")
    print(f"    {predict_query.strip()}\n")
    print("PREDICT SQL:")
    print(f"    {result.predict.sql}")
    print("TRAIN Query:")
    print(f"    {train_query.strip()}\n")
    print("TRAIN SQL:")
    print(result.train.format_sql_list())
    print("VALIDATE SQL:")
    print(f"    {result.validate.sql}")
    print("(Not specified)")


def level_III():
    """Level III: VALIDATE - OPTIONAL"""
    print("\nLevel III: PREDICT, TRAIN and VALIDATE")

    predict_query = """
    PREDICT VALUE(users.Age, CLF)
    FROM users
    WHERE users.Gender='F'
    """

    train_query = """
    TRAIN WITH (users.*, movies.*, ratings.*)
    FROM users, movies, ratings
    WHERE users.Gender='M' and users.userID BETWEEN 1 AND 3000
    """

    validate_query = """
    VALIDATE WITH (users.Age)
    FROM users
    WHERE users.Gender='M' and users.userID>3000
    """

    result = tlsql.convert(
        predict_query=predict_query,
        train_query=train_query,
        validate_query=validate_query
    )
    print("PREDICT Query:")
    print(f"    {predict_query.strip()}\n")
    print("PREDICT SQL:")
    print(f"    {result.predict.sql}")
    print("TRAIN Query:")
    print(f"    {train_query.strip()}\n")
    print("TRAIN SQL:")
    print(result.train.format_sql_list())
    print("VALIDATE Query:")
    print(f"    {validate_query.strip()}\n")
    print("VALIDATE SQL:")
    print(f"    {result.validate.sql}")


if __name__ == "__main__":
    level_I()
    level_II()
    level_III()
