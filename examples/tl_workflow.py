"""TLSQL Workflow Demo

Demonstrates three-level logic for TLSQL statements:
- Level I: PREDICT only (required)
- Level II: PREDICT + TRAIN (TRAIN optional)
- Level III: PREDICT + TRAIN + VALIDATE (both TRAIN and VALIDATE optional)

This work is conducted on the TML1M dataset from SJTUTables, a dataset of users, movies, and ratings.
"""


def level_I():
    """Level I: PREDICT - REQUIRED"""
    print("Level I: Only PREDICT")
    predict_tlsql = """
    PREDICT VALUE(users.Age, CLF)
    FROM users
    WHERE users.Gender='F'
    """

    print("PREDICT:")
    print(f"    {predict_tlsql.strip()}")
    print("\nTRAIN: Not specified, default to using all data except PREDICT data")
    print("\nVALIDATE: Not specified, default to using k=5 fold cross validation on train data")
    print()


def level_II():
    """Level II: TRAIN - OPTIONAL, defaults to all data except PREDICT"""
    print("Level II: PREDICT and TRAIN")

    predict_tlsql = """
    PREDICT VALUE(users.Age, CLF)
    FROM users
    WHERE users.Gender='F'
    """

    train_tlsql = """
    TRAIN WITH (users.*, movies.*, ratings.*)
    FROM users, movies, ratings
    WHERE users.Gender='M' and users.userID BETWEEN 1 AND 3000
    """

    print("PREDICT:")
    print(f"    {predict_tlsql.strip()}")
    print("\nTRAIN:")
    print(f"    {train_tlsql.strip()}")
    print("\nVALIDATE: Not specified, default to using k=5 fold cross validation on train data")
    print()


def level_III():
    """Level III: VALIDATE - OPTIONAL, defaults to k=5 fold cross validation"""
    print("Level III: PREDICT, TRAIN and VALIDATE")

    predict_tlsql = """
    PREDICT VALUE(users.Age, CLF)
    FROM users
    WHERE users.Gender='F'
    """

    train_tlsql = """
    TRAIN WITH (users.*, movies.*, ratings.*)
    FROM users, movies, ratings
    WHERE users.Gender='M' and users.userID BETWEEN 1 AND 3000
    """

    validate_tlsql = """
    VALIDATE WITH (users.*)
    FROM users
    WHERE users.Gender='M' and users.userID>3000
    """

    print("PREDICT:")
    print(f"    {predict_tlsql.strip()}")
    print("\nTRAIN:")
    print(f"    {train_tlsql.strip()}")
    print("\nVALIDATE:")
    print(f"    {validate_tlsql.strip()}")
    print()


if __name__ == "__main__":
    level_I()
    level_II()
    level_III()
