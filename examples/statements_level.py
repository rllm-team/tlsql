"""Three-Level Logic Demo in TLSQL

"""


def level_I():
    """Level I: PREDICT - REQUIRED"""
    print("Level I: Only PREDICT")
    predict_sql = """
    PREDICT VALUE(users.Age, CLF)
    FROM users
    WHERE users.Gender='F'
    """

    train_sql = None
    validate_sql = None

    print("PREDICT:")
    print(f"\t{predict_sql.strip()}")
    print("\nTRAIN:None")
    print("Uses all data except PREDICT data")
    print("\nVALIDATE:None")
    print("Uses k=5 fold cross validation on train data")
    print()


def level_II():
    """Level II: TRAIN - OPTIONAL, defaults to all data except PREDICT"""
    print("Level II: PREDICT and TRAIN")

    predict_sql = """
    PREDICT VALUE(users.Age, CLF)
    FROM users
    WHERE users.Gender='F'
    """

    train_sql = """
    TRAIN WITH (users.*, movies.*, ratings.*)
    FROM users, movies, ratings
    WHERE users.Gender='M' and users.userID BETWEEN 1 AND 3000
    """

    validate_sql = None

    print("PREDICT:")
    print(f"\t{predict_sql.strip()}")
    print("\nTRAIN:")
    print(f"\t{train_sql.strip()}")
    print("\nVALIDATE:None")
    print("Uses k=5 fold cross validation on train data")
    print()


def level_III():
    """Level III: VALIDATE - OPTIONAL, defaults to k=5 fold cross validation"""
    print("Level III: PREDICT, TRAIN and VALIDATE")

    predict_sql = """
    PREDICT VALUE(users.Age, CLF)
    FROM users
    WHERE users.Gender='F'
    """

    train_sql = """
    TRAIN WITH (users.*, movies.*, ratings.*)
    FROM users, movies, ratings
    WHERE users.Gender='M' and users.userID BETWEEN 1 AND 3000
    """

    validate_sql = """
    VALIDATE WITH (users.*)
    FROM users
    WHERE users.Gender='M' and users.userID>3000
    """

    print("PREDICT:")
    print(f"\t{predict_sql.strip()}")
    print("\nTRAIN:")
    print(f"\t{train_sql.strip()}")
    print("\nVALIDATE:")
    print(f"\t{validate_sql.strip()}")
    print()


def main():
    level_I()
    level_II()
    level_III()


if __name__ == "__main__":
    main()
