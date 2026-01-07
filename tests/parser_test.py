import sys

sys.path.append("./")
sys.path.append("../")
sys.path.append("../../")

from tlsql import Parser


def test_predict_syntax():
    """Test PREDICT VALUE statement with full AST tree"""
    print("\n\nTest 1: PREDICT VALUE statement")

    query = """
    PREDICT VALUE(user.age, CLF)
    FROM user
    WHERE user.loc = 'Florida'
    """

    try:
        parser = Parser(query)
        ast = parser.parse()

        print(f"Statement type: {ast.statement_type}")

        predict = ast.predict

        print("Basic Information:")
        print("VALUE clause:")
        print(f"  Target: {predict.value.target}")
        print(f"  Type: {predict.value.predict_type.type_name}")
        print(f"  Is classifier: {predict.value.predict_type.is_classifier}")
        print(f"  Is regressor: {predict.value.predict_type.is_regressor}")

        print("\nFROM clause:")
        print(f"  Table: {predict.from_table.table}")

        if predict.where:
            print("\nWHERE clause:")
            print(f"  Type: {type(predict.where.condition).__name__}")
        print("AST String Representation:")
        print(predict)

    except Exception as e:
        print(f"\nParse failed: {e}")
        import traceback
        traceback.print_exc()


def test_train_syntax():
    """Test TRAIN USING statement"""
    print("\nTest 2: TRAIN USING statement")

    query = """
    TRAIN with  (user.*, movie.title, rating.*)
    FROM user, movie, rating
    WHERE user.loc = 'BJ' AND movie.year > 1990 AND rating.score > 3
    """

    try:
        parser = Parser(query)
        ast = parser.parse()

        print(f"Statement type: {ast.statement_type}")

        train = ast.train

        print("USING clause:")
        print(f"  Selector count: {len(train.with_clause.selectors)}")
        for i, selector in enumerate(train.with_clause.selectors, 1):
            print(f"  {i}. {selector.table}.{selector.column} (is wildcard: {selector.is_wildcard})")

        print("FROM clause:")
        print(f"  Tables: {', '.join(train.tables.tables)}")
        print(f"  Count: {len(train.tables.tables)}")

        if train.where:
            print("\nWHERE clause:")
            print(f"  Type: {type(train.where.condition).__name__}")
            print("  Condition: <expression tree>")

        print("Full AST:")
        print(train)

    except Exception as e:
        print(f"\nParse failed: {e}")
        import traceback
        traceback.print_exc()


def test_validate_syntax():
    """Test VALIDATE WITH statement"""
    print("\nTest 3: VALIDATE WITH statement")

    query = """
    VALIDATE WITH  (user.*, movie.title, rating.*)
    FROM user, movie, rating
    WHERE user.loc = 'BJ' AND movie.year > 1990 AND rating.score > 3
    """

    try:
        parser = Parser(query)
        ast = parser.parse()

        print("[SUCCESS] Parsed successfully")
        print(f"Statement type: {ast.statement_type}")

        validate = ast.validate

        print("USING clause:")
        print(f"  Selector count: {len(validate.with_clause.selectors)}")
        for i, selector in enumerate(validate.with_clause.selectors, 1):
            print(f"  {i}. {selector.table}.{selector.column} (is wildcard: {selector.is_wildcard})")

        print("FROM clause:")
        print(f"  Tables: {', '.join(validate.tables.tables)}")
        print(f"  Count: {len(validate.tables.tables)}")

        if validate.where:
            print("\nWHERE clause:")
            print(f"  Type: {type(validate.where.condition).__name__}")
            print("  Condition: <expression tree>")

        print("Full AST:")
        print(validate)

    except Exception as e:
        print(f"\nParse failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_predict_syntax()
    test_train_syntax()
    test_validate_syntax()
