"""Comprehensive tests for PREDICT statements
"""

import sys

sys.path.append("./")
sys.path.append("../")
sys.path.append("../../")

from tlsql import Parser, convert


def print_test_header(test_name: str):
    """Print test section header"""
    print(f"Test: {test_name}")


def test_predict_basic():
    """Test basic PREDICT statements"""
    print_test_header("Basic PREDICT Statements")

    test_cases = [
        {
            "name": "Simple classification",
            "sql": "PREDICT VALUE(users.Age, CLF) FROM users WHERE users.Gender='M'",
        },
        {
            "name": "Simple regression",
            "sql": "PREDICT VALUE(users.Age, REG) FROM users WHERE users.Gender='F'",
        },
        {
            "name": "Without table prefix in column",
            "sql": "PREDICT VALUE(Age, CLF) FROM users WHERE Gender='M'",
        },
        {
            "name": "No WHERE clause",
            "sql": "PREDICT VALUE(users.Age, CLF) FROM users",
        },
    ]

    for test_case in test_cases:
        print(f"\n--- {test_case['name']} ---")
        print(f"SQL: {test_case['sql']}")
        try:
            parser = Parser(test_case['sql'])
            ast = parser.parse()

            if not ast.predict:
                print(" Not a PREDICT statement")
                continue

            predict = ast.predict
            print(f"  Target: {predict.value.target.table}.{predict.value.target.column}")
            print(f"  Task type: {predict.value.predict_type.type_name}")
            print(f"  Table: {predict.from_table.table}")
            print(f"  Has WHERE: {predict.where is not None}")

            result = convert(predict_query=test_case['sql'])
            print(f"  Generated condition: {result.predict_result.where_condition or 'None'}")

        except Exception as e:
            print(f"{e}")
            import traceback
            traceback.print_exc()


def test_predict_where_conditions():
    """Test PREDICT with various WHERE conditions"""
    print_test_header("PREDICT with Various WHERE Conditions")

    test_cases = [
        {
            "name": "Simple equality",
            "sql": "PREDICT VALUE(users.Age, CLF) FROM users WHERE users.Gender='M'",
        },
        {
            "name": "Inequality",
            "sql": "PREDICT VALUE(users.Age, CLF) FROM users WHERE users.Gender != 'F'",
        },
        {
            "name": "Greater than",
            "sql": "PREDICT VALUE(users.Age, CLF) FROM users WHERE users.Age > 25",
        },
        {
            "name": "Less than or equal",
            "sql": "PREDICT VALUE(users.Age, CLF) FROM users WHERE users.Age <= 50",
        },
        {
            "name": "AND condition",
            "sql": "PREDICT VALUE(users.Age, CLF) FROM users WHERE users.Gender='M' AND users.Age > 25",
        },
        {
            "name": "OR condition",
            "sql": "PREDICT VALUE(users.Age, CLF) FROM users WHERE users.Gender='M' OR users.Age < 20",
        },
        {
            "name": "Complex AND/OR",
            "sql": (
                "PREDICT VALUE(users.Age, CLF) FROM users "
                "WHERE users.Gender='M' AND (users.Age > 25 OR users.Occupation = 1)"
            ),
        },
        {
            "name": "BETWEEN",
            "sql": "PREDICT VALUE(users.Age, CLF) FROM users WHERE users.Age BETWEEN 25 AND 50",
        },
        {
            "name": "IN with numbers",
            "sql": "PREDICT VALUE(users.Age, CLF) FROM users WHERE users.Occupation IN (1, 2, 3, 4)",
        },
        {
            "name": "IN with strings",
            "sql": "PREDICT VALUE(users.Age, CLF) FROM users WHERE users.Gender IN ('M', 'F')",
        },
        {
            "name": "Multiple conditions",
            "sql": (
                "PREDICT VALUE(users.Age, CLF) FROM users "
                "WHERE users.Gender='M' AND users.Age > 25 "
                "AND users.Occupation IN (1, 2, 3)"
            ),
        },
    ]

    for test_case in test_cases:
        print(f"\n--- {test_case['name']} ---")
        print(f"SQL: {test_case['sql']}")
        try:
            parser = Parser(test_case['sql'])
            ast = parser.parse()

            if not ast.predict:
                print(" Not a PREDICT statement")
                continue

            # Test SQL generation
            result = convert(predict_query=test_case['sql'])

            print("Parsed and generated")
            print(f"  SQL Condition: {result.predict_result.where_condition or 'None'}")

        except Exception as e:
            print(f" {e}")
            import traceback
            traceback.print_exc()


def test_predict_column_references():
    """Test PREDICT with different column reference formats"""
    print_test_header("PREDICT with Different Column References")

    test_cases = [
        {
            "name": "Table.Column format",
            "sql": "PREDICT VALUE(users.Age, CLF) FROM users WHERE users.Gender='M'",
        },
        {
            "name": "Column only (no table prefix)",
            "sql": "PREDICT VALUE(Age, CLF) FROM users WHERE Gender='M'",
        },
        {
            "name": "Mixed references in WHERE",
            "sql": "PREDICT VALUE(users.Age, CLF) FROM users WHERE Gender='M' AND users.Occupation > 1",
        },
    ]
    for test_case in test_cases:
        print(f"\n--- {test_case['name']} ---")
        print(f"SQL: {test_case['sql']}")
        try:
            parser = Parser(test_case['sql'])
            ast = parser.parse()

            if not ast.predict:
                print(" Not a PREDICT statement")
                continue

            predict = ast.predict
            target = predict.value.target

            print("Parsed successfully")
            target_table_str = (
                target.table if target.table else 'None (defaults to FROM table)'
            )
            print(f"  Target table: {target_table_str}")
            print(f"  Target column: {target.column}")

            # Test SQL generation
            result = convert(predict_query=test_case['sql'])
            print(f"  Generated condition: {result.predict_result.where_condition or 'None'}")

        except Exception as e:
            print(f" {e}")
            import traceback
            traceback.print_exc()


def test_predict_task_types():
    """Test PREDICT with different task types"""
    print_test_header("PREDICT with Different Task Types")

    test_cases = [
        {
            "name": "Classification (CLF)",
            "sql": "PREDICT VALUE(users.Gender, CLF) FROM users WHERE users.Age > 25",
        },
        {
            "name": "Regression (REG)",
            "sql": "PREDICT VALUE(users.Age, REG) FROM users WHERE users.Gender='M'",
        },
    ]

    for test_case in test_cases:
        print(f"\n--- {test_case['name']} ---")
        print(f"SQL: {test_case['sql']}")
        try:
            parser = Parser(test_case['sql'])
            ast = parser.parse()

            if not ast.predict:
                print(" Not a PREDICT statement")
                continue

            predict = ast.predict
            predict_type = predict.value.predict_type

            print("Parsed successfully")
            print(f"  Task type: {predict_type.type_name}")
            print(f"  Is classifier: {predict_type.is_classifier}")
            print(f"  Is regressor: {predict_type.is_regressor}")

        except Exception as e:
            print(f" {e}")
            import traceback
            traceback.print_exc()


def test_predict_edge_cases():
    """Test PREDICT edge cases and error handling"""
    print_test_header("PREDICT Edge Cases")

    test_cases = [
        {
            "name": "No WHERE clause",
            "sql": "PREDICT VALUE(users.Age, CLF) FROM users",
            "should_pass": True,
        },
        {
            "name": "Empty WHERE (should fail)",
            "sql": "PREDICT VALUE(users.Age, CLF) FROM users WHERE",
            "should_pass": False,
        },
        {
            "name": "Nested parentheses",
            "sql": (
                "PREDICT VALUE(users.Age, CLF) FROM users "
                "WHERE (users.Gender='M' AND users.Age > 25) "
                "OR users.Occupation = 1"
            ),
            "should_pass": True,
        },
        {
            "name": "Multiple IN clauses",
            "sql": (
                "PREDICT VALUE(users.Age, CLF) FROM users "
                "WHERE users.Occupation IN (1, 2, 3) "
                "AND users.Age IN (25, 30, 35)"
            ),
            "should_pass": True,
        },
        {
            "name": "BETWEEN with AND in WHERE",
            "sql": "PREDICT VALUE(users.Age, CLF) FROM users WHERE users.Age BETWEEN 25 AND 50 AND users.Gender='M'",
            "should_pass": True,
        },
    ]

    for test_case in test_cases:
        print(f"\n--- {test_case['name']} ---")
        print(f"SQL: {test_case['sql']}")
        print(f"Expected: {'PASS' if test_case['should_pass'] else 'FAIL'}")

        try:
            parser = Parser(test_case['sql'])
            ast = parser.parse()

            if not ast.predict:
                if test_case['should_pass']:
                    print(" Not a PREDICT statement")
                else:
                    print("  [EXPECTED FAIL] Correctly rejected")
                continue

            if not test_case['should_pass']:
                print("  [UNEXPECTED PASS] Should have failed but passed")
                continue

            # Test SQL generation
            result = convert(predict_query=test_case['sql'])

            print("Parsed and generated")
            print(f"  Condition: {result.predict_result.where_condition or 'None'}")

        except Exception as e:
            if test_case['should_pass']:
                print(f" {e}")
                import traceback
                traceback.print_exc()
            else:
                print(f"  [EXPECTED FAIL] {e}")


def run_all_tests():
    """Run all PREDICT statement tests"""
    print("PREDICT Statement Tests")
    test_predict_edge_cases()


if __name__ == "__main__":
    run_all_tests()
