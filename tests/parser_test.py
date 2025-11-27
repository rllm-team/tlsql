import sys
import os

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from tl_sql.core.parser import Parser


def test_train_syntax():
    """Test TRAIN USING statement"""
    print("Test 1: TRAIN USING statement ")

    
    query = """
    TRAIN with  (user.*, movie.title, rating.*)
    FROM Tables(user, movie, rating) 
    WHERE user.loc = 'BJ' AND movie.year > 1990 AND rating.score > 3
    """
    
    try:
        parser = Parser(query)
        ast = parser.parse()
        
        print("[SUCCESS] Parsed successfully ")
        print(f"Statement type: {ast.statement_type}")
        
        # Access TRAIN statement / 访问TRAIN语句
        train = ast.train
        
        print("USING clause:")
        print(f"  Selector count: {len(train.with_clause.selectors)}")
        for i, selector in enumerate(train.with_clause.selectors, 1):
            print(f"  {i}. {selector.table}.{selector.column} (is wildcard: {selector.is_wildcard})")
        
        print("FROM Tables:")
        print(f"  Tables : {', '.join(train.tables.tables)}")
        print(f"  Count: {len(train.tables.tables)}")
        
        if train.where:
            print("\nWHERE clause:")
            print(f"  Type: {type(train.where.condition).__name__}")
            print("  Condition: <expression tree>")
        
        print("Full AST:")
        print(train)
        
    except Exception as e:
        print(f"\n[FAILED] Parse failed: {e}")
        import traceback
        traceback.print_exc()
def test_validate_syntax():
    """Test VALIDATE WITH statement"""
    print("Test 1: VALIDATE WITH statement")

    
    query = """
    VALIDATE WITH  (user.*, movie.title, rating.*)
    FROM Tables(user, movie, rating) 
    WHERE user.loc = 'BJ' AND movie.year > 1990 AND rating.score > 3
    """
    
    try:
        parser = Parser(query)
        ast = parser.parse()
        
        print("[SUCCESS] Parsed successfully")
        print(f"Statement type / 语句类型: {ast.statement_type}")
        
        # Access VALIDATE statement
        validate = ast.validate
        
        print("USING clause:")
        print(f"  Selector count: {len(validate.with_clause.selectors)}")
        for i, selector in enumerate(validate.with_clause.selectors, 1):
            print(f"  {i}. {selector.table}.{selector.column} (is wildcard: {selector.is_wildcard})")
        
        print("FROM Tables:")
        print(f"  Tables : {', '.join(validate.tables.tables)}")
        print(f"  Count : {len(validate.tables.tables)}")
        
        if validate.where:
            print("\nWHERE clause :")
            print(f"  Type : {type(validate.where.condition).__name__}")
            print("  Condition : <expression tree / 表达式树>")
        
        print("Full AST:")
        print(validate)
        
    except Exception as e:
        print(f"\n[FAILED] Parse failed: {e}")
        import traceback
        traceback.print_exc()



def test_predict_syntax():
    """Test PREDICT VALUE statement"""
    print("\n\nTest 2: PREDICT VALUE statement")

    
    query = """
    PREDICT VALUE(user.age, CLF)
    FROM user
    WHERE user.loc = 'Florida'
    """
    
    try:
        parser = Parser(query)
        ast = parser.parse()
        
        print("[SUCCESS] Parsed successfully")
        print(f"Statement type / 语句类型: {ast.statement_type}")
        
        # Access PREDICT statement 
        predict = ast.predict
        
        print("VALUE clause:")
        print(f"  Target : {predict.value.target}")
        print(f"  Type: {predict.value.predict_type.type_name}")
        print(f"  Is classifier : {predict.value.predict_type.is_classifier}")
        print(f"  Is regressor: {predict.value.predict_type.is_regressor}")
        
        print("\nFROM clause:")
        print(f"  Table: {predict.from_table.table}")
        
        if predict.where:
            print("\nWHERE clause:")
            print(f"  Type : {type(predict.where.condition).__name__}")
        
        print("\nFull AST:")
        print(predict)
        
    except Exception as e:
        print(f"\n[FAILED] Parse failed: {e}")
        import traceback
        traceback.print_exc()


def test_predict_regression():
    """Test PREDICT VALUE regression statement """
    print("\n\nTest 3: PREDICT VALUE regression")
    
    query = """
    PREDICT VALUE(house.price, REG)
    FROM house
    WHERE house.area > 100
    """
    
    try:
        parser = Parser(query)
        ast = parser.parse()
        
        print("\n[SUCCESS] Parsed successfully")
        print(f"Statement type: {ast.statement_type}")
        
        predict = ast.predict
        print(f"Target: {predict.value.target}")
        print(f"Type: {predict.value.predict_type.type_name} (regressor)")
        print(f"Table : {predict.from_table.table}")
        
    except Exception as e:
        print(f"\n[FAILED] Parse failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # test_train_syntax()
    test_validate_syntax()
    # test_predict_syntax()
    # test_predict_regression()
    

