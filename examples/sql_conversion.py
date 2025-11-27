"""TL-SQL to Standard SQL Conversion Example

This example demonstrates how TL-SQL statements are converted to standard SQL.
"""

import sys
import os

# Add project root to path to find tl_sql package 
# File structure: project_root/tl_sql/examples/sql_conversion.py
current_dir = os.path.dirname(os.path.abspath(__file__))
tl_sql_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(tl_sql_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import from tl_sql package 
from tl_sql.core.parser import Parser
from tl_sql.executor.sql_generator import SQLGenerator


def demonstrate_sql_conversion():
    """Demonstrate TL-SQL to SQL conversion """
    
    print("TL-SQL to Standard SQL Conversion ")


    # TRAIN statement 
    print("\nTRAIN Statement ")
    
    train_sql = """
    TRAIN WITH (users.*, movies.title, ratings.*)
    FROM Tables(users, movies, ratings)
    WHERE users.Age > 25 AND movies.Year > 2000 AND ratings.Rating >= 4
    """
    
    print("TL-SQL:")
    print(train_sql.strip())
    
    parser = Parser(train_sql)
    ast = parser.parse()
    generator = SQLGenerator()
    sql_list = generator.generate(ast)
    
    print("\nGenerated Standard SQL")
    for i, gen_sql in enumerate(sql_list, 1):
        print(f"{i}. Table: {gen_sql.table}")
        print(f"   SQL: {gen_sql.sql}")
    
    # PREDICT statement 
    print("\nPREDICT Statement ")
    
    predict_sql = """
    PREDICT VALUE(users.Age, CLF)
    FROM users
    WHERE users.Gender = 'M' AND users.Occupation IN (1, 2,4)
    """
    
    print("TL-SQL:")
    print(predict_sql.strip())
    
    try:
        parser = Parser(predict_sql)
        ast = parser.parse()
        generator = SQLGenerator()
        filter_cond = generator.generate(ast)
        
        print("Generated Filter Condition ")
        print(f"Table: {filter_cond.table}")
        print(f"SQL Condition: {filter_cond.condition}")
        print(f"SELECT * FROM {filter_cond.table} WHERE {filter_cond.condition}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    # VALIDATE statement 
    print("\nVALIDATE Statement ")
    
    validate_sql = """
    VALIDATE WITH (users.*, movies.*)
    FROM Tables(users, movies)
    WHERE users.Age BETWEEN 30 AND 50
    """
    
    print("TL-SQL:")
    print(validate_sql.strip())
    
    try:
        parser = Parser(validate_sql)
        ast = parser.parse()
        generator = SQLGenerator()
        sql_list = generator.generate(ast)
        
        print("\nGenerated Standard SQL")
        for i, gen_sql in enumerate(sql_list, 1):
            print(f"{i}. Table: {gen_sql.table}")
            print(f"   SQL: {gen_sql.sql}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    


if __name__ == "__main__":
    demonstrate_sql_conversion()





