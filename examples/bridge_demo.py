"""BRIDGE Model Demo with TL-SQL 

This example demonstrates how to use TL-SQL to train a BRIDGE model.


Note: This example requires the rllm package to be installed.

Usage :
    python -m tl_sql.examples.bridge_demo --epochs 100 --lr 0.001
"""

import argparse
import numpy as np
import torch
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
tl_sql_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(tl_sql_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from tl_sql.examples.bridge.train_with_tl_sql import (
    prepare_data_from_tl_sql,
    build_bridge_model,
    train_bridge_model,
    train_bridge_model_with_kfold,
    RandomBaseline,
    evaluate_random_baseline
)


def main():
    """Main demo function """
    parser = argparse.ArgumentParser(
        description="BRIDGE Model Demo with TL-SQL "
    )
    parser.add_argument("--epochs", type=int, default=10, 
                       help="Training epochs")
    parser.add_argument("--lr", type=float, default=0.001, 
                       help="Learning rate")
    parser.add_argument("--wd", type=float, default=1e-4, 
                       help="Weight decay")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    db_config = {
        'db_type': 'mysql',
        'host': 'localhost',
        'port': 3306,
        'database': 'tacm12k', 
        'username': 'root',   
        'password': 'cfy1007'  
    }
    

    # PREDICT
    predict_sql = """
    PREDICT VALUE(papers.conference, CLF)
    FROM papers
    WHERE  
    year =2004
    """
    
    # predict_sql = """
    # PREDICT VALUE(Age, CLF)
    # FROM users
    # WHERE users.Gender='F' OR users.userID IN (1,2,3,4,5,6,7)
    # """
    
    # TRAIN 
    # train_sql = """
    # TRAIN WITH (users.*, movies.*, ratings.*)
    # FROM Tables(users, movies, ratings)
    # WHERE users.Gender='M' AND movies.Year >= 2000 AND ratings.rating > 4
    # """
    
    train_sql = None
    
    # VALIDATE
    # validate_sql = """
    # VALIDATE WITH (users.*, movies.*, ratings.userID)
    # FROM Tables(users, movies, ratings)
    # WHERE users.Gender='M' AND movies.Year < 2000 AND ratings.rating < 4
    # """
    validate_sql = None
    
    # You can also set validate_sql = None to use k-fold cross-validation
    # validate_sql = None
    
    target_table, non_table_embeddings, adj, emb_size, use_kfold = prepare_data_from_tl_sql(
        train_sql=train_sql,
        validate_sql=validate_sql,
        predict_sql=predict_sql,
        db_config=db_config,
        device=device
    )
        
    print(f"Data loaded successfully")
    print(f"Target table size: {len(target_table)}")
    print(f"Number of classes: {target_table.num_classes}")
    print(f"Use k-fold: {use_kfold}")
    
    results = {}
    
    print("\nBRIDGE Model")
    bridge_model = build_bridge_model(
        target_table.num_classes, 
        target_table.metadata, 
        emb_size
    ).to(device)
    
    if use_kfold:
        print("Training with k-fold cross-validation ")
        fold_results = train_bridge_model_with_kfold(
            bridge_model, target_table, non_table_embeddings, adj, 
            args.epochs, args.lr, args.wd, k_folds=5
        )
        # Extract average accuracy from k-fold results 
        val_accs = [r['best_val_acc'] for r in fold_results]
        test_accs = [r['test_acc'] for r in fold_results]
        val_acc_bridge = np.mean(val_accs)
        test_acc_bridge = np.mean(test_accs)
        print(f"K-fold Results:")
        print(f"Validation Accuracy - Mean: {val_acc_bridge:.4f}, Std: {np.std(val_accs):.4f}")
        print(f"Test Accuracy - Mean: {test_acc_bridge:.4f}, Std: {np.std(test_accs):.4f}")
    else:
        _, val_acc_bridge, test_acc_bridge = train_bridge_model(
            bridge_model, target_table, non_table_embeddings, adj, 
            args.epochs, args.lr, args.wd
        )
        print(f"Validation Accuracy: {val_acc_bridge:.4f}")
        print(f"Test Accuracy: {test_acc_bridge:.4f}")
    
    results['BRIDGE'] = {'val_acc': val_acc_bridge, 'test_acc': test_acc_bridge}



if __name__ == "__main__":
    main()





