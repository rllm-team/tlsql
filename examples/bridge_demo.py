"""BRIDGE Model Demo with TLSQL.

This work is conducted on the TML1M dataset from SJTUTables, a dataset of users, movies, and ratings.
"""

import time

import torch
import torch.nn.functional as F

from tlsql.examples.bridge.data_loader import prepare_data_from_tlsql
from tlsql.examples.bridge.model import build_bridge_model
from tlsql.examples.bridge.baselines import run_random_baseline, run_mlp_baseline


def train_bridge(model, optimizer, target_table, non_table_embeddings, adj, y, train_mask):
    """Train BRIDGE model for one epoch"""
    model.train()
    optimizer.zero_grad()
    logits = model(table=target_table, non_table=non_table_embeddings, adj=adj)
    loss = F.cross_entropy(logits[train_mask].squeeze(), y[train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def test_bridge(model, target_table, non_table_embeddings, adj, y, masks):
    """Test BRIDGE model"""
    model.eval()
    logits = model(table=target_table, non_table=non_table_embeddings, adj=adj)
    preds = logits.argmax(dim=1)
    accs = []
    for mask in masks:
        correct = float(preds[mask].eq(y[mask]).sum().item())
        accs.append(correct / int(mask.sum()))
    return accs


def train_bridge_model(model, target_table, non_table_embeddings, adj, epochs=10, lr=0.001, wd=1e-4):
    """Train BRIDGE model

    Args:
        model: BRIDGE model instance
        target_table: Target table data
        non_table_embeddings: Non-table embeddings
        adj: Adjacency matrix
        epochs: Number of training epochs
        lr: Learning rate
        wd: Weight decay

    Returns:
        tuple: (model, best_val_acc, test_acc)
    """
    y = target_table.y.long() if target_table.y.dtype != torch.long else target_table.y
    train_mask, val_mask, test_mask = (
        target_table.train_mask,
        target_table.val_mask,
        target_table.test_mask,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    metric = "Acc"
    best_val_acc = test_acc = 0
    times = []
    
    for epoch in range(1, epochs + 1):
        start = time.time()
        train_loss = train_bridge(model, optimizer, target_table, non_table_embeddings, adj, y, train_mask)
        train_acc, val_acc, tmp_test_acc = test_bridge(
            model, target_table, non_table_embeddings, adj, y,
            [train_mask, val_mask, test_mask]
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = tmp_test_acc

        times.append(time.time() - start)
        print(
            f"BRIDGE Epoch: [{epoch}/{epochs}] "
            f"Train Loss: {train_loss:.4f} Train {metric}: {train_acc:.4f} "
            f"Val {metric}: {val_acc:.4f}, Test {metric}: {tmp_test_acc:.4f} "
        )

    print(f"BRIDGE Mean time per epoch: {torch.tensor(times).mean():.4f}s")
    print(f"BRIDGE Total time: {sum(times):.4f}s")
    print(f"Test {metric} at best Val: {test_acc:.4f}")
    return model, best_val_acc, test_acc


def main():
    """Main demo function for tml1m dataset"""
    print("This work is conducted on the TML1M dataset from SJTUTables, a dataset of users, movies, and ratings.")
    print()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    db_config = {
        'db_type': 'mysql',
        'host': 'localhost',
        'port': 3306,
        'database': 'tml1m',
        'username': 'root',
        'password': 'cfy1007'
    }
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

    predict_tlsql = """
    PREDICT VALUE(users.Age, CLF)
    FROM users
    WHERE users.Gender='F'
    """

    target_table, non_table_embeddings, adj, emb_size = prepare_data_from_tlsql(
        train_tlsql=train_tlsql,
        validate_tlsql=validate_tlsql,
        predict_tlsql=predict_tlsql,
        db_config=db_config,
        device=device
    )

    print(f"Data loaded: {len(target_table)} samples, {target_table.num_classes} classes")
    print(f"Training samples: {target_table.train_mask.sum().item()}")
    print(f"Validation samples: {target_table.val_mask.sum().item()}")
    print(f"Test samples: {target_table.test_mask.sum().item()}")

    # Random Guess
    print("\nRunning Random Guess Baseline...")
    train_mask, val_mask, test_mask = (
        target_table.train_mask,
        target_table.val_mask,
        target_table.test_mask,
    )
    random_train_acc, random_val_acc, random_test_acc = run_random_baseline(
        target_table.y, train_mask, val_mask, test_mask
    )
    print(f"Random Guess - Train Acc: {random_train_acc:.4f}, Val Acc: {random_val_acc:.4f}, Test Acc: {random_test_acc:.4f}")
    
    # MLP
    print("\nRunning MLP Baseline...")
    mlp_val_acc, mlp_test_acc = run_mlp_baseline(
        target_table, non_table_embeddings,
        device=device
    )
    
    # BRIDGE Model
    print("\nRunning BRIDGE Model...")
    bridge_model = build_bridge_model(
        target_table.num_classes,
        target_table.metadata,
        emb_size
    ).to(device)

    bridge_model, bridge_val_acc, bridge_test_acc = train_bridge_model(
        bridge_model, target_table, non_table_embeddings, adj
    )

    print(f"{'Method':<20} {'Val Acc':<15} {'Test Acc':<15}")
    print(f"{'Random Guess':<20} {random_val_acc:<15.4f} {random_test_acc:<15.4f}")
    print(f"{'MLP':<20} {mlp_val_acc:<15.4f} {mlp_test_acc:<15.4f}")
    print(f"{'BRIDGE':<20} {bridge_val_acc:<15.4f} {bridge_test_acc:<15.4f}")
    print("="*60)


if __name__ == "__main__":
    main()
