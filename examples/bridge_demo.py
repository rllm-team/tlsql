"""BRIDGE Model Demo with TLSQL.

The demo uses the TML1M dataset with three relational tables: users, movies, and ratings.
"""

import time
import torch
import torch.nn.functional as F
import numpy as np
from tlsql.examples.bridge.data_loader import prepare_data_from_tlsql
from tlsql.examples.bridge.model import build_bridge
from tlsql.examples.bridge.baselines import run_random, run_mlp
from tlsql.examples.bridge.plot import create_training_plots


def train_bridge_model(target_table, non_table_embeddings, adj, emb_size, train_mask, val_mask, test_mask, epochs=100, lr=0.001, wd=1e-4, device=None):
    """Train BRIDGE model

    Args:
        target_table: Target table data
        non_table_embeddings: Non-table embeddings
        adj: Adjacency matrix
        emb_size: Embedding size
        train_mask: Training mask
        val_mask: Validation mask
        test_mask: Test mask
        epochs: Number of training epochs
        lr: Learning rate
        wd: Weight decay
        device: Device (CPU/GPU)

    Returns:
        tuple: (train_acc, best_val_acc, test_acc, history_dict)
        where history_dict contains: {'epochs', 'train_loss', 'val_loss', 'train_acc', 'val_acc', 'times', 'mean_time', 'total_time'}
    """
    device = device or target_table.y.device
    model = build_bridge(target_table.num_classes, target_table.metadata, emb_size).to(device)
    y = target_table.y.long()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    best_val_acc = test_acc = train_acc = 0
    times = []
    history = {
        'epochs': [],
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }

    for epoch in range(1, epochs + 1):
        start_time = time.time()

        # Train
        model.train()
        optimizer.zero_grad()
        logits = model(table=target_table, non_table=non_table_embeddings, adj=adj)
        train_loss = F.cross_entropy(logits[train_mask].squeeze(), y[train_mask])
        train_loss.backward()
        optimizer.step()

        # Test
        model.eval()
        with torch.no_grad():
            logits = model(table=target_table, non_table=non_table_embeddings, adj=adj)
            preds = logits.argmax(dim=1)
            train_acc = (preds[train_mask] == y[train_mask]).float().mean().item()
            val_acc = (preds[val_mask] == y[val_mask]).float().mean().item()
            tmp_test_acc = (preds[test_mask] == y[test_mask]).float().mean().item()

            # Calculate validation loss
            val_loss = F.cross_entropy(logits[val_mask].squeeze(), y[val_mask]).item()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = tmp_test_acc

        epoch_time = time.time() - start_time
        times.append(epoch_time)
        history['epochs'].append(epoch)
        history['train_loss'].append(train_loss.item())
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc * 100)
        history['val_acc'].append(val_acc * 100)

        print(f"BRIDGE Epoch: [{epoch}/{epochs}] Train Loss: {train_loss.item():.4f} Train Acc: {train_acc:.4f} Val Acc: {val_acc:.4f}, Test Acc: {tmp_test_acc:.4f}")

    mean_time = torch.tensor(times).mean().item()
    total_time = sum(times)
    print(f"Mean time per epoch: {mean_time:.4f}s")
    print(f"Total time: {total_time:.4f}s")
    print(f"BRIDGE Final - Train Acc: {train_acc:.4f}, Val Acc: {best_val_acc:.4f}, Test Acc: {test_acc:.4f}")

    history['times'] = times
    history['mean_time'] = mean_time
    history['total_time'] = total_time
    return train_acc, best_val_acc, test_acc, history


def main():
    """Main demo function for tml1m dataset"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    db_config = {
        'db_type': 'mysql',
        'host': 'localhost',
        'port': 3306,
        'database': 'tml1m',
        'username': 'your username',
        'password': 'your password'
    }

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
    VALIDATE WITH (users.*)
    FROM users
    WHERE users.Gender='M' and users.userID>3000
    """

    target_table, non_table_embeddings, adj, emb_size = prepare_data_from_tlsql(
        predict_query=predict_query,
        train_query=train_query,
        validate_query=validate_query,
        db_config=db_config,
        device=device
    )

    train_mask = target_table.train_mask
    val_mask = target_table.val_mask
    test_mask = target_table.test_mask

    print(f"Data loaded: {len(target_table)} samples, {target_table.num_classes} classes")
    print(f"Training samples: {train_mask.sum().item()}, Validation: {val_mask.sum().item()}, Test: {test_mask.sum().item()}")

    # Random Guess
    print("\nRunning Random Guess...")
    random_train_acc, random_val_acc, random_test_acc = run_random(target_table.y, train_mask, val_mask, test_mask)

    # MLP
    print("\nRunning MLP...")
    mlp_train_acc, mlp_val_acc, mlp_test_acc = run_mlp(target_table, train_mask, val_mask, test_mask, device=device)

    # BRIDGE Model
    print("\nRunning BRIDGE Model...")
    bridge_train_acc, bridge_val_acc, bridge_test_acc, training_history = train_bridge_model(
        target_table, non_table_embeddings, adj, emb_size, train_mask, val_mask, test_mask, epochs=100, device=device
    )

    plot_data = {
        'epochs': np.array(training_history['epochs']),
        'train_loss': np.array(training_history['train_loss']),
        'val_loss': np.array(training_history['val_loss']),
        'train_acc': np.array(training_history['train_acc']),
        'val_acc': np.array(training_history['val_acc'])
    }

    create_training_plots(plot_data, save_path='./plot.png')

    # Final comparison
    print(f"\n{'Method':<20} {'Train Acc':<15} {'Val Acc':<15} {'Test Acc':<15}")
    print(f"{'=' * 65}")
    print(f"{'Random Guess':<20} {random_train_acc:<15.4f} {random_val_acc:<15.4f} {random_test_acc:<15.4f}")
    print(f"{'MLP':<20} {mlp_train_acc:<15.4f} {mlp_val_acc:<15.4f} {mlp_test_acc:<15.4f}")
    print(f"{'BRIDGE':<20} {bridge_train_acc:<15.4f} {bridge_val_acc:<15.4f} {bridge_test_acc:<15.4f}")


if __name__ == "__main__":
    main()
