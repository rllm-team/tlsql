"""BRIDGE Model Training Utilities

This module provides utilities for training BRIDGE models.

Note: This module requires the rllm package to be available.
"""

import time

import torch
import torch.nn.functional as F


def train(model, optimizer, target_table, non_table_embeddings, adj, y, train_mask):
    """Train for one epoch"""
    model.train()
    optimizer.zero_grad()
    logits = model(table=target_table, non_table=non_table_embeddings, adj=adj)

    if logits.dim() > 2:
        logits = logits.view(logits.size(0), -1)
    elif logits.dim() == 1:
        raise ValueError(f"Logits dimension error: {logits.shape}")

    loss = F.cross_entropy(logits[train_mask], y[train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def test(model, target_table, non_table_embeddings, adj, y, masks):
    """Test"""
    model.eval()
    logits = model(table=target_table, non_table=non_table_embeddings, adj=adj)
    preds = logits.argmax(dim=1)
    return [(preds[mask] == y[mask]).float().mean().item() for mask in masks]


def train_bridge_model(model, target_table, non_table_embeddings, adj, epochs, lr, wd):
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
    y = target_table.y
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
        train_loss = train(model, optimizer, target_table, non_table_embeddings, adj, y, train_mask)
        train_acc, val_acc, tmp_test_acc = test(
            model, target_table, non_table_embeddings, adj, y,
            [train_mask, val_mask, test_mask]
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = tmp_test_acc

        times.append(time.time() - start)
        print(
            f"Epoch: [{epoch}/{epochs}] "
            f"Train Loss: {train_loss:.4f} Train {metric}: {train_acc:.4f} "
            f"Val {metric}: {val_acc:.4f}, Test {metric}: {tmp_test_acc:.4f} "
        )

    print(f"Mean time per epoch: {torch.tensor(times).mean():.4f}s")
    print(f"Total time: {sum(times):.4f}s")
    return model, best_val_acc, test_acc
