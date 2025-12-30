"""Baseline Methods for Comparison

This module provides baseline methods to compare with BRIDGE model performance.
"""

import time

import torch
import torch.nn as nn


class MLP(nn.Module):
    """MLP model"""
    def __init__(self, input_dim, num_classes):
        super(MLP, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)
        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize weights"""
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return self.linear(x)


class RandomGuess:
    """Random guessing baseline """
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, y, mask):
        """Random prediction """
        n = int(mask.sum())
        preds = torch.randint(0, self.num_classes, (n,), device=y.device)
        return preds


def extract_features(target_table, device):
    """Extract features from TableData for MLP model"""
    target_table.lazy_materialize()

    feature_list = []
    for feat_tensor in target_table.feat_dict.values():
        if isinstance(feat_tensor, tuple):
            feat_tensor = feat_tensor[0]
        if feat_tensor.dim() > 2:
            feat_tensor = feat_tensor.flatten(start_dim=1)
        feature_list.append(feat_tensor.float())

    table_features = torch.cat(feature_list, dim=1).to(device)
    feature_mean = table_features.mean(dim=0, keepdim=True)
    feature_std = table_features.std(dim=0, keepdim=True) + 1e-8
    table_features = (table_features - feature_mean) / feature_std

    return table_features


def train_mlp(model, features, y, train_mask, val_mask, test_mask, epochs, lr, wd, device):
    """Train MLP baseline model"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.CrossEntropyLoss()

    y_long = y.long()

    best_val_acc = test_acc = 0
    times = []

    for epoch in range(1, epochs + 1):
        start = time.time()
        model.train()
        optimizer.zero_grad()

        logits = model(features[train_mask])
        loss = criterion(logits, y_long[train_mask])
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits_all = model(features)
            preds = logits_all.argmax(dim=1)

            train_acc = float(preds[train_mask].eq(y_long[train_mask]).sum()) / int(train_mask.sum())
            val_acc = float(preds[val_mask].eq(y_long[val_mask]).sum()) / int(val_mask.sum())
            test_acc_curr = float(preds[test_mask].eq(y_long[test_mask]).sum()) / int(test_mask.sum())

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = test_acc_curr

        times.append(time.time() - start)
        print(
            f"MLP Epoch: [{epoch}/{epochs}] "
            f"Train Loss: {loss.item():.4f} Train Acc: {train_acc:.4f} "
            f"Val Acc: {val_acc:.4f}, Test Acc: {test_acc_curr:.4f} "
        )

    print(f"MLP Mean time per epoch: {torch.tensor(times).mean():.4f}s")
    print(f"MLP Total time: {sum(times):.4f}s")

    return best_val_acc, test_acc


def run_mlp_baseline(target_table, epochs=20, lr=0.01, wd=1e-4, device=None):
    """Run MLP baseline model"""
    device = target_table.y.device if device is None else device

    features = extract_features(target_table, device)
    model = MLP(features.size(1), target_table.num_classes).to(device)

    return train_mlp(
        model, features, target_table.y,
        target_table.train_mask, target_table.val_mask, target_table.test_mask,
        epochs, lr, wd, device
    )


def run_random_baseline(y, train_mask, val_mask, test_mask):
    """Run random baseline"""
    y_long = y.long()
    num_classes = y_long.max().item() + 1
    random_baseline = RandomGuess(num_classes)

    train_preds = random_baseline(y, train_mask)
    val_preds = random_baseline(y, val_mask)
    test_preds = random_baseline(y, test_mask)

    train_acc = float(train_preds.eq(y_long[train_mask]).sum()) / int(train_mask.sum())
    val_acc = float(val_preds.eq(y_long[val_mask]).sum()) / int(val_mask.sum())
    test_acc = float(test_preds.eq(y_long[test_mask]).sum()) / int(test_mask.sum())

    return train_acc, val_acc, test_acc
