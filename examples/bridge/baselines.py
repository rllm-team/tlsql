"""Methods for Comparison"""

import torch
import torch.nn as nn


class MLP(nn.Module):
    """MLP model"""
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return self.linear(x)


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
    mean, std = table_features.mean(dim=0, keepdim=True), table_features.std(dim=0, keepdim=True) + 1e-8
    return (table_features - mean) / std


def run_mlp(target_table, train_mask, val_mask, test_mask, epochs=20, lr=0.01, wd=1e-4, device=None):
    """Run MLP model"""
    device = device or target_table.y.device
    features = extract_features(target_table, device)
    model = MLP(features.size(1), target_table.num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.CrossEntropyLoss()
    y = target_table.y.long()

    best_val_acc = test_acc = train_acc = 0

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(features[train_mask])
        loss = criterion(logits, y[train_mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits_all = model(features)
            preds = logits_all.argmax(dim=1)
            train_acc = (preds[train_mask] == y[train_mask]).float().mean().item()
            val_acc = (preds[val_mask] == y[val_mask]).float().mean().item()
            tmp_test_acc = (preds[test_mask] == y[test_mask]).float().mean().item()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = tmp_test_acc

        print(f"MLP Epoch: [{epoch}/{epochs}] Train Loss: {loss.item():.4f} Train Acc: {train_acc:.4f} Val Acc: {val_acc:.4f}, Test Acc: {tmp_test_acc:.4f}")

    print(f"MLP Final - Train Acc: {train_acc:.4f}, Val Acc: {best_val_acc:.4f}, Test Acc: {test_acc:.4f}")
    return train_acc, best_val_acc, test_acc


def run_random(y, train_mask, val_mask, test_mask):
    """Run random method"""
    y = y.long()
    num_classes = y.max().item() + 1

    train_preds = torch.randint(0, num_classes, (int(train_mask.sum()),), device=y.device)
    val_preds = torch.randint(0, num_classes, (int(val_mask.sum()),), device=y.device)
    test_preds = torch.randint(0, num_classes, (int(test_mask.sum()),), device=y.device)

    train_acc = (train_preds == y[train_mask]).float().mean().item()
    val_acc = (val_preds == y[val_mask]).float().mean().item()
    test_acc = (test_preds == y[test_mask]).float().mean().item()

    print(f"Random Guess - Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}")
    return train_acc, val_acc, test_acc
