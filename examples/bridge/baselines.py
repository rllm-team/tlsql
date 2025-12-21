"""Baseline Methods for Comparison

This module provides baseline methods to compare with BRIDGE model performance.
"""

import time

import torch
import torch.nn as nn


class MLPBaseline(nn.Module):
    """MLP model"""
    def __init__(self, input_dim, num_classes):
        super(MLPBaseline, self).__init__()
        # Single linear layer 
        self.linear = nn.Linear(input_dim, num_classes)
        # Initialize weights properly
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize weights using Xavier uniform and zeros for bias"""
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
    
    def forward(self, x):
        return self.linear(x)


class RandomBaseline:
    """Random guessing baseline """
    def __init__(self, num_classes):
        self.num_classes = num_classes
    
    def __call__(self, y, mask):
        """Random prediction """
        n = int(mask.sum())
        preds = torch.randint(0, self.num_classes, (n,), device=y.device)
        return preds
    
    def eval(self):
        pass
    
    def train(self):
        pass


def extract_features_for_mlp(target_table, non_table_embeddings, device):
    """Extract features from TableData for MLP model"""
    if not target_table.if_materialized():
        target_table.lazy_materialize()
    
    feature_list = []
    for col_type, feat_tensor in target_table.feat_dict.items():
        # Handle tuple (for TransTab format)
        if isinstance(feat_tensor, tuple):
            feat_tensor = feat_tensor[0]
        
        # Ensure tensor is 2D [batch_size, features]
        if feat_tensor.dim() == 1:
            feat_tensor = feat_tensor.unsqueeze(1)
        elif feat_tensor.dim() > 2:
            feat_tensor = feat_tensor.flatten(start_dim=1)
        
        # Ensure same batch size
        if len(feature_list) > 0:
            batch_size = feature_list[0].size(0)
            if feat_tensor.size(0) != batch_size:
                continue  # Skip if batch size doesn't match
        
        feature_list.append(feat_tensor.float())
    
    if not feature_list:
        raise ValueError("No valid features extracted from target_table")
    
    table_features = torch.cat(feature_list, dim=1).to(device)
    
    # Handle non_table_embeddings
    if non_table_embeddings is not None:
        non_table_emb = non_table_embeddings.to(device).float()
        if table_features.size(0) == non_table_emb.size(0):
            table_features = torch.cat([table_features, non_table_emb], dim=1)
    
    # Normalize features: standardize to mean=0, std=1
    feature_mean = table_features.mean(dim=0, keepdim=True)
    feature_std = table_features.std(dim=0, keepdim=True) + 1e-8
    table_features = (table_features - feature_mean) / feature_std
    
    return table_features


def train_mlp(model, features, y, train_mask, val_mask, test_mask, epochs, lr, wd, device):
    """Train MLP baseline model"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.CrossEntropyLoss()
    
    # Ensure y is Long type for accuracy calculation
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
        
        # Gradient clipping to prevent exploding gradients
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


def run_mlp_baseline(target_table, non_table_embeddings, epochs=10, lr=0.005, wd=1e-4, device=None):
    """Run MLP baseline model"""
    if device is None:
        device = target_table.y.device
    
    features = extract_features_for_mlp(target_table, non_table_embeddings, device)
    
    # Check feature statistics
    print(f"MLP Features shape: {features.shape}")
    print(f"MLP Features mean: {features.mean().item():.4f}, std: {features.std().item():.4f}")
    print(f"MLP Features min: {features.min().item():.4f}, max: {features.max().item():.4f}")
    
    model = MLPBaseline(features.size(1), target_table.num_classes).to(device)
    
    return train_mlp(
        model, features, target_table.y,
        target_table.train_mask, target_table.val_mask, target_table.test_mask,
        epochs, lr, wd, device
    )


def run_random_baseline(y, train_mask, val_mask, test_mask):
    """Run random baseline"""
    y_long = y.long()
    num_classes = y_long.max().item() + 1
    random_baseline = RandomBaseline(num_classes)
    
    train_preds = random_baseline(y, train_mask)
    val_preds = random_baseline(y, val_mask)
    test_preds = random_baseline(y, test_mask)
    
    train_acc = float(train_preds.eq(y_long[train_mask]).sum()) / int(train_mask.sum())
    val_acc = float(val_preds.eq(y_long[val_mask]).sum()) / int(val_mask.sum())
    test_acc = float(test_preds.eq(y_long[test_mask]).sum()) / int(test_mask.sum())
    
    return train_acc, val_acc, test_acc
