"""
ChemBERTa Fine-tuning for ADMET Prediction - FIXED VERSION
Matches preprocessing from optimized_gnn.py:
- Uses TDC 2-way split then manual 90/10 train/val split
- Uses clip_min=1e-3 for non-Caco2 datasets
- Uses y_all (train+val) for computing mu/sigma
"""

import os
import sys
import json
import warnings
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_auc_score, r2_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# ============== CONFIGURATION ==============

DATASETS = {
    'Caco2_Wang': 'regression',
    'Half_Life_Obach': 'regression',
    'Clearance_Hepatocyte_AZ': 'regression',
    'Clearance_Microsome_AZ': 'regression',
    'tox21': 'classification',
    'herg': 'classification',
}

SEED = 42
MAX_EPOCHS = 30
PATIENCE = 10
BATCH_SIZE = 32
UNFREEZE_LAYERS = 2  # Unfreeze last 2 transformer layers
LR_ENCODER = 1e-5    # Lower LR for pretrained layers
LR_HEAD = 1e-3       # Normal LR for prediction head
OUTPUT_DIR = os.path.join(project_root, 'results', 'chemberta_finetune')


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_data(dataset_name, task_type, seed=42):
    """
    Prepare data with preprocessing matching optimized_gnn.py:
    - TDC 2-way split (train/test) then manual 90/10 train/val
    - clip_min=1e-3 for non-Caco2 datasets
    - y_all (train+val) for computing mu/sigma
    """
    # CANONICAL split (scripts/canonical_split.py): identical scaffold test set and
    # molecule-level validation carve as the GNN benchmark -> fine-tuning is evaluated
    # on exactly the same partitions as every other model family.
    from scripts.canonical_split import canonical_split
    cs = canonical_split(dataset_name, seed=seed)
    train_smiles = [s for s, _ in cs['train']]
    train_labels = [y for _, y in cs['train']]
    val_smiles = [s for s, _ in cs['val']]
    val_labels = [y for _, y in cs['val']]
    test_smiles = [s for s, _ in cs['test']]
    test_labels = [y for _, y in cs['test']]

    # Regression target transform; normalization stats fit on TRAIN labels only.
    mu, sigma = 0.0, 1.0
    if task_type == 'regression':
        y_train = np.array(train_labels)
        is_caco2 = dataset_name.lower() == 'caco2_wang'
        if is_caco2:
            # Caco2_Wang: values already in log space (all negative)
            y_log = y_train.astype(np.float32)
            mu = float(y_log.mean()); sigma = float(y_log.std())
            if sigma < 1e-6:
                sigma = 1.0
            train_labels = [(y - mu) / sigma for y in train_labels]
            val_labels = [(y - mu) / sigma for y in val_labels]
            test_labels = [(y - mu) / sigma for y in test_labels]
        else:
            clip_min = 1e-3
            y_log = np.log(np.clip(y_train, clip_min, None))
            mu = float(y_log.mean()); sigma = float(y_log.std())
            if sigma < 1e-6:
                sigma = 1.0
            train_labels = [(np.log(max(clip_min, y)) - mu) / sigma for y in train_labels]
            val_labels = [(np.log(max(clip_min, y)) - mu) / sigma for y in val_labels]
            test_labels = [(np.log(max(clip_min, y)) - mu) / sigma for y in test_labels]
        print(f"  Normalization (train-only): mu={mu:.4f}, sigma={sigma:.4f}")

    return {
        'train_smiles': train_smiles,
        'train_labels': train_labels,
        'val_smiles': val_smiles,
        'val_labels': val_labels,
        'test_smiles': test_smiles,
        'test_labels': test_labels,
        'mu': mu,
        'sigma': sigma,
        'split_hash': cs['hashes'],
    }


class ChemBERTaDataset(Dataset):
    """Dataset for ChemBERTa with SMILES tokenization"""

    def __init__(self, smiles_list, labels, tokenizer, max_length=128):
        self.smiles = smiles_list
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.smiles[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': torch.tensor(self.labels[idx], dtype=torch.float32)
        }


class ChemBERTaFineTuned(nn.Module):
    """ChemBERTa with fine-tuning support"""

    def __init__(self, model_name="seyonec/ChemBERTa-zinc-base-v1",
                 unfreeze_layers=2, dropout=0.2, task_type='regression'):
        super().__init__()

        print(f"  Loading ChemBERTa from {model_name}...")
        self.encoder = AutoModel.from_pretrained(model_name)
        self.task_type = task_type
        self.hidden_size = self.encoder.config.hidden_size

        # Freeze all layers first
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Unfreeze last N transformer layers
        total_layers = len(self.encoder.encoder.layer)
        for i in range(total_layers - unfreeze_layers, total_layers):
            for param in self.encoder.encoder.layer[i].parameters():
                param.requires_grad = True

        # Also unfreeze pooler if present
        if hasattr(self.encoder, 'pooler') and self.encoder.pooler is not None:
            for param in self.encoder.pooler.parameters():
                param.requires_grad = True

        # Prediction head
        self.head = nn.Sequential(
            nn.Linear(self.hidden_size, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

        # Count parameters
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"  Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        # Use CLS token representation
        pooled = outputs.last_hidden_state[:, 0, :]

        logits = self.head(pooled)
        return logits  # Raw logits - sigmoid applied in loss or evaluation


def train_epoch(model, loader, optimizer, criterion, device, scheduler=None):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    n_samples = 0

    for batch in loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs.squeeze(), labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if scheduler:
            scheduler.step()

        total_loss += loss.item() * len(labels)
        n_samples += len(labels)

    return total_loss / n_samples


def evaluate(model, loader, device, task_type, mu=0.0, sigma=1.0, return_all_metrics=False):
    """Evaluate model with proper inverse transform for regression"""
    model.eval()
    preds, labels = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(input_ids, attention_mask)
            preds.extend(outputs.cpu().numpy().flatten())
            labels.extend(batch['label'].numpy().flatten())

    preds = np.array(preds)
    labels = np.array(labels)

    if task_type == 'classification':
        # Apply sigmoid to get probabilities (logits -> probabilities)
        preds = 1 / (1 + np.exp(-preds))
        auc = roc_auc_score(labels, preds)
        if return_all_metrics:
            return {'auc': auc}
        return auc
    else:
        # Denormalize to log scale
        preds_log = preds * sigma + mu
        labels_log = labels * sigma + mu

        # Log-scale metrics (before exp transform)
        rmse_log = np.sqrt(mean_squared_error(labels_log, preds_log))
        mae_log = mean_absolute_error(labels_log, preds_log)

        # Original-scale metrics (after exp transform)
        preds_orig = np.exp(preds_log)
        labels_orig = np.exp(labels_log)
        rmse_orig = np.sqrt(mean_squared_error(labels_orig, preds_orig))
        # R2 on both scales; orig-scale matches optimized_gnn.evaluate (the GNN convention).
        try:
            r2_orig = float(r2_score(labels_orig, preds_orig))
        except Exception:
            r2_orig = float('nan')
        r2_log = float(r2_score(labels_log, preds_log))

        if return_all_metrics:
            return {
                'rmse_orig': rmse_orig,
                'rmse_log': rmse_log,
                'mae_log': mae_log,
                'r2_orig': r2_orig,
                'r2_log': r2_log,
            }
        # For validation/early stopping, use log-scale RMSE (more stable)
        return rmse_log


def run_finetune_for_dataset(dataset_name, task_type, tokenizer, device):
    """Run fine-tuning for a single dataset"""

    # Load and prepare data with correct preprocessing
    print(f"  Loading {dataset_name}...")
    data = prepare_data(dataset_name, task_type, seed=SEED)

    mu = data['mu']
    sigma = data['sigma']

    print(f"  Train: {len(data['train_smiles'])}, Val: {len(data['val_smiles'])}, Test: {len(data['test_smiles'])}")

    # Create datasets
    train_dataset = ChemBERTaDataset(data['train_smiles'], data['train_labels'], tokenizer)
    val_dataset = ChemBERTaDataset(data['val_smiles'], data['val_labels'], tokenizer)
    test_dataset = ChemBERTaDataset(data['test_smiles'], data['test_labels'], tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE * 2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE * 2)

    # Create model
    model = ChemBERTaFineTuned(
        unfreeze_layers=UNFREEZE_LAYERS,
        dropout=0.2,
        task_type=task_type
    ).to(device)

    # Separate parameter groups
    encoder_params = [p for n, p in model.named_parameters() if 'encoder' in n and p.requires_grad]
    head_params = [p for n, p in model.named_parameters() if 'head' in n]

    optimizer = torch.optim.AdamW([
        {'params': encoder_params, 'lr': LR_ENCODER, 'weight_decay': 0.01},
        {'params': head_params, 'lr': LR_HEAD, 'weight_decay': 0.01}
    ])

    # Scheduler
    total_steps = len(train_loader) * MAX_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    # Loss - with class weighting for imbalanced classification
    if task_type == 'classification':
        # Calculate pos_weight for imbalanced datasets (e.g., Tox21 has ~3.5% positive)
        n_pos = sum(1 for y in data['train_labels'] if y > 0.5)
        n_neg = len(data['train_labels']) - n_pos
        if n_pos > 0:
            pos_weight = torch.tensor([n_neg / n_pos], device=device)
            print(f"  Class balance: {n_pos} pos / {n_neg} neg, pos_weight={pos_weight.item():.2f}")
        else:
            pos_weight = torch.tensor([1.0], device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.MSELoss()

    # Training
    best_val = float('inf') if task_type == 'regression' else 0
    patience_counter = 0
    best_model_state = None
    history = {'train_loss': [], 'val_metric': []}

    print("  Training...")
    for epoch in range(MAX_EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, scheduler)
        val_metric = evaluate(model, val_loader, device, task_type, mu, sigma)

        history['train_loss'].append(train_loss)
        history['val_metric'].append(val_metric)

        metric_name = 'AUC' if task_type == 'classification' else 'RMSE'
        print(f"    Epoch {epoch+1}: Loss={train_loss:.4f}, Val {metric_name}={val_metric:.4f}")

        # Early stopping
        if task_type == 'regression':
            improved = val_metric < best_val
        else:
            improved = val_metric > best_val

        if improved:
            best_val = val_metric
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"    Early stopping at epoch {epoch+1}")
                break

    # Load best model and evaluate on test
    if best_model_state:
        model.load_state_dict(best_model_state)

    test_metrics = evaluate(model, test_loader, device, task_type, mu, sigma, return_all_metrics=True)

    if task_type == 'classification':
        print(f"  Test AUC: {test_metrics['auc']:.4f}")
    else:
        print(f"  Test RMSE (orig): {test_metrics['rmse_orig']:.4f}")
        print(f"  Test RMSE (log):  {test_metrics['rmse_log']:.4f}")
        print(f"  Test MAE (log):   {test_metrics['mae_log']:.4f}")

    n_trainable = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    n_total = int(sum(p.numel() for p in model.parameters()))
    result = {
        'dataset': dataset_name,
        'task_type': task_type,
        'best_val_metric': best_val,
        'epochs_trained': len(history['train_loss']),
        'history': history,
        'finetune': {
            'protocol': 'partial task-specific fine-tuning',
            'unfreeze_layers': UNFREEZE_LAYERS,
            'lr_encoder': LR_ENCODER,
            'lr_head': LR_HEAD,
            'max_epochs': MAX_EPOCHS,
            'trainable_params': n_trainable,
            'total_params': n_total,
            'trainable_frac': n_trainable / n_total,
        },
        'preprocessing': {
            'split_method': 'canonical (scripts/canonical_split.py): scaffold test + molecule-level val carve; identical partition to the GNN benchmark',
            'split_hash': data['split_hash'],
            'norm_fit_on': 'train-only',
            'clip_min': 1e-3 if dataset_name.lower() != 'caco2_wang' else None,
            'mu': mu,
            'sigma': sigma
        }
    }

    # Add test metrics
    if task_type == 'classification':
        result['test_auc'] = test_metrics['auc']
        result['test_metric'] = test_metrics['auc']
    else:
        result['test_rmse_orig'] = test_metrics['rmse_orig']
        result['test_rmse_log'] = test_metrics['rmse_log']
        result['test_mae_log'] = test_metrics['mae_log']
        result['test_r2_orig'] = test_metrics['r2_orig']
        result['test_r2_log'] = test_metrics['r2_log']
        result['test_metric'] = test_metrics['rmse_log']

    return result


def run_chemberta_finetune_benchmark(datasets=None):
    """Run ChemBERTa fine-tuning on all (or the given) datasets"""

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ds_items = ([(d, DATASETS[d]) for d in datasets] if datasets
                else list(DATASETS.items()))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n{'='*70}")
    print("CHEMBERTA FINE-TUNING BENCHMARK (FIXED PREPROCESSING)")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Datasets: {len(DATASETS)}")
    print(f"Unfreeze layers: {UNFREEZE_LAYERS}")
    print(f"LR (encoder): {LR_ENCODER}, LR (head): {LR_HEAD}")
    print(f"\nProtocol:")
    print(f"  - Split: CANONICAL (scripts/canonical_split.py) -- identical scaffold test")
    print(f"           and molecule-level val carve as the GNN benchmark")
    print(f"  - Clip min: 1e-3 for non-Caco2")
    print(f"  - Normalization: fit on TRAIN labels only")
    print(f"{'='*70}\n")

    set_seed(SEED)

    # Load tokenizer once
    tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")

    all_results = []

    for dataset_name, task_type in ds_items:
        print(f"\n{'='*60}")
        print(f"DATASET: {dataset_name} ({task_type})")
        print(f"{'='*60}")

        try:
            result = run_finetune_for_dataset(dataset_name, task_type, tokenizer, device)
            all_results.append(result)

            # Save individual result
            result_to_save = {k: v for k, v in result.items() if k != 'history'}
            with open(f"{OUTPUT_DIR}/chemberta_ft_{dataset_name}_results_fixed.json", 'w') as f:
                json.dump(result_to_save, f, indent=2)

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

        # Clear cache
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Save summary with all metrics
    summary_rows = []
    for r in all_results:
        row = {
            'dataset': r['dataset'],
            'model': 'ChemBERTa-FT',
            'task_type': r['task_type'],
            'best_val_metric': r['best_val_metric'],
            'epochs': r['epochs_trained']
        }
        if r['task_type'] == 'classification':
            row['test_auc'] = r['test_auc']
            row['test_rmse_orig'] = None
            row['test_rmse_log'] = None
            row['test_mae_log'] = None
        else:
            row['test_auc'] = None
            row['test_rmse_orig'] = r['test_rmse_orig']
            row['test_rmse_log'] = r['test_rmse_log']
            row['test_mae_log'] = r['test_mae_log']
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(f"{OUTPUT_DIR}/chemberta_finetune_summary_fixed.csv", index=False)

    # Print summary
    print(f"\n{'='*70}")
    print("CHEMBERTA FINE-TUNING SUMMARY (FIXED)")
    print(f"{'='*70}")
    print(f"\n{'Dataset':<25} {'Task':<12} {'RMSE_orig':<12} {'RMSE_log':<12} {'MAE_log':<12} {'AUC':<8}")
    print("-"*85)

    for r in all_results:
        if r['task_type'] == 'classification':
            print(f"{r['dataset']:<25} {r['task_type']:<12} {'-':<12} {'-':<12} {'-':<12} {r['test_auc']:.4f}")
        else:
            print(f"{r['dataset']:<25} {r['task_type']:<12} {r['test_rmse_orig']:<12.4f} {r['test_rmse_log']:<12.4f} {r['test_mae_log']:<12.4f} {'-':<8}")

    print(f"\n{'='*70}")
    print(f"Results saved to: {OUTPUT_DIR}")
    print(f"{'='*70}")

    return all_results


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=None)
    a = ap.parse_args()
    run_chemberta_finetune_benchmark(datasets=a.datasets)
