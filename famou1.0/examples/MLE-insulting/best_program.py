import os
import numpy as np
import pandas as pd
import warnings
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import gc
import copy
import random
import re

# Suppress all warnings
warnings.filterwarnings("ignore")

# EVOLVE-BLOCK-START
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoConfig,
    get_linear_schedule_with_warmup,
)


# Set random seeds for reproducibility
def set_seed(seed_value=42):
    """Sets random seeds for numpy, torch, and random for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


set_seed(42)


# --- 1. Data Preprocessing ---
def clean_text(text: str) -> str:
    """Cleans the input text for better processing."""
    if not isinstance(text, str):
        return ""
    text = text.strip('"')
    text = text.replace(r"\xa0", " ").replace(r"\xc2\xa0", " ").replace("\\n", " ")
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


# --- 2. Custom Dataset for PyTorch ---
class InsultDataset(Dataset):
    """A PyTorch Dataset for handling text data with a transformer tokenizer."""

    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx] if self.labels is not None else -1

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.float),
        }


# --- 3. Adversarial Weight Perturbation (AWP) ---
class AWP:
    """
    Implements Adversarial Weight Perturbation for regularization.
    Reference: https://www.kaggle.com/code/wht1996/feedback-awp-fgm-finetuning-with-spell-check
    """

    def __init__(self, model, optimizer, adv_lr=1e-4, adv_eps=1e-2):
        self.model = model
        self.optimizer = optimizer
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.backup = {}

    def attack(self):
        e = 1e-6
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.backup[name] = param.data.clone()
                norm_grad = torch.norm(param.grad)
                if norm_grad != 0 and not torch.isnan(norm_grad):
                    r_at = self.adv_lr * param.grad / (norm_grad + e)
                    param.data.add_(r_at)

    def restore(self):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}


# --- 4. Custom Model Architecture ---
class MeanMaxPooling(nn.Module):
    """Mean and Max pooling layer for sequence data."""

    def __init__(self):
        super(MeanMaxPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        )
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask

        max_embeddings = last_hidden_state.clone()
        max_embeddings[input_mask_expanded == 0] = -1e9
        max_embeddings = torch.max(max_embeddings, 1)[0]

        return torch.cat((mean_embeddings, max_embeddings), 1)


class CustomDeBERTa(nn.Module):
    """Custom DeBERTa model with a Mean-Max pooling head."""

    def __init__(self, model_name, num_labels=1, dropout=0.1):
        super(CustomDeBERTa, self).__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.deberta = AutoModel.from_pretrained(model_name)
        self.pooler = MeanMaxPooling()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.config.hidden_size * 2, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        pooled_output = self.pooler(last_hidden_state, attention_mask)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


# --- 5. Training and Prediction Functions ---
def train_epoch(
    model, data_loader, optimizer, device, scheduler, criterion, awp_manager
):
    """Trains the model for one epoch using AWP."""
    model.train()
    for d in data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        labels = d["labels"].to(device)

        # Normal forward and backward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(outputs.squeeze(1), labels)
        loss.backward()

        # AWP attack and adversarial backward pass
        awp_manager.attack()
        optimizer.zero_grad()
        outputs_adv = model(input_ids=input_ids, attention_mask=attention_mask)
        loss_adv = criterion(outputs_adv.squeeze(1), labels)
        loss_adv.backward()
        awp_manager.restore()

        # Gradient clipping and optimizer step
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    return


def predict(model, data_loader, device):
    """Generates predictions for a given dataset."""
    model.eval()
    all_preds = []
    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            all_preds.extend(torch.sigmoid(outputs).cpu().numpy())
    return np.array(all_preds).flatten()


def main(train_df, test_df, save_path):
    print(
        "Starting insult detection with DeBERTa-v3-base, AWP, and Mean-Max Pooling..."
    )

    # --- Configuration ---
    MODEL_NAME = "microsoft/deberta-v3-base"
    MAX_SEQ_LEN = 160
    BATCH_SIZE = 16
    N_EPOCHS = 4
    N_FOLDS = 5
    LR_BACKBONE = 2e-5
    LR_HEAD = 1e-4

    # --- Preprocessing ---
    submission_template_df = test_df.copy()
    print("Cleaning text data...")
    train_df["Comment"] = train_df["Comment"].apply(clean_text)
    test_df["Comment"] = test_df["Comment"].apply(clean_text)

    train_df = train_df[train_df["Comment"].str.strip() != ""].reset_index(drop=True)
    test_df_cleaned = test_df[test_df["Comment"].str.strip() != ""].copy()

    print("Initializing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # --- Test DataLoader ---
    test_dataset = InsultDataset(
        test_df_cleaned["Comment"].tolist(),
        [-1] * len(test_df_cleaned),
        tokenizer,
        MAX_SEQ_LEN,
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE * 2, num_workers=2, shuffle=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- K-Fold Setup ---
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    models, optimizers, schedulers, train_dataloaders, awp_managers = [], [], [], [], []
    criterion = nn.BCEWithLogitsLoss()

    print(f"Setting up {N_FOLDS} folds...")
    for fold, (train_idx, _) in enumerate(skf.split(X=train_df, y=train_df["Insult"])):
        print(f"  Fold {fold+1}/{N_FOLDS}")

        fold_train_df = train_df.iloc[train_idx]
        train_dataset = InsultDataset(
            fold_train_df["Comment"].tolist(),
            fold_train_df["Insult"].tolist(),
            tokenizer,
            MAX_SEQ_LEN,
        )
        train_loader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
        )
        train_dataloaders.append(train_loader)

        model = CustomDeBERTa(MODEL_NAME).to(device)
        models.append(model)

        optimizer_parameters = [
            {"params": model.deberta.parameters(), "lr": LR_BACKBONE},
            {"params": model.pooler.parameters(), "lr": LR_HEAD},
            {"params": model.classifier.parameters(), "lr": LR_HEAD},
        ]
        optimizer = optim.AdamW(optimizer_parameters)
        optimizers.append(optimizer)

        awp = AWP(model, optimizer, adv_lr=1e-4)
        awp_managers.append(awp)

        total_steps = len(train_loader) * N_EPOCHS
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * 0.1),
            num_training_steps=total_steps,
        )
        schedulers.append(scheduler)

    # --- Training Loop ---
    print(f"\nStarting training for {N_EPOCHS} epochs across {N_FOLDS} folds...")
    for epoch in range(N_EPOCHS):
        print(f"\n===== Epoch {epoch + 1}/{N_EPOCHS} =====")

        for fold in range(N_FOLDS):
            print(f"  Training Fold {fold + 1}/{N_FOLDS}...")
            train_epoch(
                models[fold],
                train_dataloaders[fold],
                optimizers[fold],
                device,
                schedulers[fold],
                criterion,
                awp_managers[fold],
            )

        print(f"  Generating predictions for epoch {epoch + 1}...")
        all_fold_preds = []
        for fold in range(N_FOLDS):
            fold_preds = predict(models[fold], test_dataloader, device)
            all_fold_preds.append(fold_preds)

        avg_preds = np.mean(all_fold_preds, axis=0)
        pred_map = dict(zip(test_df_cleaned.index, avg_preds))
        submission_template_df["Insult"] = submission_template_df.index.map(
            pred_map
        ).fillna(0.0)
        submission_df_epoch = submission_template_df[["Insult", "Date", "Comment"]]
        submission_filename = os.path.join(save_path, f"submission_{epoch+1}.csv")
        submission_df_epoch.to_csv(submission_filename, index=False)
        print(f"  Epoch {epoch + 1} submission saved to {submission_filename}")

        gc.collect()
        torch.cuda.empty_cache()

    # --- Final Submission ---
    final_submission_filename = os.path.join(save_path, "submission.csv")
    submission_df_epoch.to_csv(final_submission_filename, index=False)
    print(f"\nFinal submission (from last epoch) saved to {final_submission_filename}")
    print("Task completed.")


# EVOLVE-BLOCK-END


if __name__ == "__main__":
    DATA_ROOT = "***"
    SAVE_PATH = "***"

    os.makedirs(SAVE_PATH, exist_ok=True)

    train_df = pd.read_csv(os.path.join(DATA_ROOT, "train.csv"))
    test_df = pd.read_csv(os.path.join(DATA_ROOT, "test.csv"))

    main(train_df, test_df, SAVE_PATH)
