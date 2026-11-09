import argparse
import csv
import gc
import inspect
import json
import math
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

MODEL_NAME = "FacebookAI/xlm-roberta-base"
RAW_FILENAME = "subtask-1-annotations-with-paragraphs.txt"
TEXT_COLUMN = "paragraph"

NUM_LABELS = 14
MAX_LENGTH = 256
VAL_SIZE = 0.10
SPLIT_SEED = 42
TRAINING_SEED = 42

LEARNING_RATE = 2e-5
NUM_TRAIN_EPOCHS = 8
WEIGHT_DECAY = 0.01
LR_SCHEDULER_TYPE = "linear"
WARMUP_FRACTION = 0.10

TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 16
GRADIENT_ACCUMULATION_STEPS = 2

POS_WEIGHT_POWER = 0.5
POS_WEIGHT_MAX = 4.0

CHECKPOINT_THRESHOLD = 0.25

INFERENCE_THRESHOLD = 0.50

EARLY_STOPPING_PATIENCE = 2
EARLY_STOPPING_THRESHOLD = 0.001

def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)

def find_raw_files(root: Path) -> list[Path]:
    files = sorted(root.rglob(RAW_FILENAME))
    if not files:
        raise FileNotFoundError(
            f"No '{RAW_FILENAME}' files found recursively under: {root}"
        )
    return files

def read_raw_records(root: Path) -> tuple[list[dict], list[tuple]]:

    records = []
    bad_rows = []

    for path in find_raw_files(root):
        with path.open(
            "r",
            encoding="utf-8-sig",
            errors="replace",
            newline="",
        ) as handle:
            reader = csv.reader(handle, delimiter="\t")

            for line_number, row in enumerate(reader, start=1):
                if len(row) < 5:
                    bad_rows.append((str(path), line_number, row))
                    continue

                labels = [
                    label.strip()
                    for label in row[3:-1]
                    if label.strip()
                ]
                paragraph = row[-1].strip()

                if not labels or not paragraph:
                    bad_rows.append((str(path), line_number, row))
                    continue

                records.append({
                    TEXT_COLUMN: paragraph,
                    "labels": labels,
                })

    if not records:
        raise RuntimeError(f"No valid labeled examples found under: {root}")

    return records, bad_rows

def prepare_train_dataframe(train_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    records, bad_rows = read_raw_records(train_dir)
    raw_df = pd.DataFrame(records)

    mlb = MultiLabelBinarizer()
    encoded = mlb.fit_transform(raw_df["labels"])
    label_columns = mlb.classes_.tolist()

    if len(label_columns) != NUM_LABELS:
        raise ValueError(
            f"Expected {NUM_LABELS} labels in train data, "
            f"found {len(label_columns)}: {label_columns}"
        )

    labels_df = pd.DataFrame(
        encoded,
        columns=label_columns,
        dtype="int8",
    )

    df = pd.concat(
        [
            raw_df[[TEXT_COLUMN]].reset_index(drop=True),
            labels_df.reset_index(drop=True),
        ],
        axis=1,
    )

    print(
        f"Processed train data: {len(df):,} rows, "
        f"{len(label_columns)} labels, {len(bad_rows)} skipped rows."
    )
    return df, label_columns

def prepare_test_dataframe(
    test_dir: Path,
    label_columns: list[str],
) -> pd.DataFrame:
    records, bad_rows = read_raw_records(test_dir)

    known = set(label_columns)
    unknown = sorted({
        label
        for record in records
        for label in record["labels"]
        if label not in known
    })

    if unknown:
        raise ValueError(
            "Test data contains labels not present in train/model metadata: "
            + ", ".join(unknown)
        )

    rows = []
    for record in records:
        row = {TEXT_COLUMN: record[TEXT_COLUMN]}
        labels = set(record["labels"])
        for label in label_columns:
            row[label] = int(label in labels)
        rows.append(row)

    df = pd.DataFrame(
        rows,
        columns=[TEXT_COLUMN] + label_columns,
    )

    print(
        f"Processed test data: {len(df):,} rows, "
        f"{len(bad_rows)} skipped rows."
    )
    return df

class ParagraphDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        tokenizer,
        label_columns: list[str],
        max_length: int,
    ):
        self.texts = (
            dataframe[TEXT_COLUMN]
            .fillna("")
            .astype(str)
            .tolist()
        )
        self.labels = (
            dataframe[label_columns]
            .astype("float32")
            .to_numpy()
        )
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        encoding = self.tokenizer(
            self.texts[index],
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )
        encoding["labels"] = torch.tensor(
            self.labels[index],
            dtype=torch.float32,
        )
        return encoding

class MultilabelDataCollator:
    def __init__(self, tokenizer, pad_to_multiple_of=None):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features):
        labels = torch.stack([
            feature["labels"]
            if isinstance(feature["labels"], torch.Tensor)
            else torch.tensor(feature["labels"], dtype=torch.float32)
            for feature in features
        ])

        input_features = [
            {
                key: value
                for key, value in feature.items()
                if key != "labels"
            }
            for feature in features
        ]

        batch = self.tokenizer.pad(
            input_features,
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        batch["labels"] = labels.float()
        return batch

class WeightedMultilabelTrainer(Trainer):
    def __init__(self, *args, pos_weight=None, **kwargs):
        super().__init__(*args, **kwargs)
        if pos_weight is None:
            raise ValueError("pos_weight is required for weighted training.")
        self.pos_weight = pos_weight

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        labels = inputs.pop("labels")
        outputs = model(**inputs)

        loss = F.binary_cross_entropy_with_logits(
            outputs.logits,
            labels.float(),
            pos_weight=self.pos_weight.to(outputs.logits.device),
        )

        return (loss, outputs) if return_outputs else loss

def build_training_arguments(
    output_dir: Path,
    warmup_steps: int,
) -> TrainingArguments:
    supports_bf16 = (
        torch.cuda.is_available()
        and torch.cuda.is_bf16_supported()
    )

    kwargs = {
        "output_dir": str(output_dir),
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "lr_scheduler_type": LR_SCHEDULER_TYPE,
        "warmup_steps": warmup_steps,
        "num_train_epochs": NUM_TRAIN_EPOCHS,
        "per_device_train_batch_size": TRAIN_BATCH_SIZE,
        "per_device_eval_batch_size": EVAL_BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "save_strategy": "epoch",
        "logging_strategy": "steps",
        "logging_steps": 50,
        "load_best_model_at_end": True,
        "metric_for_best_model": "macro_f1",
        "greater_is_better": True,
        "save_total_limit": 1,
        "bf16": supports_bf16,
        "fp16": torch.cuda.is_available() and not supports_bf16,
        "report_to": "none",
        "seed": TRAINING_SEED,
        "data_seed": SPLIT_SEED,
        "dataloader_num_workers": 2,
        "dataloader_pin_memory": True,
        "label_names": ["labels"],
    }

    parameters = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in parameters:
        kwargs["eval_strategy"] = "epoch"
    else:
        kwargs["evaluation_strategy"] = "epoch"

    return TrainingArguments(**kwargs)

def add_tokenizer_to_trainer_kwargs(kwargs: dict, tokenizer) -> None:
    parameters = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in parameters:
        kwargs["processing_class"] = tokenizer
    else:
        kwargs["tokenizer"] = tokenizer

def checkpoint_metrics(eval_prediction):
    logits = eval_prediction.predictions
    if isinstance(logits, tuple):
        logits = logits[0]

    probabilities = torch.sigmoid(torch.tensor(logits)).numpy()
    y_true = eval_prediction.label_ids.astype(np.int32)
    y_pred = (probabilities >= CHECKPOINT_THRESHOLD).astype(np.int32)

    return {
        "micro_f1": f1_score(
            y_true,
            y_pred,
            average="micro",
            zero_division=0,
        ),
        "macro_f1": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
    }

def compute_pos_weight(
    train_df: pd.DataFrame,
    label_columns: list[str],
) -> torch.Tensor:
    positive_counts = (
        train_df[label_columns]
        .sum(axis=0)
        .to_numpy(dtype=np.float32)
    )
    negative_counts = len(train_df) - positive_counts

    raw_ratio = (
        negative_counts
        / np.maximum(positive_counts, 1.0)
    )

    pos_weight_values = np.power(
        raw_ratio,
        POS_WEIGHT_POWER,
    )
    pos_weight_values = np.clip(
        pos_weight_values,
        1.0,
        POS_WEIGHT_MAX,
    )

    print("Positive-class weights:")
    for label, count, weight in zip(
        label_columns,
        positive_counts,
        pos_weight_values,
    ):
        print(f"  {label}: positives={int(count)}, pos_weight={weight:.4f}")

    return torch.tensor(
        pos_weight_values,
        dtype=torch.float32,
    )

def finetune(
    train_dir: Path,
    model_dir: Path,
    output_dir: Path,
) -> list[str]:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "--ft requires a CUDA GPU. "
            "The notebook run being reproduced was GPU-trained."
        )

    set_all_seeds(TRAINING_SEED)

    full_df, label_columns = prepare_train_dataframe(train_dir)

    splitter = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=VAL_SIZE,
        random_state=SPLIT_SEED,
    )
    train_indices, val_indices = next(
        splitter.split(
            full_df[TEXT_COLUMN],
            full_df[label_columns].to_numpy(),
        )
    )

    train_df = full_df.iloc[train_indices].reset_index(drop=True)
    val_df = full_df.iloc[val_indices].reset_index(drop=True)

    print(f"Fine-tuning rows: {len(train_df):,}")
    print(f"Validation rows:  {len(val_df):,}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_dataset = ParagraphDataset(
        train_df,
        tokenizer,
        label_columns,
        MAX_LENGTH,
    )
    val_dataset = ParagraphDataset(
        val_df,
        tokenizer,
        label_columns,
        MAX_LENGTH,
    )

    collator = MultilabelDataCollator(
        tokenizer,
        pad_to_multiple_of=8,
    )

    id2label = {
        index: label
        for index, label in enumerate(label_columns)
    }
    label2id = {
        label: index
        for index, label in id2label.items()
    }

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label_columns),
        problem_type="multi_label_classification",
        id2label=id2label,
        label2id=label2id,
    )

    optimizer_steps_per_epoch = math.ceil(
        len(train_dataset)
        / (
            TRAIN_BATCH_SIZE
            * GRADIENT_ACCUMULATION_STEPS
        )
    )
    total_optimizer_steps = (
        optimizer_steps_per_epoch
        * NUM_TRAIN_EPOCHS
    )
    warmup_steps = max(
        1,
        int(WARMUP_FRACTION * total_optimizer_steps),
    )

    checkpoint_dir = output_dir / "_training_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    training_args = build_training_arguments(
        checkpoint_dir,
        warmup_steps,
    )

    pos_weight = compute_pos_weight(
        train_df,
        label_columns,
    )

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": val_dataset,
        "data_collator": collator,
        "compute_metrics": checkpoint_metrics,
        "callbacks": [
            EarlyStoppingCallback(
                early_stopping_patience=EARLY_STOPPING_PATIENCE,
                early_stopping_threshold=EARLY_STOPPING_THRESHOLD,
            )
        ],
        "pos_weight": pos_weight,
    }
    add_tokenizer_to_trainer_kwargs(
        trainer_kwargs,
        tokenizer,
    )

    trainer = WeightedMultilabelTrainer(**trainer_kwargs)

    print(
        "\nTraining weighted XLM-RoBERTa "
        f"(lr={LEARNING_RATE}, epochs={NUM_TRAIN_EPOCHS}, "
        f"checkpoint threshold={CHECKPOINT_THRESHOLD})..."
    )
    trainer.train()

    model_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))

    metadata = {
        "base_model": MODEL_NAME,
        "text_column": TEXT_COLUMN,
        "label_columns": label_columns,
        "max_length": MAX_LENGTH,
        "inference_threshold": INFERENCE_THRESHOLD,
        "checkpoint_selection_threshold": CHECKPOINT_THRESHOLD,
        "split_seed": SPLIT_SEED,
        "training_seed": TRAINING_SEED,
        "validation_size": VAL_SIZE,
        "learning_rate": LEARNING_RATE,
        "num_train_epochs": NUM_TRAIN_EPOCHS,
        "weight_decay": WEIGHT_DECAY,
        "lr_scheduler_type": LR_SCHEDULER_TYPE,
        "warmup_fraction": WARMUP_FRACTION,
        "train_batch_size": TRAIN_BATCH_SIZE,
        "eval_batch_size": EVAL_BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "pos_weight_power": POS_WEIGHT_POWER,
        "pos_weight_max": POS_WEIGHT_MAX,
        "best_checkpoint_metric": "macro_f1",
        "best_validation_macro_f1_at_0.25": (
            float(trainer.state.best_metric)
            if trainer.state.best_metric is not None
            else None
        ),
    }

    with (model_dir / "task_metadata.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            metadata,
            handle,
            indent=2,
            ensure_ascii=False,
        )

    shutil.rmtree(checkpoint_dir, ignore_errors=True)

    del trainer, model, tokenizer, train_dataset, val_dataset
    gc.collect()
    torch.cuda.empty_cache()

    print(f"\nSaved best model to: {model_dir}")
    return label_columns

def evaluate_test(
    test_dir: Path,
    model_dir: Path,
    output_dir: Path,
) -> None:
    metadata_path = model_dir / "task_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Missing model metadata: {metadata_path}. "
            "Run with --ft first or provide a valid --model-dir."
        )

    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    label_columns = metadata["label_columns"]
    max_length = int(metadata.get("max_length", MAX_LENGTH))
    threshold = float(
        metadata.get(
            "inference_threshold",
            INFERENCE_THRESHOLD,
        )
    )

    test_df = prepare_test_dataframe(
        test_dir,
        label_columns,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    test_dataset = ParagraphDataset(
        test_df,
        tokenizer,
        label_columns,
        max_length,
    )
    collator = MultilabelDataCollator(
        tokenizer,
        pad_to_multiple_of=8 if torch.cuda.is_available() else None,
    )

    eval_kwargs = {
        "output_dir": str(output_dir / "_eval_tmp"),
        "per_device_eval_batch_size": EVAL_BATCH_SIZE,
        "report_to": "none",
    }

    if torch.cuda.is_available():
        supports_bf16 = torch.cuda.is_bf16_supported()
        eval_kwargs["bf16"] = supports_bf16
        eval_kwargs["fp16"] = not supports_bf16

    eval_args = TrainingArguments(**eval_kwargs)

    trainer_kwargs = {
        "model": model,
        "args": eval_args,
        "data_collator": collator,
    }
    add_tokenizer_to_trainer_kwargs(
        trainer_kwargs,
        tokenizer,
    )
    trainer = Trainer(**trainer_kwargs)

    print(f"\nRunning test inference at threshold {threshold:.2f}...")
    prediction_output = trainer.predict(test_dataset)

    logits = prediction_output.predictions
    if isinstance(logits, tuple):
        logits = logits[0]

    probabilities = torch.sigmoid(
        torch.tensor(logits)
    ).numpy()

    y_true = prediction_output.label_ids.astype(np.int32)
    y_pred = (
        probabilities >= threshold
    ).astype(np.int32)

    micro_f1 = f1_score(
        y_true,
        y_pred,
        average="micro",
        zero_division=0,
    )
    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    precision, recall, per_label_f1, support = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average=None,
            zero_division=0,
        )
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    predictions_df = pd.DataFrame(
        y_pred,
        columns=label_columns,
        dtype="int8",
    )
    predictions_df.to_csv(
        output_dir / "test_predictions.csv",
        index=False,
        encoding="utf-8",
    )

    metrics_df = pd.DataFrame([{
        "n_examples": len(test_df),
        "threshold": threshold,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
    }])
    metrics_df.to_csv(
        output_dir / "test_metrics.csv",
        index=False,
        encoding="utf-8",
    )

    per_label_df = pd.DataFrame({
        "label": label_columns,
        "support": support.astype(int),
        "precision": precision,
        "recall": recall,
        "f1": per_label_f1,
    })
    per_label_df.to_csv(
        output_dir / "test_per_label_metrics.csv",
        index=False,
        encoding="utf-8",
    )

    shutil.rmtree(output_dir / "_eval_tmp", ignore_errors=True)

    print("\nTEST RESULTS")
    print(f"Micro-F1: {micro_f1:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")
    print(f"\nSaved predictions: {output_dir / 'test_predictions.csv'}")
    print(f"Saved metrics:     {output_dir / 'test_metrics.csv'}")
    print(
        "Saved per-label:   "
        f"{output_dir / 'test_per_label_metrics.csv'}"
    )

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Weighted XLM-RoBERTa multilabel fine-tuning/inference "
            "for the SemEval paragraph-frame data."
        )
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--ft",
        action="store_true",
        help=(
            "Process raw train data, fine-tune the weighted model, "
            "save it, then evaluate on raw test data."
        ),
    )
    mode.add_argument(
        "--inference",
        action="store_true",
        help=(
            "Load the saved model and apply it to raw test data only."
        ),
    )

    parser.add_argument(
        "--train-dir",
        type=Path,
        default=Path("train"),
        help="Raw training-data folder (default: ./train).",
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=Path("test"),
        help="Raw test-data folder (default: ./test).",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("model"),
        help="Saved model folder (default: ./model).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Test prediction/metric folder (default: ./output).",
    )

    return parser.parse_args()

def main():
    args = parse_args()

    if args.ft:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        finetune(
            train_dir=args.train_dir,
            model_dir=args.model_dir,
            output_dir=args.output_dir,
        )
        evaluate_test(
            test_dir=args.test_dir,
            model_dir=args.model_dir,
            output_dir=args.output_dir,
        )
    else:
        evaluate_test(
            test_dir=args.test_dir,
            model_dir=args.model_dir,
            output_dir=args.output_dir,
        )

if __name__ == "__main__":
    main()