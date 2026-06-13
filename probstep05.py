from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path
import os
import csv

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from pipeline_exam.src.utils import configure_logging, load_dotenv_file
from pipeline_exam.src.models import get_model
from pipeline_exam.src.NERDataset import NERDataset, TransformerNERDataset, pad_collate, transformer_collate
from pipeline_exam.src.schemas import CANONICAL_MODELS

LOGGER = logging.getLogger(__name__)


def collect_errors(model: nn.Module, dataloader: DataLoader, dev: torch.device, vocab) -> list[dict]:
    model.eval()
    errors = []

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(dev), batch_y.to(dev)

            logits = model(batch_x)
            preds = torch.argmax(logits, dim=-1)

            preds_flat = preds.view(-1).cpu().numpy()
            labels_flat = batch_y.view(-1).cpu().numpy()
            tokens_flat = batch_x.view(-1).cpu().numpy()

            mask = (labels_flat != 0) & (labels_flat != -100)

            for token_id, true_id, pred_id in zip(tokens_flat[mask], labels_flat[mask], preds_flat[mask]):
                true_tag = vocab.idx2tag[true_id]
                pred_tag = vocab.idx2tag[pred_id]

                if true_tag != pred_tag:
                    token = vocab.idx2word.get(token_id, "[UNK]")

                    errors.append({
                        "token": token,
                        "true_tag": true_tag,
                        "predicted_tag": pred_tag,
                        "error_type": f"{true_tag} -> {pred_tag}"
                    })

    return errors


def build_step05_parser(default_repo_root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Step05 Error Analysis for trained NER models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    processed_dir = default_repo_root / "pipeline_exam" / "data" / "processed"
    models_dir = default_repo_root / "pipeline_exam" / "models"
    reports_dir = default_repo_root / "pipeline_exam" / "reports"

    parser.add_argument("--tokenized-dataset-path", default=str(processed_dir / "perizie_bio_test.tsv"))
    parser.add_argument("--vocab-path", default=str(processed_dir / "vocab.pkl"))
    parser.add_argument("--models-dir", default=str(models_dir))
    parser.add_argument("--output-reports-dir", default=str(reports_dir))

    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)

    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--logging-level", default="INFO")

    return parser


def run_step05(args: argparse.Namespace) -> None:
    configure_logging(args.logging_level)
    load_dotenv_file(".env")

    dev = torch.device(args.device)
    LOGGER.info(f"Avvio Error Analysis sul dispositivo: {dev.type.upper()}")

    reports_dir = Path(args.output_reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    vocab_path = Path(args.vocab_path)
    tokenized_dataset_path = Path(args.tokenized_dataset_path)
    models_dir = Path(args.models_dir)

    if not vocab_path.exists():
        raise FileNotFoundError(f"Vocabulary not found: {vocab_path}")

    if not tokenized_dataset_path.exists():
        raise FileNotFoundError(f"Test dataset not found: {tokenized_dataset_path}")

    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)

    huggingface_api_key = os.environ.get("HUGGINGFACE_API_KEY")

    model_files = list(models_dir.glob("*_model.pth"))
    if not model_files:
        LOGGER.warning(f"No trained models found in {models_dir}")
        return

    for model_path in model_files:
        model_id = model_path.name.replace("_model.pth", "")
        LOGGER.info(f"Error analysis for model: {model_id.upper()}")

        match model_id:
            case "bert_ner" | "biobert_ner":
                canonical_model = CANONICAL_MODELS["bert"] if model_id == "bert_ner" else CANONICAL_MODELS["biobert"]

                dataset = TransformerNERDataset(
                    file_path=tokenized_dataset_path,
                    model_name=canonical_model,
                    hf_token=huggingface_api_key,
                    vocab=vocab,
                )
                dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=transformer_collate)

            case _:
                dataset = NERDataset(tokenized_dataset_path, vocab=vocab)
                dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=pad_collate)

        model = get_model(
            model_id=model_id,
            vocab_size=len(vocab.word2idx),
            num_classes=len(vocab.tag2idx),
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            hf_token=huggingface_api_key,
        ).to(dev)

        checkpoint = torch.load(model_path, map_location=dev)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

        errors = collect_errors(model, dataloader, dev, vocab)

        output_path = reports_dir / f"{model_id}_error_analysis.csv"

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["token", "true_tag", "predicted_tag", "error_type"]
            )
            writer.writeheader()
            writer.writerows(errors)

        LOGGER.info(f"Saved {len(errors)} errors to {output_path}")