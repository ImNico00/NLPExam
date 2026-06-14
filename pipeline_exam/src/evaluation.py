from __future__ import annotations

import os
import argparse
import logging
import pickle
from pathlib import Path
import csv

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from typing import cast

from sklearn.metrics import classification_report, confusion_matrix # type: ignore
import matplotlib.pyplot as plt
import seaborn as sns

from pipeline_exam.src.utils import configure_logging, load_dotenv_file
from pipeline_exam.src.models import get_model
from pipeline_exam.src.NERDataset import NERDataset, TransformerNERDataset, Vocabulary, pad_collate, transformer_collate
from pipeline_exam.src.schemas import CANONICAL_MODELS

LOGGER = logging.getLogger(__name__)

def evaluate_and_collect_errors(model: nn.Module, dataloader: DataLoader, dev: torch.device, vocab : Vocabulary) -> tuple[list, list, list[dict]]:
    """
    Esegue l'inferenza sul test set, restituendo tag veri, predetti e log degli errori
    comprensivi della frase intera di contesto (senza ripetizioni di sub-token).
    """
    model.eval()
    
    all_labels_tags = []
    all_preds_tags = []
    errors = []

    with torch.no_grad():
        for batch_x, batch_y, flat_raw_words in dataloader:
            if isinstance(batch_x, dict):
                batch_x = {k: v.to(dev) for k, v in batch_x.items()}
                batch_y = batch_y.to(dev)
                logits = model(**batch_x)
            else:
                batch_x = batch_x.to(dev)
                batch_y = batch_y.to(dev)
                logits = model(batch_x)
                
            if hasattr(logits, "logits"):
                logits = logits.logits
            
            preds = torch.argmax(logits, dim=-1)

            # Invece di "schiacciare" tutto, prendiamo le dimensioni del batch
            batch_size, seq_len = batch_y.shape

            # Analizziamo una frase alla volta
            for b in range(batch_size):
                
                # 1. Ricostruiamo la frase originale pulita
                sentence_tokens = []
                last_added_word = None  # <--- VARIABILE AGGIUNTA PER IL FIX DELL'ECO
                
                for s in range(seq_len):
                    word = flat_raw_words[b * seq_len + s]
                    if word not in ["[PAD]", "[SPECIAL]"]:
                        # Se la parola è diversa dall'ultima inserita, la aggiungiamo!
                        if word != last_added_word:
                            sentence_tokens.append(word)
                            last_added_word = word
                
                full_sentence = " ".join(sentence_tokens)

                # 2. Iteriamo sulle parole della frase per trovare gli errori
                for s in range(seq_len):
                    true_id = int(batch_y[b, s].item())
                    pred_id = int(preds[b, s].item())
                    
                    # Ignoriamo il padding per il calcolo delle metriche
                    if true_id != vocab.pad_tag_idx and true_id != -100:
                        true_tag = vocab.idx2tag[true_id]
                        pred_tag = vocab.idx2tag[pred_id]
                        real_word = flat_raw_words[b * seq_len + s]

                        # Raccogliamo i tag per il Classification Report
                        all_labels_tags.append(true_tag)
                        all_preds_tags.append(pred_tag)

                        # Se c'è un errore, salviamo tutto, inclusa la frase!
                        if true_tag != pred_tag:
                            errors.append({
                                "token": real_word,
                                "true_tag": true_tag,
                                "predicted_tag": pred_tag,
                                "error_type": f"{true_tag} -> {pred_tag}",
                                "sentence": full_sentence  # <--- QUINTO LABEL
                            })

    return all_labels_tags, all_preds_tags, errors

def format_pipeline_step04_summary(
    *,
    model_id: str,
    model_path: str,
    cm_full_path: str,
    cm_entity_path: str,
    macro_f1: float,
    accuracy: float
) -> str:
    summary = (
        f"Pipeline Step04 summary ({model_id.upper()}):\n"
        f"- Model ID: {model_id}\n"
        f"- Model Path: {model_path}\n"
        f"- Confusion Matrix Full: {cm_full_path}\n"
        f"- Confusion Matrix Entity: {cm_entity_path}\n"
        f"- Overall Accuracy: {accuracy:.4f}\n"
        f"- Macro F1-Score (No 'O'): {macro_f1:.4f}"
    )
    return summary

def build_step04_parser(default_repo_root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Start Evaluation Loop for all trained NER Models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    processed_dir = default_repo_root / "pipeline_exam" / "data" / "processed"
    models_dir = default_repo_root / "pipeline_exam" / "models"
    evaluation_dir = default_repo_root / "pipeline_exam" / "evaluations"
    
    parser.add_argument("--models-evaluation-summary-path", default=str(evaluation_dir / "models_evaluation_summary.csv"))
    parser.add_argument("--tokenized-dataset-path", default=str(processed_dir / "perizie_bio_test.tsv"))
    parser.add_argument("--vocab-path", default=str(processed_dir / "vocab.pkl"))
    parser.add_argument("--models-dir", default=str(models_dir), help="Cartella contenente i file .pth")
    parser.add_argument("--output-eval-dir", default=str(evaluation_dir), help="Cartella dove salvare le Matrici di Confusione")
    
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--logging-level", default="INFO")
    return parser

def run_step04(args: argparse.Namespace) -> None:
    configure_logging(args.logging_level)
    load_dotenv_file(".env")
    
    dev = torch.device(args.device)
    LOGGER.info(f"🚀 Avvio Evaluation sul dispositivo: {dev.type.upper()}")
    
    evaluation_dir = Path(args.output_eval_dir)
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    models_dir = Path(args.models_dir)
    
    huggingface_api_key = os.environ.get("HUGGINGFACE_API_KEY")
    if not huggingface_api_key:
        LOGGER.warning("HUGGINGFACE_API_KEY not set -- unauthenticated may fail.")
    
    vocab_path = Path(args.vocab_path)
    if not vocab_path.exists():
        LOGGER.error(f"Vocabolario non trovato in {vocab_path}!")
        return
        
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)
        
    tokenized_dataset_path = Path(args.tokenized_dataset_path)
    if not tokenized_dataset_path.exists():
        raise FileNotFoundError(f"Test dataset not found: {tokenized_dataset_path}")

    model_files = sorted(models_dir.glob("*_model.pth"))
    if not model_files:
        LOGGER.warning(f"Nessun modello trovato in {models_dir}. Esegui prima lo Step 03.")
        return
        
    LOGGER.info(f"Trovati {len(model_files)} modelli da valutare. Inizio ciclo...")

    summary_rows = []
    
    for model_path in model_files:
        # Estraiamo l'ID dinamico (es: "bilstm_model.pth" -> "bilstm")
        model_id = model_path.name.replace("_model.pth", "")
        LOGGER.info(f"\n{'='*50}\nValutazione Modello: {model_id.upper()}\n{'='*50}")

        match(model_id):
            case "bert_ner" | "biobert_ner":
                canonical_model = CANONICAL_MODELS["bert"] if model_id == "bert_ner" else CANONICAL_MODELS["biobert"]
                dataset = TransformerNERDataset(
                    file_path=tokenized_dataset_path, 
                    model_name=canonical_model,
                    hf_token=huggingface_api_key,
                    vocab=vocab
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
            hf_token=huggingface_api_key
        ).to(dev)

        checkpoint = torch.load(model_path, map_location=dev, weights_only=False)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

        labels_tags, preds_tags, errors = evaluate_and_collect_errors(model, dataloader, dev, vocab)

        all_labels = sorted(set(labels_tags) | set(preds_tags))
        entity_labels = [
            label for label in all_labels
            if label not in {"O", vocab.pad_token}
        ]
            
        report_str = classification_report(
            labels_tags,
            preds_tags,
            labels=entity_labels,
            zero_division=0,
        )
        raw_report = classification_report(
            labels_tags, 
            preds_tags, 
            labels=entity_labels, 
            zero_division=0, 
            output_dict=True
            )
        report_dict = cast(dict, raw_report)
        macro_f1 = report_dict['macro avg']['f1-score']
        
        # Per l'Accuracy consideriamo tutti i tag, inclusi "O" (giusto per avere una metrica globale)
        full_raw_report_dict = classification_report(labels_tags, preds_tags, zero_division=0, output_dict=True)
        full_report_dict = cast(dict, full_raw_report_dict)
        accuracy = full_report_dict['accuracy']
        
        LOGGER.info(f"Classification Report ({model_id.upper()}):\n{report_str}")
        
        cm_full = confusion_matrix(labels_tags, preds_tags, labels=all_labels)
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm_full,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=all_labels,
            yticklabels=all_labels,
        )
        plt.title(f"Full Confusion Matrix - {model_id.upper()}")
        plt.ylabel('True BIO Tag')
        plt.xlabel('Predicted BIO Tag')
        
        cm_full_path = evaluation_dir / f"{model_id}_confusion_matrix_full.png"
        plt.savefig(cm_full_path, bbox_inches="tight")
        plt.close()

        cm_entity = confusion_matrix(labels_tags, preds_tags, labels=entity_labels)
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm_entity,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=entity_labels,
            yticklabels=entity_labels,
        )
        plt.title(f"Entity Confusion Matrix - {model_id.upper()}")
        plt.ylabel('True BIO Tag')
        plt.xlabel('Predicted BIO Tag')
        
        cm_entity_path = evaluation_dir / f"{model_id}_confusion_matrix_entity.png"
        plt.savefig(cm_entity_path, bbox_inches="tight")
        plt.close()

        output_path = evaluation_dir / f"{model_id}_error_analysis.csv"
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["token", "true_tag", "predicted_tag", "error_type", "sentence"]
            )
            writer.writeheader()
            writer.writerows(errors)

        LOGGER.info(f"Saved {len(errors)} errors to {output_path}")

        summary_rows.append({
            "model_id": model_id,
            "accuracy": accuracy,
            "macro_f1_no_o": macro_f1,
            "num_errors": len(errors)
        })
        
        LOGGER.info(
            "\n%s",
            format_pipeline_step04_summary(
                model_id=model_id,
                model_path=str(model_path),
                cm_full_path=str(cm_full_path),
                cm_entity_path=str(cm_entity_path),
                macro_f1=macro_f1,
                accuracy=accuracy
            )
        )

    summary_path = args.models_evaluation_summary_path
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model_id",
                "accuracy",
                "macro_f1_no_o",
                "num_errors"
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    LOGGER.info(f"Saved evaluation summary to {summary_path}")