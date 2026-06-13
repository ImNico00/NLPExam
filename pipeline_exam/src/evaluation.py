from __future__ import annotations

import os
import argparse
import logging
import pickle
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import device
from typing import cast

from sklearn.metrics import classification_report, confusion_matrix # type: ignore
import matplotlib.pyplot as plt
import seaborn as sns

from pipeline_exam.src.utils import configure_logging, load_dotenv_file
from pipeline_exam.src.models import get_model
from pipeline_exam.src.NERDataset import NERDataset, TransformerNERDataset, pad_collate, transformer_collate
from pipeline_exam.src.schemas import CANONICAL_MODELS

LOGGER = logging.getLogger(__name__)

def evaluate_model(model: nn.Module, dataloader: DataLoader, dev: device, vocab) -> tuple[list, list]:
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(dev), batch_y.to(dev)
            
            logits = model(batch_x)
            preds = torch.argmax(logits, dim=-1)
            
            preds_flat = preds.view(-1).cpu().numpy()
            labels_flat = batch_y.view(-1).cpu().numpy()
            
            # CRITICO: Ignoriamo sia il padding della BiLSTM (0) sia quello dei Transformers (-100)
            mask = (labels_flat != 0) & (labels_flat != -100)
            
            all_preds.extend(preds_flat[mask])
            all_labels.extend(labels_flat[mask])
            
    # Mappiamo i numeri ai tag stringa
    preds_tags = [vocab.idx2tag[p] for p in all_preds]
    labels_tags = [vocab.idx2tag[l] for l in all_labels]
    
    return labels_tags, preds_tags

def format_pipeline_step04_summary(
    *,
    model_id: str,
    model_path: str,
    confusion_matrix_path: str,
    macro_f1: float,
    accuracy: float
) -> str:
    summary = (
        f"Pipeline Step04 summary ({model_id.upper()}):\n"
        f"- Model ID: {model_id}\n"
        f"- Model Path: {model_path}\n"
        f"- Confusion Matrix: {confusion_matrix_path}\n"
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
    plots_dir = default_repo_root / "pipeline_exam" / "plots"
    
    parser.add_argument("--tokenized-dataset-path", default=str(processed_dir / "perizie_bio_test.tsv"))
    parser.add_argument("--vocab-path", default=str(processed_dir / "vocab.pkl"))
    parser.add_argument("--models-dir", default=str(models_dir), help="Cartella contenente i file .pth")
    parser.add_argument("--output-plots-dir", default=str(plots_dir), help="Cartella dove salvare le Matrici di Confusione")
    
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
    
    plots_dir = Path(args.output_plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    models_dir = Path(args.models_dir)
    
    huggingface_api_key = os.environ.get("HUGGINGFACE_API_KEY")
    if not huggingface_api_key:
        LOGGER.warning("HUGGINGFACE_API_KEY not set -- unauthenticated may fail.")
    
    # 1. Caricamento del Vocabolario (serve per tutti i modelli)
    vocab_path = Path(args.vocab_path)
    if not vocab_path.exists():
        LOGGER.error(f"Vocabolario non trovato in {vocab_path}!")
        return
        
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)
        
    tokenized_dataset_path = Path(args.tokenized_dataset_path)
    if not tokenized_dataset_path.exists():
        raise FileNotFoundError(f"Test dataset not found: {tokenized_dataset_path}")

    # 2. Cerchiamo tutti i modelli salvati nella cartella
    model_files = list(models_dir.glob("*_model.pth"))
    if not model_files:
        LOGGER.warning(f"Nessun modello trovato in {models_dir}. Esegui prima lo Step 03.")
        return
        
    LOGGER.info(f"Trovati {len(model_files)} modelli da valutare. Inizio ciclo...")

    # 3. LOOP SUI MODELLI
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
        model.load_state_dict(checkpoint["model_state_dict"])

        labels_tags, preds_tags = evaluate_model(model, dataloader, dev, vocab)
        labels = sorted(set(labels_tags))
        if "O" in labels: 
            labels.remove("O") # Togliamo "O" per calcolare la F1 vera sulle entità
            
        report_str = classification_report(labels_tags, preds_tags, labels=labels, zero_division=0)
        raw_report = classification_report(labels_tags, preds_tags, labels=labels, zero_division=0, output_dict=True)
        report_dict = cast(dict, raw_report)
        macro_f1 = report_dict['macro avg']['f1-score']
        
        # Per l'Accuracy consideriamo tutti i tag, inclusi "O" (giusto per avere una metrica globale)
        full_raw_report_dict = classification_report(labels_tags, preds_tags, zero_division=0, output_dict=True)
        full_report_dict = cast(dict, full_raw_report_dict)
        accuracy = full_report_dict['accuracy']
        
        LOGGER.info(f"Classification Report ({model_id.upper()}):\n{report_str}")
        
        
        cm = confusion_matrix(labels_tags, preds_tags, labels=labels)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.title(f'Confusion Matrix - {model_id.upper()}')
        plt.ylabel('True BIO Tag')
        plt.xlabel('Predicted BIO Tag')
        
        cm_path = plots_dir / f"{model_id}_confusion_matrix.png"
        plt.savefig(cm_path)
        plt.close()
        
        LOGGER.info(
            "\n%s",
            format_pipeline_step04_summary(
                model_id=model_id,
                model_path=str(model_path),
                confusion_matrix_path=str(cm_path),
                macro_f1=macro_f1,
                accuracy=accuracy
            )
        )