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
from typing import cast

from seqeval.metrics import classification_report, f1_score, accuracy_score
from sklearn.metrics import confusion_matrix # type: ignore
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from pipeline_exam.src.utils import configure_logging, load_dotenv_file, extract_error_logs, stanza_entities_to_bio, align_gold_label_for_model
from pipeline_exam.src.models import get_pytorch_model, StanzaMedicalNER
from pipeline_exam.src.NERDataset import NERDataset, TransformerNERDataset, Vocabulary, pad_collate, transformer_collate
from pipeline_exam.src.schemas import CANONICAL_MODELS

LOGGER = logging.getLogger(__name__)

def confusion_matrix_figure(cm_path : Path, 
                            all_labels_tags : list, 
                            all_preds_tags : list, 
                            labels : list,
                            title : str) -> None:
    cm = confusion_matrix(all_labels_tags, all_preds_tags, labels=labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.ylabel('True BIO Tag')
    plt.xlabel('Predicted BIO Tag')
    plt.savefig(cm_path, bbox_inches="tight")
    plt.close()

def evaluate_and_collect_errors(model: nn.Module, dataloader: DataLoader, dev: torch.device, vocab : Vocabulary, gt_model_name: str) -> tuple[list[list[str]], list[list[str]], list[dict]]:
    model.eval()
    
    all_labels_tags = []
    all_preds_tags = []
    errors = []

    with torch.no_grad():
        for batch_x, batch_y, flat_raw_words, flat_raw_tags in dataloader:
            
            # 1. Spostamento dati sul device
            if isinstance(batch_x, dict):
                batch_x = {k: v.to(dev) for k, v in batch_x.items()}
            else:
                batch_x = batch_x.to(dev)
            batch_y = batch_y.to(dev)

            # 2. FORWARD UNIFICATO (Come nello step 03)
            if isinstance(batch_x, dict):
                outputs = model(**batch_x, labels=batch_y)
            else:
                outputs = model(input_ids=batch_x, labels=batch_y)

            # 3. Estrazione predizioni
            # N.B. Per Transformers/LSTM è un Tensore. Per CRF è una Lista di Liste di interi.
            preds = outputs["predictions"]

            batch_size, seq_len = batch_y.shape

            for b in range(batch_size):
                valid_tokens = []
                valid_true_tags = []
                valid_pred_tags = []

                for s in range(seq_len):
                    true_tag = flat_raw_tags[b * seq_len + s]
                    
                    # Estraiamo il pred_id in modo sicuro (gestendo sia Tensori che Liste da CRF)
                    if isinstance(preds, torch.Tensor):
                        pred_id = int(preds[b, s].item())
                    else:
                        # Se è la lista del CRF, il padding è già stato rimosso dalla decodifica.
                        # Evitiamo IndexError se 's' va oltre i token validi.
                        pred_id = int(preds[b][s]) if s < len(preds[b]) else 0

                    # Filtriamo il padding
                    if true_tag != "[PAD]":
                        pred_tag = vocab.idx2tag.get(pred_id, "O")
                        real_word = flat_raw_words[b * seq_len + s]
                        true_tag_aligned = align_gold_label_for_model(true_tag, gt_model_name.lower())
                        pred_tag_aligned = align_gold_label_for_model(pred_tag, gt_model_name.lower())

                        valid_tokens.append(real_word)
                        valid_true_tags.append(true_tag_aligned)
                        valid_pred_tags.append(pred_tag_aligned)

                all_labels_tags.append(valid_true_tags)
                all_preds_tags.append(valid_pred_tags)

                errors.extend(extract_error_logs(
                    tokens=valid_tokens,
                    true_tags=valid_true_tags,
                    pred_tags=valid_pred_tags
                ))

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
        description="Start Evaluation Loop for all trained NER Models and Ground Truths",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    processed_dir = default_repo_root / "pipeline_exam" / "data" / "processed"
    models_dir = default_repo_root / "pipeline_exam" / "models"
    evaluation_dir = default_repo_root / "pipeline_exam" / "evaluations"
    
    parser.add_argument("--processed-dir", default=str(processed_dir), help="Cartella con il golden-test o i test set e i vocab.pkl")
    parser.add_argument("--models-dir", default=str(models_dir), help="Cartella base contenente le sottocartelle dei modelli")
    parser.add_argument("--output-eval-base-dir", default=str(evaluation_dir), help="Cartella base dove salvare le valutazioni")

    parser.add_argument("--use-golden-test", action="store_true", default=True, help="Flag per usare il test set 'golden' in processed/golden-test invece di un test set creato con la ground truth corrente. Utile per confrontare modelli diversi sullo stesso test set.")
    parser.add_argument("--no-golden-test", dest="use_golden_test", action="store_false")

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
    
    models_base_dir = Path(args.models_dir)
    processed_base_dir = Path(args.processed_dir)
    evaluations_base_dir = Path(args.output_eval_base_dir)

    use_golden_test = args.use_golden_test
    
    huggingface_api_key = os.environ.get("HUGGINGFACE_API_KEY")
    if not huggingface_api_key:
        LOGGER.warning("HUGGINGFACE_API_KEY not set -- unauthenticated may fail.")
    
    if not models_base_dir.exists():
        LOGGER.error(f"Cartella modelli non trovata in {models_base_dir}. Esegui prima lo Step 03.")
        return

    # Trova tutte le sottocartelle dentro models/ (es: scibc5cdr, dictionary, scibionlp, stanza, ecc.) che rappresentano le Ground Truth
    gt_folders = [d for d in models_base_dir.iterdir() if d.is_dir()]
    
    if not gt_folders:
        LOGGER.warning(f"Nessuna cartella di Ground Truth trovata in {models_base_dir}.")
        return

    LOGGER.info(f"Trovate {len(gt_folders)} Ground Truth da valutare. Inizio ciclo...")

    # --- CICLO SULLE GROUND TRUTH ---
    for gt_folder in gt_folders:
        gt_model_name = gt_folder.name
        LOGGER.info(f"\n{'*'*60}\nInizio Valutazione per Ground Truth: {gt_model_name.upper()}\n{'*'*60}")

        # 1. Definiamo i percorsi specifici per questa Ground Truth
        vocab_path = processed_base_dir / gt_model_name / "vocab.pkl"
        tokenized_dataset_path = processed_base_dir / "golden-test.tsv" if use_golden_test else processed_base_dir / gt_model_name / "perizie_bio_test.tsv"
        
        # ECCO LA CARTELLA DINAMICA CHE CHIEDEVI:
        evaluation_dir = evaluations_base_dir / f"evaluations_groundtruth_{gt_model_name}"
        evaluation_dir.mkdir(parents=True, exist_ok=True)

        if not vocab_path.exists():
            LOGGER.warning(f"Vocab mancante per {gt_model_name}. Salto...")
            continue
        if not tokenized_dataset_path.exists():
            LOGGER.warning(f"File di test mancante per {gt_model_name}. Controlla il file golden-test.tsv esista e che il flag --use-golden-test sia impostato correttamente. Salto...")
            continue

        with open(vocab_path, "rb") as f:
            vocab = pickle.load(f)

        model_files = sorted(gt_folder.glob("*_model.pth"))
        if not model_files:
            LOGGER.warning(f"Nessun modello trovato in {gt_folder}. Salto...")
            continue

        summary_rows = []
        
        # --- CICLO SUI MODELLI ADDESTRATI (BiLSTM, BERT, BioBERT, Stanza) ---
        for model_path in model_files:
            model_id = model_path.name.replace("_model.pth", "")
            LOGGER.info(f"\n{'='*50}\nValutazione Modello: {model_id.upper()}\n{'='*50}")

            # Carichiamo il dizionario salvato in fase di training
            checkpoint = torch.load(model_path, map_location=dev, weights_only=False)

            if model_id == "stanza":
                LOGGER.info("Valutazione Stanza (non-tensor)...")
                model = StanzaMedicalNER(custom_model_path=model_path)
                df_test = pd.read_csv(tokenized_dataset_path, sep="\t", keep_default_na=False, dtype={"Token": str})
                
                all_labels_tags = []
                all_preds_tags = []
                errors = []
                
                for _, group in df_test.groupby("Sentence_ID", sort=False):
                    tokens = group["Token"].tolist()
                    raw_text = " ".join(tokens)
                    true_tags = group["BIO_Tag"].tolist()
                    
                    # Inferenza Stanza
                    entities = model.predict([raw_text])[0]
                    pred_tags = stanza_entities_to_bio(tokens, entities)
                    
                    aligned_true = [align_gold_label_for_model(t, gt_model_name.lower()) for t in true_tags]
                    aligned_pred = [align_gold_label_for_model(p, gt_model_name.lower()) for p in pred_tags]
                    
                    all_labels_tags.append(aligned_true)
                    all_preds_tags.append(aligned_pred)
                    
                    errors.extend(extract_error_logs(
                        tokens=tokens,
                        true_tags=aligned_true,
                        pred_tags=aligned_pred
                    ))
            else:
                # --- SETUP MODELLI PYTORCH ---
                match(model_id):
                    case "bert_ner" | "biobert_ner":
                        canonical_model = CANONICAL_MODELS["bert"] if model_id == "bert_ner" else CANONICAL_MODELS["biobert"]
                        dataset = TransformerNERDataset(
                            file_path=tokenized_dataset_path, 
                            model_name=canonical_model,
                            hf_token=huggingface_api_key,
                            gt_model_name=gt_model_name,
                            vocab=vocab
                        )
                        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=transformer_collate)
                    case _:
                        dataset = NERDataset(
                            tokenized_dataset_path, 
                            vocab=vocab, 
                            gt_model_name=gt_model_name
                        )
                        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=pad_collate)
                
                # Inizializziamo il modello PyTorch
                model = get_pytorch_model(
                    model_id=model_id,
                    vocab_size=len(vocab.word2idx),
                    num_classes=len(vocab.tag2idx),
                    embedding_dim=args.embedding_dim,
                    hidden_dim=args.hidden_dim,
                    hf_token=huggingface_api_key
                ).to(dev)

                # Carichiamo i pesi
                if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                    model.load_state_dict(checkpoint["model_state_dict"])
                elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                    model.load_state_dict(checkpoint)
                
                # Lanciamo l'inferenza usando la funzione riadattata
                all_labels_tags, all_preds_tags, errors = evaluate_and_collect_errors(model, dataloader, dev, vocab, gt_model_name)

            # ==========================================
            # CALCOLO METRICHE (COMUNE A TUTTI I MODELLI)
            # ==========================================
            flat_labels = [tag for sentence in all_labels_tags for tag in sentence]
            flat_preds = [tag for sentence in all_preds_tags for tag in sentence]

            all_labels_sorted = sorted(set(flat_labels) | set(flat_preds))
            entity_labels = [
                label for label in all_labels_sorted
                if label not in {"O", vocab.pad_token}
            ]

            if not entity_labels:
                LOGGER.warning(f"Nessuna entità valida trovata o predetta per {model_id}. Metriche impostate a 0.")
                summary_rows.append({
                    "model_id": model_id,
                    "accuracy": 0.0,
                    "recall": 0.0,
                    "precision": 0.0,
                    "macro_f1_no_o": 0.0,
                    "num_errors": len(errors)
                })
                continue
                
            report_str = classification_report(
                all_labels_tags,
                all_preds_tags,
                zero_division=0,
            )
            raw_report = classification_report(
                all_labels_tags, 
                all_preds_tags, 
                zero_division=0, 
                output_dict=True
            )

            report_dict = cast(dict, raw_report)
            # Estraiamo le metriche macro (che già escludono le "O" di default in seqeval)
            macro_f1 = report_dict['macro avg']['f1-score']
            recall = report_dict['macro avg']['recall']
            precision = report_dict['macro avg']['precision']
            # Calcoliamo l'Accuracy totale (inclusi i tag "O") usando l'apposita funzione di seqeval
            accuracy = accuracy_score(all_labels_tags, all_preds_tags)
            
            LOGGER.info(f"Classification Report ({model_id.upper()}):\n{report_str}")
            
            # Matrice di Confusione FULL
            cm_full_path = evaluation_dir / f"{model_id}_confusion_matrix_full.png"
            confusion_matrix_figure(cm_full_path, 
                        flat_labels,
                        flat_preds,
                        all_labels_sorted, 
                        title=f"Full Confusion Matrix - {model_id.upper()} ({gt_model_name.upper()})")

            true_entities = set(flat_labels) - {"O", vocab.pad_token}
            
            if not true_entities:
                LOGGER.warning(f"Attenzione: Nessuna entità vera trovata per {model_id}! Impossibile generare la matrice ENTITY.")
                summary_rows.append({
                    "model_id": model_id,
                    "accuracy": 0.0,
                    "recall": 0.0,
                    "precision": 0.0,
                    "macro_f1_no_o": 0.0,
                    "num_errors": len(errors)
                })
                LOGGER.info(
                    "\n%s",
                    format_pipeline_step04_summary(
                        model_id=model_id,
                        model_path=str(model_path),
                        cm_full_path="Not Generated (No Entities)",
                        cm_entity_path="Not Generated (No Entities)",
                        macro_f1=0.0,
                        accuracy=0.0
                    )
                )
                continue
            else:
                cm_entity_path = evaluation_dir / f"{model_id}_confusion_matrix_entity.png"
                confusion_matrix_figure(cm_entity_path, 
                                        flat_labels,
                                        flat_preds,
                                        entity_labels, 
                                        title=f"Entity Confusion Matrix - {model_id.upper()} ({gt_model_name.upper()})")

            # Estrazione Errori CSV
            output_path = evaluation_dir / f"{model_id}_error_analysis.csv"
            with open(output_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["token", "true_tag", "predicted_tag", "error_type", "sentence"])
                writer.writeheader()
                writer.writerows(errors)

            LOGGER.info(f"Saved {len(errors)} errors to {output_path}")

            summary_rows.append({
                "model_id": model_id,
                "accuracy": accuracy,
                "recall": recall,
                "precision": precision,
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

        # Salvataggio del Summary Globale per QUESTA Ground Truth (nella sua cartella!)
        summary_path = evaluation_dir / "models_evaluation_summary.csv"
        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["model_id", "accuracy", "recall", "precision", "macro_f1_no_o", "num_errors"],
            )
            writer.writeheader()
            writer.writerows(summary_rows)

        LOGGER.info(f"[{gt_model_name.upper()}] Saved evaluation summary to {summary_path}")


