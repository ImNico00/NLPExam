from __future__ import annotations

from html import parser
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

from sklearn.metrics import classification_report, confusion_matrix # type: ignore
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from pipeline_exam.src.utils import configure_logging, load_dotenv_file, extract_error_logs, stanza_entities_to_bio, align_gold_label_for_model
from pipeline_exam.src.models import get_model, BiLSTM_CRF_NER
from pipeline_exam.src.NERDataset import NERDataset, TransformerNERDataset, Vocabulary, pad_collate, transformer_collate
from pipeline_exam.src.schemas import CANONICAL_MODELS

LOGGER = logging.getLogger(__name__)

def evaluate_and_collect_errors(model: nn.Module, dataloader: DataLoader, dev: torch.device, vocab : Vocabulary, gt_model_name: str) -> tuple[list, list, list[dict]]:
    """
    Esegue l'inferenza sul test set, restituendo tag veri, predetti e log degli errori
    comprensivi della frase intera di contesto (senza ripetizioni di sub-token).
    """
    model.eval()
    
    all_labels_tags = []
    all_preds_tags = []
    errors = []

    # Controllo dinamico: stiamo usando il CRF?
    is_crf_model = isinstance(model, BiLSTM_CRF_NER)

    with torch.no_grad():
        for batch_x, batch_y, flat_raw_words in dataloader:
            
            # 1. Spostiamo i tensori sul device
            if isinstance(batch_x, dict):
                batch_x = {k: v.to(dev) for k, v in batch_x.items()}
            else:
                batch_x = batch_x.to(dev)
            batch_y = batch_y.to(dev)

            # 2. Otteniamo le predizioni (Viterbi vs Argmax)
            if is_crf_model:
                # Inferenza CRF con Algoritmo di Viterbi
                # Ritorna una lista di liste (escludendo i [PAD] mascherati)
                assert isinstance(batch_x, torch.Tensor), "batch_x deve essere un Tensor per il modello CRF"
                best_paths = model.predict(batch_x)
                
                # Ricreiamo un tensore "preds" delle stesse dimensioni di batch_y
                # riempito inizialmente con il padding index per compatibilità con il tuo loop
                preds = torch.full_like(batch_y, fill_value=vocab.pad_tag_idx)
                
                # Inseriamo i percorsi predetti nel tensore
                for b_idx, path in enumerate(best_paths):
                    path_len = len(path)
                    preds[b_idx, :path_len] = torch.tensor(path, device=dev)
                    
            else:
                # Inferenza standard per BiLSTM pura o Transformer
                if isinstance(batch_x, dict):
                    logits = model(**batch_x)
                else:
                    logits = model(batch_x)
                    
                if hasattr(logits, "logits"):
                    logits = logits.logits
                
                preds = torch.argmax(logits, dim=-1)

            batch_size, seq_len = batch_y.shape

            for b in range(batch_size):
                # Usiamo liste temporanee per la singola frase
                valid_tokens = []
                valid_true_tags = []
                valid_pred_tags = []

                for s in range(seq_len):
                    true_id = int(batch_y[b, s].item())
                    pred_id = int(preds[b, s].item())
                    
                    # Ignoriamo il padding e i sub-token mascherati (-100)
                    if true_id != vocab.pad_tag_idx and true_id != -100:
                        true_tag = vocab.idx2tag[true_id]
                        pred_tag = vocab.idx2tag[pred_id]
                        real_word = flat_raw_words[b * seq_len + s]

                        true_tag = align_gold_label_for_model(true_tag, gt_model_name)
                        pred_tag = align_gold_label_for_model(pred_tag, gt_model_name)

                        valid_tokens.append(real_word)
                        valid_true_tags.append(true_tag)
                        valid_pred_tags.append(pred_tag)

                        # Popoliamo le metriche globali (Accuracy, F1, ecc.)
                        all_labels_tags.append(true_tag)
                        all_preds_tags.append(pred_tag)

                # Ora che abbiamo allineato tutto per questa frase, usiamo la funzione di utility
                # Nota: la funzione ricostruirà internamente la frase con " ".join(valid_tokens)
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
    
    # Non chiediamo più i percorsi esatti dei file, ma solo le cartelle base!
    parser.add_argument("--processed-dir", default=str(processed_dir), help="Cartella con i test set e i vocab.pkl")
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

    # Trova tutte le sottocartelle dentro models/ (es: scibc5cdr, dictionary, scibert)
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
                model = get_model(
                    model_id=model_id,
                    custom_model_path=model_path,
                    vocab_size=len(vocab.word2idx),
                    num_classes=len(vocab.tag2idx),
                    hf_token=huggingface_api_key
                )
                df_test = pd.read_csv(tokenized_dataset_path, sep="\t")
                
                all_labels_tags = []
                all_preds_tags = []
                errors = []
                
                for _, group in df_test.groupby("Sentence_ID"):
                    tokens = group["Token"].tolist()
                    raw_text = " ".join(tokens)
                    true_tags = group["BIO_Tag"].tolist()
                    
                    # Inferenza Stanza
                    entities = model([raw_text])[0]
                    pred_tags = stanza_entities_to_bio(tokens, entities)
                    
                    aligned_true = [align_gold_label_for_model(t, gt_model_name) for t in true_tags]
                    aligned_pred = [align_gold_label_for_model(p, gt_model_name) for p in pred_tags]
                    
                    all_labels_tags.extend(aligned_true)
                    all_preds_tags.extend(aligned_pred)
                    
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
                model = get_model(
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
            all_labels_sorted = sorted(set(all_labels_tags) | set(all_preds_tags))
            entity_labels = [
                label for label in all_labels_sorted
                if label not in {"O", vocab.pad_token}
            ]
                
            report_str = classification_report(
                all_labels_tags,
                all_preds_tags,
                labels=entity_labels,
                zero_division=0,
            )
            raw_report = classification_report(
                all_labels_tags, 
                all_preds_tags, 
                labels=entity_labels, 
                zero_division=0, 
                output_dict=True
            )
            report_dict = cast(dict, raw_report)
            macro_f1 = report_dict['macro avg']['f1-score']
            
            # Per l'Accuracy consideriamo tutti i tag, inclusi "O"
            full_raw_report_dict = classification_report(all_labels_tags, all_preds_tags, zero_division=0, output_dict=True)
            full_report_dict = cast(dict, full_raw_report_dict)
            accuracy = full_report_dict['accuracy']
            
            LOGGER.info(f"Classification Report ({model_id.upper()}):\n{report_str}")
            
            # Matrice di Confusione FULL
            cm_full = confusion_matrix(all_labels_tags, all_preds_tags, labels=all_labels_sorted)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm_full, annot=True, fmt="d", cmap="Blues", xticklabels=all_labels_sorted, yticklabels=all_labels_sorted)
            plt.title(f"Full Confusion Matrix - {model_id.upper()} ({gt_model_name.upper()})")
            plt.ylabel('True BIO Tag')
            plt.xlabel('Predicted BIO Tag')
            
            cm_full_path = evaluation_dir / f"{model_id}_confusion_matrix_full.png"
            plt.savefig(cm_full_path, bbox_inches="tight")
            plt.close()

            # Matrice di Confusione ENTITY
            cm_entity = confusion_matrix(all_labels_tags, all_preds_tags, labels=entity_labels)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm_entity, annot=True, fmt="d", cmap="Blues", xticklabels=entity_labels, yticklabels=entity_labels)
            plt.title(f"Entity Confusion Matrix - {model_id.upper()} ({gt_model_name.upper()})")
            plt.ylabel('True BIO Tag')
            plt.xlabel('Predicted BIO Tag')
            
            cm_entity_path = evaluation_dir / f"{model_id}_confusion_matrix_entity.png"
            plt.savefig(cm_entity_path, bbox_inches="tight")
            plt.close()

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
                fieldnames=["model_id", "accuracy", "macro_f1_no_o", "num_errors"],
            )
            writer.writeheader()
            writer.writerows(summary_rows)

        LOGGER.info(f"[{gt_model_name.upper()}] Saved evaluation summary to {summary_path}")