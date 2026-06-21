from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, Tuple

import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import subprocess
import sys

from pipeline_exam.src.utils import configure_logging, load_dotenv_file
from pipeline_exam.src.models import get_pytorch_model
from pipeline_exam.src.NERDataset import TransformerNERDataset, transformer_collate, NERDataset, pad_collate
from pipeline_exam.src.schemas import CANONICAL_MODELS


LOGGER = logging.getLogger(__name__)

def training(
    model: nn.Module, 
    epochs: int, 
    train_dataloader: DataLoader, 
    val_dataloader: DataLoader, 
    dev: torch.device, 
    optimizer: Optimizer
) -> Tuple[Dict[str, Any] | None, float, float]:
    
    best_val_loss = float('inf')
    best_model_state = None
    avg_train_loss = 0.0

    for epoch in range(epochs):
        # ==========================================
        # 1. FASE DI TRAINING
        # ==========================================
        model.train()
        epoch_train_loss = 0.0
        
        for batch_x, batch_y, _, _ in train_dataloader:
            # Spostamento dati sul device
            if isinstance(batch_x, dict):
                batch_x = {k: v.to(dev) for k, v in batch_x.items()}
            else:
                batch_x = batch_x.to(dev)
            
            batch_y = batch_y.to(dev)

            optimizer.zero_grad()

            # FORWARD UNIFICATO
            # Passiamo le labels direttamente al modello.
            # Il modello calcolerà la loss al suo interno.
            if isinstance(batch_x, dict):
                outputs = model(**batch_x, labels=batch_y)
            else:
                outputs = model(input_ids=batch_x, labels=batch_y)

            # Estrazione della loss dal dizionario standardizzato
            loss = outputs["loss"]
            
            # Sicurezza extra nel caso in cui tu usi DataParallel (più GPU) in futuro:
            # DataParallel restituisce un vettore di loss, quindi prendiamo la media
            if loss.dim() > 0:
                loss = loss.mean()

            # Backpropagation
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
            
        avg_train_loss = epoch_train_loss / len(train_dataloader)
        
        # ==========================================
        # 2. FASE DI VALIDATION
        # ==========================================
        model.eval()
        epoch_val_loss = 0.0
        
        with torch.no_grad():
            for batch_x, batch_y, _, _ in val_dataloader:
                # Spostamento dati sul device
                if isinstance(batch_x, dict):
                    batch_x = {k: v.to(dev) for k, v in batch_x.items()}
                else:
                    batch_x = batch_x.to(dev)
                
                batch_y = batch_y.to(dev)

                # FORWARD UNIFICATO (Identico al training)
                if isinstance(batch_x, dict):
                    outputs = model(**batch_x, labels=batch_y)
                else:
                    outputs = model(input_ids=batch_x, labels=batch_y)

                loss = outputs["loss"]
                
                if loss.dim() > 0:
                    loss = loss.mean()
                    
                epoch_val_loss += loss.item()
                
        avg_val_loss = epoch_val_loss / len(val_dataloader)
        
        LOGGER.info(f"Epoca [{epoch+1:02d}/{epochs}] - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        # ==========================================
        # 3. MODEL CHECKPOINTING
        # ==========================================
        if avg_val_loss < best_val_loss:
            LOGGER.info(f"✨ Validation Loss migliorata ({best_val_loss:.4f} -> {avg_val_loss:.4f}). Salvataggio modello in corso...")
            best_val_loss = avg_val_loss
            best_model_state = copy.deepcopy(model.state_dict())

    return best_model_state, avg_train_loss, best_val_loss

def format_pipeline_step03_summary(
    *,
    model_id: str,
    huggingface_api_key_setted: str,
    train_path: str,
    val_path: str,
    vocab_path: str,
    output_model_path: str,
    batch_size: int,
    epochs: int,
    lr: float,
    embedding_dim: int,
    hidden_dim: int,
    device_used: str,
    logging_level: str,
    final_train_loss: float,
    best_val_loss: float
) -> str:
    summary = (
        "Pipeline Step03 summary:\n"
        f"- Model ID: {model_id}\n"
        f"- HuggingFace API Key Set: {huggingface_api_key_setted}\n"
        f"- Train Dataset Path: {train_path}\n"
        f"- Val Dataset Path: {val_path}\n"
        f"- Vocabulary Path: {vocab_path}\n"
        f"- Output Model Path: {output_model_path}\n"
        f"- Epochs: {epochs}\n"
        f"- Batch Size: {batch_size}\n"
        f"- Embedding Dimension: {embedding_dim}\n"
        f"- Hidden Layers Dimension: {hidden_dim}\n"
        f"- Learning Rate: {lr}\n"
        f"- Device: {device_used}\n"
        f"- Final Train Loss: {final_train_loss:.4f}\n"
        f"- Logging Level: {logging_level}\n"
        f"- Best Validation Loss: {best_val_loss:.4f}"
    )
    return summary

def build_step03_parser(default_repo_root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Start Model Training Loop for NER",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    processed_dir = default_repo_root / "pipeline_exam" / "data" / "processed"
    models_dir = default_repo_root / "pipeline_exam" / "models"
    
    parser.add_argument("--model-id", type=str, default="bilstm", choices=["bilstm", "bilstm_crf", "bert_ner", "biobert_ner", "stanza"],
                        help="ID del modello da addestrare.")
    
    parser.add_argument("--tokenized-train-path", default=None)
    parser.add_argument("--tokenized-val-path", default=None)
    parser.add_argument("--vocab-path", default=None)
    
    parser.add_argument("--processed-dir", default=str(processed_dir))
    parser.add_argument("--output-model-dir", default=str(models_dir))
    
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=0.001) # Learning Rate
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--logging-level", default="INFO")
    return parser

def run_step03(args: argparse.Namespace) -> None:
    configure_logging(args.logging_level)
    load_dotenv_file(".env")
    
    dev = torch.device(args.device)
    LOGGER.info(f"🚀 Avvio training sul dispositivo: {dev.type.upper()}")
    
    model_id: str = args.model_id
    embedding_dim = args.embedding_dim
    hidden_dim = args.hidden_dim
    batch_size = args.batch_size
    epochs = int(args.epochs)
    lr_to_use = args.lr
    output_model_dir = Path(args.output_model_dir)
    
    huggingface_api_key = os.environ.get("HUGGINGFACE_API_KEY")
    if not huggingface_api_key:
        LOGGER.warning("HUGGINGFACE_API_KEY not set -- unauthenticated may fail.")

    tasks = []
    
    # Se l'utente ha passato TUTTI i percorsi specifici, facciamo una singola run
    if args.tokenized_train_path and args.tokenized_val_path and args.vocab_path:
        tasks.append({
            "train": Path(args.tokenized_train_path),
            "val": Path(args.tokenized_val_path),
            "vocab": Path(args.vocab_path),
            "out_dir": output_model_dir
        })
    else:
        # Altrimenti, Auto-Discovery su tutte le sottocartelle
        processed_dir = Path(args.processed_dir)
        if not processed_dir.exists():
            LOGGER.error(f"Directory non trovata: {processed_dir}")
            return
            
        train_files = list(processed_dir.glob("*/perizie_bio_train.tsv"))
        if not train_files:
            LOGGER.error("Nessun dataset di training trovato per l'auto-discovery.")
            return
            
        for t_path in train_files:
            folder = t_path.parent
            tasks.append({
                "train": t_path,
                "val": folder / "perizie_bio_val.tsv",
                "vocab": folder / "vocab.pkl",
                "out_dir": output_model_dir / folder.name
            })
            
    LOGGER.info(f"Trovate {len(tasks)} configurazioni di dataset. Inizio ciclo di addestramento...")
    final_train_loss = 0.0
    best_val_loss = 0.0
    output_model_path = ""

    for task in tasks:
        train_dataset_path = task["train"]
        val_dataset_path = task["val"]
        vocab_path = task["vocab"]
        out_models_dir = task["out_dir"]
        
        # Crea la cartella di output se non esiste (es: models/scibc5cdr)
        out_models_dir.mkdir(parents=True, exist_ok=True)
        
        dataset_name = train_dataset_path.parent.name
        LOGGER.info(f"\n{'='*50}\nAddestramento Modello: {model_id.upper()} su Dataset: {dataset_name.upper()}\n{'='*50}")

        if not vocab_path.exists() or not train_dataset_path.exists() or not val_dataset_path.exists():
            LOGGER.error(f"File mancanti per il dataset {dataset_name}. Salto questa configurazione...")
            continue
            
        with open(vocab_path, "rb") as f:
            vocab = pickle.load(f)

        canonical_model = CANONICAL_MODELS["bert"] if model_id == "bert_ner" else CANONICAL_MODELS["biobert"]
        
        # Inizializzazione Dataloaders
        match(model_id):
            case "bert_ner" | "biobert_ner":
                train_dataset = TransformerNERDataset(
                    file_path=train_dataset_path, 
                    model_name=canonical_model, 
                    hf_token=huggingface_api_key, 
                    vocab=vocab,
                    gt_model_name=dataset_name
                    )
                train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=transformer_collate)
                val_dataset = TransformerNERDataset(
                    file_path=val_dataset_path, 
                    model_name=canonical_model, 
                    hf_token=huggingface_api_key, 
                    vocab=vocab,
                    gt_model_name=dataset_name
                    )
                val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=transformer_collate)
                lr_to_use = 0.00002 if args.lr == 0.001 else args.lr
            case "stanza":
                LOGGER.info(f"[{dataset_name.upper()}] Avvio Training di Stanza (Transformer) sui file JSON...")
                
                json_train_path = (train_dataset_path.parent / "stanza_train.json").resolve()
                json_val_path = (val_dataset_path.parent / "stanza_val.json").resolve()
                out_models_dir_abs = out_models_dir.resolve()
                
                if not json_train_path.exists() or not json_val_path.exists():
                    LOGGER.error(f"File JSON per Stanza mancanti in {json_train_path.parent}!")
                    continue

                stanza_custom_model_name = f"{model_id}_model.pth"
                
                stanza_cmd = [
                    sys.executable, "-m", "stanza.models.ner_tagger",
                    "--train_file", str(json_train_path),
                    "--eval_file", str(json_val_path),
                    "--mode", "train",
                    "--shorthand", "en_custom",
                    "--bert_model", "dmis-lab/biobert-v1.1",
                    "--no_pretrain",
                    "--save_dir", str(out_models_dir_abs),
                    "--save_name", stanza_custom_model_name,
                    "--max_steps", str(epochs * sum(1 for line in open(train_dataset_path, encoding="utf-8") if not line.strip()) // batch_size), # brutale mi piace
                    "--max_steps_no_improve", "1000",
                    "--hidden_dim", str(hidden_dim),
                    "--cuda" if dev.type == "cuda" else "--cpu",
                    "--batch_size", str(batch_size),
                    "--lr", str(lr_to_use),
                    "--bert_learning_rate", "5e-5",
                    "--eval_interval", "50",
                    "--optim", "adamw",
                    "--bert_finetune"
                ]
                
                result = subprocess.run(stanza_cmd, capture_output=False)
                
                if result.returncode == 0:
                    LOGGER.info(f"[{dataset_name.upper()}] Addestramento Stanza completato con successo!")
                    output_model_path = out_models_dir / stanza_custom_model_name
                else:
                    LOGGER.error(f"[{dataset_name.upper()}] Errore durante l'addestramento di Stanza. Controlla i log.")
            case _:
                train_dataset = NERDataset(
                    train_dataset_path, 
                    vocab=vocab,
                    gt_model_name=dataset_name
                    )
                train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate)
                val_dataset = NERDataset(
                    val_dataset_path, 
                    vocab=vocab,
                    gt_model_name=dataset_name
                )
                val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=pad_collate)


        if model_id != "stanza":
            # Inizializzazione Modello e Ottimizzatore (Ricreati da zero per ogni dataset!)
            model = get_pytorch_model(
                model_id=model_id,
                vocab_size=len(vocab.word2idx),
                num_classes=len(vocab.tag2idx),
                embedding_dim=embedding_dim,
                hidden_dim=hidden_dim,
                hf_token=huggingface_api_key
            ).to(dev)

            optimizer = (
                optim.AdamW(model.parameters(), lr=lr_to_use, weight_decay=0.01)
                if model_id in {"bert_ner", "biobert_ner"}
                else optim.Adam(model.parameters(), lr=lr_to_use)
            )
            
            output_model_path = out_models_dir / f"{model_id}_model.pth"
            
            LOGGER.info(f"Inizio Addestramento per {epochs} epoche...")
            best_model_state, final_train_loss, best_val_loss = training(
                model=model,
                epochs=epochs,
                train_dataloader=train_dataloader,
                val_dataloader=val_dataloader,
                dev=dev,
                optimizer=optimizer
            )

            if best_model_state is not None:
                torch.save({
                    "model_state_dict": best_model_state,
                    "model_id": model_id,
                    "vocab": vocab,
                    "num_classes": len(vocab.tag2idx),
                    "embedding_dim": embedding_dim,
                    "hidden_dim": hidden_dim,
                    "best_val_loss": best_val_loss,
                    "canonical_model": canonical_model if model_id in {"bert_ner", "biobert_ner"} else None
                }, output_model_path)
                LOGGER.info(f"[{dataset_name.upper()}] Modello salvato su {output_model_path}")

        LOGGER.info(
            "\n%s",
            format_pipeline_step03_summary(
                model_id=model_id,
                huggingface_api_key_setted="YES" if huggingface_api_key else "NO",
                train_path=str(train_dataset_path),
                val_path=str(val_dataset_path),
                vocab_path=str(vocab_path),
                output_model_path=str(output_model_path),
                device_used=dev.type.upper(),
                logging_level=args.logging_level,
                batch_size=batch_size,
                epochs=epochs,
                lr=lr_to_use,
                hidden_dim=hidden_dim,
                embedding_dim=embedding_dim if model_id != "stanza" else 0.0,
                final_train_loss=final_train_loss if model_id != "stanza" else 0.0,
                best_val_loss=best_val_loss if model_id != "stanza" else 0.0
            )
        )