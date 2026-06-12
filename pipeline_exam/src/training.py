from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path

import os
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.optim as optim
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch import device

from pipeline_exam.src.utils import configure_logging, load_dotenv_file
from pipeline_exam.src.models import get_model
from pipeline_exam.src.NERDataset import TransformerNERDataset, transformer_collate, NERDataset, pad_collate

LOGGER = logging.getLogger(__name__)

def training(model : nn.Module, epochs : int, dataloader : DataLoader, dev : device, 
             optimizer : Optimizer, criterion : CrossEntropyLoss) -> float:
    model.train()
    avg_loss = 0.0
    for epoch in range(epochs):
        epoch_loss = 0.0
        
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(dev), batch_y.to(dev)
            optimizer.zero_grad()
            logits = model(batch_x)
            logits_flat = logits.view(-1, logits.shape[-1]) 
            batch_y_flat = batch_y.view(-1)                 
            
            loss = criterion(logits_flat, batch_y_flat)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(dataloader)
        LOGGER.info(f"Epoca [{epoch+1:02d}/{epochs}] - Loss Media: {avg_loss:.4f}")
    return avg_loss

def format_pipeline_step03_summary(
    *,
    model_id: str,
    huggingface_api_key_setted: str,
    tokenized_dataset_path: str,
    vocab_path: str,
    output_model_path: str,
    batch_size: int,
    epochs: int,
    lr: float,
    embedding_dim: int,
    hidden_dim: int,
    device_used: str,
    logging_level: str,
    final_loss: float
) -> str:
    summary = (
        "Pipeline Step03 summary:\n"
        f"- Model ID: {model_id}\n"
        f"- HuggingFace API Key Setted: {huggingface_api_key_setted}\n"
        f"- Tokenized Dataset Path: {tokenized_dataset_path}\n"
        f"- Vocabulary Path: {vocab_path}\n"
        f"- Output Model Path: {output_model_path}\n"
        f"- Epochs: {epochs}\n"
        f"- Batch Size: {batch_size}\n"
        f"- Learning Rate: {lr}\n"
        f"- Embedding Dim: {embedding_dim}\n"
        f"- Hidden Dim: {hidden_dim}\n"
        f"- Device: {device_used}\n"
        f"- Logging Level: {logging_level}\n"
        f"- Final Training Loss: {final_loss:.4f}"
    )
    return summary

def build_step03_parser(default_repo_root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Start Bi-LSTM Training Loop for NER",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    processed_dir = default_repo_root / "pipeline_exam" / "data" / "processed"
    models_dir = default_repo_root / "pipeline_exam" / "models"
    parser.add_argument("--model-id", type=str, default="bilstm", choices=["bilstm", "bert_ner"],
                        help="ID del modello da addestrare.")
    
    parser.add_argument("--tokenized-dataset-path", default=str(processed_dir / "perizie_bio_train.tsv"))
    
    parser.add_argument("--vocab-path", default=str(processed_dir / "vocab.pkl"))
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
    
    out_models_dir = Path(args.output_model_dir)
    out_models_dir.mkdir(parents=True, exist_ok=True)
    model_id : str = args.model_id
    embedding_dim = args.embedding_dim
    hidden_dim = args.hidden_dim
    batch_size = args.batch_size

    epochs = int(args.epochs)
    huggingface_api_key = os.environ.get("HUGGINGFACE_API_KEY")
    if not huggingface_api_key:
        LOGGER.warning("HUGGINGFACE_API_KEY not set -- unauthenticated may fail.")
    
    vocab_path = Path(args.vocab_path)
    if not vocab_path.exists():
        LOGGER.error(f"Vocabolario non trovato in {vocab_path}. Esegui prima lo script di build_vocab!")
        return
    
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)
        
    tokenized_dataset_path = Path(args.tokenized_dataset_path)

    if model_id == "bert_ner":
        dataset = TransformerNERDataset(
            file_path=tokenized_dataset_path, 
            model_name="dbmdz/bert-base-italian-cased",
            hf_token=huggingface_api_key,
            vocab=vocab
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=transformer_collate)
        criterion = nn.CrossEntropyLoss(ignore_index=-100)
        lr_to_use = 0.00002 if args.lr == 0.001 else args.lr 
    else:
        dataset = NERDataset(tokenized_dataset_path, vocab=vocab)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate)
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        lr_to_use = args.lr
    
    LOGGER.info(f"Inizializzazione della rete {model_id.upper()}...")
    model = get_model(
        model_id=model_id,
        vocab_size=len(vocab.word2idx),
        num_classes=len(vocab.tag2idx),
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        hf_token=huggingface_api_key
    ).to(dev)
    
    optimizer = optim.Adam(model.parameters(), lr=lr_to_use)
    
    LOGGER.info(f"Inizio Addestramento per {epochs} epoche...")
    final_loss = training(
        model=model,
        epochs=epochs,
        dataloader=dataloader,
        dev=dev,
        optimizer=optimizer,
        criterion=criterion
    )

    output_model_path = out_models_dir / f"{model_id}_model.pth"
    torch.save(model.state_dict(), output_model_path)
    
    LOGGER.info(
        "\n%s",
        format_pipeline_step03_summary(
            model_id=model_id,
            huggingface_api_key_setted="YES" if huggingface_api_key else "NO",
            tokenized_dataset_path=str(tokenized_dataset_path),
            vocab_path=str(vocab_path),
            output_model_path=str(output_model_path),
            batch_size=batch_size,
            epochs=epochs,
            lr=args.lr,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            device_used=dev.type.upper(),
            logging_level=args.logging_level,
            final_loss=final_loss
        ),
    )