from __future__ import annotations

import argparse
import logging
from pathlib import Path
import pickle

from pipeline_exam.src.utils import configure_logging, load_dotenv_file
from pipeline_exam.src.NERDataset import NERDataset

LOGGER = logging.getLogger(__name__)

def format_pipeline_step02_summary(
    *,
    tokenized_dataset_path: str,
    vocab_path: str,
    vocab_size: int,
    num_classes: int,
    output_dir: str
    ) -> str:
    summary = (
        "Pipeline Step02 summary:\n"
        f"- Train Dataset Path (for Vocab): {tokenized_dataset_path}\n"
        f"- Vocabulary Path: {vocab_path}\n"
        f"- Vocabulary Size: {vocab_size} unique tokens\n"
        f"- Classes Detected: {num_classes} BIO tags\n"
        f"- Output Directory: {output_dir}"
    )
    return summary

def build_step02_parser(default_repo_root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Step02 Vocabulary Creation from Training BIO Dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    processed_dir = default_repo_root / "pipeline_exam" / "data" / "processed"
    parser.add_argument("--output-dir", default=str(processed_dir))
    parser.add_argument("--logging-level", default="INFO")
    return parser

def run_step02(args: argparse.Namespace) -> None:
    configure_logging(args.logging_level)
    load_dotenv_file(".env")
    
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_files = list(out_dir.glob("*/perizie_bio_train.tsv"))
    
    if not train_files:
        LOGGER.error(f"Nessun file 'perizie_bio_train.tsv' trovato nelle sottocartelle di {out_dir}!")
        return
    
    LOGGER.info(f"Trovate {len(train_files)} cartelle di dataset da processare. Inizio generazione vocabolari...")

    for train_path in train_files:
        current_folder = train_path.parent
        vocab_path = current_folder / "vocab.pkl"
        
        LOGGER.info(f"[{current_folder.name.upper()}] Costruzione Vocabolario...")
        gt_model_name = current_folder.name 
        dataset = NERDataset(train_path, gt_model_name=gt_model_name)
        
        with open(vocab_path, "wb") as f:
            pickle.dump(dataset.vocab, f)
            
        LOGGER.info(f"[{current_folder.name.upper()}] Vocabolario salvato in: {vocab_path}")
        LOGGER.info(
            "\n%s",
            format_pipeline_step02_summary(
                tokenized_dataset_path=str(train_path),
                vocab_path=str(vocab_path),
                vocab_size=len(dataset.vocab.word2idx),
                num_classes=len(dataset.vocab.tag2idx),
                output_dir=str(current_folder)
            ),
        )
        LOGGER.info("-" * 50) # Riga di separazione per i log