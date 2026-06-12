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
    batch_size: int,
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
        description="Start Exam Step02 PyTorch Data Loading and Vocabulary Creation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    processed_dir = default_repo_root / "pipeline_exam" / "data" / "processed"
    
    parser.add_argument("--tokenized-dataset-path", default=str(processed_dir / "perizie_bio_train.tsv"))
    
    parser.add_argument("--output-dir", default=str(processed_dir))
    parser.add_argument("--vocab-path", default=str(processed_dir / "vocab.pkl"))
    parser.add_argument("--batch-size", type=int, default=16) 
    parser.add_argument("--logging-level", default="INFO")
    return parser

def run_step02(args: argparse.Namespace) -> None:
    configure_logging(args.logging_level)
    load_dotenv_file(".env")
    
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenized_dataset_path = Path(args.tokenized_dataset_path)
    vocab_path = Path(args.vocab_path)
    batch_size = args.batch_size

    LOGGER.info(f"Costruzione Vocabolario utilizzando ESCLUSIVAMENTE il set di addestramento: {tokenized_dataset_path.name}...")
    dataset = NERDataset(tokenized_dataset_path)
    
    with open(vocab_path, "wb") as f:
        pickle.dump(dataset.vocab, f)
    LOGGER.info(f"Vocabolario salvato in: {vocab_path}")

    LOGGER.info(
        "%s",
        format_pipeline_step02_summary(
            tokenized_dataset_path=str(tokenized_dataset_path),
            vocab_path=str(vocab_path),
            vocab_size=len(dataset.vocab.word2idx),
            num_classes=len(dataset.vocab.tag2idx),
            batch_size=int(batch_size),
            output_dir=str(out_dir)
        ),
    )