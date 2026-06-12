from __future__ import annotations

import argparse
import logging
from pathlib import Path
import pandas as pd
from datasets import load_dataset
from huggingface_hub import login
import os

from pipeline_exam.src.utils import configure_logging, load_dotenv_file

LOGGER = logging.getLogger(__name__)

def format_pipeline_step00_summary(
    *,
    dataset_name: str,
    dataset_path: str,
    num_samples: int,
    output_dir: str
) -> str:
    summary = (
        "Pipeline Step00 summary (Hugging Face Data Ingestion):\n"
        f"- Source Dataset: {dataset_name}\n"
        f"- Created Dataset CSV: {dataset_path}\n"
        f"- Samples Extracted: {num_samples}\n"
        f"- Output Directory: {output_dir}"
    )
    return summary

def build_step00_parser(default_repo_root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Start Exam Step00 Data Ingestion from Hugging Face",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    raw_dir = default_repo_root / "pipeline_exam" / "data" / "raw"
    parser.add_argument("--dataset-path", default=str(raw_dir / "perizie_mediche_sintetiche.csv"))
    parser.add_argument("--num-samples", type=int, default=2000, help="Numero di frasi da scaricare")
    parser.add_argument("--hf-dataset", type=str, default="bc5cdr", help="Nome del dataset su HF")
    parser.add_argument("--logging-level", default="INFO")
    
    return parser

def run_step00(args: argparse.Namespace) -> None:
    configure_logging(args.logging_level)
    load_dotenv_file(".env")
    dataset_path = Path(args.dataset_path)
    out_dir = dataset_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    huggingface_api_key = os.environ.get("HUGGINGFACE_API_KEY")
    if huggingface_api_key:
        LOGGER.info("Autenticazione con Hugging Face in corso...")
        login(token=huggingface_api_key)
    else:
        LOGGER.warning("HUGGINGFACE_API_KEY non trovata. Il download potrebbe fallire se il dataset è privato.")

    LOGGER.info(f"Scaricamento del dataset '{args.hf_dataset}' da Hugging Face...")
    hf_dataset = load_dataset(args.hf_dataset)
    
    data_rows = []
    total_extracted = 0
    
    for split_name in hf_dataset.keys():
        if "en" not in split_name: 
            continue
        LOGGER.info(f"Processing split: {split_name}...")
        split_data = hf_dataset[split_name]
        limit = min(len(split_data), args.num_samples)
        for i in range(limit):
            row = split_data[i]
            tokens = row.get('tokens')
            if tokens:
                testo_intero = " ".join(tokens)
                data_rows.append({
                    "id_referto": f"HF_{split_name}_{i:05d}",
                    "testo_perizia": testo_intero
                })
                total_extracted += 1
    df = pd.DataFrame(data_rows)
    df.to_csv(dataset_path, index=False)

    LOGGER.info("\n%s", format_pipeline_step00_summary(
        dataset_name=args.hf_dataset,
        dataset_path=str(dataset_path),
        num_samples=total_extracted,
        output_dir=str(out_dir)
    ))