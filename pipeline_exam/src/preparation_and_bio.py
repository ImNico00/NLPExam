from __future__ import annotations

import argparse
import logging
from pathlib import Path
import spacy
from spacy.language import Language
from sklearn.model_selection import train_test_split #type: ignore
import pandas as pd

from pipeline_exam.src.utils import configure_logging

LOGGER = logging.getLogger(__name__)

def initialize_ner_dataset(nlp: Language, df: pd.DataFrame) -> pd.DataFrame:
    required_columns = {"id_referto", "testo_perizia"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    sentences_array = []
    for _, row in df.iterrows():
        id_referto = row['id_referto']
        testo = str(row["testo_perizia"]) if pd.notna(row["testo_perizia"]) else ""
        doc = nlp(testo)
        for token in doc:
            if not token.is_space:
                if token.ent_iob_ == "O" or not token.ent_type_:
                    bio_tag = "O"
                else:
                    bio_tag = f"{token.ent_iob_}-{token.ent_type_}"
                sentences_array.append([id_referto, token.text, bio_tag])
        sentences_array.append(["", "", ""])
    return pd.DataFrame(sentences_array, columns=["Sentence_ID", "Token", "BIO_Tag"])

def format_pipeline_step01_summary(
    *,
    dataset_path: str,
    train_path: str,
    val_path: str,
    test_path: str,
    size_train: int,
    size_val: int,
    size_test: int,
    output_dir: str
) -> str:
    summary = (
        "Pipeline Step01 summary (Data Split & Tokenization):\n"
        f"- Original Dataset: {dataset_path}\n"
        f"- Train TSV Path: {train_path} ({size_train} referti)\n"
        f"- Validation TSV Path: {val_path} ({size_val} referti)\n"
        f"- Test TSV Path: {test_path} ({size_test} referti)\n"
        f"- Output Directory: {output_dir}"
    )
    return summary

def build_step01_parser(default_repo_root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Step01 Data Split, Tokenization and Automatic BIO Annotation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    processed_dir = default_repo_root / "pipeline_exam" / "data" / "processed"
    raw_dir = default_repo_root / "pipeline_exam" / "data" / "raw"
    parser.add_argument("--dataset-path", default=str(raw_dir / "medical_reports_english_translated.csv"))
    parser.add_argument("--output-dir", default=str(processed_dir))
    parser.add_argument("--train-output", default=str(processed_dir / "perizie_bio_train.tsv"))
    parser.add_argument("--val-output", default=str(processed_dir / "perizie_bio_val.tsv"))
    parser.add_argument("--test-output", default=str(processed_dir / "perizie_bio_test.tsv"))
    parser.add_argument("--seed-split", default=int(42))
    parser.add_argument("--logging-level", default="INFO")
    return parser

def run_step01(args: argparse.Namespace) -> None:
    configure_logging(args.logging_level)
    
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = Path(args.dataset_path)
    train_output = Path(args.train_output)
    val_output = Path(args.val_output)
    test_output = Path(args.test_output)
    seed = args.seed_split

    LOGGER.info("Caricamento del modello medico pre-addestrato (ScispaCy BC5CDR)...")
    nlp = spacy.load("en_ner_bc5cdr_md")

    LOGGER.info("Lettura del dataset grezzo...")
    df = pd.read_csv(dataset_path)

    LOGGER.info("Splitting del dataset (80% Train, 10% Val, 10% Test)...")
    df_train, df_temp = train_test_split(df, test_size=0.2, random_state=seed)
    df_val, df_test = train_test_split(df_temp, test_size=0.5, random_state=seed)

    LOGGER.info("Estrazione automatica delle entità (DISEASE, CHEMICAL) e BIO tagging...")
    df_train_tokens = initialize_ner_dataset(nlp, df_train)
    df_train_tokens.to_csv(train_output, sep='\t', index=False)

    df_val_tokens = initialize_ner_dataset(nlp, df_val)
    df_val_tokens.to_csv(val_output, sep='\t', index=False)

    df_test_tokens = initialize_ner_dataset(nlp, df_test)
    df_test_tokens.to_csv(test_output, sep='\t', index=False)

    LOGGER.info(
        "\n%s",
        format_pipeline_step01_summary(
            dataset_path=str(dataset_path),
            train_path=str(train_output),
            val_path=str(val_output),
            test_path=str(test_output),
            size_train=len(df_train),
            size_val=len(df_val),
            size_test=len(df_test),
            output_dir=str(out_dir)
        ),
    )