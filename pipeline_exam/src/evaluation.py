from __future__ import annotations

import argparse
import logging
from pathlib import Path
import spacy
from spacy.language import Language
import pandas as pd

from pipeline_exam.src.utils import configure_logging, load_dotenv_file, patterns

LOGGER = logging.getLogger(__name__)

def initialize_ner_dataset(nlp: Language, df: pd.DataFrame) -> pd.DataFrame:
    sentences_array = []
    for _, row in df.iterrows():
        id_referto = row['id_referto']
        testo = row['testo_perizia']
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

def format_pipeline_step_summary(
    *,
    dataset_path,
    output_tokenized_bio,
    size_df_tokens,
    output_dir
    ) -> str:
    summary = (
        "Pipeline Step summary:\n"
        f"- Dataset Path: {dataset_path}\n"
        f"- Dataset BIO Tokenized: {output_tokenized_bio}\n"
        f"- Size Dataset BIO Tokenized: {size_df_tokens}\n"
        f"- Output Directory: {output_dir}"
    )
    return summary

def build_step03_parser(default_repo_root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Start Exam Step01 Data Preparation With Ground Truth Using Rule-Based Approach",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    processed_dir = default_repo_root / "pipeline_exam" / "data" / "processed"
    dataset_dir = default_repo_root / "data"
    parser.add_argument("--dataset-path", default=str(dataset_dir / "perizie_mediche_sintetiche.csv"))
    parser.add_argument("--output-dir", default=str(processed_dir))
    parser.add_argument("--output-tokenized-bio", default=str(processed_dir / "perizie_tokenizzate_BIO.tsv"))
    parser.add_argument("--logging-level", default="INFO")
    return parser

def run_step03(args: argparse.Namespace) -> None:
    configure_logging(args.logging_level)
    load_dotenv_file(".env")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = Path(args.dataset_path)
    output_tokenized_bio = Path(args.output_tokenized_bio)

    nlp = spacy.blank("it")
    ruler = nlp.add_pipe("entity_ruler")

    ruler.add_patterns(patterns)
    df = pd.read_csv(dataset_path)
    df_tokens = initialize_ner_dataset(nlp, df)
    df_tokens.to_csv(output_tokenized_bio, sep='\t', index=False)

    LOGGER.info(
        "%s",
        format_pipeline_step_summary(
            dataset_path = str(dataset_path),
            output_tokenized_bio = str(output_tokenized_bio),
            size_df_tokens = len(df_tokens),
            output_dir=str(out_dir)
        ),
    )