from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )

def ensure_parent_dir(path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target

def load_dotenv_file(path: str | Path) -> None:
    env_path = Path(path)
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


def load_yaml_file(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        return {}
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]

    for lineno, raw_line in enumerate(config_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue

        indent = len(raw_line) - len(raw_line.lstrip(" "))
        if indent % 2 != 0:
            raise ValueError(f"Invalid indentation in YAML config at {config_path}:{lineno}")

        line = raw_line.strip()
        if ":" not in line:
            raise ValueError(f"Expected key/value mapping in YAML config at {config_path}:{lineno}")

        key, raw_value = line.split(":", 1)
        key = key.strip()
        raw_value = raw_value.strip()

        while len(stack) > 1 and indent <= stack[-1][0]:
            stack.pop()
        current = stack[-1][1]

        if not raw_value:
            child: dict[str, Any] = {}
            current[key] = child
            stack.append((indent, child))
            continue

        value: Any
        lowered = raw_value.lower()
        if lowered in {"true", "false"}:
            value = lowered == "true"
        else:
            try:
                value = int(raw_value)
            except ValueError:
                try:
                    value = float(raw_value)
                except ValueError:
                    value = raw_value.strip("'\"")
        current[key] = value

    return root

patterns = [
    # --- DRUG (Farmaci) ---
    {"label": "DRUG", "pattern": [{"LOWER": "aspirin"}]},
    {"label": "DRUG", "pattern": [{"LOWER": "ibuprofen"}]},
    {"label": "DRUG", "pattern": [{"LOWER": "metformin"}]},
    {"label": "DRUG", "pattern": [{"LOWER": "paracetamol"}]},
    {"label": "DRUG", "pattern": [{"LOWER": "heparin"}]},
    {"label": "DRUG", "pattern": [{"LOWER": "trastuzumab"}, {"LOWER": "emtansine"}]},
    {"label": "DRUG", "pattern": [{"LOWER": "clavulanic"}, {"LOWER": "acid"}]},
    # Extra trovati nel tuo dataset:
    {"label": "DRUG", "pattern": [{"LOWER": "losartan"}]},
    {"label": "DRUG", "pattern": [{"LOWER": "amoxicillin"}]},
    {"label": "DRUG", "pattern": [{"LOWER": "albuterol"}]},
    
    # --- DISEASE (Malattie e Sintomi) ---
    {"label": "DISEASE", "pattern": [{"LOWER": "covid-19"}]},
    {"label": "DISEASE", "pattern": [{"LOWER": "pneumonia"}]},
    {"label": "DISEASE", "pattern": [{"LOWER": "type"}, {"LOWER": "2"}, {"LOWER": "diabetes"}]},
    {"label": "DISEASE", "pattern": [{"LOWER": "arterial"}, {"LOWER": "hypertension"}]},
    {"label": "DISEASE", "pattern": [{"LOWER": "hypertension"}]},
    {"label": "DISEASE", "pattern": [{"LOWER": "acute"}, {"LOWER": "appendicitis"}]},
    {"label": "DISEASE", "pattern": [{"LOWER": "acute"}, {"LOWER": "myocardial"}, {"LOWER": "infarction"}]},
    {"label": "DISEASE", "pattern": [{"LOWER": "major"}, {"LOWER": "depressive"}, {"LOWER": "disorder"}]},
    {"label": "DISEASE", "pattern": [{"LOWER": "bronchial"}, {"LOWER": "asthma"}]},
    {"label": "DISEASE", "pattern": [{"LOWER": "celiac"}, {"LOWER": "disease"}]},
    {"label": "DISEASE", "pattern": [{"LOWER": "osteoarthritis"}]},
    
    # --- PROCEDURE (Procedure Mediche/Esami) ---
    {"label": "PROCEDURE", "pattern": [{"LOWER": "magnetic"}, {"LOWER": "resonance"}, {"LOWER": "imaging"}]},
    {"label": "PROCEDURE", "pattern": [{"LOWER": "mri"}]},
    {"label": "PROCEDURE", "pattern": [{"LOWER": "ct"}, {"LOWER": "scan"}]},
    {"label": "PROCEDURE", "pattern": [{"LOWER": "liver"}, {"LOWER": "biopsy"}]},
    {"label": "PROCEDURE", "pattern": [{"LOWER": "electrocardiogram"}]},
    {"label": "PROCEDURE", "pattern": [{"LOWER": "ecg"}]},
    {"label": "PROCEDURE", "pattern": [{"LOWER": "colonoscopy"}]},
    {"label": "PROCEDURE", "pattern": [{"LOWER": "endoscopy"}]},
    {"label": "PROCEDURE", "pattern": [{"LOWER": "x-ray"}]},
    
    # --- ANATOMY (Anatomia) ---
    {"label": "ANATOMY", "pattern": [{"LOWER": "right"}, {"LOWER": "femur"}]},
    {"label": "ANATOMY", "pattern": [{"LOWER": "left"}, {"LOWER": "lung"}]},
    {"label": "ANATOMY", "pattern": [{"LOWER": "right"}, {"LOWER": "lung"}]},
    {"label": "ANATOMY", "pattern": [{"LOWER": "myocardium"}]},
    {"label": "ANATOMY", "pattern": [{"LOWER": "chest"}]},
    {"label": "ANATOMY", "pattern": [{"LOWER": "abdomen"}]},
    {"label": "ANATOMY", "pattern": [{"LOWER": "left"}, {"LOWER": "eye"}]},
    {"label": "ANATOMY", "pattern": [{"LOWER": "left"}, {"LOWER": "knee"}]},
    {"label": "ANATOMY", "pattern": [{"LOWER": "kidneys"}]},
    {"label": "ANATOMY", "pattern": [{"LOWER": "liver"}]}
]