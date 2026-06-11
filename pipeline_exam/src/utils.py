from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any
from PIL import Image

def imposta_sfondo_bianco(image):
    """Converte in modo sicuro qualsiasi immagine in RGB, riempiendo la trasparenza di bianco."""
    
    if image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info):
        image = image.convert('RGBA')
        bg = Image.new("RGB", image.size, (255, 255, 255))
        bg.paste(image, mask=image.split()[3])
        return bg
    return image.convert("RGB")

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
    {"label": "DRUG", "pattern": [{"LOWER": "aspirina"}]},
    {"label": "DRUG", "pattern": [{"LOWER": "ibuprofene"}]},
    {"label": "DRUG", "pattern": [{"LOWER": "metformina"}]},
    {"label": "DRUG", "pattern": [{"LOWER": "paracetamolo"}]},
    {"label": "DRUG", "pattern": [{"LOWER": "eparina"}]},
    {"label": "DRUG", "pattern": [{"LOWER": "trastuzumab"}, {"LOWER": "emtansine"}]},
    {"label": "DRUG", "pattern": [{"LOWER": "acido"}, {"LOWER": "clavulanico"}]},
    
    # --- DISEASE (Malattie e Sintomi) ---
    {"label": "DISEASE", "pattern": [{"LOWER": "covid-19"}]},
    {"label": "DISEASE", "pattern": [{"LOWER": "polmonite"}]},
    {"label": "DISEASE", "pattern": [{"LOWER": "diabete"}, {"LOWER": "di"}, {"LOWER": "tipo"}, {"LOWER": "2"}]},
    {"label": "DISEASE", "pattern": [{"LOWER": "ipertensione"}, {"LOWER": "arteriosa"}]},
    {"label": "DISEASE", "pattern": [{"LOWER": "appendicite"}, {"LOWER": "acuta"}]},
    {"label": "DISEASE", "pattern": [{"LOWER": "infarto"}, {"LOWER": "miocardico"}, {"LOWER": "acuto"}]},
    
    # --- PROCEDURE (Procedure Mediche/Esami) ---
    {"label": "PROCEDURE", "pattern": [{"LOWER": "risonanza"}, {"LOWER": "magnetica"}]},
    {"label": "PROCEDURE", "pattern": [{"LOWER": "mri"}]},
    {"label": "PROCEDURE", "pattern": [{"LOWER": "tac"}]},
    {"label": "PROCEDURE", "pattern": [{"LOWER": "biopsia"}, {"LOWER": "epatica"}]},
    {"label": "PROCEDURE", "pattern": [{"LOWER": "elettrocardiogramma"}]},
    
    # --- ANATOMY (Anatomia) ---
    {"label": "ANATOMY", "pattern": [{"LOWER": "femore"}, {"LOWER": "destro"}]},
    {"label": "ANATOMY", "pattern": [{"LOWER": "polmone"}, {"LOWER": "sinistro"}]},
    {"label": "ANATOMY", "pattern": [{"LOWER": "miocardio"}]}
]