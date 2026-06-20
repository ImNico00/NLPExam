from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any
import json
import pandas as pd

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


def align_gold_label_for_model(gold_label: str, gt_taxonomy: str) -> str:
    """
    Adatta l'etichetta del Gold Standard (ricca e completa) alla tassonomia
    su cui il modello specifico è stato addestrato.
    
    gt_taxonomy indica la 'scuola di pensiero' del modello:
    - 'scibert': solo ENTITY
    - 'scibc5cdr': solo CHEMICAL e DISEASE
    - 'dictionary': tassonomia completa (DRUG, DISEASE, PROCEDURE, ANATOMY)
    """
    if gold_label in ["O", "[PAD]", "", None] or not isinstance(gold_label, str):
        return "O"
        
    parts = gold_label.split("-", 1)
    if len(parts) == 2:
        prefix, base_label = parts
    else:
        prefix, base_label = "", gold_label

    # Se stiamo valutando un modello addestrato con SciBERT
    if gt_taxonomy == "scibert":
        return f"{prefix}-ENTITY" if prefix else "ENTITY"

    elif gt_taxonomy == "scibc5cdr":
        if base_label in ["DRUG", "CHEMICAL"]:
            return f"{prefix}-CHEMICAL"
        elif base_label == "DISEASE":
            return f"{prefix}-DISEASE"
        else:
            return "O"

    return gold_label

def pandas_to_stanza_json(df: pd.DataFrame, output_json_path: Path):
    stanza_data = []
    
    for _, group in df.groupby("Sentence_ID", sort=False):
        sentence = []
        for _, row in group.iterrows():
            word = str(row["Token"]).strip()
            tag = str(row["BIO_Tag"]).strip()
            
            if not word:
                continue
                
            # LA CORREZIONE È QUI: Creiamo un dizionario invece di una lista
            sentence.append({"text": word, "ner": tag})
            
        if sentence:
            stanza_data.append(sentence)
            
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(stanza_data, f, ensure_ascii=False, indent=2)
        
    return stanza_data

def stanza_entities_to_bio(tokens: list[str], entities: list[dict]) -> list[str]:
    """
    Converte le entità di Stanza (span-based) in tag BIO allineati ai tuoi token.
    """
    bio_tags = ["O"] * len(tokens)
    
    for ent in entities:
        ent_tokens = ent["text"].split()
        ent_type = ent["type"]
        
        for i in range(len(tokens) - len(ent_tokens) + 1):
            if tokens[i : i + len(ent_tokens)] == ent_tokens:
                bio_tags[i] = f"B-{ent_type}"
                for j in range(1, len(ent_tokens)):
                    bio_tags[i + j] = f"I-{ent_type}"
                break
    return bio_tags


def extract_error_logs(tokens: list[str], true_tags: list[str], pred_tags: list[str]) -> list[dict]:
    """
    Confronta i tag veri e predetti e restituisce una lista di dizionari con i dettagli degli errori.
    Presuppone che tokens, true_tags e pred_tags abbiano tutti la stessa lunghezza.
    """
    full_sentence = " ".join(tokens)
    errors_list = []
    
    for token, true_tag, pred_tag in zip(tokens, true_tags, pred_tags):
        if true_tag != pred_tag:
            errors_list.append({
                "token": token,
                "true_tag": true_tag,
                "predicted_tag": pred_tag,
                "error_type": f"{true_tag} -> {pred_tag}",
                "sentence": full_sentence
            })
            
    return errors_list

patterns = [
    # ==========================================
    # --- DRUG (Farmaci e Principi Attivi) ---
    # ==========================================
    {"label": "DRUG", "pattern": [{"LOWER": "aspirin"}]},
    {"label": "DRUG", "pattern": [{"LOWER": "ibuprofen"}]},
    {"label": "DRUG", "pattern": [{"LOWER": "metformin"}]},
    {"label": "DRUG", "pattern": [{"LOWER": "paracetamol"}]},
    {"label": "DRUG", "pattern": [{"LOWER": "heparin"}]},
    {"label": "DRUG", "pattern": [{"LOWER": "trastuzumab"}, {"LOWER": "emtansine"}]},
    {"label": "DRUG", "pattern": [{"LOWER": "clavulanic"}, {"LOWER": "acid"}]},
    {"label": "DRUG", "pattern": [{"LOWER": "losartan"}]},
    {"label": "DRUG", "pattern": [{"LOWER": "amoxicillin"}]},
    {"label": "DRUG", "pattern": [{"LOWER": "albuterol"}]},
    {"label": "DRUG", "pattern": [{"LOWER": "nitroglycerin"}]},
    {"label": "DRUG", "pattern": [{"LOWER": "statin"}]},
    {"label": "DRUG", "pattern": [{"LOWER": "omeprazole"}]},
    {"label": "DRUG", "pattern": [{"LOWER": "diazepam"}]},
    {"label": "DRUG", "pattern": [{"LOWER": "salbutamol"}]},
    {"label": "DRUG", "pattern": [{"LOWER": "penicillin"}]},
    {"label": "DRUG", "pattern": [{"LOWER": "levothyroxine"}]},
    {"label": "DRUG", "pattern": [{"LOWER": "tricyclic"}, {"LOWER": "antidepressants"}]},
    {"label": "DRUG", "pattern": [{"LOWER": "ceftriaxone"}]},
    {"label": "DRUG", "pattern": [{"LOWER": "corticosteroids"}]},
    {"label": "DRUG", "pattern": [{"LOWER": "morphine"}]},
    {"label": "DRUG", "pattern": [{"LOWER": "pantoprazole"}]},
    {"label": "DRUG", "pattern": [{"LOWER": "citalopram"}]},
    {"label": "DRUG", "pattern": [{"LOWER": "ace"}, {"LOWER": "inhibitors"}]},
    {"label": "DRUG", "pattern": [{"LOWER": "timolol"}]},
    {"label": "DRUG", "pattern": [{"LOWER": "influenza"}, {"LOWER": "vaccine"}]},
    {"label": "DRUG", "pattern": [{"LOWER": "ciprofloxacin"}]},
    {"label": "DRUG", "pattern": [{"LOWER": "oral"}, {"LOWER": "contraceptives"}]},
    {"label": "DRUG", "pattern": [{"LOWER": "propofol"}]},
    {"label": "DRUG", "pattern": [{"LOWER": "mefloquine"}]},
    {"label": "DRUG", "pattern": [{"LOWER": "sertraline"}]},
    {"label": "DRUG", "pattern": [{"LOWER": "lisinopril"}]},
    {"label": "DRUG", "pattern": [{"LOWER": "atorvastatin"}]},
    {"label": "DRUG", "pattern": [{"LOWER": "gabapentin"}]},
    {"label": "DRUG", "pattern": [{"LOWER": "amlodipine"}]},
    {"label": "DRUG", "pattern": [{"LOWER": "aminoglycoside"}, {"LOWER": "antibiotics"}]},
    {"label": "DRUG", "pattern": [{"LOWER": "antibiotic"}, {"LOWER": "therapy"}]},
    {"label": "DRUG", "pattern": [{"LOWER": "vitamin"}]},

    # ==========================================
    # --- DISEASE (Malattie, Condizioni e Sintomi) ---
    # ==========================================
    {"label": "DISEASE", "pattern": [{"LOWER": "covid-19"}]},
    {"label": "DISEASE", "pattern": [{"LOWER": "sars-cov-2"}]},
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
    {"label": "DISEASE", "pattern": [{"LOWER": "hypercholesterolemia"}]},
    {"label": "DISEASE", "pattern": [{"LOWER": "ischemic"}, {"LOWER": "alterations"}]},
    {"label": "DISEASE", "pattern": [{"LOWER": "hemorrhages"}]},
    {"label": "DISEASE", "pattern": [{"LOWER": "migraine"}]},
    {"label": "DISEASE", "pattern": [{"LOWER": "hyperglycemia"}]},
    {"label": "DISEASE", "pattern": [{"LOWER": "gallstones"}]},
    {"label": "DISEASE", "pattern": [{"LOWER": "ototoxicity"}]},
    {"label": "DISEASE", "pattern": [{"LOWER": "chronic"}, {"LOWER": "renal"}, {"LOWER": "failure"}]},
    {"label": "DISEASE", "pattern": [{"LOWER": "chronic"}, {"LOWER": "kidney"}, {"LOWER": "disease"}]},
    {"label": "DISEASE", "pattern": [{"LOWER": "bone"}, {"LOWER": "metastases"}]},
    {"label": "DISEASE", "pattern": [{"LOWER": "gastroesophageal"}, {"LOWER": "reflux"}, {"LOWER": "disease"}]},
    {"label": "DISEASE", "pattern": [{"LOWER": "atrial"}, {"LOWER": "fibrillation"}]},
    {"label": "DISEASE", "pattern": [{"LOWER": "bronchopneumonia"}]},
    {"label": "DISEASE", "pattern": [{"LOWER": "glaucoma"}]},
    {"label": "DISEASE", "pattern": [{"LOWER": "open-angle"}, {"LOWER": "glaucoma"}]},
    {"label": "DISEASE", "pattern": [{"LOWER": "epileptic"}, {"LOWER": "seizure"}]},
    {"label": "DISEASE", "pattern": [{"LOWER": "dyspnea"}]},
    {"label": "DISEASE", "pattern": [{"LOWER": "dry"}, {"LOWER": "cough"}]},
    {"label": "DISEASE", "pattern": [{"LOWER": "atheromatous"}, {"LOWER": "plaques"}]},
    {"label": "DISEASE", "pattern": [{"LOWER": "carcinoma"}]},
    {"label": "DISEASE", "pattern": [{"LOWER": "infiltrating"}, {"LOWER": "ductal"}, {"LOWER": "carcinoma"}]},
    {"label": "DISEASE", "pattern": [{"LOWER": "anemia"}]},
    {"label": "DISEASE", "pattern": [{"LOWER": "iron-deficiency"}, {"LOWER": "anemia"}]},
    {"label": "DISEASE", "pattern": [{"LOWER": "macrocytic"}, {"LOWER": "anemia"}]},
    {"label": "DISEASE", "pattern": [{"LOWER": "melanoma"}]},
    {"label": "DISEASE", "pattern": [{"LOWER": "cancer"}]},
    {"label": "DISEASE", "pattern": [{"LOWER": "breast"}, {"LOWER": "cancer"}]},
    {"label": "DISEASE", "pattern": [{"LOWER": "metastatic"}, {"LOWER": "breast"}, {"LOWER": "cancer"}]},
    {"label": "DISEASE", "pattern": [{"LOWER": "ovarian"}, {"LOWER": "cancer"}]},
    {"label": "DISEASE", "pattern": [{"LOWER": "hypothyroidism"}]},
    {"label": "DISEASE", "pattern": [{"LOWER": "subclinical"}, {"LOWER": "hypothyroidism"}]},
    {"label": "DISEASE", "pattern": [{"LOWER": "polyp"}]},
    {"label": "DISEASE", "pattern": [{"LOWER": "bacterial"}, {"LOWER": "sinusitis"}]},
    {"label": "DISEASE", "pattern": [{"LOWER": "crohn's"}, {"LOWER": "disease"}]},
    {"label": "DISEASE", "pattern": [{"LOWER": "skin"}, {"LOWER": "rash"}]},
    {"label": "DISEASE", "pattern": [{"LOWER": "renal"}, {"LOWER": "colic"}]},
    {"label": "DISEASE", "pattern": [{"LOWER": "pulmonary"}, {"LOWER": "nodules"}]},
    {"label": "DISEASE", "pattern": [{"LOWER": "pulmonary"}, {"LOWER": "embolism"}]},
    {"label": "DISEASE", "pattern": [{"LOWER": "hashimoto's"}, {"LOWER": "thyroiditis"}]},
    {"label": "DISEASE", "pattern": [{"LOWER": "pleural"}, {"LOWER": "effusion"}]},
    {"label": "DISEASE", "pattern": [{"LOWER": "alzheimer's"}, {"LOWER": "disease"}]},
    {"label": "DISEASE", "pattern": [{"LOWER": "lymphadenopathies"}]},
    {"label": "DISEASE", "pattern": [{"LOWER": "meningioma"}]},
    {"label": "DISEASE", "pattern": [{"LOWER": "major"}, {"LOWER": "depression"}]},
    {"label": "DISEASE", "pattern": [{"LOWER": "rheumatoid"}, {"LOWER": "arthritis"}]},
    {"label": "DISEASE", "pattern": [{"LOWER": "amyotrophic"}, {"LOWER": "lateral"}, {"LOWER": "sclerosis"}]},
    {"label": "DISEASE", "pattern": [{"LOWER": "als"}]},
    {"label": "DISEASE", "pattern": [{"LOWER": "copd"}]},
    {"label": "DISEASE", "pattern": [{"LOWER": "acute"}, {"LOWER": "hepatitis"}]},
    {"label": "DISEASE", "pattern": [{"LOWER": "bone"}, {"LOWER": "fractures"}]},
    {"label": "DISEASE", "pattern": [{"LOWER": "paresthesia"}]},
    {"label": "DISEASE", "pattern": [{"LOWER": "diabetic"}, {"LOWER": "neuropathy"}]},
    {"label": "DISEASE", "pattern": [{"LOWER": "diabetic"}, {"LOWER": "retinopathy"}]},
    {"label": "DISEASE", "pattern": [{"LOWER": "peptic"}, {"LOWER": "ulcer"}]},
    {"label": "DISEASE", "pattern": [{"LOWER": "bleeding"}, {"LOWER": "peptic"}, {"LOWER": "ulcer"}]},
    {"label": "DISEASE", "pattern": [{"LOWER": "gastric"}, {"LOWER": "ulcer"}]},
    {"label": "DISEASE", "pattern": [{"LOWER": "renal"}, {"LOWER": "cyst"}]},
    {"label": "DISEASE", "pattern": [{"LOWER": "cortical"}, {"LOWER": "renal"}, {"LOWER": "cyst"}]},
    {"label": "DISEASE", "pattern": [{"LOWER": "generalized"}, {"LOWER": "anxiety"}, {"LOWER": "disorder"}]},
    {"label": "DISEASE", "pattern": [{"LOWER": "deep"}, {"LOWER": "vein"}, {"LOWER": "thrombosis"}]},
    {"label": "DISEASE", "pattern": [{"LOWER": "dvt"}]},
    {"label": "DISEASE", "pattern": [{"LOWER": "retinal"}, {"LOWER": "detachment"}]},
    {"label": "DISEASE", "pattern": [{"LOWER": "obstructive"}, {"LOWER": "sleep"}, {"LOWER": "apnea"}, {"LOWER": "syndrome"}]},
    {"label": "DISEASE", "pattern": [{"LOWER": "osas"}]},
    {"label": "DISEASE", "pattern": [{"LOWER": "coxarthrosis"}]},
    {"label": "DISEASE", "pattern": [{"LOWER": "hemorrhagic"}, {"LOWER": "cystitis"}]},
    {"label": "DISEASE", "pattern": [{"LOWER": "spontaneous"}, {"LOWER": "abortion"}]},
    {"label": "DISEASE", "pattern": [{"LOWER": "aortic"}, {"LOWER": "aneurysm"}]},
    {"label": "DISEASE", "pattern": [{"LOWER": "traumatic"}, {"LOWER": "brain"}, {"LOWER": "injury"}]},
    {"label": "DISEASE", "pattern": [{"LOWER": "psoriasis"}]},
    {"label": "DISEASE", "pattern": [{"LOWER": "sensorineural"}, {"LOWER": "hearing"}, {"LOWER": "loss"}]},
    {"label": "DISEASE", "pattern": [{"LOWER": "myasthenia"}, {"LOWER": "gravis"}]},
    {"label": "DISEASE", "pattern": [{"LOWER": "acute"}, {"LOWER": "otitis"}, {"LOWER": "media"}]},
    {"label": "DISEASE", "pattern": [{"LOWER": "peritonitis"}]},
    {"label": "DISEASE", "pattern": [{"LOWER": "hyperlipidemia"}]},
    {"label": "DISEASE", "pattern": [{"LOWER": "childhood"}, {"LOWER": "asthma"}]},
    {"label": "DISEASE", "pattern": [{"LOWER": "obstructive"}, {"LOWER": "deficit"}]},
    {"label": "DISEASE", "pattern": [{"LOWER": "adverse"}, {"LOWER": "reaction"}]},
    {"label": "DISEASE", "pattern": [{"LOWER": "asthma"}, {"LOWER": "attacks"}]},

    # ==========================================
    # --- PROCEDURE (Procedure Mediche, Esami, Test) ---
    # ==========================================
    {"label": "PROCEDURE", "pattern": [{"LOWER": "magnetic"}, {"LOWER": "resonance"}, {"LOWER": "imaging"}]},
    {"label": "PROCEDURE", "pattern": [{"LOWER": "mri"}]},
    {"label": "PROCEDURE", "pattern": [{"LOWER": "ct"}, {"LOWER": "scan"}]},
    {"label": "PROCEDURE", "pattern": [{"LOWER": "follow-up"}, {"LOWER": "ct"}, {"LOWER": "scan"}]},
    {"label": "PROCEDURE", "pattern": [{"LOWER": "emergency"}, {"LOWER": "ct"}, {"LOWER": "scan"}]},
    {"label": "PROCEDURE", "pattern": [{"LOWER": "liver"}, {"LOWER": "biopsy"}]},
    {"label": "PROCEDURE", "pattern": [{"LOWER": "biopsy"}]},
    {"label": "PROCEDURE", "pattern": [{"LOWER": "emergency"}, {"LOWER": "biopsy"}]},
    {"label": "PROCEDURE", "pattern": [{"LOWER": "electrocardiogram"}]},
    {"label": "PROCEDURE", "pattern": [{"LOWER": "ecg"}]},
    {"label": "PROCEDURE", "pattern": [{"LOWER": "colonoscopy"}]},
    {"label": "PROCEDURE", "pattern": [{"LOWER": "endoscopy"}]},
    {"label": "PROCEDURE", "pattern": [{"LOWER": "x-ray"}]},
    {"label": "PROCEDURE", "pattern": [{"LOWER": "emergency"}, {"LOWER": "x-ray"}]},
    {"label": "PROCEDURE", "pattern": [{"LOWER": "complete"}, {"LOWER": "blood"}, {"LOWER": "count"}]},
    {"label": "PROCEDURE", "pattern": [{"LOWER": "cbc"}]},
    {"label": "PROCEDURE", "pattern": [{"LOWER": "abdominal"}, {"LOWER": "ultrasound"}]},
    {"label": "PROCEDURE", "pattern": [{"LOWER": "ultrasound"}]},
    {"label": "PROCEDURE", "pattern": [{"LOWER": "histological"}, {"LOWER": "examination"}]},
    {"label": "PROCEDURE", "pattern": [{"LOWER": "bone"}, {"LOWER": "scan"}]},
    {"label": "PROCEDURE", "pattern": [{"LOWER": "doppler"}]},
    {"label": "PROCEDURE", "pattern": [{"LOWER": "radiotherapy"}]},
    {"label": "PROCEDURE", "pattern": [{"LOWER": "blood"}, {"LOWER": "chemistry"}, {"LOWER": "tests"}]},
    {"label": "PROCEDURE", "pattern": [{"LOWER": "blood"}, {"LOWER": "chemistry"}, {"LOWER": "panel"}]},
    {"label": "PROCEDURE", "pattern": [{"LOWER": "calprotectin"}, {"LOWER": "test"}]},
    {"label": "PROCEDURE", "pattern": [{"LOWER": "electroencephalogram"}]},
    {"label": "PROCEDURE", "pattern": [{"LOWER": "eeg"}]},
    {"label": "PROCEDURE", "pattern": [{"LOWER": "spirometry"}]},
    {"label": "PROCEDURE", "pattern": [{"LOWER": "throat"}, {"LOWER": "swab"}]},
    {"label": "PROCEDURE", "pattern": [{"LOWER": "psa"}, {"LOWER": "level"}]},
    {"label": "PROCEDURE", "pattern": [{"LOWER": "hrct"}]},
    {"label": "PROCEDURE", "pattern": [{"LOWER": "antibiogram"}]},
    {"label": "PROCEDURE", "pattern": [{"LOWER": "pet-ct"}]},
    {"label": "PROCEDURE", "pattern": [{"LOWER": "esophagogastroduodenoscopy"}]},
    {"label": "PROCEDURE", "pattern": [{"LOWER": "egd"}]},
    {"label": "PROCEDURE", "pattern": [{"LOWER": "ct"}, {"LOWER": "angiography"}]},
    {"label": "PROCEDURE", "pattern": [{"LOWER": "fundus"}, {"LOWER": "examination"}]},
    {"label": "PROCEDURE", "pattern": [{"LOWER": "uterine"}, {"LOWER": "curettage"}]},
    {"label": "PROCEDURE", "pattern": [{"LOWER": "pure"}, {"LOWER": "tone"}, {"LOWER": "audiometry"}]},
    {"label": "PROCEDURE", "pattern": [{"LOWER": "echocardiogram"}]},
    {"label": "PROCEDURE", "pattern": [{"LOWER": "surgical"}, {"LOWER": "reconstruction"}]},
    {"label": "PROCEDURE", "pattern": [{"LOWER": "ophthalmological"}, {"LOWER": "examination"}]},
    {"label": "PROCEDURE", "pattern": [{"LOWER": "echo-color"}, {"LOWER": "doppler"}]},
    {"label": "PROCEDURE", "pattern": [{"LOWER": "rapid"}, {"LOWER": "test"}]},
    {"label": "PROCEDURE", "pattern": [{"LOWER": "orthopedic"}, {"LOWER": "evaluation"}]},
    {"label": "PROCEDURE", "pattern": [{"LOWER": "urological"}, {"LOWER": "examination"}]},
    {"label": "PROCEDURE", "pattern": [{"LOWER": "suturing"}]},
    {"label": "PROCEDURE", "pattern": [{"LOWER": "receptor"}, {"LOWER": "antibody"}, {"LOWER": "test"}]},
    {"label": "PROCEDURE", "pattern": [{"LOWER": "dermatological"}, {"LOWER": "examination"}]},
    {"label": "PROCEDURE", "pattern": [{"LOWER": "abdominal"}, {"LOWER": "physical"}, {"LOWER": "examination"}]},
    {"label": "PROCEDURE", "pattern": [{"LOWER": "physical"}, {"LOWER": "examination"}]},

    # ==========================================
    # --- ANATOMY (Parti anatomiche) ---
    # ==========================================
    {"label": "ANATOMY", "pattern": [{"LOWER": "right"}, {"LOWER": "femur"}]},
    {"label": "ANATOMY", "pattern": [{"LOWER": "left"}, {"LOWER": "lung"}]},
    {"label": "ANATOMY", "pattern": [{"LOWER": "right"}, {"LOWER": "lung"}]},
    {"label": "ANATOMY", "pattern": [{"LOWER": "myocardium"}]},
    {"label": "ANATOMY", "pattern": [{"LOWER": "chest"}]},
    {"label": "ANATOMY", "pattern": [{"LOWER": "abdomen"}]},
    {"label": "ANATOMY", "pattern": [{"LOWER": "left"}, {"LOWER": "eye"}]},
    {"label": "ANATOMY", "pattern": [{"LOWER": "right"}, {"LOWER": "eye"}]},
    {"label": "ANATOMY", "pattern": [{"LOWER": "left"}, {"LOWER": "knee"}]},
    {"label": "ANATOMY", "pattern": [{"LOWER": "kidneys"}]},
    {"label": "ANATOMY", "pattern": [{"LOWER": "liver"}]},
    {"label": "ANATOMY", "pattern": [{"LOWER": "frontal"}, {"LOWER": "lobe"}]},
    {"label": "ANATOMY", "pattern": [{"LOWER": "right"}, {"LOWER": "lower"}, {"LOWER": "quadrant"}]},
    {"label": "ANATOMY", "pattern": [{"LOWER": "right"}, {"LOWER": "breast"}]},
    {"label": "ANATOMY", "pattern": [{"LOWER": "sigmoid"}, {"LOWER": "colon"}]},
    {"label": "ANATOMY", "pattern": [{"LOWER": "medial"}, {"LOWER": "meniscus"}]},
    {"label": "ANATOMY", "pattern": [{"LOWER": "cranial"}, {"LOWER": "nerves"}]},
    {"label": "ANATOMY", "pattern": [{"LOWER": "right"}, {"LOWER": "lobe"}]},
    {"label": "ANATOMY", "pattern": [{"LOWER": "pelvis"}]},
    {"label": "ANATOMY", "pattern": [{"LOWER": "hips"}]},
    {"label": "ANATOMY", "pattern": [{"LOWER": "lower"}, {"LOWER": "limbs"}]},
    {"label": "ANATOMY", "pattern": [{"LOWER": "distal"}, {"LOWER": "radius"}]},
    {"label": "ANATOMY", "pattern": [{"LOWER": "scalp"}]},
    {"label": "ANATOMY", "pattern": [{"LOWER": "brain"}]},
    {"label": "ANATOMY", "pattern": [{"LOWER": "thyroid"}, {"LOWER": "gland"}]},
    {"label": "ANATOMY", "pattern": [{"LOWER": "thyroid"}]},
    {"label": "ANATOMY", "pattern": [{"LOWER": "cervical"}, {"LOWER": "spine"}]},
    {"label": "ANATOMY", "pattern": [{"LOWER": "lumbar"}, {"LOWER": "spine"}]},
    {"label": "ANATOMY", "pattern": [{"LOWER": "right"}, {"LOWER": "shoulder"}]},
    {"label": "ANATOMY", "pattern": [{"LOWER": "ulnar"}, {"LOWER": "collateral"}, {"LOWER": "ligament"}]},
    {"label": "ANATOMY", "pattern": [{"LOWER": "supra-aortic"}, {"LOWER": "trunks"}]},
    {"label": "ANATOMY", "pattern": [{"LOWER": "left"}, {"LOWER": "eyebrow"}, {"LOWER": "arch"}]}
]