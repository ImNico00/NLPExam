from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any

from transformers import BlipProcessor, BlipForConditionalGeneration

from pipeline_exam.src.chat_model_class import ChatModel
from pipeline_exam.src.utils import configure_logging, load_dotenv_file

LOGGER = logging.getLogger(__name__)

def load_visual_model(device : str, visual_model_id : str, hf_token: str | None) -> tuple[Any, BlipForConditionalGeneration]:
    processor = BlipProcessor.from_pretrained(visual_model_id, token=hf_token, use_fast=True)
    vision_model = BlipForConditionalGeneration.from_pretrained(visual_model_id, token=hf_token).to(device) # type: ignore
    vision_model.eval() # Modalità inferenza
    return processor, vision_model

def start_conversation(chatModel : ChatModel):
    prompt = "Puoi dirmi cosa vedi in questa immagine? https://upload.wikimedia.org/wikipedia/commons/a/ae/225Ac-PSMA-617.svg"
    LOGGER.info("Avvio della conversazione...")
    while(prompt != "STOP"):
        chatModel.user_turn(prompt)
        print("Continua la conversazione: ")
        prompt = input()

def format_pipeline_step_summary(
    *,        
    visual_processor,
    visual_model_id, 
    llm_model_id,
    device,
    huggingface_setted,
    output_dir,
    cache_dir
    ) -> str:
    summary = (
        "Pipeline Step summary:\n"
        f"- LLM Used: {llm_model_id}\n"
        f"- Visual Model Used: {visual_model_id}\n"
        f"- Processor Model Used: {visual_processor}\n"
        f"- Device Used: {device}\n"
        f"- HuggingFace API Key Setted: {huggingface_setted}\n"
        f"- Output Directory: {output_dir}\n"
        f"- Cache Directory: {cache_dir}\n"
    )
    return summary

def build_pipeline_parser(default_repo_root: Path) -> argparse.ArgumentParser:
    _ = default_repo_root
    parser = argparse.ArgumentParser(
        description="Start exam pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--llm-model-id", default="llama3.2")
    parser.add_argument("--visual-model-id", default="Salesforce/blip-image-captioning-base")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="pipeline_exam/src/data/processed")
    parser.add_argument("--cache-dir", default="pipeline_exam/src/data/raw")
    parser.add_argument("--logging-level", default="INFO")
    return parser

def run_pipeline(args: argparse.Namespace) -> None:

    configure_logging(args.logging_level)
    load_dotenv_file(".env")
    out_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    visual_model_id = args.visual_model_id
    llm_model_id = args.llm_model_id
    device = args.device

    huggingface_api_key = os.environ.get("HUGGINGFACE_API_KEY")
    if not huggingface_api_key:
        LOGGER.warning("HUGGINGFACE_API_KEY not set -- unauthenticated may fail.")

    visual_processor, visual_model = load_visual_model(device=device, visual_model_id=visual_model_id, hf_token=huggingface_api_key)

    #system_prompt = {
    #    "role": "system", 
    #    "content": "Sei un assistente AI avanzato. Hai a disposizione un tool chiamato 'analyze_image' per "
    #               "guardare le immagini (sia file locali che link http/https). Se l'utente ti fa domande "
    #               "su un'immagine, DEVI usare il tool per scoprire cosa contiene prima di rispondere."
    #}
    chatModel = ChatModel(
        visual_processor=visual_processor,
        visual_model=visual_model, 
        model_id=llm_model_id,
        device=device,
        msgs=[]
    )

    start_conversation(chatModel)

    LOGGER.info(
        "%s",
        format_pipeline_step_summary(
            visual_processor=visual_processor,
            visual_model_id=visual_model_id, 
            llm_model_id=llm_model_id,
            device=device,
            huggingface_setted=("Yes" if huggingface_api_key else "No"),
            output_dir=out_dir,
            cache_dir=cache_dir
        ),
    )