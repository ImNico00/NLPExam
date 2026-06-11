
import requests
from io import BytesIO
import cairosvg # type: ignore
from ollama._client import Client
from PIL import Image
import logging
from typing import Any
import os

from pipeline_exam.src.utils import imposta_sfondo_bianco

LOGGER = logging.getLogger(__name__)

class ChatModel(Client):
    def __init__(self, model_id: str, visual_model: Any, visual_processor: Any, device: str, msgs: list | None = None):
        super().__init__()
        self.model_id = model_id
        self.msgs = msgs if msgs is not None else [] 
        self.visual_processor = visual_processor
        self.visual_model = visual_model
        self.device = device
        self.headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
                    "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
                    "Accept-Encoding": "gzip, deflate, br",
                    "Accept-Language": "it-IT,it;q=0.9,en-US;q=0.8,en;q=0.7",
                    "Connection": "keep-alive"
                }

    def user_turn(self, input) -> None:
        """Turno utente"""
        LOGGER.info(input)
        self.msgs.append({"role": "user", "content": input})
        self.do_chat()

    def do_chat(self) -> None:
        """Inizio chat LLM"""
        response = self.chat(model=self.model_id, messages=self.msgs, tools=[self.analyze_image])
        LOGGER.info(response.message.content)
        self.msgs.append(response.message)
        
        if response.message.tool_calls:
            for call in response.message.tool_calls:                
                if call.function.name == "analyze_image":
                    path_grezzo = call.function.arguments.get("image_source", "")
                    path_pulito = path_grezzo.strip().strip("\"'") 
                    descrizione = self.analyze_image(path_pulito)
                    self.msgs.append({"role": "tool", "name": "analyze_image", "content": descrizione})
                    self.do_chat()


    # START TOOLS DA POTER CHIAMARE
    def analyze_image(self, image_source: str) -> str:
        """Analizza l'immagine al percorso locale o all'URL specificato e restituisce una descrizione testuale."""
        LOGGER.info(f"[TOOL LOG] Analisi visiva avviata per: {image_source}")
        try:
            if image_source.startswith(("http://", "https://")):
                LOGGER.debug("[TOOL LOG] Download dell'immagine in corso...")
                response = requests.get(image_source, headers=self.headers, timeout=10)
                response.raise_for_status()
                if "svg" in response.headers.get("Content-Type", "") or image_source.lower().endswith(".svg"):
                    img_data = cairosvg.svg2png(bytestring=response.content)
                else:
                    img_data = response.content
                image = Image.open(BytesIO(img_data))
            else:
                if not os.path.exists(image_source):
                    errore = f"ERRORE: Il file {image_source} non esiste. Chiedi all'utente un percorso valido."
                    LOGGER.error(f"[TOOL LOG] {errore}")
                    return errore
                if image_source.lower().endswith('.svg'):
                    LOGGER.debug("[TOOL LOG] Rilevato SVG locale, conversione in PNG in corso...")
                    img_data = cairosvg.svg2png(url=image_source)
                    image = Image.open(BytesIO(img_data))
                else:
                    image = Image.open(image_source)
            image = imposta_sfondo_bianco(image)

            inputs = self.visual_processor(image, return_tensors="pt").to(self.device)
            out = self.visual_model.generate(**inputs, max_new_tokens=50)
            descrizione = self.visual_processor.decode(out[0], skip_special_tokens=True)
            return f"L'immagine mostra: {descrizione}"
            
        except requests.exceptions.RequestException as e:
            errore = f"ERRORE DI RETE: Impossibile scaricare l'immagine. Dettagli: {str(e)}"
            LOGGER.error(f"[TOOL LOG] {errore}")
            return errore
        except Exception as e:
            errore = f"ERRORE durante l'analisi dell'immagine: {str(e)}"
            LOGGER.error(f"[TOOL LOG] {errore}")
            return errore
