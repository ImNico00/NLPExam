# Multimodal AI Agent - NLP Exam Pipeline

Repository contenente l'implementazione di un Agente IA Multimodale sviluppato per l'esame di NLP. La pipeline integra un LLM locale (tramite Ollama) con un modello di Image Captioning (tramite Hugging Face) utilizzando il paradigma del *Tool Calling* per analizzare e descrivere immagini fornite dall'utente (sia file locali che URL).

## 🚀 Architettura del Sistema

Il progetto non utilizza modelli nativamente multimodali pesanti, ma implementa un'architettura agentica modulare:
- **Cervello Logico (LLM):** `llama3.2` eseguito localmente tramite Ollama. Gestisce la conversazione e decide autonomamente quando chiamare lo strumento visivo.
- **Occhi (Vision Tool):** `Salesforce/blip-image-captioning-base` eseguito in locale su GPU tramite PyTorch/Transformers. Traduce i tensori delle immagini in testo naturale.
- **Integrazione:** Classe `ChatModel` personalizzata che eredita dal client Ollama, gestisce lo stato della chat e scarica le immagini eludendo i blocchi anti-bot (es. Wikipedia).

## ⚙️ Setup dell'Ambiente

Il progetto utilizza **Poetry** per la gestione rigorosa delle dipendenze e dell'ambiente virtuale.

### Prerequisiti
- [Python 3.12](https://www.python.org/)
- [Poetry](https://python-poetry.org/)
- [Ollama](https://ollama.com/) installato e in esecuzione (`ollama serve`).

### Installazione

1. Clona il repository.
2. Assicurati di scaricare il modello LLM corretto tramite Ollama:
   ```bash ollama pull llama3.2```
3. Installa le dipendenze Python (inclusi PyTorch e Transformers):
   ```poetry install```

### 🖥 Esecuzione della Pipeline

La pipeline è pensata per essere eseguita tramite riga di comando. Il punto di ingresso principale espone diversi argomenti per configurare i modelli e l'hardware.

Per avviare l'agente interattivo con le impostazioni di default (Llama 3.2 + BLIP su CUDA):
```poetry run python pipeline_exam/src/start_exam.py```

### Configurazione Avanzata (CLI)
Il parser `argparse` permette di sovrascrivere i parametri predefiniti:
- `--llm-model-id`: Cambia il modello testuale (default: `llama3.2`). Nota: il modello deve supportare nativamente i tool.
- `--visual-model-id`: Cambia l'encoder visivo (default: `Salesforce/blip-image-captioning-base`).
- `--device`: Seleziona l'hardware di inferenza visiva (`cuda` o `cpu`).

Esempio per eseguire l'analisi visiva solo su CPU:
```poetry run python pipeline_exam/src/start_exam.py --device cpu```

### 🔄 Flusso di Interazione
All'avvio, l'agente riceverà un System Prompt che gli conferisce il tool `analyze_image`.
L'utente può interagire inserendo query di testo puro o richiedendo l'analisi di immagini passandone il percorso.

Input supportati per le immagini:
- **Path Locali**: Percorsi assoluti o relativi nel file system (es. `data/images/foto.png`).
- **URL Online**: Link diretti all'immagine (es. `https://sito.com/immagine.png`). Il tool integra un bypass per gli User-Agent al fine di scaricare immagini da repository come Wikimedia Commons senza incorrere in errori `403 Forbidden`.

Digita `STOP` nel terminale per terminare l'esecuzione.