# Clinical NER Pipeline - NLP Exam

Repository contenente l'implementazione di una pipeline end-to-end di Machine Learning per l'estrazione di entità cliniche (Named Entity Recognition) sviluppata per l'esame di NLP. La pipeline automatizza la generazione di dati, l'annotazione (BIO-tagging) e l'addestramento comparativo di reti neurali classiche e modelli Transformer (Domain Adaptation) per l'estrazione di `DISEASE` e `CHEMICAL` da referti medici.

## 🚀 Architettura del Sistema

Il progetto non si limita al training di un singolo modello, ma implementa un'architettura modulare industriale (MLOps):
- **Data Engineering & Annotation (Step 00 & 01):** Generazione procedurale di referti medici sintetici in lingua inglese. L'annotazione (Gold Standard) non è manuale, ma è effettuata tramite `scispacy` (`en_ner_bc5cdr_md`), sfruttando il pre-training su letteratura biomedica.
- **Deep Learning Baseline (Step 03):** Rete **BiLSTM** sviluppata custom in PyTorch con embedding statici per stabilire una baseline di performance.
- **Attention & Domain Adaptation (Step 03):** Fine-tuning di Transformer Hugging Face. Confronto tra un modello generico (`bert-base-cased`) e un modello di dominio clinico (`emilyalsentzer/Bio_ClinicalBERT` addestrato su MIMIC-III).
- **Evaluation Engine (Step 04):** Modulo di validazione rigorosa che calcola lo *Strict Entity F1-Score* (escludendo programmaticamente l'over-rappresentazione della classe `O`) e genera Matrici di Confusione comparative.

## ⚙️ Setup dell'Ambiente

Il progetto utilizza **Poetry** per la gestione rigorosa delle dipendenze, gestendo attentamente l'allineamento delle versioni per i modelli biomedici.

### Prerequisiti
- [Python 3.12](https://www.python.org/)
- [Poetry](https://python-poetry.org/)
- Supporto CUDA (opzionale ma fortemente raccomandato per il fine-tuning dei Transformer).

### Installazione

1. Clona il repository.
2. Installa le dipendenze di base tramite Poetry:
   ```bash
   poetry install
   ```
3. **Setup Architettura Biomedica:** Installa la versione corretta di spaCy e il modello clinico ScispaCy. Per garantire la riproducibilità ed evitare conflitti di tokenizzazione, il sistema fissa spaCy alla versione 3.7.x:
   ```bash
   poetry add "spacy>=3.7.0,<3.8.0"
   poetry run pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz
   ```

## 🖥 Esecuzione della Pipeline

La pipeline è divisa in step logici pensati per essere eseguiti tramite riga di comando. 

### 1. Preparazione Dati e Annotazione (Step 00, 01, 02)
Esegui in sequenza i moduli per generare il dataset, annotarlo con tag BIO e costruire i vocabolari tensoriali:
```bash
poetry run python pipeline_exam/src/step00_creation.py
poetry run python pipeline_exam/src/step01_annotation.py
poetry run python pipeline_exam/src/step02_build_vocab.py
```

### 2. Addestramento Modelli (Step 03)
Avvia il training loop. Esempio per addestrare il modello Bio_ClinicalBERT di default su CUDA:
```bash
poetry run python pipeline_exam/src/step03_train.py --model-id biobert_ner
```

### Configurazione Avanzata (CLI) per il Training
Il parser `argparse` permette di sovrascrivere gli iperparametri per sperimentare:
- `--model-id`: Architettura da addestrare (`bilstm`, `bert_ner`, `biobert_ner`).
- `--epochs`: Epoche di addestramento (default: `25`). Il sistema implementa l'*Early Checkpointing* salvando automaticamente i pesi relativi alla migliore Validation Loss.
- `--lr`: Learning rate. Il codice lo scala dinamicamente (es. `2e-5`) in caso di fine-tuning Transformer per evitare *catastrophic forgetting*.
- `--batch-size`: Dimensione del batch (default: `16`).

Esempio per una BiLSTM veloce su CPU:
```bash
poetry run python pipeline_exam/src/step03_train.py --model-id bilstm --device cpu --epochs 10 --lr 0.001
```

## 🔄 Flusso di Valutazione (Step 04)
Una volta addestrati i modelli, la valutazione processa automaticamente tutti i file `.pth` salvati nella cartella `models/`.

```bash
poetry run python pipeline_exam/src/step04_evaluate.py
# Oppure tramite lo script bash con logging:
./run_step04_with_logs.sh
```

**Output generati:**
1. **Classification Report:** Precision, Recall e F1-Score in console per ogni modello.
2. **Strict Metrics:** Le metriche vengono calcolate rimuovendo il tag vuoto `O` per fornire un quadro reale sulla predizione delle malattie ed evitare l'illusione statistica da class imbalance.
3. **Confusion Matrices:** Vengono salvate automaticamente in `plots/` delle heatmap (`.png`) generate tramite Seaborn, per confrontare visivamente i falsi positivi/negativi e la capacità predittiva delle varie architetture.