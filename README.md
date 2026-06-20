# Clinical NER Pipeline - NLP Exam

Repository contenente l’implementazione di una pipeline end-to-end di Machine Learning per l’estrazione automatica di entità cliniche tramite Named Entity Recognition, sviluppata per l’esame di NLP.
La pipeline automatizza l’intero flusso di lavoro: annotazione in formato BIO e addestramento comparativo di reti neurali classiche (BiLSTM-CRF) e modelli Transformer con tecniche di domain adaptation. Il sistema consente inoltre di generare diverse pseudo-Ground Truth a partire da approcci rule-based (Dizionari/Pattern) e modelli pre-addestrati (SciBERT, BC5CDR), permettendo il confronto tra differenti strategie di annotazione e training (Weak Supervision).

## 🚀 Architettura del Sistema

Il progetto non si limita al training di un singolo modello, ma implementa una pipeline modulare e riproducibile in stile MLOps:
- **Data Engineering & Annotation (Step 01 & 02):** Generazione procedurale di referti medici sintetici. L'annotazione non è fissa, ma modulare: il sistema permette di generare Ground Truth diverse usando approcci Rule-Based (Dizionario) o pre-trained Transformer (scispacy BC5CDR e SciBERT). Lo Step 02 implementa una logica di Auto-Discovery per processare automaticamente tutte le varianti generate.
- **Deep Learning Baselines (Step 03):** 
   - **BiLSTM Pura:** Sviluppata custom in PyTorch con embedding statici per stabilire una baseline iniziale di performance basata su decisioni locali.
   - **BiLSTM-CRF:** BiLSTM custom in PyTorch con l'aggiunta di un livello *Conditional Random Field* (CRF) sfruttando una matrice di transizione ottimizzata apprendendo le restrizioni strutturali dello spazio delle etichette BIO ed effettua la decodifica della sequenza globale migliore a runtime tramite l'**Algoritmo di Viterbi**.
- **Attention & Domain Adaptation (Step 03):** Fine-tuning di Transformer Hugging Face. Confronto tra un modello generico (`bert-base-cased`) e un modello di dominio clinico (`emilyalsentzer/Bio_ClinicalBERT` addestrato su MIMIC-III).
- **Evaluation Engine & Error Analysis (Step 04):** Modulo di validazione rigorosa che calcola lo *Strict Entity F1-Score* (escludendo programmaticamente l'over-rappresentazione della classe `O`) e genera Matrici di Confusione comparative. Inoltre, il modulo estrae automaticamente un log degli errori (.csv) contenente l'intera frase di contesto, facilitando l'ispezione visiva dei falsi positivi e dei disallineamenti di dominio.

## 🏆 Metodologia di Valutazione: Il Gold Standard (Single Source of Truth)
Una delle sfide principali nell'utilizzo di diverse strategie di annotazione automatica (Silver Standard) è garantire che la valutazione finale dei modelli sia scientificamente rigorosa, equa e confrontabile (apples-to-apples).
Per risolvere questo problema, il progetto adotta un approccio basato su un'unica Single Source of Truth:
- `Costruzione Manuale`: Un sottoinsieme isolato di referti medici è stato escluso dalle fasi di training automatico e annotato meticolosamente a mano seguendo rigorose linee guida cliniche (tassonomia completa: DRUG, DISEASE, PROCEDURE, ANATOMY).
- `Test Set Fisso e Condiviso`: Questo dataset costituisce il vero e proprio Gold Standard Test Set del progetto. È un file fisso e immutabile (formato CoNLL TSV) che risiede permanentemente all'interno della cartella data/processed/ (es. processed/golden-test.tsv).
- `Allineamento Dinamico`: Durante la fase di Evaluation (Step 04), tutti i modelli addestrati – indipendentemente dalla pseudo-Ground Truth da cui derivano – vengono valutati esclusivamente su questo file. Uno script di allineamento della tassonomia si occupa di adattare dinamicamente le etichette del Gold Standard alle capacità specifiche di ciascun modello in fase di inferenza, eliminando la necessità di avere file di test multipli e garantendo metriche di confronto assolute e inattaccabili.

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
3. **Setup Architettura Biomedica:** Installa la versione corretta di spaCy e il modello clinico scispaCy. Per garantire la riproducibilità ed evitare conflitti di tokenizzazione, il sistema fissa spaCy alla versione 3.7.x:
   ```bash
   poetry run pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz
   poetry run pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_scibert-0.5.4.tar.gz
   ```

## 🖥 Esecuzione della Pipeline

La pipeline è divisa in step logici pensati per essere eseguiti tramite riga di comando. 

### 1. Preparazione Dati e Annotazione (Step 01, 02)
Puoi specificare quale modello usare per generare la Ground Truth tramite il flag `--gt-model` (`scibc5cdr`, `scibert`, `dictionary`):
```bash
./script/run_step01_with_logs.sh --gt-model scibert
./script/run_step02_with_logs.sh
```

### 2. Addestramento Modelli (Step 03)
Avvia il training loop. Grazie all'Auto-Discovery, se non specifichi percorsi manuali, lo script addestrerà il modello selezionato su tutte le Ground Truth generate in precedenza, salvando i pesi in sottocartelle dedicate (models/<gt_model>/<model_id>_model.pth):
```bash
./script/run_step03_with_logs.sh --model-id biobert_ner
```

### Configurazione Avanzata (CLI) per il Training
Il parser `argparse` permette di sovrascrivere gli iperparametri per sperimentare:
- `--model-id`: Architettura da addestrare (`bilstm`, `bilstm_crf`, `bert_ner`, `biobert_ner`, `stanza`).
- `--epochs`: Epoche di addestramento (default: `25`). Il sistema implementa l'*Early Checkpointing* salvando automaticamente i pesi relativi alla migliore Validation Loss.
- `--lr`: Learning rate. Il codice lo scala dinamicamente (es. `2e-5`) in caso di fine-tuning Transformer per evitare *catastrophic forgetting*.
- `--batch-size`: Dimensione del batch (default: `16`).

Esempio per una BiLSTM veloce su CPU:
```bash
./script/run_step03_with_logs.sh --model-id bilstm --device cpu --epochs 10 --lr 0.001
```

## 🔄 Flusso di Valutazione (Step 04)
Una volta addestrati i modelli, la valutazione processa automaticamente tutte le sottocartelle in `models/`, accoppiandole con i rispettivi Test Set e Vocabolari.

```bash
./script/run_step04_with_logs.sh
```

**Output generati (salvati in `evaluations/evaluations_groundtruth_<gt_model>/`):**
1. **Classification Report:** Precision, Recall e F1-Score globale salvati in un recap file `models_evaluation_summary.csv`.
2. **Confusion Matrices:** Heatmap (`.png`) Full ed Entity-Only per confrontare visivamente la capacità predittiva delle architetture.
3. **Contextual Error Analysis:** Un file `_error_analysis.csv` che traccia ogni singola misclassificazione affiancandola alla frase originale pulita, essenziale per l'analisi clinica del modello.