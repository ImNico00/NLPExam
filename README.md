# Clinical NER Pipeline - NLP Exam

Repository contenente l’implementazione di una pipeline end-to-end di Machine Learning per l’estrazione automatica di entità cliniche tramite Named Entity Recognition, sviluppata per l’esame di NLP.
La pipeline automatizza l’intero flusso di lavoro: annotazione in formato BIO e addestramento comparativo di reti neurali classiche (BiLSTM-CRF), modelli Transformer e framework avanzati (Stanza) con tecniche di domain adaptation. Il sistema consente inoltre di generare diverse pseudo-Ground Truth a partire da approcci rule-based (Dizionari/Pattern) e modelli pre-addestrati (scispaCy), permettendo il confronto tra differenti strategie di annotazione e training (Weak Supervision).

## 🚀 Architettura del Sistema

Il progetto non si limita al training di un singolo modello, ma implementa una pipeline modulare e riproducibile in stile MLOps:
- **Data Engineering & Annotation (Step 01 & 02):** Generazione procedurale di referti medici e annotazione BIO. L'annotazione non è fissa, ma modulare: il sistema permette di generare Ground Truth diverse usando approcci Rule-Based (Dizionario custom) o modelli pre-trained scispaCy (en_ner_bc5cdr_md e en_ner_bionlp13cg_md). Lo step prevede lo split automatico (80-10-10) e l'esportazione sia in formato tabulare (TSV) che JSON per compatibilità con framework multipli (es. Stanza). Lo Step 02 implementa una logica di Auto-Discovery per processare automaticamente tutte le varianti generate, estraendo le mappature e serializzando i vocabolari (vocab.pkl) necessari ai layer di embedding in PyTorch.
- **Deep Learning Baselines (Step 03):**
   - **BiLSTM Pura:** Sviluppata custom in PyTorch con embedding statici per stabilire una baseline iniziale di performance basata su decisioni locali.
   - **BiLSTM-CRF:** BiLSTM custom in PyTorch con l'aggiunta di un livello Conditional Random Field (CRF) sfruttando una matrice di transizione ottimizzata apprendendo le restrizioni strutturali dello spazio delle etichette BIO ed effettua la decodifica della sequenza globale migliore a runtime tramite l'Algoritmo di Viterbi.
- **Attention & Domain Adaptation (Step 03):** 
   - **Transformers:** Fine-tuning PyTorch di Transformer tramite Hugging Face. Confronto tra un modello generico (bert_ner) e un modello di dominio clinico (biobert_ner).
   - **Stanza NER:** Integrazione del framework NLP di Stanford. Il sistema avvia automaticamente un sottoprocesso per eseguire il fine-tuning della sua architettura basata su BioBERT, consumando direttamente i file JSON generati nello Step 01.
- **Evaluation Engine & Error Analysis (Step 04):** Modulo di validazione rigorosa che calcola il Macro F1-Score (escludendo programmaticamente l'over-rappresentazione della classe O e dei sub-token) e genera Matrici di Confusione comparative. Inoltre, il modulo estrae automaticamente un log degli errori contestuale (.csv) che ricostruisce l'intera frase originale, facilitando l'ispezione visiva dei falsi positivi e dei disallineamenti di dominio.

## 🏆 Metodologia di Valutazione: Il Gold Standard (Single Source of Truth)
Una delle sfide principali nell'utilizzo di diverse strategie di annotazione automatica (Silver Standard) è garantire che la valutazione finale dei modelli sia scientificamente rigorosa, equa e confrontabile (apples-to-apples).
Per risolvere questo problema, il progetto adotta un approccio basato su un'unica Single Source of Truth:
- Costruzione Manuale: Un sottoinsieme isolato di referti medici è stato escluso dalle fasi di training automatico e annotato meticolosamente a mano seguendo rigorose linee guida cliniche (tassonomia completa: DRUG, DISEASE, PROCEDURE, ANATOMY).
- Test Set Fisso e Condiviso: Questo dataset costituisce il vero e proprio Gold Standard Test Set del progetto. È un file fisso e immutabile (formato CoNLL TSV) che risiede permanentemente all'interno della cartella data/processed/ (es. processed/golden-test.tsv).
- Allineamento Dinamico: Durante la fase di Evaluation (Step 04), tutti i modelli addestrati – indipendentemente dalla pseudo-Ground Truth da cui derivano – vengono valutati esclusivamente su questo file. Uno script di allineamento della tassonomia si occupa di adattare dinamicamente le etichette del Gold Standard alle capacità specifiche di ciascun modello in fase di inferenza, eliminando la necessità di avere file di test multipli e garantendo metriche di confronto assolute e inattaccabili.

## ⚙️ Setup dell'Ambiente

Il progetto utilizza Poetry per la gestione rigorosa delle dipendenze, gestendo attentamente l'allineamento delle versioni per i modelli biomedici.

### Prerequisiti
- Python 3.12
- Poetry
- Chiave API di Hugging Face salvata in un file .env (HUGGINGFACE_API_KEY=...) per scaricare i modelli protetti/custom.
- Supporto CUDA (opzionale ma fortemente raccomandato per il fine-tuning dei Transformer).

### Installazione

1. Clona il repository.
2. Installa le dipendenze di base tramite Poetry:
   poetry install
3. Setup Architettura Biomedica: Installa la versione corretta di spaCy e i modelli clinici scispaCy per la generazione della Ground Truth. Per garantire la riproducibilità ed evitare conflitti di tokenizzazione, il sistema fissa spaCy alla versione 3.7.x:
   poetry run pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz
   poetry run pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bionlp13cg_md-0.5.4.tar.gz

## 🖥 Esecuzione della Pipeline

La pipeline è divisa in step logici pensati per essere eseguiti tramite riga di comando. 

### 1. Preparazione Dati e Annotazione (Step 01 & Step 02)

Lo Step 01 esegue il data split (80% Train, 10% Validation, 10% Test) dal file grezzo medical_reports_english_translated.csv, tokenizza il testo e genera i tag BIO per le entità mediche. Crea inoltre file JSON dedicati al training per i modelli stanza.
Puoi specificare quale modello usare per generare la Ground Truth tramite il flag --gt-model (scibc5cdr [default], scibionlp, dictionary):
./script/run_step01_with_logs.sh --gt-model scibionlp

*I dati processati verranno salvati in una sottocartella dedicata alla strategia scelta: data/processed/<gt-model>/*

Lo Step 02 implementa l'Auto-Discovery: scansiona automaticamente tutte le sottocartelle generate nello step precedente, analizza i dataset di training (perizie_bio_train.tsv) ed estrae i metadati necessari per l'addestramento. Costruisce le mappature in indici (Token-to-Index, Tag-to-Index) e serializza i dizionari tramite Pickle.
./script/run_step02_with_logs.sh

*Output generato: Un file vocab.pkl salvato dinamicamente all'interno di ogni sottocartella di Ground Truth.*

### 2. Addestramento Modelli (Step 03)
Avvia il training loop. Grazie all'Auto-Discovery, lo script individua automaticamente le Ground Truth generate ed esegue l'addestramento iterativo per ciascuna, salvando l'output in sottocartelle dedicate (models/<dataset_name>/<model_id>_model.pth). Il sistema implementa nativamente l'Early Checkpointing, salvando alla fine delle epoche lo stato del modello che ha registrato la Validation Loss migliore.
./script/run_step03_with_logs.sh --model-id biobert_ner

### Configurazione Avanzata (CLI) per il Training
Il parser permette di sovrascrivere gli iperparametri per sperimentare:
- --model-id: Architettura da addestrare (bilstm, bilstm_crf, bert_ner, biobert_ner, stanza).
- --epochs: Epoche di addestramento (default: 25).
- --lr: Learning rate (default: 0.001). Il codice include un adattamento dinamico automatico: se si addestra un Transformer (bert_ner o biobert_ner) o stanza, scende in automatico a un rate prudente (es. 2e-5 o 5e-5) applicando lo scheduler AdamW per evitare il catastrophic forgetting.
- --batch-size: Dimensione del batch (default: 16).
- --embedding-dim / --hidden-dim: Parametri di rete (usati principalmente per le reti custom BiLSTM).

Esempio per l'addestramento di una rete BiLSTM pura su CPU:
./script/run_step03_with_logs.sh --model-id bilstm --device cpu --epochs 10

## 🔄 Flusso di Valutazione (Step 04)
Una volta addestrati i modelli, la valutazione processa automaticamente tutte le sottocartelle in models/. Di default, lo script impone la Single Source of Truth valutando tutti i modelli generati sul file golden-test.tsv. 

./script/run_step04_with_logs.sh

*Nota: Puoi disabilitare la valutazione incrociata e testare i modelli sui loro rispettivi test set estratti nello Step 01 (Silver Standard) passando il flag --no-golden-test.*

**Output generati (salvati in evaluations/evaluations_groundtruth_<gt_model>/):**
1. Classification Report: Recap globale delle performance (Accuracy e Macro F1-Score senza classe O) salvato in models_evaluation_summary.csv.
2. Confusion Matrices (.png): Generazione doppia per modello:
   - _confusion_matrix_full.png: Heatmap completa (inclusa la classe maggioritaria O).
   - _confusion_matrix_entity.png: Heatmap purificata sulle sole entità estratte, ideale per l'analisi dei misallineamenti semantici.
3. Contextual Error Analysis (_error_analysis.csv): File dettagliato che traccia ogni singola misclassificazione affiancandola alla frase originale pulita ricostruita parola per parola, fondamentale per la validazione clinica visiva.