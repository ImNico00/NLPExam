import torch
import torch.nn as nn
from transformers import AutoModelForTokenClassification, AutoConfig
from pipeline_exam.src.schemas import CANONICAL_MODELS

def get_model(model_id: str, vocab_size: int, num_classes: int, hf_token : str | None, **kwargs) -> nn.Module:
    """
    Factory function per istanziare il modello corretto in base all'ID.
    """
    model_id = model_id.lower()
    
    if model_id == "bilstm":
        return BiLSTM_NER(
            vocab_size=vocab_size,
            num_classes=num_classes,
            embedding_dim=kwargs.get("embedding_dim", 128),
            hidden_dim=kwargs.get("hidden_dim", 256)
        )
    elif model_id == "bert_ner":
        return TransformerNER(
            model_name=CANONICAL_MODELS["bert"], 
            num_classes=num_classes,
            hf_token=hf_token
        )
    elif model_id == "biobert_ner":
        return TransformerNER(
            model_name=CANONICAL_MODELS["biobert"], 
            num_classes=num_classes,
            hf_token=hf_token
        )
        
    elif model_id == "dummy_baseline":
        raise NotImplementedError("Baseline non implementata.")
        
    else:
        raise ValueError(f"Errore: model_id '{model_id}' non riconosciuto. Scegli tra: ['bilstm', 'bert_ner', 'biobert_ner']")

class BiLSTM_NER(nn.Module):
    def __init__(
        self, 
        vocab_size: int, 
        num_classes: int, 
        embedding_dim: int = 128, 
        hidden_dim: int = 256, 
        padding_idx: int = 0
    ):
        super(BiLSTM_NER, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # 1. Layer di Embedding
        # padding_idx=0 indica a PyTorch di ignorare il token [PAD] e lasciarlo a 0
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, 
            embedding_dim=embedding_dim, 
            padding_idx=padding_idx
        )
        
        # 2. Layer Bi-LSTM
        # batch_first=True si aspetta i tensori in formato (Batch, Sequenza, Feature)
        self.lstm = nn.LSTM(
            input_size=embedding_dim, 
            hidden_size=hidden_dim, 
            num_layers=1,             # Possiamo aumentare il numero di strati (es. 2)
            bidirectional=True,       # Cruciale per leggere il contesto a destra e sinistra
            batch_first=True
        )
        
        # 3. Layer Lineare
        # hidden_dim * 2 perché abbiamo due direzioni (forward e backward)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Definisce il flusso dei dati (Forward Pass).
        x è il nostro tensore di token ID: forma (batch_size, seq_len)
        """
        # (batch_size, seq_len) -> (batch_size, seq_len, embedding_dim)
        embedded = self.embedding(x)
        
        # Passiamo gli embeddings alla LSTM
        # lstm_out: (batch_size, seq_len, hidden_dim * 2)
        lstm_out, _ = self.lstm(embedded)
        
        # Passiamo l'output al layer finale per ottenere le probabilità per ogni classe BIO
        # logits: (batch_size, seq_len, num_classes)
        logits = self.fc(lstm_out)
        
        return logits

class TransformerNER(nn.Module):
    def __init__(self, model_name: str, num_classes: int, hf_token : str | None):
        super(TransformerNER, self).__init__()
        self.hf_token = hf_token
        self.config = AutoConfig.from_pretrained(model_name, num_labels=num_classes, token=self.hf_token)
        self.transformer = AutoModelForTokenClassification.from_pretrained(
            model_name, 
            config=self.config,
            token=self.hf_token
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Calcoliamo la maschera dinamicamente (0 è il padding di BERT)
        attention_mask = (x != 0).long()
        outputs = self.transformer(input_ids=x, attention_mask=attention_mask)
        return outputs.logits