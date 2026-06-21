from pathlib import Path

import torch
import torch.nn as nn
from torchcrf import CRF
import stanza
from transformers import AutoModelForTokenClassification, AutoConfig
from pipeline_exam.src.schemas import CANONICAL_MODELS
from typing import Dict, List, Optional, Any

def get_pytorch_model(model_id: str, vocab_size: int, num_classes: int, hf_token: Optional[str] = None, **kwargs) -> nn.Module:
    """Factory esclusiva per i modelli PyTorch."""
    model_id = model_id.lower()
    
    if model_id == "bilstm":
        return BiLSTM_NER(
            vocab_size=vocab_size,
            num_classes=num_classes,
            embedding_dim=kwargs.get("embedding_dim", 128),
            hidden_dim=kwargs.get("hidden_dim", 256),
            padding_idx=kwargs.get("padding_idx", 0),
            padding_tag_idx=kwargs.get("padding_tag_idx", 0)
        )
    elif model_id == "bilstm_crf":
        return BiLSTM_CRF_NER(
            vocab_size=vocab_size,
            num_classes=num_classes,
            embedding_dim=kwargs.get("embedding_dim", 128),
            hidden_dim=kwargs.get("hidden_dim", 256),
            padding_idx=kwargs.get("padding_idx", 0),
            padding_tag_idx=kwargs.get("padding_tag_idx", 0)
        )  
    elif model_id in ["bert_ner", "biobert_ner"]:
        model_name = CANONICAL_MODELS["bert"] if model_id == "bert_ner" else CANONICAL_MODELS["biobert"]
        return TransformerNER(
            model_name=model_name,
            num_classes=num_classes,
            hf_token=hf_token
        )
    else:
        raise ValueError(f"Modello PyTorch '{model_id}' non riconosciuto. Scegli tra: bilstm, bilstm_crf, bert_ner, biobert_ner")

class BiLSTM_NER(nn.Module):
    def __init__(
            self, 
            vocab_size: int, 
            num_classes: int, 
            embedding_dim: int = 128, 
            hidden_dim: int = 256, 
            padding_idx: int = 0, 
            padding_tag_idx: int = 0,
            dropout_p: float = 0.5
            ):
        super().__init__()
        self.num_classes = num_classes
        self.padding_idx = padding_idx
        self.padding_tag_idx = padding_tag_idx

        self.dropout = nn.Dropout(p=dropout_p)
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=1, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        # Loss function integrata
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=self.padding_tag_idx)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None, **kwargs) -> Dict[str, Any]:

        embedded = self.embedding(input_ids)
        embedded = self.dropout(embedded)

        lstm_out, _ = self.lstm(embedded)
        lstm_out = self.dropout(lstm_out)
        logits = self.fc(lstm_out)
        
        loss = None
        if labels is not None:
            # Flattening per CrossEntropy: (batch_size * seq_len, num_classes)
            loss = self.loss_fct(logits.view(-1, self.num_classes), labels.view(-1))
            
        return {"loss": loss, "logits": logits, "predictions": torch.argmax(logits, dim=-1)}

class BiLSTM_CRF_NER(BiLSTM_NER):
    def __init__(
            self, 
            vocab_size: int, 
            num_classes: int, 
            embedding_dim: int = 128, 
            hidden_dim: int = 256, 
            padding_idx: int = 0, 
            padding_tag_idx: int = 0,
            dropout_p: float = 0.5
            ):
        super().__init__(vocab_size, num_classes, embedding_dim, hidden_dim, padding_idx, padding_tag_idx, dropout_p)
        self.crf = CRF(num_tags=num_classes, batch_first=True)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None, **kwargs) -> Dict[str, Any]:
        embedded = self.embedding(input_ids)
        embedded = self.dropout(embedded)

        lstm_out, _ = self.lstm(embedded)
        lstm_out = self.dropout(lstm_out)
        emissions = self.fc(lstm_out)
        
        # Gestione unificata della maschera
        if attention_mask is None:
            mask = (input_ids != self.padding_idx).bool()
        else:
            mask = attention_mask.bool()

        loss = None
        if labels is not None:
            labels_cloned = labels.clone()
            labels_cloned[~mask] = self.padding_tag_idx
            loss = -self.crf(emissions, labels_cloned, mask=mask, reduction="mean")

        # Il CRF restituisce una lista di liste di interi come predizioni decodificate
        predictions = self.crf.decode(emissions, mask=mask)
        
        return {"loss": loss, "logits": emissions, "predictions": predictions}

class TransformerNER(nn.Module):
    def __init__(self, model_name: str, num_classes: int, hf_token: Optional[str]):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name, num_labels=num_classes, token=hf_token)
        self.transformer = AutoModelForTokenClassification.from_pretrained(model_name, config=self.config, token=hf_token)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None, **kwargs) -> Dict[str, Any]:
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )
        predictions = torch.argmax(outputs.logits, dim=-1) if outputs.logits is not None else None
        return {"loss": outputs.loss, "logits": outputs.logits, "predictions": predictions}
    

class StanzaMedicalNER:
    """Non eredita da nn.Module, poiché è una pipeline di inferenza basata su testo."""
    def __init__(self, lang: str = "en", custom_model_path: Optional[Path] = None):
        self.lang = lang
        if custom_model_path:
            self.pipeline = stanza.Pipeline(lang=self.lang, processors='tokenize,ner', ner_model_path=str(custom_model_path), tokenize_no_ssplit=True)
        else:
            stanza.download(lang=self.lang, package="mimic", processors={'ner': 'i2b2'})
            self.pipeline = stanza.Pipeline(lang=self.lang, package="mimic", processors={'ner': 'i2b2'}, tokenize_no_ssplit=True)

    def predict(self, text_batch: List[str]) -> List[List[Dict[str, Any]]]:
        batch_entities = []
        for text in text_batch:
            doc = self.pipeline(text)
            entities = [{"text": ent.text, "type": ent.type, "start_char": ent.start_char, "end_char": ent.end_char} for ent in doc.entities]
            batch_entities.append(entities)
        return batch_entities