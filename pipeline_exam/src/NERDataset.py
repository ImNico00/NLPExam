import pandas as pd

from torch.utils.data import Dataset
from transformers import AutoTokenizer
from typing import List, Tuple, Dict
import torch
from pathlib import Path
from torch.nn.utils.rnn import pad_sequence

class Vocabulary:
    def __init__(self):
        # [PAD] = 0 (Padding), [UNK] = 1 (Sconosciuto)
        self.word2idx: Dict[str, int] = {"[PAD]": 0, "[UNK]": 1}
        self.idx2word: Dict[int, str] = {0: "[PAD]", 1: "[UNK]"}
        
        # Per i tag, l'ID 0 è sempre il padding
        self.tag2idx: Dict[str, int] = {"[PAD]": 0}
        self.idx2tag: Dict[int, str] = {0: "[PAD]"}
        
    def build_vocab(self, df: pd.DataFrame):
        for token in df['Token'].dropna().unique():
            if token not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[token] = idx
                self.idx2word[idx] = token
                
        for tag in df['BIO_Tag'].dropna().unique():
            if tag not in self.tag2idx:
                idx = len(self.tag2idx)
                self.tag2idx[tag] = idx
                self.idx2tag[idx] = tag

class NERDataset(Dataset):
    def __init__(self, file_path: Path, vocab: Vocabulary | None = None):
        self.sentences: List[List[str]] = []
        self.labels: List[List[str]] = []
        
        self.load_tokenized_dataset(file_path)
        
        self.vocab = vocab if vocab else Vocabulary()
        if not vocab:
            df = pd.read_csv(file_path, sep='\t', keep_default_na=False)
            df = df[df['Token'] != '']
            self.vocab.build_vocab(df)

    def load_tokenized_dataset(self, file_path: Path):
        df = pd.read_csv(file_path, sep='\t')
        df = df.dropna(subset=['Token', 'BIO_Tag'])
        for _, group in df.groupby('Sentence_ID'):
            tokens = group['Token'].tolist()
            tags = group['BIO_Tag'].tolist()
            self.sentences.append(tokens)
            self.labels.append(tags)

    def __len__(self) -> int:
        return len(self.sentences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sentence = self.sentences[idx]
        tags = self.labels[idx]
        
        token_ids = [self.vocab.word2idx.get(w, self.vocab.word2idx["[UNK]"]) for w in sentence]
        tag_ids = [self.vocab.tag2idx[t] for t in tags]
        
        return torch.tensor(token_ids, dtype=torch.long), torch.tensor(tag_ids, dtype=torch.long)
    
class TransformerNERDataset(Dataset):
    def __init__(self, file_path: Path, model_name: str, hf_token: str | None, vocab: Vocabulary | None = None):
        """Dataset specifico per modelli HuggingFace (Sub-Word Tokenization)"""
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        
        self.sentences = []
        self.labels = []
        
        df = self.load_tokenized_dataset(file_path)
        
        self.vocab = vocab if vocab else Vocabulary()
        if not vocab:
            self.vocab.build_vocab(df)

    def load_tokenized_dataset(self, file_path: Path) -> pd.DataFrame:
        df = pd.read_csv(file_path, sep='\t', keep_default_na=False)
        df = df[df['Token'] != ''] 
        for _, group in df.groupby('Sentence_ID'):
            tokens = group['Token'].tolist()
            tags = group['BIO_Tag'].tolist()
            self.sentences.append(tokens)
            self.labels.append(tags)
        return df

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        words = self.sentences[idx]
        tags = self.labels[idx]

        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            truncation=True,
            max_length=256
        )
        
        # Allineamento delle etichette!
        word_ids = encoding.word_ids() # Mappa i sub-words all'indice della parola originale
        label_ids = []
        previous_word_idx = None
        
        for word_idx in word_ids:
            if word_idx is None:
                # Token speciali ([CLS], [SEP]) -> ID -100 per essere ignorati dalla Loss
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                # È il PRIMO sub-token di una parola: gli diamo l'etichetta vera
                tag_str = tags[word_idx]
                label_ids.append(self.vocab.tag2idx[tag_str])
            else:
                # È un sub-token successivo di una parola già etichettata: lo ignoriamo
                label_ids.append(-100)
            previous_word_idx = word_idx

        return torch.tensor(encoding["input_ids"], dtype=torch.long), torch.tensor(label_ids, dtype=torch.long)
    

def transformer_collate(batch):
    """Il collate_fn specifico per i Transformers"""
    input_ids = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    
    # Pad input_ids con 0
    padded_input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    # Pad labels con -100 (fondamentale per non alterare la Loss di BERT)
    padded_labels = pad_sequence(labels, batch_first=True, padding_value=-100)
    
    return padded_input_ids, padded_labels

def pad_collate(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    sentences = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    
    padded_sentences = pad_sequence(sentences, batch_first=True, padding_value=0)
    padded_labels = pad_sequence(labels, batch_first=True, padding_value=0)
    
    return padded_sentences, padded_labels