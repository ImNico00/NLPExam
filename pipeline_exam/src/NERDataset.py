import pandas as pd

from torch.utils.data import Dataset
from transformers import AutoTokenizer
from typing import List, Tuple, Dict
import torch
from pathlib import Path
from torch.nn.utils.rnn import pad_sequence
from pipeline_exam.src.utils import align_gold_label_for_model

class Vocabulary:
    def __init__(self):
        self.pad_token = "[PAD]"
        self.unk_token = "[UNK]"

        self.pad_word_idx = 0
        self.unk_word_idx = 1
        self.pad_tag_idx = 0

        self.word2idx: Dict[str, int] = {
            self.pad_token: self.pad_word_idx,
            self.unk_token: self.unk_word_idx,
        }

        self.idx2word: Dict[int, str] = {
            self.pad_word_idx: self.pad_token,
            self.unk_word_idx: self.unk_token,
        }

        self.tag2idx: Dict[str, int] = {
            self.pad_token: self.pad_tag_idx,
        }

        self.idx2tag: Dict[int, str] = {
            self.pad_tag_idx: self.pad_token,
        }
        
    def build_vocab(self, df: pd.DataFrame):
        for token in df['Token'].dropna().unique():
            if token not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[token] = idx
                self.idx2word[idx] = token
                
        for tag in sorted(df["BIO_Tag"].dropna().unique()):
            if tag not in self.tag2idx:
                idx = len(self.tag2idx)
                self.tag2idx[tag] = idx
                self.idx2tag[idx] = tag

class NERDataset(Dataset):
    def __init__(self, file_path: Path, vocab: Vocabulary | None = None, gt_model_name: str = ""):
        self.sentences: List[List[str]] = []
        self.labels: List[List[str]] = []
        self.gt_model_name = gt_model_name
        
        df = self.load_tokenized_dataset(file_path)
        
        self.vocab = vocab if vocab else Vocabulary()
        if not vocab:
            self.vocab.build_vocab(df)

    def load_tokenized_dataset(self, file_path: Path):
        df = pd.read_csv(file_path, sep="\t", keep_default_na=False, dtype={"Token": str})
        df = df[(df["Token"] != "") & (df["BIO_Tag"] != "")]
        
        if self.gt_model_name:
            df["BIO_Tag"] = df["BIO_Tag"].apply(lambda t: align_gold_label_for_model(t, self.gt_model_name))
            
        for _, group in df.groupby("Sentence_ID", sort=False):
            tokens = [str(t) for t in group["Token"].tolist()] 
            tags = group["BIO_Tag"].tolist()
                
            self.sentences.append(tokens)
            self.labels.append(tags) 
            
        return df

    def __len__(self) -> int:
        return len(self.sentences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, List[str], List[str]]:
        sentence = self.sentences[idx]
        tags = self.labels[idx]
        
        token_ids = [self.vocab.word2idx.get(w, self.vocab.word2idx["[UNK]"]) for w in sentence]
        tag_ids = [self.vocab.tag2idx.get(t, self.vocab.tag2idx.get("O", 0)) for t in tags]
        
        return torch.tensor(token_ids, dtype=torch.long), torch.tensor(tag_ids, dtype=torch.long), sentence, tags
    
class TransformerNERDataset(NERDataset):
    def __init__(self, file_path: Path, model_name: str, hf_token: str | None, vocab: Vocabulary | None = None, gt_model_name: str = ""):
        """Dataset specifico per modelli HuggingFace. Eredita caricamento e stato da NERDataset."""
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            token=hf_token,
            use_fast=True
        )
        super().__init__(file_path=file_path, vocab=vocab, gt_model_name=gt_model_name)

    def __getitem__(self, idx):
        words = self.sentences[idx]
        tags = self.labels[idx]

        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            truncation=True,
            max_length=256,
            return_attention_mask=True
        )
        
        word_ids = encoding.word_ids()
        label_ids = []
        raw_tokens = [] 
        raw_tags = []
        
        previous_word_idx = None
        
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
                raw_tokens.append("[SPECIAL]") 
                raw_tags.append("[PAD]")
            elif word_idx != previous_word_idx:
                tag_str = tags[word_idx]
                
                # Pulizia: i tag sono GIÀ allineati dalla classe madre!
                label_id = self.vocab.tag2idx.get(tag_str, self.vocab.tag2idx.get("O", 0))
                label_ids.append(label_id)
                
                raw_tokens.append(words[word_idx]) 
                raw_tags.append(tag_str)
            else:
                label_ids.append(-100)
                raw_tokens.append(words[word_idx]) 
                raw_tags.append("[PAD]")
            previous_word_idx = word_idx

        return (
            {
                "input_ids": torch.tensor(encoding["input_ids"], dtype=torch.long),
                "attention_mask": torch.tensor(encoding["attention_mask"], dtype=torch.long),
            },
            torch.tensor(label_ids, dtype=torch.long),
            raw_tokens,
            raw_tags
        )
    

def transformer_collate(batch):
    input_ids = [item[0]["input_ids"] for item in batch]
    attention_masks = [item[0]["attention_mask"] for item in batch]
    labels = [item[1] for item in batch]
    raw_sentences = [item[2] for item in batch]
    raw_tags = [item[3] for item in batch]

    padded_input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    padded_attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)
    padded_labels = pad_sequence(labels, batch_first=True, padding_value=-100)

    max_len = padded_input_ids.size(1)
    flat_raw_words = []
    flat_raw_tags = []

    for seq, tags_seq in zip(raw_sentences, raw_tags):
        flat_raw_words.extend(seq)
        flat_raw_words.extend(["[PAD]"] * (max_len - len(seq)))
        flat_raw_tags.extend(tags_seq)
        flat_raw_tags.extend(["[PAD]"] * (max_len - len(tags_seq)))

    batch_x = {
        "input_ids": padded_input_ids,
        "attention_mask": padded_attention_masks,
    }

    return batch_x, padded_labels, flat_raw_words, flat_raw_tags


def pad_collate(batch):
    sentences = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    raw_sentences = [item[2] for item in batch] 
    raw_tags = [item[3] for item in batch]
    
    padded_sentences = pad_sequence(sentences, batch_first=True, padding_value=0)
    padded_labels = pad_sequence(labels, batch_first=True, padding_value=0)
    
    max_len = padded_sentences.size(1)
    flat_raw_words = []
    flat_raw_tags = []
    
    for seq, tags_seq in zip(raw_sentences, raw_tags):
        flat_raw_words.extend(seq)
        flat_raw_words.extend(["[PAD]"] * (max_len - len(seq)))
        flat_raw_tags.extend(tags_seq)
        flat_raw_tags.extend(["[PAD]"] * (max_len - len(tags_seq)))
        
    return padded_sentences, padded_labels, flat_raw_words, flat_raw_tags