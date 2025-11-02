import re
from typing import Any, List

import nltk
import torch
from torch.utils.data import Dataset
from torchtext.data import Dataset as TorchTextDataset
import numpy as np
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_context_average_embedding(
    sentence_tokens: List[str],
    oov_token: str,
    glove: Any = None,
    embedding_matrix: np.ndarray = None,
    index_from_word: dict = None,
) -> np.ndarray:
    """Generates an approximate embedding for an OOV word by averaging the
    embeddings of surrounding context words.

    Args:
        sentence_tokens (List[str]): A list of tokens from a sentence.
        oov_token (str): The OOV word for which an approximate embedding is needed.
        glove : A pretrained glove embedding model
        embedding_matrix (np.ndarray, optional): Precomputed embedding matrix.
        index_from_word (dict, optional): Dictionary mapping words to row indices
            in the embedding_matrix.

    Returns:
        np.ndarray: The averaged embedding vector for the OOV word.
        Returns a zero vector if no in-vocabulary context words are found.
    """

    def _get_embedding(word: str) -> np.ndarray:
        """Internal helper to get embedding from any model and convert to numpy."""
        if word and word in glove.stoi:
            return glove[word]
        elif (
            embedding_matrix is not None
            and index_from_word is not None
            and word in index_from_word
        ):
            return embedding_matrix[index_from_word[word]]

        return None

    context_embeddings = [
        _get_embedding(word)
        for word in sentence_tokens
        if word != oov_token and _get_embedding(word) is not None
    ]

    if context_embeddings:
        return np.mean(context_embeddings, axis=0)

    print("ZERO")
    return np.zeros(glove.dim)

class SentenceDataset(Dataset):
    def __init__(self, torchtext_examples, index_from_word, label_vocab):
        self.examples = torchtext_examples
        self.index_from_word = index_from_word
        self.label_vocab = label_vocab
        
        # Use index 1 for <unk> (unknown). 0 is often <pad>.
        # Check your index_from_word to see what your <unk> token is.
        # If it doesn't have one, define it.
        if "<unk>" in self.index_from_word:
            self.UNK_INDEX = self.index_from_word["<unk>"]
        else:
            self.UNK_INDEX = 0 # A common default

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # --- THIS IS THE CHANGE ---
        # Convert tokens to indexes using the JSON-loaded dict
        # Use .get(token, self.UNK_INDEX) to handle out-of-vocabulary words
        tokens = [self.index_from_word.get(t, self.UNK_INDEX) for t in example.text]
        # --- END OF CHANGE ---
        
        length = len(tokens)
        
        # This part is still correct (using the LABEL.vocab)
        label = self.label_vocab.stoi[example.label]
        
        return {
            "indexes": torch.LongTensor(tokens),
            "original_len": length,
            "label": torch.LongTensor([label]).squeeze(0) # This is fine
        }

def collate_fn(batch):
    """Pads sequences to max length in batch and returns tensors"""
    batch_sorted = sorted(batch, key=lambda x: x['original_len'], reverse=True)
    sequences = [item['indexes'] for item in batch_sorted]
    lengths = [item['original_len'] for item in batch_sorted]
    labels = torch.stack([item['label'] for item in batch_sorted])

    sequences_padded = nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=1)
    lengths_tensor = torch.tensor(lengths, dtype=torch.long)

    return {
        'indexes': sequences_padded.to(device),
        'original_len': lengths_tensor.to(device),
        'label': labels.to(device)
    }

