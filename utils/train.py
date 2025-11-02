# import os
# from dataclasses import dataclass
# from pathlib import Path

# import lightning as L
# import numpy as np
# import torch
# from torch.utils.data import Dataset, dataloader 
# from torchtext.data.dataset import Dataset as TorchTextDataset
# from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
# from lightning.pytorch.loggers import TensorBoardLogger
# from models.RNN import RNN, RNNClassifier
# from torch.utils.data import DataLoader

# from utils.analytics import get_result_from_file
# from utils.config import Config

# class TorchTextDatasetWrapper(Dataset):
#     """Wrapper to convert TorchText Dataset to PyTorch Dataset"""
    
#     def __init__(self, torchtext_dataset: TorchTextDataset):
#         self.dataset = torchtext_dataset
#         self.examples = list(torchtext_dataset.examples)
    
#     def __len__(self):
#         return len(self.examples)
    
#     def __getitem__(self, idx):
#         example = self.examples[idx]
#         # Assuming your TorchText dataset has 'text' and 'label' fields
#         # Adjust field names based on your actual dataset structure
#         text_indices = example.text  # Should already be numericalized
#         label = example.label
        
#         return {
#             'indexes': torch.tensor(text_indices, dtype=torch.long),
#             'label': torch.tensor(label, dtype=torch.long),
#             'original_len': len(text_indices)
#         }


# def collate_fn(batch):
#     """Custom collate function to handle variable length sequences"""
#     indexes = [item['indexes'] for item in batch]
#     labels = torch.stack([item['label'] for item in batch])
#     original_lens = torch.tensor([item['original_len'] for item in batch])
    
#     # Pad sequences to the same length
#     max_len = max(original_lens)
#     padded_indexes = torch.zeros(len(indexes), max_len, dtype=torch.long)
    
#     for i, seq in enumerate(indexes):
#         padded_indexes[i, :len(seq)] = seq
    
#     return {
#         'indexes': padded_indexes,
#         'label': labels,
#         'original_len': original_lens
#     }

# def train_rnn_model_with_parameters(
#     embedding_matrix: np.ndarray,
#     train_dataset: TorchTextDataset,
#     val_dataset: TorchTextDataset,
#     batch_size: int,
#     learning_rate: float,
#     optimizer_name: str,
#     hidden_dim: int,
#     num_layers: int,
#     sentence_representation_type: str = "last",
#     show_progress: bool = True,
#     seed: int = Config.SEED,
#     log_dir: str = "rnn/test",
#     early_stopping_patience: int = 3,
#     freeze_embedding: bool = False,
#     rnn_type: str = "RNN",
#     bidirectional: bool = False,
# ):

#     min_epochs = 0
#     max_epochs = 10_000
#     num_workers = os.cpu_count() // 2

#     L.seed_everything(seed)

#     _rnn_model = RNN(
#         embedding_matrix=embedding_matrix,
#         hidden_dim=hidden_dim,
#         num_layers=num_layers,
#         output_dim=2,
#         sentence_representation_type=sentence_representation_type,
#         freeze_embedding=freeze_embedding,
#         rnn_type=rnn_type,
#         bidirectional=bidirectional,
#     )

#     model = RNNClassifier(
#         rnn_model=_rnn_model,
#         optimizer_name=optimizer_name,
#         lr=learning_rate,
#         show_progress=show_progress,
#     )

#     wrap_train_dataset = TorchTextDatasetWrapper(train_dataset)
#     wrap_val_dataset = TorchTextDatasetWrapper(val_dataset)

#     train_dataloader = DataLoader(
#         wrap_train_dataset,
#         batch_size=batch_size,
#         shuffle=True,
#         num_workers=num_workers,
#     )
#     val_dataloader = DataLoader(
#         val_dataset,
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=num_workers,
#     )

#     # Train model.
#     log_file_name = f"{log_dir}/batch_size_{batch_size}-lr_{learning_rate}-optimizer_{optimizer_name}-hidden_dim_{hidden_dim}-num_layers_{num_layers}-sr_type_{sentence_representation_type}-freeze_{freeze_embedding}-rnn_type_{rnn_type}-bidirectional_{bidirectional}"

#     # Skip if run before
#     if list(Path().rglob(log_file_name)):
#         print(f"[Skipping] {log_file_name}")
#         result = get_result_from_file(f"tb_logs/{log_file_name}")

#         return result["val_acc"]  # for optuna

#     logger = TensorBoardLogger("tb_logs", name=log_file_name)

#     callbacks = [
#         EarlyStopping(
#             monitor="val_loss",
#             mode="min",
#             patience=early_stopping_patience,
#             min_delta=1e-4,
#         ),
#         EarlyStopping(
#             monitor="val_acc",
#             mode="max",
#             patience=early_stopping_patience * 5,
#             min_delta=1e-4,
#         ),
#         ModelCheckpoint(
#             monitor="val_loss",
#             save_top_k=1,
#             mode="min",
#         ),
#     ]
#     trainer = L.Trainer(
#         default_root_dir="models/",
#         callbacks=callbacks,
#         min_epochs=min_epochs,
#         max_epochs=max_epochs,
#         logger=logger,
#         accelerator="cpu",
#         log_every_n_steps=5,
#     )

#     trainer.fit(
#         model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
#     )

#     result = get_result_from_file(f"tb_logs/{log_file_name}")

#     return result["val_acc"]  # for optuna


# @dataclass
# class OptimizerArgs:
#     optimizer_name: str = "Adam"
#     learning_rate: float = 1e-3

#     def __post_init__(self):
#         if self.learning_rate <= 0:
#             raise ValueError("learning_rate must be a positive float")


# @dataclass
# class DataArgs:
#     batch_size: int
#     train_dataset: TorchTextDataset
#     val_dataset: TorchTextDataset
#     shuffle_train: bool = True
#     shuffle_val: bool = False

import os
from dataclasses import dataclass
from pathlib import Path

import lightning as L
import numpy as np
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, Dataset
from torchtext.data.dataset import Dataset as TorchTextDataset


class TorchTextDatasetWrapper(Dataset):
    """Wrapper to convert TorchText Dataset to PyTorch Dataset"""
    
    def __init__(self, torchtext_dataset: TorchTextDataset, text_field, label_field):
        """
        Args:
            torchtext_dataset: TorchText Dataset object
            text_field: The TEXT field object with vocab
            label_field: The LABEL field object with vocab
        """
        self.dataset = torchtext_dataset
        self.text_field = text_field
        self.label_field = label_field
        self.examples = list(torchtext_dataset.examples)
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Get the raw text and label from the example
        # TorchText stores them as attributes of the example object
        text = example.text
        label = example.label
        
        # Convert text to indices if not already done
        if isinstance(text, str):
            # Text is still a string, need to process it
            text_tokens = self.text_field.preprocess(text)
            text_indices = [self.text_field.vocab.stoi[token] for token in text_tokens]
        elif isinstance(text, list):
            # Check if it's a list of strings (tokens) or integers (indices)
            if len(text) > 0 and isinstance(text[0], str):
                # List of tokens, convert to indices
                text_indices = [self.text_field.vocab.stoi[token] for token in text]
            else:
                # Already indices
                text_indices = text
        else:
            raise ValueError(f"Unexpected text type: {type(text)}")
        
        # Convert label to index if it's a string
        if isinstance(label, str):
            label_idx = self.label_field.vocab.stoi[label]
        else:
            label_idx = label
        
        return {
            'indexes': torch.tensor(text_indices, dtype=torch.long),
            'label': torch.tensor(label_idx, dtype=torch.long),
            'original_len': len(text_indices)
        }


def collate_fn(batch):
    """Custom collate function to handle variable length sequences"""
    indexes = [item['indexes'] for item in batch]
    labels = torch.stack([item['label'] for item in batch])
    original_lens = torch.tensor([item['original_len'] for item in batch])
    
    # Pad sequences to the same length
    max_len = max(original_lens)
    padded_indexes = torch.zeros(len(indexes), max_len, dtype=torch.long)
    
    for i, seq in enumerate(indexes):
        padded_indexes[i, :len(seq)] = seq
    
    return {
        'indexes': padded_indexes,
        'label': labels,
        'original_len': original_lens
    }


def train_rnn_model_with_parameters(
    embedding_matrix: np.ndarray,
    train_dataset: TorchTextDataset,
    val_dataset: TorchTextDataset,
    batch_size: int,
    learning_rate: float,
    optimizer_name: str,
    hidden_dim: int,
    num_layers: int,
    text_field=None,  # ADD THIS: Pass the TEXT field
    label_field=None,  # ADD THIS: Pass the LABEL field
    sentence_representation_type: str = "last",
    show_progress: bool = True,
    seed: int = 42,
    log_dir: str = "rnn/test",
    early_stopping_patience: int = 3,
    freeze_embedding: bool = False,
    rnn_type: str = "RNN",
    bidirectional: bool = False,
):
    """
    Train RNN model with TorchText datasets.
    
    Args:
        text_field: The TEXT field object with vocabulary (required for numericalization)
        label_field: The LABEL field object with vocabulary (required for label conversion)
    """
    if text_field is None or label_field is None:
        raise ValueError("text_field and label_field must be provided for TorchText datasets")
    
    min_epochs = 0
    max_epochs = 10_000
    num_workers = os.cpu_count() // 2 if os.cpu_count() else 2
    
    L.seed_everything(seed)
    
    # Import your RNN and RNNClassifier here
    from models.RNN import RNN, RNNClassifier
    
    _rnn_model = RNN(
        embedding_matrix=embedding_matrix,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        output_dim=2,
        sentence_representation_type=sentence_representation_type,
        freeze_embedding=freeze_embedding,
        rnn_type=rnn_type,
        bidirectional=bidirectional,
    )
    
    model = RNNClassifier(
        rnn_model=_rnn_model,
        optimizer_name=optimizer_name,
        lr=learning_rate,
        show_progress=show_progress,
    )
    
    # Wrap TorchText datasets with field information
    wrapped_train = TorchTextDatasetWrapper(train_dataset, text_field, label_field)
    wrapped_val = TorchTextDatasetWrapper(val_dataset, text_field, label_field)
    
    # Create dataloaders with custom collate function
    train_dataloader = DataLoader(
        wrapped_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    
    val_dataloader = DataLoader(
        wrapped_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    
    # Train model
    log_file_name = f"{log_dir}/batch_size_{batch_size}-lr_{learning_rate}-optimizer_{optimizer_name}-hidden_dim_{hidden_dim}-num_layers_{num_layers}-sr_type_{sentence_representation_type}-freeze_{freeze_embedding}-rnn_type_{rnn_type}-bidirectional_{bidirectional}"
    
    # Skip if run before
    if list(Path().rglob(log_file_name)):
        print(f"[Skipping] {log_file_name}")
        from utils.analytics import get_result_from_file
        result = get_result_from_file(f"tb_logs/{log_file_name}")
        return result["val_acc"]
    
    logger = TensorBoardLogger("tb_logs", name=log_file_name)
    
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=early_stopping_patience,
            min_delta=1e-4,
        ),
        EarlyStopping(
            monitor="val_acc",
            mode="max",
            patience=early_stopping_patience * 5,
            min_delta=1e-4,
        ),
        ModelCheckpoint(
            monitor="val_loss",
            save_top_k=1,
            mode="min",
        ),
    ]
    
    trainer = L.Trainer(
        default_root_dir="models/",
        callbacks=callbacks,
        min_epochs=min_epochs,
        max_epochs=max_epochs,
        logger=logger,
        accelerator="cpu",
        log_every_n_steps=5,
    )
    
    trainer.fit(
        model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
    )
    
    from utils.analytics import get_result_from_file
    result = get_result_from_file(f"tb_logs/{log_file_name}")
    return result["val_acc"]


@dataclass
class OptimizerArgs:
    optimizer_name: str = "Adam"
    learning_rate: float = 1e-3
    
    def __post_init__(self):
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be a positive float")


@dataclass
class DataArgs:
    batch_size: int
    train_dataset: TorchTextDataset
    val_dataset: TorchTextDataset
    shuffle_train: bool = True
    shuffle_val: bool = False
