import os
from dataclasses import dataclass
from pathlib import Path

import lightning as L
import numpy as np
from torch.utils.data import Dataset, DataLoader
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from models.RNN import RNN, RNNClassifier

from utils.analytics import get_result_from_file
from utils.config import Config
from utils.helper import collate_fn

def train_rnn_model_with_parameters(
    embedding_matrix: np.ndarray,
    train_dataset: Dataset,
    val_dataset: Dataset,
    batch_size: int,
    learning_rate: float,
    optimizer_name: str,
    hidden_dim: int,
    num_layers: int,
    sentence_representation_type: str = "last",
    show_progress: bool = True,
    seed: int = 42,
    log_dir: str = "rnn/test",
    early_stopping_patience: int = 3,
    freeze_embedding: bool = True,
    rnn_type: str = "RNN",
    bidirectional: bool = False,
    dropout: float = 0.0,
    embedding_dropout: float = 0.0,
    weight_decay: float = 0.0,
):

    min_epochs = 0
    max_epochs = 10_000
    num_workers = os.cpu_count() // 2

    L.seed_everything(seed)

    _rnn_model = RNN(
        embedding_matrix=embedding_matrix,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        output_dim=6,
        sentence_representation_type=sentence_representation_type,
        freeze_embedding=freeze_embedding,
        rnn_type=rnn_type,
        bidirectional=bidirectional,
        dropout=dropout,
        embedding_dropout=embedding_dropout,
    )

    model = RNNClassifier(
        rnn_model=_rnn_model,
        optimizer_name=optimizer_name,
        lr=learning_rate,
        show_progress=show_progress,
        weight_decay=weight_decay,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        multiprocessing_context='spawn',
        persistent_workers=True
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        multiprocessing_context='spawn',
        persistent_workers=True
    )

    # Train model.
    log_file_name = f"{log_dir}/batch_size_{batch_size}-lr_{learning_rate}-optimizer_{optimizer_name}-hidden_dim_{hidden_dim}-num_layers_{num_layers}-sr_type_{sentence_representation_type}-freeze_{freeze_embedding}-rnn_type_{rnn_type}-bidirectional_{bidirectional}"

    # Skip if run before
    if list(Path().rglob(log_file_name)):
        print(f"[Skipping] {log_file_name}")
        result = get_result_from_file(f"tb_logs/{log_file_name}")

        return result["val_acc"]  # for optuna

    logger = TensorBoardLogger("tb_logs", name=log_file_name)

    callbacks = [
        # EarlyStopping(
        #     monitor="val_loss",
        #     mode="min",
        #     patience=early_stopping_patience,
        #     min_delta=1e-4,
        # ),
        EarlyStopping(
            monitor="val_acc",
            mode="max",
            patience=early_stopping_patience,
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
        accelerator="auto",
        log_every_n_steps=5,
        gradient_clip_val=1.0,
    )

    trainer.fit(
        model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
    )

    result = get_result_from_file(f"tb_logs/{log_file_name}")

    return result["val_acc"]  # for optuna
