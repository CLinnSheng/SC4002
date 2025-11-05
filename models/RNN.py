import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchmetrics.classification import MulticlassAccuracy


class RNN(nn.Module):
    def __init__(
        self,
        embedding_matrix: np.ndarray,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sentence_representation_type: str,
        freeze_embedding: bool = False,
        rnn_type: str = "RNN",
        bidirectional: bool = False,
        dropout: float = 0.0,
        embedding_dropout: float = 0.0,
        fc_dropout: float = 0.0,
    ):
        """
        RNN model with pretrained embeddings for text classification tasks.

        Args:
            embedding_matrix (Union[torch.Tensor, list]): Pretrained word embeddings matrix.
            hidden_dim (int): Dimensionality of the hidden layer in the RNN.
            output_dim (int): Dimensionality of the output layer.
            num_layers (int): Number of RNN layers.
            sentence_representation_type (str): Type of sentence representation to use
                                                ('last', 'max', 'average').
            freeze_embedding (Optional[bool]): Whether to freeze the embedding layer
                                               (default: True).
            rnn_type (str): Type of RNN to use, e.g., "RNN" or "GRU" or "LSTM" (default: "RNN").
            bidirectional (bool): If True, use a bidirectional RNN (default: False).
        """
        super(RNN, self).__init__()

        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.rnn_type = rnn_type
        self.dropout = dropout

        self.embedding_dropout = nn.Dropout(embedding_dropout)
        self.fc_dropout = nn.Dropout(fc_dropout)

        if sentence_representation_type not in ["last", "max", "average"]:
            raise Exception(
                "Invalid `sentence_representation_type`. Choose from 'last', 'max', or 'average'."
            )
        self.sentence_representation_type = sentence_representation_type

        # embedding layer
        _, embedding_dim = embedding_matrix.shape
        self.embedding = nn.Embedding.from_pretrained(
            torch.FloatTensor(embedding_matrix), freeze=freeze_embedding
        )

        # Choose RNN layer
        if rnn_type == "GRU":
            self.rnn = nn.GRU(
                embedding_dim,
                hidden_dim,
                num_layers,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=dropout if num_layers > 1 else 0
            )
        elif rnn_type == "LSTM":
            self.rnn = nn.LSTM(
                embedding_dim,
                hidden_dim,
                num_layers,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=dropout if num_layers > 1 else 0
            )
        elif rnn_type == "RNN":
            self.rnn = nn.RNN(
                embedding_dim, 
                hidden_dim, 
                num_layers,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=dropout if num_layers > 1 else 0 
            )
        else:
            raise ValueError("Invalid `rnn_type`. Choose from 'GRU', 'LSTM', or 'RNN'.")

        # Calculate RNN output dimension based on bidirectionality
        rnn_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        # fc layer
        self.fc = nn.Linear(rnn_output_dim, rnn_output_dim // 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(rnn_output_dim // 2, output_dim)

    def forward(self, sequences, original_len):
        embeddings = self.embedding(sequences)
        embeddings = self.embedding_dropout(embeddings)

        # Handle variable length sequences
        packed_input = pack_padded_sequence(
            embeddings,
            lengths=original_len.cpu(),
            enforce_sorted=False,
            batch_first=True,
        )

        packed_output, hidden = self.rnn(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        # extract sentence representation
        # if self.sentence_representation_type == "last":
        #     if self.rnn_type == "GRU" and self.bidirectional:
        #         sentence_representation = (
        #             hidden[-2:].transpose(0, 1).contiguous().view(sequences.size(0), -1)
        #         )
        #     elif self.rnn_type == "LSTM" and self.bidirectional:
        #         sentence_representation = torch.cat(
        #             (hidden[0][-1], hidden[1][-1]), dim=1
        #         )
        #     else:
        #         sentence_representation = hidden[-1]
        # elif self.sentence_representation_type == "max":
        #     sentence_representation, _ = torch.max(output, dim=1)
        # elif self.sentence_representation_type == "average":
        #     sentence_representation = torch.mean(output, dim=1)

        if self.sentence_representation_type == "last":
            if self.rnn_type == "LSTM":
                h_n, c_n = hidden  # Unpack LSTM hidden state tuple
                if self.bidirectional:
                    sentence_representation = torch.cat((h_n[-2], h_n[-1]), dim=1)
                else:
                    sentence_representation = h_n[-1]
            elif self.rnn_type == "GRU":
                if self.bidirectional:
                    sentence_representation = torch.cat((hidden[-2], hidden[-1]), dim=1)
                else:
                    sentence_representation = hidden[-1]
            else:  # Regular RNN
                sentence_representation = hidden[-1]
            
        # elif self.sentence_representation_type == "max":
        #     sentence_representation, _ = torch.max(output, dim=1)
        # elif self.sentence_representation_type == "average":
        #     sentence_representation = torch.mean(output, dim=1)
        elif self.sentence_representation_type == "max":
                # Create mask: [batch, seq_len]
            mask = torch.arange(output.size(1), device=output.device).unsqueeze(0) < original_len.unsqueeze(1)
            mask = mask.unsqueeze(-1)  # [batch, seq_len, 1]
            masked_output = output.masked_fill(~mask, float('-inf'))
            sentence_representation, _ = torch.max(masked_output, dim=1)
            
        elif self.sentence_representation_type == "average":
            mask = torch.arange(output.size(1), device=output.device).unsqueeze(0) < original_len.unsqueeze(1)
            mask = mask.unsqueeze(-1).float()  # [batch, seq_len, 1]
            sentence_representation = (output * mask).sum(dim=1) / mask.sum(dim=1)
            # logits = self.fc2(self.relu(self.fc(sentence_representation)))

        sentence_representation = self.fc_dropout(sentence_representation)
        hidden_out = self.relu(self.fc(sentence_representation))
        hidden_out = self.fc_dropout(hidden_out)
        logits = self.fc2(hidden_out)

        return logits

    def get_embeddings(self):
        return self.embedding.weight.data


class RNNClassifier(L.LightningModule):
    """

    Args:
        rnn_model (torch.nn.Module): The RNN model used for generating logits from inputs.
        optimizer_name (str): Name of the optimizer to use. Options include 'SGD', 'Adagrad', 'Adam', and 'RMSprop'.
        lr (float): Learning rate for the optimizer.
        show_progress (bool, optional): If True, logs additional progress information to the progress bar. Default is False.
    """

    def __init__(
        self,
        rnn_model: any,
        optimizer_name: str,
        lr: float,
        show_progress: bool = False,
        weight_decay: float = 0.0,
    ):
        super().__init__()
        self.model = rnn_model
        self.optimizer_name = optimizer_name
        self.lr = lr
        self.show_progress = show_progress
        self.weight_decay = weight_decay

        self.metric = MulticlassAccuracy(num_classes=6)
        self.save_hyperparameters()

        # self.train_metric = MulticlassAccuracy(num_classes=6)
        # self.val_metric = MulticlassAccuracy(num_classes=6)
        # self.test_metric = MulticlassAccuracy(num_classes=6)

        # self.save_hyperparameters(ignore=["model"])
        self.train_epoch_losses = []
        self.val_epoch_accs = []

    def training_step(self, batch, batch_idx):
        indexes = batch["indexes"]
        labels = batch["label"]
        original_lens = batch["original_len"]

        logits = self.model(indexes, original_lens)
        loss = F.cross_entropy(logits, labels)
        acc = self.metric(logits, labels)

        self.log("train_loss", loss, prog_bar=self.show_progress)
        self.log("train_acc", acc, prog_bar=self.show_progress)

        # Set it for plotting the graph on every epoch
        # self.log("train_loss", loss, prog_bar=self.show_progress, on_epoch=True, on_step=True)
        # self.log("train_acc", acc, prog_bar=self.show_progress, on_epoch=True, on_step=True)

        return loss

    def validation_step(self, batch, batch_idx):
        indexes = batch["indexes"]
        labels = batch["label"]
        original_lens = batch["original_len"]

        logits = self.model(indexes, original_lens)

        loss = F.cross_entropy(logits, labels)
        acc = self.metric(logits, labels)

        self.log("val_loss", loss, prog_bar=self.show_progress)
        self.log("val_acc", acc, prog_bar=self.show_progress)

        # Set it for plotting the graph on every epoch
        # self.log("val_loss", loss, prog_bar=self.show_progress, on_epoch=True, on_step=True)
        # self.log("val_acc", acc, prog_bar=self.show_progress, on_epoch=True, on_step=True)

    def test_step(self, batch, batch_idx):
        indexes = batch["indexes"]
        labels = batch["label"]
        original_lens = batch["original_len"]

        logits = self.model(indexes, original_lens)

        loss = F.cross_entropy(logits, labels)
        acc = self.metric(logits, labels)

        self.log("test_loss", loss, prog_bar=self.show_progress)
        self.log("test_acc", acc, prog_bar=self.show_progress)

    def predict_step(self, batch, batch_idx):
        indexes = batch["indexes"]
        original_lens = batch["original_len"]

        logits = self.model(indexes, original_lens)
        return logits

    def configure_optimizers(self):
        if self.optimizer_name == "SGD":
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer_name == "Adagrad":
                optimizer = torch.optim.Adagrad(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer_name == "Adam":
             optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer_name == "RMSprop":
                optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            raise Exception("Invalid optimizer name!")

        return optimizer

    # For plotting graph on every epoch
    # def on_train_epoch_end(self):
    #     if 'train_loss' in self.trainer.callback_metrics:
    #         self.train_epoch_losses.append(self.trainer.callback_metrics['train_loss'].item())
        
    # def on_validation_epoch_end(self):
    #     if  not self.trainer.sanity_checking:
    #         if 'val_acc' in self.trainer.callback_metrics:
    #             self.val_epoch_accs.append(self.trainer.callback_metrics['val_acc'].item())
