from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleTower(nn.Module):
    """
    A simplified version of the encoder that uses a single LSTM layer
    and minimal additional components.
    """

    def __init__(
        self,
        embedding_matrix: torch.Tensor,
        hidden_dim: int = 256,
        output_dim: int = 256,
    ):
        super(SimpleTower, self).__init__()
        vocab_size, embed_dim = embedding_matrix.shape

        # Create the embedding layer from the pretrained matrix
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)

        # Single-layer unidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )

        # Simple linear projection
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, lengths: Optional[torch.Tensor] = None):
        # Embed the input tokens
        embedded = self.embedding(x)

        # Pack sequence if lengths are provided
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            _, (hidden, _) = self.lstm(packed)
        else:
            _, (hidden, _) = self.lstm(embedded)

        # Get the final hidden state
        last_hidden = hidden[-1]  # Shape: (batch_size, hidden_dim)

        # Project to output dimension
        return self.fc(last_hidden)


class SimpleQueryEncoder(nn.Module):
    def __init__(
        self,
        embedding_matrix: torch.Tensor,
        hidden_dim: int = 256,
        output_dim: int = 256,
    ):
        super(SimpleQueryEncoder, self).__init__()
        self.tower = SimpleTower(embedding_matrix, hidden_dim, output_dim)

    def forward(self, query, lengths=None):
        return self.tower(query, lengths)


class SimpleDocumentEncoder(nn.Module):
    def __init__(
        self,
        embedding_matrix: torch.Tensor,
        hidden_dim: int = 256,
        output_dim: int = 256,
    ):
        super(SimpleDocumentEncoder, self).__init__()
        self.tower = SimpleTower(embedding_matrix, hidden_dim, output_dim)

    def forward(self, doc, lengths=None):
        return self.tower(doc, lengths)


# Example usage
if __name__ == "__main__":
    import numpy as np

    # Load embedding matrix (example)
    embedding_matrix_np = np.load("aligned_embedding_matrix.npy")
    embedding_matrix_tensor = torch.tensor(embedding_matrix_np, dtype=torch.float32)

    # Create encoders
    query_encoder = SimpleQueryEncoder(
        embedding_matrix_tensor, hidden_dim=256, output_dim=256
    )
    doc_encoder = SimpleDocumentEncoder(
        embedding_matrix_tensor, hidden_dim=256, output_dim=256
    )

    # Test with dummy data
    batch_size = 8
    seq_length_query = 20
    seq_length_doc = 50

    dummy_queries = torch.randint(
        0, embedding_matrix_tensor.shape[0], (batch_size, seq_length_query)
    )
    dummy_docs = torch.randint(
        0, embedding_matrix_tensor.shape[0], (batch_size, seq_length_doc)
    )

    # Forward pass
    query_out = query_encoder(dummy_queries)
    doc_out = doc_encoder(dummy_docs)

    print("Query embeddings shape:", query_out.shape)
    print("Document embeddings shape:", doc_out.shape)
