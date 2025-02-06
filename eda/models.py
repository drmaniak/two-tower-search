from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """
    A generic encoder that maps a sequence of token indices to a fixed-dimensional vector.
    It first embeds the tokens using a pretrained embedding matrix, then processes the sequence
    using a bidirectional LSTM, applies dropout, and projects the final hidden state via a linear layer.
    """

    def __init__(
        self,
        embedding_matrix: torch.Tensor,
        hidden_dim: int = 256,
        num_layers: int = 1,
        bidirectional: bool = True,
        dropout: float = 0.1,
        output_dim: Optional[int] = None,
    ):
        """
        Args:
            embedding_matrix (torch.Tensor): A tensor of shape (vocab_size, embed_dim) containing the pretrained embeddings.
            hidden_dim (int): Hidden dimension for the LSTM.
            num_layers (int): Number of LSTM layers.
            bidirectional (bool): Whether to use a bidirectional LSTM.
            dropout (float): Dropout probability.
            output_dim (Optional[int]): If specified, the final encoder output is projected to this dimension.
                                       If None, output dimension is hidden_dim * (2 if bidirectional else 1).
        """
        super(Encoder, self).__init__()
        vocab_size, embed_dim = embedding_matrix.shape

        # Create the embedding layer from the pretrained matrix.
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)

        # Determine the dimension after the LSTM
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        final_dim = output_dim if output_dim is not None else lstm_output_dim
        self.fc = nn.Linear(lstm_output_dim, final_dim)
        self.activation = nn.ReLU()  # or use nn.Tanh()

    def forward(self, x, lengths: Optional[torch.Tensor] = None):
        """
        Args:
            x (torch.Tensor): Tensor of shape (batch_size, seq_length) containing token indices.
            lengths (Optional[torch.Tensor]): Actual lengths of sequences before padding, if using pack_padded_sequence.

        Returns:
            torch.Tensor: Tensor of shape (batch_size, final_dim) representing the encoded sequences.
        """
        # Embed the input tokens
        embedded = self.embedding(x)  # shape: (batch_size, seq_length, embed_dim)

        num_zeros = torch.sum(torch.all(embedded == 0, dim=-1)).item()
        print(f"Number of zero embeddings: {num_zeros}")

        # If sequence lengths are provided, pack the sequence (helps LSTM ignore padding)
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_out, (hidden, _) = self.lstm(packed)
            # Unpack output if needed (not used here)
        else:
            _, (hidden, _) = self.lstm(embedded)

        # For bidirectional LSTM, hidden is (num_layers*2, batch, hidden_dim).
        # We typically take the final layer's hidden states from both directions and concatenate them.
        if self.lstm.bidirectional:
            hidden_cat = torch.cat(
                (hidden[-2], hidden[-1]), dim=1
            )  # shape: (batch, hidden_dim*2)
        else:
            hidden_cat = hidden[-1]  # shape: (batch, hidden_dim)

        hidden_cat = self.dropout(hidden_cat)
        out = self.fc(hidden_cat)
        out = self.activation(out)
        return out


# Define the Query Encoder and Document Encoder as separate classes.
# (They can share the same architecture; you can later decide whether to share weights.)
class QueryEncoder(nn.Module):
    def __init__(self, embedding_matrix: torch.Tensor, **kwargs):
        super(QueryEncoder, self).__init__()
        self.encoder = Encoder(embedding_matrix, **kwargs)

    def forward(self, query, lengths=None):
        return self.encoder(query, lengths)


class DocumentEncoder(nn.Module):
    def __init__(self, embedding_matrix: torch.Tensor, **kwargs):
        super(DocumentEncoder, self).__init__()
        self.encoder = Encoder(embedding_matrix, **kwargs)

    def forward(self, doc, lengths=None):
        return self.encoder(doc, lengths)


# Example usage in a training script:
if __name__ == "__main__":
    # Suppose you already have a global embedding matrix as a NumPy array
    # that was built using your global vocabulary and pretrained word2vec,
    # and you've converted it to a PyTorch tensor.
    import numpy as np

    # For example, load embedding_matrix from file (this is your aligned matrix)
    embedding_matrix_np = np.load(
        "aligned_embedding_matrix.npy"
    )  # shape: (vocab_size, 300)
    embedding_matrix_tensor = torch.tensor(embedding_matrix_np, dtype=torch.float32)

    # Create instances of your two encoders.
    query_encoder = QueryEncoder(
        embedding_matrix_tensor,
        hidden_dim=256,
        num_layers=1,
        bidirectional=True,
        dropout=0.2,
    )
    doc_encoder = DocumentEncoder(
        embedding_matrix_tensor,
        hidden_dim=256,
        num_layers=1,
        bidirectional=True,
        dropout=0.2,
    )

    # Suppose you have a batch of tokenized queries and documents (already numericalized)
    # For illustration, we'll use random indices.
    batch_size = 8
    seq_length_query = 20
    seq_length_doc = 50

    dummy_queries = torch.randint(
        0, embedding_matrix_tensor.shape[0], (batch_size, seq_length_query)
    )
    dummy_docs = torch.randint(
        0, embedding_matrix_tensor.shape[0], (batch_size, seq_length_doc)
    )

    # Optionally, if you have lengths, compute them; here we assume full length.
    query_out = query_encoder(dummy_queries)  # shape: (batch_size, final_dim)
    doc_out = doc_encoder(dummy_docs)  # shape: (batch_size, final_dim)

    print("Query embeddings shape:", query_out.shape)
    print("Document embeddings shape:", doc_out.shape)
