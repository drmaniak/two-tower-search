import torch.nn as nn


class DocTower(nn.Module):
    def __init__(self, embed_dim, lstm_hidden_dim, output_dim, embeddings_tensor=None):
        """ """
        super(DocTower, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(
            embeddings_tensor, padding_idx=0, freeze=True
        )
        self.lstm = nn.LSTM(embed_dim, lstm_hidden_dim, batch_first=True)
        self.fc = nn.Linear(lstm_hidden_dim, output_dim)

    def forward(self, x, sequence_lengths):
        x = self.embedding(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            x, sequence_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (hn, _) = self.lstm(packed)
        last_hidden = hn[-1]  # Shape: (batch_size, lstm_hidden_dim)
        return self.fc(last_hidden)


class TwoTowerModel(nn.Module):
    def __init__(
        self,
        output_dim,
        embed_dim,
        lstm_hidden_dim_query,
        lstm_hidden_dim_doc,
        embeddings_tensor=None,
    ):
        """ """
        super(TwoTowerModel, self).__init__()
        self.query_tower = DocTower(
            embed_dim, lstm_hidden_dim_query, output_dim, embeddings_tensor
        )
        self.doc_tower = DocTower(
            embed_dim, lstm_hidden_dim_doc, output_dim, embeddings_tensor
        )

    def forward(self, query, doc_positive, doc_negative, query_length, doc_length):
        """ """
        query_embed = self.query_tower(query, query_length)
        positive_embed = self.doc_tower(doc_positive, doc_length)
        negative_embed = self.doc_tower(doc_negative, doc_length)
        return query_embed, positive_embed, negative_embed
