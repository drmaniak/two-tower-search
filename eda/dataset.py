import pickle
from pathlib import Path
from typing import List, Optional

import pandas as pd
import torch
from torch.utils.data import Dataset
from utils import tokenize

from eda.vocabulary import Vocabulary

# Assume you already have your tokenizer function and Vocabulary class from your vocabulary.py


class TwoTowerDataset(Dataset):
    """
    This dataset returns one example per row from the data.
    Each example is a dictionary with three keys:
      - 'query': token indices (list[int]) for the query column.
      - 'positive': token indices (list[int]) for the positive document.
      - 'negative': token indices (list[int]) for the negative document.
    The tokens are converted to indices using your Vocabulary instance.
    """

    def __init__(
        self,
        file_path: Path,
        max_len_query: Optional[int] = None,
        max_len_docs: Optional[int] = None,
    ):
        """
        Args:
            file_path (Path): Path to the .pkl file containing tokenized data and vocabulary.
            max_len (Optional[int]): Maximum sequence length (if you wish to pad/truncate).
        """
        with open(file_path, "rb") as datafile:
            data = pickle.load(datafile)

        self.token_dict = data["token_dict"]
        self.vocab = data["vocabulary"]
        self.max_len_query = max_len_query
        self.max_len_docs = max_len_docs

    def __len__(self):
        return len(self.token_dict["query"])

    def __getitem__(self, idx):
        query_indices = self.token_dict["query"][idx]
        pos_indices = self.token_dict["positive_passage"][idx]
        neg_indices = self.token_dict["negative_passage"][idx]

        if self.max_len_query is not None:
            query_indices = query_indices[: self.max_len_query]

        if self.max_len_docs is not None:
            pos_indices = pos_indices[: self.max_len_docs]
            neg_indices = neg_indices[: self.max_len_docs]

        return {
            "query": query_indices,
            "positive": pos_indices,
            "negative": neg_indices,
        }


# ------------------------------------------------------------------------------
# Example usage:
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    pass
