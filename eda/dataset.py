import argparse
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import gensim.downloader as api
import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from utils import tokenize
from vocabulary import Vocabulary

# Assume you already have your tokenizer function and Vocabulary class from your vocabulary.py


def return_vocab_from_pickle(pickle_file: Path):
    with open(pickle_file, "rb") as f:
        data = joblib.load(f)
        logging.info(
            f"Loaded vocab from {pickle_file} with {len(data['vocabulary'])} unique words"
        )
        return data["vocabulary"]


def create_global_vocab(
    pkl_files: list[Path | str],
    vocab_dir: Path = Path("../vocab"),
    output_file: Path | str = "global_vocab.pkl",
):
    global_word2idx = {}
    global_idx2word = {}

    special_tokens = ["<PAD>", "<UNK>"]
    for token in special_tokens:
        if token not in global_word2idx:
            global_word2idx[token] = len(global_word2idx)
            global_idx2word[len(global_idx2word)] = token

    for pkl_file in tqdm(pkl_files, desc="Reading Pickle Files"):
        filepath = vocab_dir / pkl_file
        if not filepath.is_file():
            logging.info(f"{filepath} does not exist")
            continue
        vocab_object = return_vocab_from_pickle(filepath)
        for word in tqdm(
            vocab_object.word2idx.keys(), desc=f"Updating global vocab using {pkl_file}"
        ):
            if word not in global_word2idx:
                global_word2idx[word] = len(global_word2idx)
                global_idx2word[len(global_idx2word)] = word

    with open(vocab_dir / output_file, "wb") as file:
        joblib.dump({"word2idx": global_word2idx, "idx2word": global_idx2word}, file)

    return global_word2idx, global_idx2word


def build_aligned_embedding_matrix(
    word2idx: dict[str, int],
    pretrained_model,
    embedding_dim: int = 300,
    vocab_dir: Path = Path("../vocab"),
    embed_file: Path | str = "aligned_embed_matrix.pkl",
):
    vocab_size = len(word2idx)

    embedding_matrix = np.zeros((vocab_size, embedding_dim), dtype=np.float32)

    oov_count = 0
    for word, idx in word2idx.items():
        if word in pretrained_model:
            embedding_matrix[idx] = pretrained_model[word]
        else:
            oov_count += 1

    logging.info(f"Proportion of Out-Of-Vocabulary Words: {oov_count / vocab_size}")
    filepath = vocab_dir / embed_file
    logging.info(f"Writing to file {filepath}")
    with open(filepath, "wb") as file:
        joblib.dump(embedding_matrix, file)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate MS MARCO triplet datasets in various formats"
    )
    parser.add_argument(
        "--filepaths",
        nargs="+",
        default=[
            "datavocab_train.pkl",
            "datavocab_validation.pkl",
            "datavocab_test.pkl",
            "datavocab_hard_negatives.pkl",
        ],
        help="Text columns to process (default: query positive_passage negative_passage)",
    )
    parser.add_argument(
        "--vocab-dir",
        type=Path,
        default=Path("./vocab"),
        help="Directory to look for vocab files",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="global_vocab.pkl",
        help="Filename to save containing the output global vocabulary.",
    )
    parser.add_argument(
        "--embed-file",
        type=str,
        default="aligned_embed_matrix.pkl",
        help="Filename to save containing the Aligned Embedding Matrix.",
    )
    parser.add_argument(
        "--pretrained-model",
        type=str,
        default="word2vec-google-news-300",
        help="Specify which pretrained word2vec model to use when building our desired embedding matrix.",
    )
    return parser.parse_args()


class TwoTowerDataset(Dataset):
    """
    I want to modify this Dataset class to add the following functionality. Currently, I pull in a pkl file
    that has tokenized senteces (for query, positive_passage, negative_passage) and a vocabulary Object. This has its own word2idx and idx2word mappings
    The is built on the tokenized data, and then I need to generate embeddings from the pretrained word2vec from googlenews
    I'd then like to Obtain embeddings and align it with the pretrained model's.

    I understand that I'd need to align this with the pretrained word2vec model. How do I do this?
    """

    def __init__(
        self,
        token_path: Path,
        vocab_path: Path,
        embed_path: Path,
        max_len_query: Optional[int] = None,
        max_len_docs: Optional[int] = None,
    ):
        """
        Args:
            file_path (Path): Path to the .pkl file containing tokenized data and vocabulary.
            max_len (Optional[int]): Maximum sequence length (if you wish to pad/truncate).
        """
        with open(token_path, "rb") as datafile:
            tokens = joblib.load(datafile)

        with open(vocab_path, "rb") as datafile:
            vocab = joblib.load(datafile)

        with open(embed_path, "rb") as datafile:
            embeds = joblib.load(datafile)

        self.token_dict = tokens["token_dict"]
        self.word2idx = vocab["word2idx"]
        self.idx2word = vocab["idx2word"]
        self.embedding_matrix = embeds
        self.max_len_query = max_len_query
        self.max_len_docs = max_len_docs

    def numericalize(self, tokens: List[str]) -> List[int]:
        """
        Convert a list of tokens into a list of indices using word2idx.
        If a token is missing, use the <UNK> token index.
        """
        return [
            self.word2idx.get(token, self.word2idx.get("<UNK>", 1)) for token in tokens
        ]

    def __len__(self):
        return len(self.token_dict["query"])

    def __getitem__(self, idx):
        # Get raw token lists from token_dict
        query_tokens = self.token_dict["query"][idx]
        pos_tokens = self.token_dict["positive_passage"][idx]
        neg_tokens = self.token_dict["negative_passage"][idx]

        # Convert tokens to indices
        query_indices = self.numericalize(query_tokens)
        pos_indices = self.numericalize(pos_tokens)
        neg_indices = self.numericalize(neg_tokens)

        # Apply truncation if required
        if self.max_len_query is not None:
            query_indices = query_indices[: self.max_len_query]
        if self.max_len_docs is not None:
            pos_indices = pos_indices[: self.max_len_docs]
            neg_indices = neg_indices[: self.max_len_docs]

        return {
            "query": query_indices,
            "query_length": len(query_indices),
            "positive": pos_indices,
            "positive_length": len(pos_indices),
            "negative": neg_indices,
            "negative_length": len(neg_indices),
        }


def collate_fn_query(batch, pad_idx: int):
    """
    Collate function for the query tower.
    Expects each sample in the batch to have a key "query" containing a list of token indices.
    Returns a padded tensor for the query sequences.
    """
    # Convert each query list to a tensor
    query_tensors = [torch.tensor(item["query"], dtype=torch.long) for item in batch]
    query_lengths = torch.tensor([item["query_length"] for item in batch])

    # Pad all queries to the maximum length in the batch
    queries_padded = torch.nn.utils.rnn.pad_sequence(
        query_tensors, batch_first=True, padding_value=pad_idx
    )

    return queries_padded, query_lengths


def collate_fn_document(batch, pad_idx: int):
    """
    Collate function for the document tower.
    Expects each sample in the batch to have keys "positive" and "negative" containing lists of token indices.
    Returns two padded tensors: one for positives and one for negatives.
    """
    pos_tensors = [torch.tensor(item["positive"], dtype=torch.long) for item in batch]
    neg_tensors = [torch.tensor(item["negative"], dtype=torch.long) for item in batch]

    pos_lengths = torch.tensor([item["positive_length"] for item in batch])
    neg_lengths = torch.tensor([item["negative_length"] for item in batch])

    pos_padded = torch.nn.utils.rnn.pad_sequence(
        pos_tensors, batch_first=True, padding_value=pad_idx
    )
    neg_padded = torch.nn.utils.rnn.pad_sequence(
        neg_tensors, batch_first=True, padding_value=pad_idx
    )

    return (pos_padded, pos_lengths), (neg_padded, neg_lengths)


# ------------------------------------------------------------------------------
# Example usage:
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(
                f"global_vocab_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            ),
            logging.StreamHandler(),
        ],
    )

    logging.info("Building global vocab")

    global_word2idx, global_idx2word = create_global_vocab(
        pkl_files=args.filepaths, vocab_dir=args.vocab_dir, output_file=args.output_file
    )

    logging.info(f"Final updated vocab has {len(global_word2idx)} unique words")

    logging.info(f"Initializing the pretrained model")
    pretrained_model = api.load(args.pretrained_model)
    logging.info(f"{args.pretrained_model} loaded")

    build_aligned_embedding_matrix(
        global_word2idx,
        pretrained_model,
        pretrained_model.vector_size,
        vocab_dir=args.vocab_dir,
        embed_file=args.embed_file,
    )

    logging.info("Completed Aligned Embedding Matrix.")
