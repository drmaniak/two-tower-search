# train.py
import argparse
import logging
import os
import pickle
from functools import partial
from pathlib import Path

import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb

# Import your dataset, model definitions, and loss function.
# Adjust these import paths to your project structure.
from dataset import TwoTowerDataset, collate_fn_document, collate_fn_query
from loss_functions import (
    TripletLoss,
)  # The custom loss function provided by your teammate.
from models import DocumentEncoder, QueryEncoder
from torch.utils.data import DataLoader
from tqdm import tqdm
from vocabulary import Vocabulary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train two-tower model using custom TripletLoss with wandb logging."
    )
    parser.add_argument(
        "--train-token-path",
        type=Path,
        required=True,
        help="Path to the pickle file containing tokenized data.",
    )
    parser.add_argument(
        "--val-token-path",
        type=Path,
        required=True,
        help="Path to the pickle file containing tokenized data.",
    )
    parser.add_argument(
        "--hn-token-path",
        type=Path,
        required=True,
        help="Path to the pickle file containing tokenized data.",
    )
    parser.add_argument(
        "--vocab-path",
        type=Path,
        required=True,
        help="Path to the pickle file containing the vocabulary.",
    )
    parser.add_argument(
        "--embed-path",
        type=Path,
        required=True,
        help="Path to the pickle file containing the aligned embedding matrix.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for training."
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs."
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument(
        "--margin", type=float, default=1.0, help="Margin for TripletLoss."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for training.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Initialize wandb
    wandb.init(project="two-tower-training", config=vars(args))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )

    logging.info("Starting training...")

    device = torch.device(args.device)
    # Create your dataset instance.
    # Here, TwoTowerDataset loads tokenized data, vocabulary, and aligned embedding matrix.
    logging.info("Loading train dataset")
    train_dataset = TwoTowerDataset(
        token_path=args.train_token_path,
        vocab_path=args.vocab_path,
        embed_path=args.embed_path,
        max_len_query=40,
        max_len_docs=80,
    )
    logging.info("Loading val dataset")
    val_dataset = TwoTowerDataset(
        token_path=args.val_token_path,
        vocab_path=args.vocab_path,
        embed_path=args.embed_path,
        max_len_query=40,
        max_len_docs=80,
    )
    # logging.info("Loading hard-neg dataset")
    # hn_dataset = TwoTowerDataset(
    #     token_path=args.hn_token_path,
    #     vocab_path=args.vocab_path,
    #     embed_path=args.embed_path,
    #     max_len_query=128,
    #     max_len_docs=256,
    # )

    # For simplicity, we'll assume the same dataset is used for training;
    # in practice, you might have separate files for train/val.
    # Extract the PAD token index from your vocabulary. (Assuming <PAD> is always in the vocab.)
    pad_idx = train_dataset.word2idx.get("<PAD>", 0)

    # Create two DataLoaders: one for queries, one for documents.
    query_train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=partial(collate_fn_query, pad_idx=pad_idx),
    )
    document_train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=partial(collate_fn_document, pad_idx=pad_idx),
    )

    query_val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=partial(collate_fn_query, pad_idx=pad_idx),
    )
    document_val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=partial(collate_fn_document, pad_idx=pad_idx),
    )

    # Load the aligned embedding matrix from the embed pickle.
    embedding_matrix = joblib.load(args.embed_path)
    embedding_tensor = torch.tensor(embedding_matrix, dtype=torch.float32)

    # Instantiate the query and document encoders.
    # Here we use a common architecture; adjust hidden dimensions and dropout as desired.
    query_encoder = QueryEncoder(
        embedding_tensor, hidden_dim=256, num_layers=1, bidirectional=True, dropout=0.3
    )
    doc_encoder = DocumentEncoder(
        embedding_tensor, hidden_dim=256, num_layers=1, bidirectional=True, dropout=0.3
    )

    query_encoder.to(device)
    doc_encoder.to(device)

    # Create optimizer over both models' parameters.
    optimizer = optim.Adam(
        list(query_encoder.parameters()) + list(doc_encoder.parameters()), lr=args.lr
    )

    # Use the custom triplet loss.
    criterion = TripletLoss(margin=args.margin)

    # Training loop.
    for epoch in tqdm(range(args.epochs), desc="Training Loop Progress"):
        query_encoder.train()
        doc_encoder.train()
        total_loss = 0.0
        num_batches = 0

        # Zip the two dataloaders. (Ensure they yield batches in corresponding order.)
        for (query_batch, query_lengths), (
            (pos_batch, pos_lengths),
            (neg_batch, neg_lengths),
        ) in tqdm(
            zip(query_train_dataloader, document_train_dataloader),
            desc=f"Iterating through training batches in epoch {epoch}",
        ):
            # Move data to device.
            query_batch = query_batch.to(device)
            pos_batch = pos_batch.to(device)
            neg_batch = neg_batch.to(device)

            optimizer.zero_grad()

            # Forward pass: get embeddings.
            query_embeds = query_encoder(query_batch, lengths=query_lengths)
            pos_embeds = doc_encoder(pos_batch, lengths=pos_lengths)
            neg_embeds = doc_encoder(neg_batch, lengths=neg_lengths)

            # Compute the loss using the custom triplet loss function.
            loss = criterion(query_embeds, pos_embeds, neg_embeds)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            wandb.log({"batch_train_loss": loss.item()})

        avg_loss = total_loss / num_batches
        wandb.log({"epoch_train_loss": avg_loss, "epoch": epoch})
        logging.info(f"Epoch {epoch}: Avg Train Loss: {avg_loss:.4f}")

        # Optionally add a validation loop here if you have a validation dataset.
        # ----- Validation Loop -----

        query_encoder.eval()
        doc_encoder.eval()
        total_val_loss = 0.0
        num_val_batches = 0
        with torch.no_grad():
            for (query_batch, query_lengths), (
                (pos_batch, pos_lengths),
                (neg_batch, neg_lengths),
            ) in tqdm(
                zip(query_val_dataloader, document_val_dataloader),
                desc=f"Iterating through validation batches in epoch {epoch}",
            ):
                query_batch = query_batch.to(device)
                pos_batch = pos_batch.to(device)
                neg_batch = neg_batch.to(device)

                query_embeds = query_encoder(query_batch, lengths=query_lengths)
                pos_embeds = doc_encoder(pos_batch, lengths=query_lengths)
                neg_embeds = doc_encoder(neg_batch, lengths=query_lengths)

                val_loss = criterion(query_embeds, pos_embeds, neg_embeds)

                total_val_loss += val_loss.item()
                num_val_batches += 1

        avg_val_loss = total_val_loss / num_val_batches
        wandb.log({"epoch_val_loss": avg_val_loss, "epoch": epoch})
        print(f"Epoch {epoch}: Avg Val Loss: {avg_val_loss:.4f}")

    torch.save(query_encoder.state_dict(), "query_encoder.pth")
    torch.save(doc_encoder.state_dict(), "doc_encoder.pth")
    logging.info("Trained Models Saved")
    wandb.finish()
    logging.info("Training complete.")


if __name__ == "__main__":
    # Set up logging.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )
    logging.info("Importing Vocabular")
    from vocabulary import Vocabulary

    logging.info(f"Vocab imported - sample class: {Vocabulary()}")

    main()
