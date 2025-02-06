import argparse
import logging
import os
from functools import partial
from pathlib import Path
from typing import Any, Dict

import torch
import torch.optim as optim
import wandb
import yaml
from dataset import TwoTowerDataset, collate_fn_document, collate_fn_query
from loss_functions import TripletLoss
from models import DocumentEncoder, QueryEncoder
from models2 import SimpleDocumentEncoder, SimpleQueryEncoder
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_model(
    model_name: str, embedding_matrix: torch.Tensor, model_config: Dict[str, Any]
):
    """Factory function to create model based on configuration."""
    if model_name == "simple":
        return (
            SimpleQueryEncoder(
                embedding_matrix,
                hidden_dim=model_config.get("hidden_dim", 256),
                output_dim=model_config.get("output_dim", 256),
            ),
            SimpleDocumentEncoder(
                embedding_matrix,
                hidden_dim=model_config.get("hidden_dim", 256),
                output_dim=model_config.get("output_dim", 256),
            ),
        )
    elif model_name == "complex":
        return (
            QueryEncoder(
                embedding_matrix,
                hidden_dim=model_config.get("hidden_dim", 256),
                num_layers=model_config.get("num_layers", 1),
                bidirectional=model_config.get("bidirectional", True),
                dropout=model_config.get("dropout", 0.1),
            ),
            DocumentEncoder(
                embedding_matrix,
                hidden_dim=model_config.get("hidden_dim", 256),
                num_layers=model_config.get("num_layers", 1),
                bidirectional=model_config.get("bidirectional", True),
                dropout=model_config.get("dropout", 0.1),
            ),
        )
    else:
        raise ValueError(f"Unknown model type: {model_name}")


def train_epoch(
    query_encoder,
    doc_encoder,
    query_dataloader,
    doc_dataloader,
    criterion,
    optimizer,
    device,
    epoch: int,
):
    """Train for one epoch."""
    query_encoder.train()
    doc_encoder.train()
    total_loss = 0.0
    num_batches = 0

    for (query_batch, query_lengths), (
        (pos_batch, pos_lengths),
        (neg_batch, neg_lengths),
    ) in tqdm(zip(query_dataloader, doc_dataloader), desc=f"Training epoch {epoch}"):
        # Move data to device
        query_batch = query_batch.to(device)
        pos_batch = pos_batch.to(device)
        neg_batch = neg_batch.to(device)

        optimizer.zero_grad()

        # Forward pass
        query_embeds = query_encoder(query_batch, query_lengths)
        pos_embeds = doc_encoder(pos_batch, pos_lengths)
        neg_embeds = doc_encoder(neg_batch, neg_lengths)

        # Compute loss
        loss = criterion(query_embeds, pos_embeds, neg_embeds)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1
        wandb.log({"batch_train_loss": loss.item()})

    return total_loss / num_batches


def validate(
    query_encoder, doc_encoder, query_dataloader, doc_dataloader, criterion, device
):
    """Validate the model."""
    query_encoder.eval()
    doc_encoder.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for (query_batch, query_lengths), (
            (pos_batch, pos_lengths),
            (neg_batch, neg_lengths),
        ) in tqdm(zip(query_dataloader, doc_dataloader), desc="Validating"):
            query_batch = query_batch.to(device)
            pos_batch = pos_batch.to(device)
            neg_batch = neg_batch.to(device)

            query_embeds = query_encoder(query_batch, query_lengths)
            pos_embeds = doc_encoder(pos_batch, pos_lengths)
            neg_embeds = doc_encoder(neg_batch, neg_lengths)

            loss = criterion(query_embeds, pos_embeds, neg_embeds)
            wandb.log({"batch_val_loss": loss.item()})
            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train two-tower model with configuration"
    )
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument(
        "--model-type",
        type=str,
        required=True,
        choices=["simple", "complex"],
        help="Type of model to use (simple or complex)",
    )
    parser.add_argument("--train-token-path", type=Path, required=True)
    parser.add_argument("--val-token-path", type=Path, required=True)
    parser.add_argument("--vocab-path", type=Path, required=True)
    parser.add_argument("--embed-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--pretrained-query-model", type=Path)
    parser.add_argument("--pretrained-doc-model", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(args.output_dir / "training.log"),
        ],
    )

    # Initialize wandb
    wandb.init(project="two-tower-training", config={**vars(args), **config})

    # Load datasets
    train_dataset = TwoTowerDataset(
        token_path=args.train_token_path,
        vocab_path=args.vocab_path,
        embed_path=args.embed_path,
        max_len_query=config["data"]["max_len_query"],
        max_len_docs=config["data"]["max_len_docs"],
    )

    val_dataset = TwoTowerDataset(
        token_path=args.val_token_path,
        vocab_path=args.vocab_path,
        embed_path=args.embed_path,
        max_len_query=config["data"]["max_len_query"],
        max_len_docs=config["data"]["max_len_docs"],
    )

    # Create dataloaders
    pad_idx = train_dataset.word2idx.get("<PAD>", 0)
    train_query_dataloader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        collate_fn=partial(collate_fn_query, pad_idx=pad_idx),
    )
    train_doc_dataloader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        collate_fn=partial(collate_fn_document, pad_idx=pad_idx),
    )

    val_query_dataloader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        collate_fn=partial(collate_fn_query, pad_idx=pad_idx),
    )
    val_doc_dataloader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        collate_fn=partial(collate_fn_document, pad_idx=pad_idx),
    )

    # Load embedding matrix
    embedding_matrix = torch.load(args.embed_path)

    # Initialize models
    query_encoder, doc_encoder = get_model(
        args.model_type, embedding_matrix, config["model"]
    )

    query_encoder.to(args.device)
    doc_encoder.to(args.device)

    # Initialize optimizer and loss
    optimizer = optim.Adam(
        list(query_encoder.parameters()) + list(doc_encoder.parameters()),
        lr=config["training"]["learning_rate"],
    )
    criterion = TripletLoss(margin=config["training"]["margin"])

    # Training loop
    best_val_loss = float("inf")
    for epoch in range(config["training"]["epochs"]):
        # Train
        train_loss = train_epoch(
            query_encoder,
            doc_encoder,
            train_query_dataloader,
            train_doc_dataloader,
            criterion,
            optimizer,
            args.device,
            epoch,
        )

        # Validate
        val_loss = validate(
            query_encoder,
            doc_encoder,
            val_query_dataloader,
            val_doc_dataloader,
            criterion,
            args.device,
        )

        # Log metrics
        wandb.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        logger.info(
            f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}"
        )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                query_encoder.state_dict(), args.output_dir / "best_query_encoder.pth"
            )
            torch.save(
                doc_encoder.state_dict(), args.output_dir / "best_doc_encoder.pth"
            )

    # Save final model
    torch.save(query_encoder.state_dict(), args.output_dir / "final_query_encoder.pth")
    torch.save(doc_encoder.state_dict(), args.output_dir / "final_doc_encoder.pth")

    wandb.finish()
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
