import argparse
import gc
import logging
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Generator, List, Tuple, overload

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
from typing_extensions import Literal

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(f"dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


class MSMarcoTripletGenerator:
    def __init__(self, split: str = "train", batch_size: int = 100):
        """Init the MS MARCO dataset loader

        Args:
            split (str): Dataset split to use ("train", "validation")
            batch_size (int): Number of examples to process at once

        """

        self.split = split
        self.batch_size = batch_size
        self.dataset = None

    def load_dataset(self):
        """Load the MS MARCO dataset from Hugging Face"""

        logger.info(f"Loading MS MARCO dataset split: {self.split}")
        self.dataset = load_dataset(
            "ms_marco", "v2.1", split=self.split, streaming=True
        )

    @overload
    def generate_triplets(
        self, num_queries: int, url_inc: Literal[True]
    ) -> Generator[
        Tuple[str, List[str], List[str], List[str], List[str]], None, None
    ]: ...

    @overload
    def generate_triplets(
        self, num_queries: int, url_inc: Literal[False]
    ) -> Generator[Tuple[str, List[str], List[str]], None, None]: ...

    def generate_triplets(self, num_queries: int = 1000, url_inc: bool = False):
        if self.dataset is None:
            self.load_dataset()

        logger.info(f"Generating triplets for {num_queries}")
        recent_examples_buffer = []
        buffer_size = self.batch_size * 100
        examples_processed = 0
        dataset_iterator = iter(self.dataset)
        pbar = tqdm(total=num_queries, desc="Generating Triplets", unit=" query")

        while examples_processed < num_queries:
            try:
                current_example = next(dataset_iterator)
                examples_processed += 1

                query: str = current_example["query"]
                passages: List[str] = current_example["passages"]["passage_text"]
                positive_urls: List[str] = current_example["passages"]["url"]

                recent_examples_buffer.append(current_example)
                if len(recent_examples_buffer) > buffer_size:
                    recent_examples_buffer.pop(0)

                if len(recent_examples_buffer) < 2:
                    pbar.update(1)
                    continue

                negative_passages: List[str] = []
                negative_urls: List[str] = []
                while len(negative_passages) < len(passages):
                    neg_example = random.choice(recent_examples_buffer)
                    if neg_example["query"] != query:
                        rand_idx = random.randint(
                            0, len(neg_example["passages"]["passage_text"]) - 1
                        )
                        neg_passage = neg_example["passages"]["passage_text"][rand_idx]
                        neg_url = neg_example["passages"]["url"][rand_idx]
                        negative_passages.append(neg_passage)
                        negative_urls.append(neg_url)

                if url_inc:
                    yield (
                        query,
                        passages,
                        positive_urls,
                        negative_passages,
                        negative_urls,
                    )
                else:
                    yield (query, passages, negative_passages)

                pbar.update(1)
            except StopIteration:
                logger.info("Reached end of dataset")
                break

        pbar.close()

    def save_flattened_samples(
        self,
        output_path: str | Path,
        num_queries: int = 1000,
        url_inc: bool = False,
        format: str = "csv",
    ):
        logger.info(f"Saving flattened samples to {output_path}")
        os.makedirs(os.path.dirname(str(output_path)), exist_ok=True)

        rows = []
        if url_inc:
            for item in self.generate_triplets(num_queries, url_inc=True):
                query, pos_passages, pos_urls, neg_passages, neg_urls = item
                for pos_passage, pos_url, neg_passage, neg_url in zip(
                    pos_passages, pos_urls, neg_passages, neg_urls
                ):
                    rows.append(
                        {
                            "query": query,
                            "positive_passage": pos_passage,
                            "positive_url": pos_url,
                            "negative_passage": neg_passage,
                            "negative_url": neg_url,
                        }
                    )
        else:
            for item in self.generate_triplets(num_queries, url_inc=False):
                query, pos_passages, neg_passages = item
                for pos_passage, neg_passage in zip(pos_passages, neg_passages):
                    rows.append(
                        {
                            "query": query,
                            "positive_passage": pos_passage,
                            "negative_passage": neg_passage,
                        }
                    )

        df = pd.DataFrame(rows)
        if format == "parquet":
            df.to_parquet(output_path, index=False, compression="snappy")
        elif format == "json":
            df.to_json(output_path, index=False)
        else:
            df.to_csv(output_path, index=False)

        logger.info(f"Finished writing {len(rows)} samples to {output_path}")

    def process_triplets_in_batches(
        self, batch_size: int, num_queries: int = 1000, url_inc: bool = False
    ):
        """Process triplets in batches to manage memory

        Args:
            num_queries (int): Number of queries to process
            batch_size (int): Size of batches to process at once

        Yields:
            list: Batch of triplets
        """

        if not batch_size:
            batch_size = self.batch_size

        current_batch = []

        for triplet in self.generate_triplets(num_queries, url_inc=url_inc):
            current_batch.append(triplet)

            if len(current_batch) >= batch_size:
                yield current_batch
                current_batch = []

        if current_batch:
            yield current_batch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate MS MARCO triplet datasets in various formats"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./data"),
        help="Directory to save the output files (default: ./data)",
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        default=100000,
        help="Number of queries to process (default: 100000)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for processing (default: 256)",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "validation"],
        default="train",
        help="Dataset split to use (default: train)",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["csv", "parquet", "json"],
        default="parquet",
        help="Output file format (default: parquet)",
    )
    parser.add_argument(
        "--include-urls",
        action="store_true",
        help="Include URLs in the output (default: False)",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        help="Base name for output file (default: auto-generated based on parameters)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize generator
    generator = MSMarcoTripletGenerator(split=args.split, batch_size=args.batch_size)

    # Generate output filename if not specified
    if args.output_name:
        output_name = f"{args.output_name}.{args.format}"
    else:
        url_suffix = "with-urls" if args.include_urls else "no-urls"
        output_name = f"flattened_{args.num_queries}_{url_suffix}.{args.format}"

    output_path = args.output_dir / output_name

    # Generate and save dataset
    generator.save_flattened_samples(
        output_path=output_path,
        num_queries=args.num_queries,
        url_inc=args.include_urls,
        format=args.format,
    )

    logger.info("All work complete. Preparing to shut down.")
    gc.collect()
    logging.shutdown()
    sys.exit(0)
