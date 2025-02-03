import argparse
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from constants import punctuation_map
from nltk.stem import PorterStemmer, WordNetLemmatizer
from tqdm import tqdm
from utils import tokenize


class Vocabulary:
    def __init__(self, min_freq: int = 1):
        """
        Initialize vocabulary with support for priority datasets
        Args:
            min_freq (int): Minimum frequency for a word to be included from the general dataset
        """
        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}
        self.word_freq = Counter()
        self.min_freq = min_freq

        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"

        # Add special tokens
        self.add_special_tokens()

    def add_special_tokens(self):
        """Add special tokens to vocabulary"""
        special_tokens = [self.pad_token, self.unk_token]
        for token in special_tokens:
            self.word2idx[token] = len(self.word2idx)
            self.idx2word[len(self.idx2word)] = token

    def build_vocabulary(self, text_data: List[List[str]]):
        """
        Build vocabulary from tokenized text data with priority dataset support
        Args:
            text_data: List of tokenized sentences
            is_priority: If True, words from this dataset are treated as priority words
                       and use a lower frequency threshold
        """
        # Count word frequencies
        counter = self.word_freq
        for sentence in tqdm(
            text_data, desc="Building frequency table of unique words"
        ):
            counter.update(sentence)

        logging.info(counter.most_common(n=10))

        for word, freq in tqdm(counter.items(), desc="Building Vocabulary"):
            if freq >= self.min_freq and word not in self.word2idx:
                self.word2idx[word] = len(self.word2idx)
                self.idx2word[len(self.idx2word)] = word

    def finalize_vocabulary(self):
        """
        Finalize vocabulary by combining priority and general words
        Should be called after all datasets have been processed
        """

        # Then add remaining words from general dataset
        for word, freq in self.word_freq.items():
            if freq >= self.min_freq and word not in self.word2idx:
                self.word2idx[word] = len(self.word2idx)
                self.idx2word[len(self.idx2word)] = word

    def get_word_stats(self) -> dict:
        """
        Get statistics about word frequencies in both datasets
        Returns:
            dict: Dictionary containing vocabulary statistics
        """
        return {
            "total_vocab_size": len(self.word2idx),
            "general_words": len(
                [w for w, f in self.word_freq.items() if f >= self.min_freq]
            ),
            "general_unique_words": len(self.word_freq),
        }

    def __len__(self):
        return len(self.word2idx)

    def to_idx(self, word: str) -> int:
        """Convert word to index"""
        return self.word2idx.get(word, self.word2idx[self.unk_token])

    def to_word(self, idx: int) -> str:
        """Convert index to word"""
        return self.idx2word.get(idx, self.unk_token)


class DatasetProcessor:
    def __init__(
        self,
        min_freq: int = 1,
        punctuation_map: Optional[dict[str, str]] = {},
        junk_punctuations: bool = True,
    ):
        """
        Initialize the dataset processor for creating a vocabulary from text data

        Args:
            min_freq (int): Minimum Frequency for a word to be included in the vocab
            punctuation_map (Optional[dict[str, str]]): Mapping for punctuation handling
            junk_punctuations (bool): Flag to control whether to discard punctuations or not
        """

        self.vocabulary = Vocabulary(min_freq=min_freq)
        self.punctuation_map = punctuation_map
        self.junk_punctuations = junk_punctuations

    def process_text_columns(
        self,
        df: pd.DataFrame,
        columns: List[str],
        stemmer: PorterStemmer,
        lemmer: WordNetLemmatizer,
    ) -> Vocabulary:
        """
        Process multiple text columns from a dataframe to build a Vocabulary

        Args:
            df: Pandas DataFrame containing text columns
            columns: List of column names to process
        """

        all_tokens = []

        # Process each specified column
        for column in columns:
            if column not in df.columns:
                continue

            # Tokenize each text entry in the column
            column_tokens = []
            for text in tqdm(
                df[column].fillna(""), desc=f"Tokenizing the {column} column"
            ):
                tokens = tokenize(
                    text=text,
                    punctuation_map=self.punctuation_map,
                    stemmer=stemmer,
                    lemmer=lemmer,
                    junk_punctuations=self.junk_punctuations,
                )
                column_tokens.append(tokens)
            all_tokens.extend(column_tokens)

        self.vocabulary.build_vocabulary(all_tokens)
        self.vocabulary.finalize_vocabulary()

        return self.vocabulary

    def process_parquet_file(
        self,
        file_path: str | Path,
        columns: list[str],
        stemmer: PorterStemmer | Any,
        lemmer: WordNetLemmatizer | Any,
    ) -> Vocabulary:
        df = pd.read_parquet(file_path)
        return self.process_text_columns(df, columns, stemmer, lemmer)


def setup_logger(log_file: Path | Any = None):
    """Configure logging for the vocabulary building process"""
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file) if log_file else logging.NullHandler(),
        ],
    )
    return logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build vocabulary from text data in parquet files"
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        required=True,
        help="Input parquet file path",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./vocab"),
        help="Directory to save the vocabulary and logs (default: ./data)",
    )
    parser.add_argument(
        "--min-freq",
        type=int,
        default=2,
        help="Minimum word frequency threshold (default: 2)",
    )
    parser.add_argument(
        "--columns",
        nargs="+",
        default=["query", "positive_passage", "negative_passage"],
        help="Text columns to process (default: query positive_passage negative_passage)",
    )
    parser.add_argument(
        "--use-stemming",
        action="store_true",
        help="Apply Porter Stemming to tokens",
    )
    parser.add_argument(
        "--use-lemmatization",
        action="store_true",
        help="Apply WordNet Lemmatization to tokens",
    )
    parser.add_argument(
        "--save-vocab",
        action="store_true",
        help="Save vocabulary to file",
    )
    parser.add_argument(
        "--junk-punctuations",
        action="store_true",
        default=True,
        help="Remove punctuations during tokenization (default: True)",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        help="Path to log file (default: None, logs to stdout only)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Setup logging
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_file = (
        args.output_dir / f"vocab_build_{args.input_file.stem}.log"
        if args.log_file
        else None
    )
    logger = setup_logger(log_file)

    logger.info("Starting vocabulary building process")
    logger.info(f"Input file: {args.input_file}")
    logger.info(f"Processing columns: {args.columns}")

    # Initialize processors
    stemmer = PorterStemmer() if args.use_stemming else None
    lemmer = WordNetLemmatizer() if args.use_lemmatization else None

    processor = DatasetProcessor(
        min_freq=args.min_freq,
        punctuation_map=punctuation_map,
        junk_punctuations=args.junk_punctuations,
    )

    try:
        logger.info("Processing input file...")
        vocabulary = processor.process_parquet_file(
            args.input_file,
            columns=args.columns,
            stemmer=stemmer,
            lemmer=lemmer,
        )

        stats = vocabulary.get_word_stats()
        logger.info(f"Vocabulary building completed successfully")
        logger.info(f"Vocabulary size: {len(vocabulary)}")
        logger.info(f"Statistics: {stats}")

        if args.save_vocab:
            vocab_file = args.output_dir / f"vocabulary_{args.input_file.stem}.json"
            vocab_data = {
                "word2idx": vocabulary.word2idx,
                "idx2word": {str(k): v for k, v in vocabulary.idx2word.items()},
                "stats": stats,
            }
            import json

            with open(vocab_file, "w") as f:
                json.dump(vocab_data, f, indent=2)
            logger.info(f"Vocabulary saved to: {vocab_file}")

    except Exception as e:
        logger.error(f"Error during vocabulary building: {str(e)}", exc_info=True)
        raise
