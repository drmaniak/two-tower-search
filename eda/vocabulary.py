import argparse
import json
import logging
import pickle
from collections import Counter
from multiprocessing import process
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import gensim.downloader as api
import joblib
import pandas as pd
from constants import punctuation_map
from gensim.models import KeyedVectors
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
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

    def evaluate_with_pretrained(self, model_name: str):
        """
        Evaluate how much the current vocabulary aligns with a pretrained Word2Vec model.

        Args:
            model_name (str): Model name to pass into gensim.downloader's load function
        """

        # Load the pretrained word2vec model
        pretrained_model = api.load(model_name)
        pretrained_vocab = set(pretrained_model.key_to_index.keys())

        # Calculate metrics
        common_words_count = len(
            set(self.word2idx.keys()).intersection(pretrained_vocab)
        )
        unique_words_count = len(set(self.word2idx.keys()).difference(pretrained_vocab))
        missing_words_count = len(
            pretrained_vocab.difference(set(self.word2idx.keys()))
        )
        percentage_in_pretrained = (common_words_count / len(self.word2idx)) * 100

        return {
            "common_words_count": common_words_count,
            "unique_words_count": unique_words_count,
            "missing_words_count": missing_words_count,
            "total_pretrained_count": len(pretrained_vocab),
            "percentage_in_pretrained": percentage_in_pretrained,
        }

    def load_from_file(self, vocab_file: str | Path):
        """Load Vocabulary from a json file"""
        with open(vocab_file, "r") as f:
            vocab_data = json.load(f)
            self.word2idx = vocab_data["word2idx"]
            self.idx2word = {int(k): v for k, v in vocab_data["idx2word"].items()}
            self.word_freq = Counter(vocab_data.get("word_freq", {}))

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
        output_path: str | Path,
        min_freq: int = 1,
        punctuation_map: Optional[dict[str, str]] = {},
        junk_punctuations: bool = True,
        tokenizer_type: Literal["custom", "word_tokenizer"] = "custom",
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
        self.tokenizer_type = tokenizer_type
        self.output_path = output_path
        self.token_dict = {}

    def save_tokenized_data(self, tokenized_data: dict):
        """Save tokenized data and vocabulary to a file."""
        with open(self.output_path, "wb") as f:
            pickle.dump(
                {"token_dict": tokenized_data, "vocabulary": self.vocabulary}, f
            )

    def load_tokenized_data(self, file_path: str | Path) -> dict:
        """Load tokenized data and vocabulary from a file."""
        with open(file_path, "rb") as f:
            data = joblib.load(f)
            self.token_dict = data["token_dict"]
            self.vocabulary = data["vocabulary"]
            return data["token_dict"]

    def process_text_columns(
        self,
        df: pd.DataFrame,
        columns: List[str],
        stemmer: PorterStemmer,
        lemmer: WordNetLemmatizer,
        save_dataset: bool = False,
    ) -> Vocabulary:
        """
        Process multiple text columns from a dataframe to build a Vocabulary

        Args:
            df: Pandas DataFrame containing text columns
            columns: List of column names to process
        """

        all_tokens = []

        tokenized_data = {}

        # Process each specified column
        for column in columns:
            if column not in df.columns:
                continue

            logging.info(f"Tokenizing {column} column")
            tqdm.pandas(desc=f"Tokenizing {column} column")
            # Tokenize each text entry in the column
            if self.tokenizer_type.lower() == "custom":
                token_series = (
                    df[column]
                    .fillna("")
                    .progress_apply(
                        lambda text: tokenize(
                            text=text,
                            punctuation_map=self.punctuation_map,
                            stemmer=stemmer,
                            lemmer=lemmer,
                            junk_punctuations=self.junk_punctuations,
                        )
                    )
                )
            else:
                # token_series = df[column].fillna("").apply(word_tokenize)
                token_series = df[column].fillna("").progress_apply(word_tokenize)

            tokenized_data[column] = token_series.tolist()
            # all_tokens.extend(token_series.tolist())

        logging.info(f"Tokenized data added to {self.__class__.__name__} instance")
        self.token_dict = tokenized_data
        self.vocabulary.build_vocabulary(sum(tokenized_data.values(), []))
        # self.vocabulary.build_vocabulary(all_tokens)
        self.vocabulary.finalize_vocabulary()

        if save_dataset:
            logging.info(f"Writing Tokens and Vocabulary to {self.output_path}")
            self.save_tokenized_data(tokenized_data)
            logging.info(f"Finished saving tokens & vocabulary to {self.output_path}")

        return self.vocabulary

    def process_parquet_file(
        self,
        file_path: str | Path,
        columns: list[str],
        stemmer: PorterStemmer | Any,
        lemmer: WordNetLemmatizer | Any,
        save_dataset: bool | Any,
    ) -> Vocabulary:
        df = pd.read_parquet(file_path)
        return self.process_text_columns(df, columns, stemmer, lemmer, save_dataset)


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
        help="Input parquet file path",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./vocab"),
        help="Directory to save the vocabulary and logs (default: ./data)",
    )
    parser.add_argument(
        "--output-file",
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
        "--tokenizer-type",
        type=str,
        default="custom",
        help="Specify the tokenizer type []",
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
        "--save-data",
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
    parser.add_argument(
        "--load-data",
        type=Path,
        help="Path to a saved vocabulary file to load and compare",
    )
    parser.add_argument(
        "--pretrained-model",
        type=str,
        help="Name of the pretrained Word2Vec model to download for comparison",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_filepath = args.output_dir / args.output_file
    if args.load_data:
        log_file = (
            args.output_dir / f"vocab_load_{args.load_data.stem}.log"
            if args.log_file
            else None
        )
    else:
        log_file = (
            args.output_dir / f"vocab_build_{args.input_file.stem}.log"
            if args.log_file
            else None
        )
    logger = setup_logger(log_file)

    logger.info("Starting vocabulary building process")

    try:
        if args.load_data:
            logger.info(f"Loading Tokens/Vocab from {args.load_data}")
            processor = DatasetProcessor(
                output_path=output_filepath,
                min_freq=args.min_freq,
                punctuation_map=punctuation_map,
                junk_punctuations=args.junk_punctuations,
                tokenizer_type="word_tokenizer",
            )
            processor.load_tokenized_data(args.load_data)
            logger.info("Tokens/Vocabulary loaded successfully")
            vocabulary = processor.vocabulary

        else:
            logger.info(f"Input file: {args.input_file}")
            logger.info(f"Processing columns: {args.columns}")

            # Initialize processors
            stemmer = PorterStemmer() if args.use_stemming else None
            lemmer = WordNetLemmatizer() if args.use_lemmatization else None

            processor = DatasetProcessor(
                output_path=output_filepath,
                min_freq=args.min_freq,
                punctuation_map=punctuation_map,
                junk_punctuations=args.junk_punctuations,
                tokenizer_type=args.tokenizer_type,
            )
            logger.info("Processing input file...")
            vocabulary = processor.process_parquet_file(
                args.input_file,
                columns=args.columns,
                stemmer=stemmer,
                lemmer=lemmer,
                save_dataset=args.save_data,
            )

        stats = vocabulary.get_word_stats()
        logger.info("Vocabulary building completed successfully")
        logger.info(f"Vocabulary size: {len(vocabulary)}")
        logger.info(f"Statistics: {stats}")

        if args.pretrained_model:
            logger.info(f"Comparing with pretrained model: {args.pretrained_model}")
            alignment_stats = vocabulary.evaluate_with_pretrained(args.pretrained_model)
            logger.info("Alignment with pretrained model:")
            logger.info(alignment_stats)

    except Exception as e:
        logger.error(f"Error during vocabulary building: {str(e)}", exc_info=True)
        raise
