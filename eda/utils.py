"""
Utility functions for text preprocessing and tokenization.

This module provides functions for splitting, tokenizing, and processing text data.
It supports various text preprocessing steps including:
- String splitting
- Punctuation handling
- Word stemming
- Lemmatization

The module is designed to be flexible, allowing optional use of stemming and lemmatization,
and customizable punctuation handling.
"""

import re
from typing import Any, LiteralString

import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer

nltk.download("stopwords")
from nltk.corpus import stopwords

STOP_WORDS = set(stopwords.words("english"))


def split_string(text: str) -> list[str]:
    """
    Split a string into words, filtering out unwanted tokens.

    Uses regex to split text and applies filtering rules:
    - Removes pure numbers
    - Removes very short strings (length < 2)
    - Removes strings with mixed alphanumeric characters
    - Preserves words with apostrophes (e.g., "don't")

    Args:
        text (str): The input text to be split.

    Returns:
        list[str]: A list of filtered tokens.

    Example:
        >>> split_string("Hello, world! 123 test123")
        ['Hello', 'world']
    """
    # First, normalize apostrophes
    text = text.replace("'", "'")

    # Split on whitespace and punctuation, but preserve contractions
    words = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|\S", text)

    # Filter tokens
    filtered_words = []
    for word in words:
        # Skip if token is too short
        if len(word) < 2:
            continue

        # Skip if token is a number or contains numbers
        if re.search(r"\d", word):
            continue

        # Skip if token contains special characters (except apostrophe)
        if re.search(r"[^a-zA-Z\']", word):
            continue

        filtered_words.append(word)

    return filtered_words


def tokenize_word(
    word: str,
    punctuation_map: dict[str, str],
    stemmer: PorterStemmer | Any = None,
    lemmer: WordNetLemmatizer | Any = None,
    junk_punctuations: bool = False,
) -> list[str] | list[LiteralString]:
    """
    Process a single word with optional stemming, lemmatization, and punctuation handling.

    This function applies several text preprocessing steps in the following order:
    1. Converts the word to lowercase
    2. Handles punctuation marks according to the punctuation_map
    3. Applies stemming (if stemmer is provided)
    4. Applies lemmatization (if lemmer is provided)

    Args:
        word (str): The input word to be processed.
        punctuation_map (dict[str, str]): A mapping of punctuation marks to their desired
            representations (e.g., {'?': ' ? '}).
        stemmer (PorterStemmer | Any, optional): A stemmer instance for word stemming.
            Defaults to None.
        lemmer (WordNetLemmatizer | Any, optional): A lemmatizer instance for word
            lemmatization. Defaults to None.
        junk_punctuations (bool, optional): If True, removes punctuation instead of
            replacing it. Defaults to False.

    Returns:
        list[str]: A list of processed tokens derived from the input word.

    Example:
        >>> punctuation_map = {'.': ' . ', '!': ' ! '}
        >>> tokenize_word("Hello!", punctuation_map)
        ['hello', '!']
        >>> tokenize_word("running", punctuation_map, PorterStemmer())
        ['run']

    Note:
        - Returns ['<nan>'] for empty strings or "nan" input
        - Stemming and lemmatization are optional and can be used independently
        - The function splits on whitespace after all processing steps
    """

    if word == "nan" or word.strip() == "":
        return ["<nan>"]  # Handle missing or NaN values explicitly

    # Lowercase the word
    word_token = word.lower()

    def replace_punctuation_with_token(
        text: str, pmap: dict[str, str], junk: bool = junk_punctuations
    ) -> str:
        if junk:
            return "".join(["" if char in pmap.keys() else char for char in text])
        return "".join([pmap.get(char, char) for char in text])

    word_token = replace_punctuation_with_token(word_token, punctuation_map)

    # Stemming
    if stemmer:
        word_token = " ".join([stemmer.stem(w) for w in word_token.split()])

    # Lemmatization
    if lemmer:
        word_token = " ".join(
            [lemmer.lemmatize(w, pos="v") for w in word_token.split()]
        )

    # Return a list of tokens
    return word_token.split()


def tokenize(
    text: str,
    punctuation_map: dict[str, str] | None,
    stemmer: PorterStemmer | Any = None,
    lemmer: WordNetLemmatizer | Any = None,
    junk_punctuations: bool = False,
) -> list[str]:
    """
    Tokenize a complete text string with full preprocessing pipeline.

    This function serves as the main entry point for text preprocessing, combining
    string splitting and word tokenization into a complete pipeline. It processes
    text through the following steps:
    1. Splits text into individual words and punctuation marks
    2. Processes each token with lowercase conversion, punctuation handling,
       optional stemming, and optional lemmatization

    Args:
        text (str): The input text to be tokenized.
        punctuation_map (dict[str, str]): A mapping of punctuation marks to their desired
            representations (e.g., {'?': ' ? '}).
        stemmer (PorterStemmer | Any, optional): A stemmer instance for word stemming.
            Defaults to None.
        lemmer (WordNetLemmatizer | Any, optional): A lemmatizer instance for word
            lemmatization. Defaults to None.
        junk_punctuations (bool, optional): If True, removes punctuation instead of
            replacing it. Defaults to False.

    Returns:
        list[str]: A list of processed tokens from the input text.

    Example:
        >>> punct_map = {'.': ' . ', '!': ' ! '}
        >>> tokenize("Hello world!", punct_map)
        ['hello', 'world', '!']
        >>> tokenize("Running fast!", punct_map, PorterStemmer())
        ['run', 'fast', '!']

    Note:
        - This is the main function to use for text preprocessing
        - Combines split_string() and tokenize_word() into a single pipeline
        - Handles missing values and empty strings appropriately
        - Returns a flattened list of tokens
    """

    words = split_string(text.lower())

    filtered_words = [word for word in words if word not in STOP_WORDS]

    tokens = []
    for word in filtered_words:
        token = tokenize_word(word, punctuation_map, stemmer, lemmer, junk_punctuations)
        tokens.extend(token)

    return tokens
