import re
from nltk.tokenize import word_tokenize
from typing import *

def pad_truncate(data: str, max_len: int) -> List[str]:
    "word_tokenize and pad/truncate a sentence to max_len"
    data = data.lower()
    data = re.sub(r'[^a-zA-Z0-9]', ' ', data)
    clean_sent = word_tokenize(data)
    tokens = clean_sent[:max_len]
    tokens += ['<unk>'] * (max_len - len(tokens))
    return tokens

def convert_to_indices(data: List[str], word_to_id: dict) -> List[int]:
    "Convert a list of words to a list of indices using a word_to_id dictionary"
    return [word_to_id.get(w, word_to_id['<unk>']) for w in data]
