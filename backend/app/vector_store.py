from langchain_chroma import Chroma

from config import CHROMA_DB_NAME, CHROMA_DB_PATH
from database_embedding import Embeddings
from model import load_model

# Load mode & embedding function
model, word2idx = load_model()
embedding_function = Embeddings(model, word2idx)

vector_store = Chroma(
    collection_name=CHROMA_DB_NAME,
    embedding_function=embedding_function,  # type: ignore
    persist_directory=CHROMA_DB_PATH,
)
