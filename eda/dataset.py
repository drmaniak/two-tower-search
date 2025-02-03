import logging
import random
from datetime import datetime

from datasets import load_dataset

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
    def __init__(self, split="train"):
        """Init the MS MARCO dataset loader

        Args:
            split (str): Dataset split to use ("train", "validation")

        """

        self.split = split
        self.dataset = None
        self.all_passages = []
        self.passage_to_query_map = {}

    def load_dataset(self):
        """Load the MS MARCO dataset from Hugging Face"""

        logger.info(f"Loading MS MARCO dataset split: {self.split}")
        self.dataset = load_dataset("ms_marco", "v2.1", split=self.split)

        # Create a lookup dictionary for passages
        logger.info("Creating passage lookup dictionary...")
        for idx, example in enumerate(self.dataset):
            query = example["query"]
            passages = example["passges"]["passage_text"]

            # Add all passages to the global list
            self.all_passages.extend(passages)

            # Map each passage to its query
            for passage in passages:
                self.passage_to_query_map[passage] = query

        logger.info(f"Total number of passages collected: {len(self.all_passages)}")

    def generate_triplets(self, num_queries: int = 1000):
        """Generate triplets of (query, positive_passages, negative_passages)

        Args:
            num_queries (int): Number of queries to process

        Returns:
            list: List of triplets (query, list[positive_passages], list[negative_passages])
        """
        if self.dataset is None:
            self.load_dataset()

        triplets = []
        logger.info(f"Generating triplets for {num_queries}")
