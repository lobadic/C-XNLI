from typing import List, Optional
from loguru import logger

import pandas as pd
import numpy as np
import pyarrow as pa
from torch.utils.data import Dataset


class ClassificationDataset(Dataset):
    def __init__(
        self,
        filepath: str,
        text_columns: List[str],
        label_column: str,
        tokenizer,
        possible_labels: list,
        max_seq_length: int = 128,
        padding="max_length",
        truncation=True,
    ):

        """
        Sentence pairs Classification Dataset.
        """

        self.filepath = filepath
        logger.info(f"Loading dataset from {self.filepath}")

        self.text_columns = text_columns
        self.label_column = label_column

        self.tokenizer = tokenizer
        logger.info(f"Tokenizer: {self.tokenizer}")

        self.possible_labels = possible_labels
        logger.info(f"Possible labels: {self.possible_labels}")

        self.max_seq_length = max_seq_length
        logger.info(f"Max sequence length: {self.max_seq_length}")

        self.padding = padding
        self.truncation = truncation

        assert len(self.text_columns) == 2

        self.df = self.load_data()

        logger.info(
            f"Number of NaN samples in the text columns: {self.df[self.text_columns].isna().sum()}"
        )
        logger.info("Filling NaN samples in text columns with empty string")
        self.text = self.df[self.text_columns].fillna("")
        logger.info(f"Dataset length: {len(self.text)}")
        logger.info(f"Dataset sample: {self.text.iloc[:5]}")

        self.label_to_idx_map, self.idx_to_label_map = self.get_label_maps()
        logger.info(f"label_to_idx_map: {self.label_to_idx_map}")

        self.labels_sanity_check()

        self.df = pa.Table.from_pandas(self.df)
        self.text = pa.Table.from_pandas(self.text)

    def load_data(self):

        if self.filepath.endswith(".csv"):
            sep = ","
        elif self.filepath.endswith(".tsv"):
            sep = "\t"

        return pd.read_csv(
            self.filepath, sep=sep, usecols=self.text_columns + [self.label_column]
        )

    def get_label_maps(
        self,
    ):

        label_to_idx_map = {}
        idx_to_label_map = {}

        for i, label in enumerate(self.possible_labels):
            label_to_idx_map[str(label)] = i
            idx_to_label_map[i] = str(label)

        return label_to_idx_map, idx_to_label_map

    def labels_sanity_check(
        self,
    ):

        unique_labels = self.df[self.label_column].unique()

        for uq_label in unique_labels:
            assert (
                str(uq_label) in self.label_to_idx_map
            ), f"Label '{uq_label}' missing from the list of possible labels {self.possible_labels}."

    def __len__(
        self,
    ):
        return len(self.df)

    def __getitem__(self, idx):

        label = self.df[self.label_column][idx]
        label_idx = int(self.label_to_idx_map[str(label)])

        encoded = self.tokenizer(
            str(self.text[self.text_columns[0]][idx]),
            str(self.text[self.text_columns[1]][idx]),
            max_length=self.max_seq_length,
            truncation=self.truncation,
            padding=self.padding,
            return_tensors="pt",
        )

        for key, value in encoded.items():
            encoded[key] = value.squeeze()

        encoded["labels"] = label_idx

        return dict(encoded)
