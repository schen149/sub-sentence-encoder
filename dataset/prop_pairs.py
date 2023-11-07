"""
Dataset class for proposition pair
"""
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from utils.data_utils import read_jsonl, convert_example_to_features

from pytorch_lightning import LightningDataModule


class PropPairDataset(LightningDataModule):
    """
    torch dataset for proposition pairs
    """

    def __init__(self,
                 train_data_path: str,
                 val_data_path: str,
                 test_data_path: str,
                 model_name_or_path: str,
                 max_seq_length: int = 128,
                 train_batch_size: int = 32,
                 eval_batch_size: int = 32,
                 sanity: int = None,
                 ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.train_data_path = train_data_path
        self.val_data_path = val_data_path
        self.test_data_path = test_data_path

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
        if not self.tokenizer.is_fast:
            raise ValueError(
                "Only models with fast tokenizers are supported, as we need to get the character offsets of each token")

        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size

        self.sanity = sanity
        self.prepare_data()

    def prepare_data(self):
        self.train_examples = read_jsonl(self.train_data_path)
        self.val_examples = read_jsonl(self.val_data_path)
        self.test_examples = read_jsonl(self.test_data_path)

        if self.sanity:
            self.train_examples = self.train_examples[:self.sanity]


    def train_dataloader(self):
        return DataLoader(self.train_examples,
                          batch_size=self.train_batch_size,
                          collate_fn=self.convert_to_features,
                          shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_examples,
                          batch_size=self.eval_batch_size,
                          collate_fn=self.convert_to_features)

    def test_dataloader(self):
        return DataLoader(self.test_examples,
                          batch_size=self.eval_batch_size,
                          collate_fn=self.convert_to_features)

    def convert_to_features(self, batch_examples):
        """

        :param batch_examples:
        :return:
        """
        return convert_example_to_features(batch_examples,
                                           tokenizer=self.tokenizer,
                                           sent1_key="sent1",
                                           sent2_key="sent2",
                                           sent1_props_key="sent1_props_spans",
                                           sent2_props_key="sent2_props_spans",
                                           max_seq_len=self.max_seq_length,
                                           generate_labels=True)
