from torch.utils.data import Dataset
import torch
import numpy as np
from collections import defaultdict
import random

MASK = "<MASK>"
UNK = "<UNK>"
unk_threshold = 10

class MyDataset(Dataset):
    # TODO: Create masked Penn Treebank dataset.
    #       You can change signature of the initializer.
    def __init__(self, file_path, seq_len, word2id=None):
        super().__init__()
        # TODO: Read data from file
        tokens = read_file(file_path)

        # TODO: Split data into fixed length sequences

        # TODO: Mask part of the data for BERT training
        if not word2id:
            self.word2id = dict()
            self.word2id[MASK] = 0
            self.word2id[UNK] = 2

            count = defaultdict(int)
            for token in tokens:
                count[token] += 1
                if count[token] >= unk_threshold and token not in self.word2id:
                    self.word2id[token] = len(self.word2id)

        else:
            self.word2id = word2id

        self.all_data = list()
        for token in tokens:
            if not self.all_data or len(self.all_data[-1]) >= seq_len:
                self.all_data.append(list())

            tid = self.word2id[token] if token in self.word2id else self.word2id[UNK]
            self.all_data[-1].append(tid)
        if self.all_data[-1] < seq_len:
            self.all_data.pop()

    def __len__(self):
        """
        __len__ should return a the length of the dataset

        :return: an integer length of the dataset
        """
        # TODO: Override method to return length of dataset
        return len(self.all_data)

    def __getitem__(self, i):
        """
        __getitem__ should return a tuple or dictionary of the data at some
        index

        :param idx: the index for retrieval

        :return: tuple or dictionary of the data
        """
        # TODO: Override method to return the ith item in dataset
        data = self.all_data[i]
        num_masked = int(len(data) * 0.15)
        masked_ind = random.sample(list(range(len(data))), num_masked)
        label = list()
        for ind in masked_ind:
            label.append(data[ind])
            data[ind] = self.word2id[MASK]
        
        return {
            "data": torch.LongTensor(data),
            "label": torch.LongTensor(label),
            "masked_ind": torch.LongTensor(masked_ind),
        }


def read_file(fname):
    content = []
    with open(fname) as f:
        for line in f.readlines():
            content += line.lower().strip().split()
    return content
