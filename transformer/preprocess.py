from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch
import numpy as np
from collections import Counter, defaultdict

PAD = "PAD"
START = "START"
UNK = "UNK_"
STOP = "STOP"
# unk_threshold = 5

class TransDataset(Dataset):
    def __init__(self, input_file, word2id=None):
        if not word2id:
            self.word2id = dict()
            self.word2id[PAD] = 0
            self.word2id[START] = 1
            self.word2id[UNK] = 2
        else:
            self.word2id = word2id

        with open(input_file, "r") as f:
            sentences = f.readlines()

        # count = defaultdict(int)
        # sentences = [sent.strip().split() for sent in sentences]
        # for sentence in sentences:
        #     for token in sentence:
        #         count[token] += 1
        #         if count[token] >= unk_threshold and token not in self.word2id:
        #             self.word2id[token] = len(self.word2id)

        sentences = [sent.strip().split() for sent in sentences]
        for sentence in sentences:
            for token in sentence:
                if token not in self.word2id:
                    self.word2id[token] = len(self.word2id)

        self.all_data = list()
        self.all_label = list()
        self.all_length = list()
        for sentence in sentences:
            data, label = list(), list()
            data.append(self.word2id[START])
            for token in sentence:
                tid = self.word2id[token] if token in self.word2id else self.word2id[UNK]
                data.append(tid)
                label.append(tid)
            data.pop()

            self.all_length.append(len(data))
            self.all_data.append(torch.LongTensor(data))
            self.all_label.append(torch.LongTensor(label))

        self.all_data = pad_sequence(self.all_data, batch_first=True, padding_value=self.word2id[PAD])
        self.all_label = pad_sequence(self.all_label, batch_first=True, padding_value=self.word2id[PAD])

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):
        return {
            "data": self.all_data[idx],
            "label": self.all_label[idx],
            "length": self.all_length[idx],
        }

def load_dataset(fn, batch_size):
    """
    :param fn: filename for the dataset
    :return: (torch.utils.data.DataLoader, torch.utils.data.DataLoader) for train and test
    :Comment: You don't have to shuffle the test dataset
    """
    train_set = TransDataset(fn[0])
    test_set = TransDataset(fn[1], train_set.word2id)
    vocab_size = len(train_set.word2id)
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, vocab_size