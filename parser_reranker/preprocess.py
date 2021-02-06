from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch
import numpy as np
from collections import Counter, defaultdict

PAD = "PAD"
START = "START"
UNK = "UNK_"
STOP = "STOP"
unk_threshold = 10

class ParsingDataset(Dataset):
    def __init__(self, input_file):
        """
        Read and parse the train file line by line. Create a vocabulary
        dictionary that maps all the unique tokens from your data as
        keys to a unique integer value. Then vectorize your
        data based on your vocabulary dictionary.

        :param input_file: the data file pathname
        """
        # TODO: read the input file line by line and put the lines in a list.

        # TODO: split the whole file (including both training and validation
        # data) into words and create the corresponding vocab dictionary.

        # TODO: create inputs and labels for both training and validation data
        #       and make sure you pad your inputs.

        # Hint: remember to add start and pad to create inputs and labels
        self.word2id = dict()
        self.word2id[PAD] = 0
        self.word2id[START] = 1
        self.word2id[UNK] = 2

        with open(input_file, "r") as f:
            sentences = f.readlines()

        count = defaultdict(int)
        sentences = [sent.strip().split() for sent in sentences]
        for sentence in sentences:
            for token in sentence:
                count[token] += 1
                if count[token] >= unk_threshold and token not in self.word2id:
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
        """
        len should return a the length of the dataset

        :return: an integer length of the dataset
        """
        # TODO: Override method to return length of dataset
        return len(self.all_data)

    def __getitem__(self, idx):
        """
        getitem should return a tuple or dictionary of the data at some index

        :param idx: the index for retrieval

        :return: tuple or dictionary of the data
        """
        # TODO: Override method to return the items in dataset
        return {
            "data": self.all_data[idx],
            "label": self.all_label[idx],
            "length": self.all_length[idx],
        }


class RerankingDataset(Dataset):
    def __init__(self, parse_file, gold_file, word2id):
        """
        Read and parse the parse files line by line. Unk all words that has not
        been encountered before (not in word2id). Split the data according to
        gold file. Calculate number of constituents from the gold file.

        :param parse_file: the file containing potential parses
        :param gold_file: the file containing the right parsings
        :param word2id: the previous mapping (dictionary) from word to its word
                        id
        """
        with open(parse_file, "r") as f:
            lines = f.readlines()

        self.batches = list()
        for line in lines:
            line = line.strip()
            if line.isdigit():
                self.batches.append({
                    "datas": [],
                    "labels": [],
                    "lengths": [],
                    "parsing_res": [],
                    "num_gold": -1,
                })
                continue

            batch_dict = self.batches[-1]
            tokens = line.split()
            data, label = list(), list()
            data.append(word2id[START])
            for token in tokens[2:]:
                if token not in word2id:
                    data.append(word2id[UNK])
                    label.append(word2id[UNK])
                else:
                    data.append(word2id[token])
                    label.append(word2id[token])
            label.append(word2id[STOP])

            batch_dict["parsing_res"].append((int(tokens[0]), int(tokens[1])))
            batch_dict["lengths"].append(len(data))
            batch_dict["datas"].append(torch.LongTensor(data))
            batch_dict["labels"].append(torch.LongTensor(label))

        for batch_dict in self.batches:
            batch_dict["datas"] = pad_sequence(batch_dict["datas"], batch_first=True, padding_value=word2id[PAD])
            batch_dict["labels"] = pad_sequence(batch_dict["labels"], batch_first=True, padding_value=word2id[PAD])
            batch_dict["lengths"] = torch.LongTensor(batch_dict["lengths"])

        with open(gold_file, "r") as f:
            gold_lines = f.readlines()

        assert(len(gold_lines) == len(self.batches))
        for batch_dict, gold_line in zip(self.batches, gold_lines):
            gold_line = gold_line.strip()
            batch_dict["num_gold"] = Counter(gold_line)["("]

    def __len__(self):
        """
        len should return a the length of the dataset

        :return: an integer length of the dataset
        """
        # TODO: Override method to return length of dataset
        return len(self.batches)

    def __getitem__(self, idx):
        """
        getitem should return a tuple or dictionary of the data at some index

        :param idx: the index for retrieval

        :return: tuple or dictionary of the data
        """
        # TODO: Override method to return the items in dataset
        return self.batches[idx]
