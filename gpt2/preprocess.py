from torch.utils.data import Dataset, DataLoader
import torch

from torch.nn.utils.rnn import pad_sequence
import numpy as np
from collections import Counter, defaultdict

PAD = "PAD"
START = "START"
UNK = "UNK_"
STOP = "STOP"

class GPTDataset(Dataset):
    def __init__(self, input_file, tokenizer, model):

        with open(input_file, "r") as f:
            sentences = f.readlines()

        sentences = [sent.strip() for sent in sentences[1:]]
        # for sentence in sentences:
        #     for token in sentence:
        #         if token not in self.word2id:
        #             self.word2id[token] = len(self.word2id)

        self.all_data = list()
        self.all_label = list()
        self.all_length = list()
        for sentence in sentences:
            label = tokenizer.encode(sentence)
            if model == "transformer":
                data = tokenizer.encode(START + " " + sentence[:-(len(STOP) + 1)])
            else:
                data = tokenizer.encode(sentence)

            self.all_length.append(len(data))
            self.all_data.append(torch.LongTensor(data))
            self.all_label.append(torch.LongTensor(label))

        self.all_data = pad_sequence(self.all_data, batch_first=True, padding_value=0)
        self.all_label = pad_sequence(self.all_label, batch_first=True, padding_value=0)

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):
        return {
            "data": self.all_data[idx],
            "label": self.all_label[idx],
            "length": self.all_length[idx],
        }


def load_dataset(fn, tokenizer, model, batch_size):
    """
    :param fn: filename for the dataset
    :return: (torch.utils.data.DataLoader, torch.utils.data.DataLoader) for train and test
    :Comment: You don't have to shuffle the test dataset
    """
    train_set = GPTDataset(fn[0], tokenizer, model)
    test_set = GPTDataset(fn[1], tokenizer, model)
    vocab_size = len(tokenizer)
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader, vocab_size


# def load_dataset(fn, tokenizer, batch_size):
#     """
#     :param fn: filename for the dataset
#     :return: (torch.utils.data.DataLoader, torch.utils.data.DataLoader) for train and test
#     :Comment: This preprocess step is different from the previous ones. In this assignment, we are interested in using a pre-trained model.
#     So, we have to use the exact vocabulary the pre-trained model was trained with. We are using the GPT-2 model, so pass your data through
#     the GPT-2 tokenizer to get the word ids. You don't need to create your own dictionary.
#     """
#     pass
