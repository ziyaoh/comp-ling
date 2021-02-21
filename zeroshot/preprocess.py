from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch
import numpy as np
import argparse
import unicodedata
import re
from collections import defaultdict
from tqdm import tqdm  # optional progress bar

PAD = "PAD"
START = "START"
STOP = "STOP"
UNK = "UNK"

ENG_TAG = "<eng>"
DEU_TAG = "<deu>"
FRA_TAG = "<fra>"

class TranslationDataset(Dataset):
    def __init__(self, input_file, enc_seq_len, dec_seq_len,
                 bpe=True, target=None, word2id=None, flip=False, zeroshot=False, test=False):
        """
        Read and parse the translation dataset line by line. Make sure you
        separate them on tab to get the original sentence and the target
        sentence. You will need to adjust implementation details such as the
        vocabulary depending on whether the dataset is BPE or not.

        :param input_file: the data file pathname
        :param enc_seq_len: sequence length of encoder
        :param dec_seq_len: sequence length of decoder
        :param bpe: whether this is a Byte Pair Encoded dataset
        :param target: the tag for target language
        :param word2id: the word2id to append upon
        :param flip: whether to flip the ordering of the sentences in each line
        """
        # TODO: read the input file line by line and put the lines in a list.

        # TODO: split the whole file (including both training and validation
        # data) into words and create the corresponding vocab dictionary.

        # TODO: create inputs and labels for both training and validation data
        #       and make sure you pad your inputs.

        # Hint: remember to add start and pad to create inputs and labels
        if zeroshot: # deu -> eng -> fra
            deu_file, fra_file = input_file

            if test:
                _, deu_lines = read_from_corpus(deu_file)
                _, fra_lines = read_from_corpus(fra_file)
                other = fra_lines
                source = [[FRA_TAG] + line for line in deu_lines]

            else:
                eng_d, deu_lines = read_from_corpus(deu_file)
                eng_f, fra_lines = read_from_corpus(fra_file)
                other = fra_lines + eng_d
                source = [[ENG_TAG] + line for line in deu_lines]
                source += [[FRA_TAG] + line for line in eng_f]
        else:
            if target:
                source, other = read_from_corpus(input_file)
                source = [[target] + line for line in source]

            else:
                source, other = read_from_corpus(input_file)

        if word2id:
            self.word2id = word2id
        else:
            self.word2id = dict()
            self.word2id[PAD] = 0
            self.word2id[START] = 1
            self.word2id[STOP] = 2
            self.word2id[UNK] = 3

        self.all_src_data = list()
        self.all_oth_data = list()
        self.all_label = list()
        self.all_src_length = list()
        self.all_oth_length = list()

        # word to wordid
        for line in source:
            for word in line:
                if word not in self.word2id:
                    self.word2id[word] = len(self.word2id)
        for line in other:
            for word in line:
                if word not in self.word2id:
                    self.word2id[word] = len(self.word2id)

        # data and label
        for sentence in other:
            data, label = list(), list()
            data.append(self.word2id[START])
            for token in sentence:
                tid = self.word2id[token]
                data.append(tid)
                label.append(tid)
            label.append(self.word2id[STOP])
            if target:
                data += [0] * (dec_seq_len - len(data))
                label += [0] * (dec_seq_len - len(label))

            self.all_oth_length.append(len(data))
            self.all_oth_data.append(torch.LongTensor(data))
            self.all_label.append(torch.LongTensor(label))
        for sentence in source:
            data = list()
            for token in sentence:
                tid = self.word2id[token]
                data.append(tid)
            data.append(self.word2id[STOP])
            if target:
                data += [0] * (enc_seq_len - len(data))

            self.all_src_length.append(len(data))
            self.all_src_data.append(torch.LongTensor(data))

        if not target:
            self.all_src_data = pad_sequence(self.all_src_data, batch_first=True, padding_value=0)
            self.all_oth_data = pad_sequence(self.all_oth_data, batch_first=True, padding_value=0)
            self.all_label = pad_sequence(self.all_label, batch_first=True, padding_value=0)

    def __len__(self):
        """
        len should return a the length of the dataset

        :return: an integer length of the dataset
        """
        # TODO: Override method to return length of dataset
        return len(self.all_src_data)

    def __getitem__(self, idx):
        """
        getitem should return a tuple or dictionary of the data at some index
        In this case, you should return your original and target sentence and
        anything else you may need in training/validation/testing.

        :param idx: the index for retrieval

        :return: tuple or dictionary of the data
        """
        # TODO: Override method to return the items in dataset
        return {
            "enc_data": self.all_src_data[idx],
            "dec_data": self.all_oth_data[idx],
            "label": self.all_label[idx],
            "enc_length": self.all_src_length[idx],
            "dec_length": self.all_oth_length[idx],
        }


def read_from_corpus(corpus_file):
    eng_lines = []
    frn_lines = []

    with open(corpus_file, 'r') as f:
        for line in f:
            words = line.split("\t")
            eng_line = words[0]
            frn_line = words[1][:-1]

            eng_line = re.sub('([.,!?():;])', r' \1 ', eng_line)
            eng_line = re.sub('\s{2,}', ' ', eng_line)
            frn_line = re.sub('([.,!?():;])', r' \1 ', frn_line)
            frn_line = re.sub('\s{2,}', ' ', frn_line)

            eng_lines.append(eng_line.split())
            frn_lines.append(frn_line.split())
    return eng_lines, frn_lines


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', help='path to BPE input')
    parser.add_argument('iterations', help='number of iterations', type=int)
    parser.add_argument('output_file', help='path to BPE output')
    args = parser.parse_args()
