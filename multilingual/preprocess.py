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
unk_threshold = 4

class TranslationDataset(Dataset):
    def __init__(self, input_file, enc_seq_len, dec_seq_len,
                 bpe=True, target=None, word2id=None):
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
        """
        # TODO: read the input file line by line and put the lines in a list.

        # TODO: split the whole file (including both training and validation
        # data) into words and create the corresponding vocab dictionary.

        # TODO: create inputs and labels for both training and validation data
        #       and make sure you pad your inputs.

        # Hint: remember to add start and pad to create inputs and labels

        self.all_eng_data = list()
        self.all_src_data = list()
        self.all_label = list()
        self.all_eng_length = list()
        self.all_src_length = list()

        if not bpe:
            english, source = preprocess_vanilla(input_file, threshold=unk_threshold)

            self.eng_word2id = dict()
            self.eng_word2id[PAD] = 0
            self.eng_word2id[START] = 1
            self.eng_word2id[STOP] = 2
            self.eng_word2id[UNK] = 3
            self.src_word2id = dict()
            self.src_word2id[PAD] = 0
            self.src_word2id[START] = 1
            self.src_word2id[STOP] = 2
            self.src_word2id[UNK] = 3

            # word to wordid
            for line in english:
                for word in line:
                    if word not in self.eng_word2id:
                        self.eng_word2id[word] = len(self.eng_word2id)
            for line in source:
                for word in line:
                    if word not in self.src_word2id:
                        self.src_word2id[word] = len(self.src_word2id)

            # data and label
            for sentence in english:
                data, label = list(), list()
                data.append(self.eng_word2id[START])
                for token in sentence:
                    tid = self.eng_word2id[token]
                    data.append(tid)
                    label.append(tid)
                label.append(self.eng_word2id[STOP])

                self.all_eng_length.append(len(data))
                self.all_eng_data.append(torch.LongTensor(data))
                self.all_label.append(torch.LongTensor(label))
            for sentence in source:
                data = list()
                for token in sentence:
                    tid = self.src_word2id[token]
                    data.append(tid)
                data.append(self.src_word2id[STOP])

                self.all_src_length.append(len(data))
                self.all_src_data.append(torch.LongTensor(data))
        else:
            if target:
                english, source = read_from_corpus(input_file)
                source = [[target] + line for line in source]

                if word2id:
                    self.word2id = word2id
                else:
                    self.word2id = dict()
                    self.word2id[PAD] = 0
                    self.word2id[START] = 1
                    self.word2id[STOP] = 2
                    self.word2id[UNK] = 3
            else:
                english, source = read_from_corpus(input_file)
                self.word2id = dict()
                self.word2id[PAD] = 0
                self.word2id[START] = 1
                self.word2id[STOP] = 2
                self.word2id[UNK] = 3

            # word to wordid
            for line in english:
                for word in line:
                    if word not in self.word2id:
                        self.word2id[word] = len(self.word2id)
            for line in source:
                for word in line:
                    if word not in self.word2id:
                        self.word2id[word] = len(self.word2id)

            # data and label
            for sentence in english:
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

                self.all_eng_length.append(len(data))
                self.all_eng_data.append(torch.LongTensor(data))
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
            self.all_eng_data = pad_sequence(self.all_eng_data, batch_first=True, padding_value=0)
            self.all_src_data = pad_sequence(self.all_src_data, batch_first=True, padding_value=0)
            self.all_label = pad_sequence(self.all_label, batch_first=True, padding_value=0)

    def __len__(self):
        """
        len should return a the length of the dataset

        :return: an integer length of the dataset
        """
        # TODO: Override method to return length of dataset
        return len(self.all_eng_data)

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
            "dec_data": self.all_eng_data[idx],
            "label": self.all_label[idx],
            "enc_length": self.all_src_length[idx],
            "dec_length": self.all_eng_length[idx],
        }

def get_stats(vocab):
    pairs = defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols)-1):
            pairs[symbols[i],symbols[i+1]] += freq
    return pairs

def merge_vocab(pair, v_in):
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]
    return v_out

def learn_bpe(train_file, iterations):
    """
    learn_bpe learns the BPE from data in the train_file and return a
    dictionary of {Byte Pair Encoded vocabulary: count}.

    Note: The original vocabulary should not include '</w>' symbols.
    Note: Make sure you use unicodedata.normalize to normalize the strings when
          reading file inputs.

    You are allowed to add helpers.

    :param train_file: file of the original version
    :param iterations: number of iterations of BPE to perform

    :return: vocabulary dictionary learned using BPE
    """
    # TODO: Please implement the BPE algorithm.
    with open(train_file, "r") as f:
        lines = f.readlines()
    
    vocab = defaultdict(int)
    for line in lines:
        line = unicodedata.normalize("NFKC", line)
        sents = line.split("\t")
        for sent in sents:
            tokens = sent.split()
            for token in tokens:
                token = " ".join(token)
                vocab[token] += 1

    for i in tqdm(range(iterations)):
        pairs = get_stats(vocab)
        best = max(pairs, key=pairs.get)
        vocab = merge_vocab(best, vocab)

    return vocab


def get_transforms(vocab):
    """
    get_transforms return a mapping from an unprocessed vocabulary to its Byte
    Pair Encoded counterpart.

    :param vocab: BPE vocabulary dictionary of {bpe-vocab: count}

    :return: dictionary of {original: bpe-version}
    """
    transforms = {}
    for vocab, count in vocab.items():
        word = vocab.replace(' ', '')
        bpe = vocab + " </w>"
        transforms[word] = bpe
    return transforms


def apply_bpe(train_file, bpe_file, vocab):
    """
    apply_bpe applies the BPE vocabulary learned from the train_file to itself
    and save it to bpe_file.

    :param train_file: file of the original version
    :param bpe_file: file to save the Byte Pair Encoded version
    :param vocab: vocabulary dictionary learned from learn_bpe
    """
    with open(train_file) as r, open(bpe_file, 'w') as w:
        transforms = get_transforms(vocab)
        for line in r:
            line = unicodedata.normalize("NFKC", line)
            words = re.split(r'(\s+)', line.strip())
            bpe_str = ""
            for word in words:
                if word.isspace():
                    bpe_str += word
                else:
                    bpe_str += transforms[word]
            bpe_str += "\n"
            w.write(bpe_str)


def count_vocabs(eng_lines, frn_lines):
    eng_vocab = defaultdict(lambda: 0)
    frn_vocab = defaultdict(lambda: 0)

    for eng_line in eng_lines:
        for eng_word in eng_line:
            eng_vocab[eng_word] += 1
    for frn_line in frn_lines:
        for frn_word in frn_line:
            frn_vocab[frn_word] += 1

    return eng_vocab, frn_vocab


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


def unk_words(eng_lines, frn_lines, eng_vocab, frn_vocab, threshold=5):
    for eng_line in eng_lines:
        for i in range(len(eng_line)):
            if eng_vocab[eng_line[i]] <= threshold:
                eng_line[i] = "UNK"

    for frn_line in frn_lines:
        for i in range(len(frn_line)):
            if frn_vocab[frn_line[i]] <= threshold:
                frn_line[i] = "UNK"


def preprocess_vanilla(corpus_file, threshold=5):
    """
    preprocess_vanilla unks the corpus and returns two lists of lists of words.

    :param corpus_file: file of the corpus
    :param threshold: threshold count to UNK
    """
    eng_lines, frn_lines = read_from_corpus(corpus_file)
    eng_vocab, frn_vocab = count_vocabs(eng_lines, frn_lines)
    unk_words(eng_lines, frn_lines, eng_vocab, frn_vocab, threshold)
    return eng_lines, frn_lines


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', help='path to BPE input')
    parser.add_argument('iterations', help='number of iterations', type=int)
    parser.add_argument('output_file', help='path to BPE output')
    args = parser.parse_args()

    vocab = learn_bpe(args.input_file, args.iterations)
    apply_bpe(args.input_file, args.output_file, vocab)
