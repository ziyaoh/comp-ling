from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch


class LSTMLM(nn.Module):
    def __init__(self, vocab_size, rnn_size, embedding_size):
        """
        The Model class implements the LSTM-LM model.
        Feel free to initialize any variables that you find necessary in the
        constructor.

        :param vocab_size: The number of unique tokens in the data
        :param rnn_size: The size of hidden cells in LSTM/GRU
        :param embedding_size: The dimension of embedding space
        """
        super().__init__()
        # TODO: initialize the vocab_size, rnn_size, embedding_size
        self.vocab_size = vocab_size
        self.rnn_size = rnn_size
        self.embedding_size = embedding_size

        # TODO: initialize embeddings, LSTM, and linear layers
        # embedding -> LSTM -> dropout -> logits
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_size, padding_idx=0)
        self.lstm = nn.LSTM(input_size=embedding_size, hidden_size=rnn_size, num_layers=1, batch_first=True, bias=True)
        self.logits = nn.Linear(in_features=rnn_size, out_features=vocab_size, bias=True)
        # self.softmax = nn.Softmax(dim=-1)

    def forward(self, inputs, lengths):

        """
        Runs the forward pass of the model.

        :param inputs: word ids (tokens) of shape (batch_size, window_size)
        :param lengths: array of actual lengths (no padding) of each input

        :return: the logits, a tensor of shape
                 (batch_size, window_size, vocab_size)
        """
        # TODO: write forward propagation

        # make sure you use pack_padded_sequence and pad_packed_sequence to
        # reduce calculation
        total_length = inputs.shape[1]

        embedding = self.embedding(inputs)
        packed_seq = pack_padded_sequence(input=embedding, lengths=lengths, batch_first=True, enforce_sorted=False)
        tmp_output, _ = self.lstm(packed_seq)
        seq, _ = pad_packed_sequence(sequence=tmp_output, batch_first=True, total_length=total_length)
        logits = self.logits(seq)
        # return self.softmax(logits)
        return logits
