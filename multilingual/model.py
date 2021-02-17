from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch


class Seq2Seq(nn.Module):
    def __init__(self, vocab_size, rnn_size, embedding_size, output_size,
                 enc_seq_len, dec_seq_len, bpe):
        """
        The Model class implements the LSTM-LM model.
        Feel free to initialize any variables that you find necessary in the
        constructor.

        :param vocab_size: The number of unique tokens in the input
        :param rnn_size: The size of hidden cells in LSTM/GRU
        :param embedding_size: The dimension of embedding space
        :param output_size: The vocab size in output sequence
        :param enc_seq_len: The sequence length of encoder
        :param dec_seq_len: The sequence length of decoder
        :param bpe: whether the data is Byte Pair Encoded (shares same vocab)
        """
        super().__init__()
        # TODO: initialize the vocab_size, rnn_size, embedding_size
        self.vocab_size = vocab_size
        self.rnn_size = rnn_size
        self.bpe = bpe

        # TODO: initialize embeddings, LSTM, and linear layers
        if bpe:
            self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_size, padding_idx=0)
        else:
            self.enc_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_size, padding_idx=0)
            self.dec_embedding = nn.Embedding(num_embeddings=output_size, embedding_dim=embedding_size, padding_idx=0)

        self.encoding_layer = nn.GRU(input_size=embedding_size, hidden_size=rnn_size, num_layers=1, batch_first=True, bias=True)
        self.decoding_layer = nn.GRU(input_size=embedding_size, hidden_size=rnn_size, num_layers=1, batch_first=True, bias=True)
        self.logits = nn.Linear(in_features=rnn_size, out_features=output_size, bias=True)

    def forward(self, encoder_inputs, decoder_inputs, encoder_lengths,
                decoder_lengths):

        """
        Runs the forward pass of the model.

        :param inputs: word ids (tokens) of shape (batch_size, seq_len)
        :param encoder_lengths: array of actual lengths (no padding) encoder
                                inputs
        :param decoder_lengths: array of actual lengths (no padding) decoder
                                inputs

        :return: the logits, a tensor of shape
                 (batch_size, seq_len, vocab_size)
        """
        # TODO: write forward propagation

        # make sure you use pack_padded_sequence and pad_padded_sequence to
        # reduce calculation
        total_length = decoder_inputs.shape[1]

        if self.bpe:
            encoder_embedding = self.embedding(encoder_inputs)
            decoder_embedding = self.embedding(decoder_inputs)
        else:
            encoder_embedding = self.enc_embedding(encoder_inputs)
            decoder_embedding = self.dec_embedding(decoder_inputs)

        encoder_packed_seq = pack_padded_sequence(input=encoder_embedding, lengths=encoder_lengths, batch_first=True, enforce_sorted=False)
        decoder_packed_seq = pack_padded_sequence(input=decoder_embedding, lengths=decoder_lengths, batch_first=True, enforce_sorted=False)

        encoder_output, context_vec = self.encoding_layer(encoder_packed_seq)
        decoder_output, _ = self.decoding_layer(decoder_packed_seq)

        seq, _ = pad_packed_sequence(sequence=decoder_output, batch_first=True, total_length=total_length)
        logits = self.logits(seq)
        return logits