from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        self.output_size = output_size
        self.rnn_size = rnn_size
        self.bpe = bpe

        # TODO: initialize embeddings, LSTM, and linear layers
        if bpe:
            self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_size, padding_idx=0)
        else:
            self.enc_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_size, padding_idx=0)
            self.dec_embedding = nn.Embedding(num_embeddings=output_size, embedding_dim=embedding_size, padding_idx=0)

        self.encoding_layer = nn.GRU(input_size=embedding_size, hidden_size=rnn_size, num_layers=1, batch_first=True, bias=True)
        self.decoding_layer = nn.GRU(input_size=embedding_size + rnn_size, hidden_size=rnn_size, num_layers=1, batch_first=False, bias=True)
        self.logits = nn.Linear(in_features=rnn_size, out_features=output_size, bias=True)

        self.attention = Attention(rnn_size)

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
        total_length = encoder_inputs.shape[1]
        decoder_inputs = decoder_inputs.transpose(1, 0)

        if self.bpe:
            encoder_embedding = self.embedding(encoder_inputs)
            decoder_embedding = self.embedding(decoder_inputs)
        else:
            encoder_embedding = self.enc_embedding(encoder_inputs)
            decoder_embedding = self.dec_embedding(decoder_inputs)

        encoder_packed_seq = pack_padded_sequence(input=encoder_embedding, lengths=encoder_lengths.cpu(), batch_first=True, enforce_sorted=False)
        encoder_output, state = self.encoding_layer(encoder_packed_seq)
        encoder_output_t, _ = pad_packed_sequence(sequence=encoder_output, batch_first=True, total_length=total_length)
        # encoder_output_t = [batch size, seq length, rnn size]
        encoder_output = encoder_output_t.transpose(1, 0)

        seq_len, batch_size = decoder_inputs.shape[0], decoder_inputs.shape[1]
        decoder_output = torch.zeros(seq_len, batch_size, self.rnn_size).to(device)
        for t, batch in enumerate(decoder_embedding):
            # batch = [batch size, embedding size]
            attn = self.attention(state, encoder_output)
            attn = attn.unsqueeze(1)
            #attention= [batch size, 1, seq len]

            context = torch.bmm(attn, encoder_output_t)
            # context = [batch size, 1, rnn size]
            context = context.transpose(1, 0)
            # context = [1, batch size, rnn size]

            batch = batch.unsqueeze(0)
            # batch = [1, batch size, embedding size]
            rnn_input = torch.cat((batch, context), dim=-1)

            output, state = self.decoding_layer(rnn_input, state)

            decoder_output[t] = output.squeeze(0)

        logits = self.logits(decoder_output)
        return logits.transpose(1, 0)



class Attention(nn.Module):
    def __init__(self, rnn_size):
        super().__init__()

        self.k = nn.Linear(rnn_size, rnn_size, bias=True)
        self.q = nn.Linear(rnn_size, rnn_size, bias = True)

    def forward(self, hidden, encoder_outputs):

        #hidden = [1, batch size, rnn size]
        #encoder_outputs = [seq len, batch size, rnn size]

        batch_size = encoder_outputs.shape[1]
        seq_len = encoder_outputs.shape[0]

        k = self.k(encoder_outputs)
        k = k.transpose(1, 0)
        q = self.q(hidden)
        q = q.permute(1, 2, 0)

        # k = [batch size, seq len, rnn size]
        # q = [batch size, rnn size, 1]

        attention = torch.bmm(k, q)
        attention = attention.squeeze(2)
        #attention= [batch size, seq len]

        return F.softmax(attention, dim=1)
