from torch import nn
from torch.nn import functional as F
import torch
import math
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BERT(nn.Module):
    def __init__(self, vocab_size, hidden_size=512, num_head=8, num_layers=6, dropout_rate=0.1):
        super().__init__()
        # TODO: Initialize BERT modules
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=hidden_size, padding_idx=0)
        self.pos_enc = PositionalEncoder(hidden_size)
        layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_head, dropout=dropout_rate)
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.logits = nn.Linear(in_features=hidden_size, out_features=vocab_size)

        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.logits.bias.data.zero_()
        self.logits.weight.data.uniform_(-initrange, initrange)

    def forward(self, x, all_masked_ind):
        # TODO: Write feed-forward step
        res = self.get_embeddings(x)

        # res = [batch_size * seq_len * hidden]
        # all_masked_ind = [batch_size * num_masked]
        masked_res = torch.empty(0).to(device)
        for seq, masked_ind in zip(res, all_masked_ind):
            # seq = [seq_len * hidden]
            masked = torch.index_select(seq, 0, masked_ind)
            # masked = [num_masked * hidden]
            masked_res = torch.cat((masked_res, masked.unsqueeze(0)))
        # masked_res = [batch_size * num_masked * hidden]

        logits = self.logits(masked_res)
        return logits

    def get_embeddings(self, x):
        # TODO: Write function that returns BERT embeddings of a sequence
        emb = self.embedding(x)
        res = self.pos_enc(emb)

        res = torch.transpose(res, 0, 1)
        res = self.transformer(res)
        res = torch.transpose(res, 0, 1)
        return res


"""
class PositionalEncoder(nn.Module):
    def __init__(self, hidden_size, max_seq_len=1000):
        super(PositionalEncoder, self).__init__()
        
        encode = torch.zeros(max_seq_len, hidden_size)
        for pos in range(max_seq_len):
            for i in range(0, hidden_size, 2):
                tmp = pos / (10000 ** (i/hidden_size))
                encode[:, i] = math.sin(tmp)
                encode[:, i + 1] = math.cos(tmp)
                
        self.encode = encode.unsqueeze(0)
        self.register_buffer('pos_enc', self.encode)
 
    
    def forward(self, x):
        seq_len = x.size(1)
        x = x + torch.autograd.Variable(self.encode[:,:seq_len], requires_grad=False).to(device)
        return x
"""


class PositionalEncoder(nn.Module):
    def __init__(self, hidden_size, max_seq_len=1000):
        super(PositionalEncoder, self).__init__()
        self.embed = nn.Embedding(max_seq_len, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        position = torch.arange(0, x.shape[1]).unsqueeze(
            0).repeat(x.shape[0], 1).to(device)
        pos_x = self.embed(position)
        # input = embedded_x * self.scale + pos_x
        embeddings = self.norm(x + pos_x)
        return embeddings


