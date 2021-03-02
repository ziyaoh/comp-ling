import math
import torch
import torch.nn as nn

FF_HIDDEN = 2048

"""
check list
1. residual & layer normalization
2. mask
3. positional encoding
"""
class Transformer(nn.Module):
    """
    word embedding
    pos encoding
    go through each layer
    logits
    """
    def __init__(self, vocab_size, hidden_size, num_head, num_layers, dropout_rate):
        super(Transformer, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=hidden_size, padding_idx=0)
        self.pos_enc = PositionalEncoder(hidden_size)
        self.layers = nn.ModuleList([TransformLayer(hidden_size, num_head, dropout_rate)])
        self.logits = nn.Linear(in_features=hidden_size, out_features=vocab_size)

    def forward(self, inputs):
        static_emb = self.embedding(inputs)
        pos_emb = self.pos_enc(static_emb)
        for layer in self.layers:
            res = layer(pos_emb)
        logits = self.logits(res)
        return logits


class TransformLayer(nn.Module):
    """
    x = multihead(input)
    return feedforward(x)
    """
    def __init__(self, hidden_size, num_head, dropout_rate):
        super(TransformLayer, self).__init__()
        self.multiheads = MultiHeadAttention(hidden_size, num_head, dropout_rate)
        self.ff = FeedForward(hidden_size, dropout_rate)

    def forward(self, inputs):
        x = self.multiheads(inputs)
        return self.ff(x)


class MultiHeadAttention(nn.Module):
    """
    x = concate([singleAttn(input)])
    x = linear(x)
    x = dropout(x)
    return norm(input + x)
    """
    def __init__(self, hidden_size, num_head, dropout_rate):
        super(MultiHeadAttention, self).__init__()
        self.hidden_size = hidden_size

        self.heads = nn.ModuleList([SingleAttention(hidden_size, num_head) for _ in range(num_head)])

        d_k = hidden_size // num_head
        self.condense = nn.Linear(in_features=d_k*num_head, out_features=hidden_size)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.norm = nn.LayerNorm(hidden_size)
    
    def forward(self, inputs):
        attns = [head(inputs) for head in self.heads]
        attns = torch.cat(attns, -1)
        condensed_attn = self.condense(attns)
        result = self.dropout(condensed_attn)
        return self.norm(inputs + result)


class SingleAttention(nn.Module):
    def __init__(self, hidden_size, num_head):
        super(SingleAttention, self).__init__()
        self.hidden_size = hidden_size
        self.d_k = hidden_size // num_head
        self.k = nn.Linear(hidden_size, self.d_k, bias=True)
        self.q = nn.Linear(hidden_size, self.d_k, bias=True)
        self.v = nn.Linear(hidden_size, self.d_k, bias=True)

    def forward(self, inputs):

        # inputs = [batch size, seq len, hidden size]
        seq_len = inputs.shape[1]

        k = self.k(inputs)
        # k = [batch size, seq len, d_k]
        q = self.q(inputs)
        q = q.transpose(1, 2)
        # q = [batch size, d_k, seq len]
        v = self.v(inputs)
        # v = [batch size, seq len, d_k]

        weight = torch.bmm(k, q) // math.sqrt(self.d_k)
        mask = torch.triu(torch.ones((seq_len, seq_len)), diagonal=1).bool()
        weight = weight.masked_fill(mask, -float("inf")).type_as(weight)
        # weight = [batch size, seq len, seq len]

        score = torch.softmax(weight, -1)
        # score = [batch size, seq len, seq len]
        attention = torch.bmm(score, v)
        # attention = [batch size, seq len, d_k]

        return attention


class FeedForward(nn.Module):
    """
    x = layer1(input)
    x = relu(x)
    x = layer2(x)
    x = dropout(x)
    return norm(input + x)
    """
    def __init__(self, hidden_size, dropout_rate):
        super(FeedForward, self).__init__()
        self.layer1 = nn.Linear(in_features=hidden_size, out_features=FF_HIDDEN)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(in_features=FF_HIDDEN, out_features=hidden_size)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, inputs):
        x = self.layer1(inputs)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.dropout(x)
        return self.norm(inputs + x)


class PositionalEncoder(nn.Module):
    def __init__(self, hidden_size, max_seq_len = 1000):
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
        x = x + torch.autograd.Variable(self.encode[:,:seq_len], requires_grad=False)
        return x