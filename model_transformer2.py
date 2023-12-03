import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.autograd import Variable

class AddEps(nn.Module):
    def __init__(self, channels):
        super(AddEps, self).__init__()

        self.channels = channels
        self.linear = nn.Sequential(
            nn.Linear(channels, channels),
            nn.Tanh()
        )

    def forward(self, x):
        eps = torch.randn_like(x)
        eps = self.linear(eps)

        return eps + x

def clones(module, n):
    """produce n identical layers."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


def attention(query, key, value, mask=None, dropout=None):
    """compute 'scaled dot product attention'"""
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = f.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class Attention(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(Attention, self).__init__()
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model

    def forward(self, query, key, value, mask=None):
        """implements figure 2"""
        nbatches = query.size(0)

        query, key, value = \
            [l(x).view(nbatches, -1, self.d_model)
             for l, x in zip(self.linears, (query, key, value))]

        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        return self.linears[-1](x)


class EncoderDecoder(nn.Module):
    """
    a standard encoder-decoder architecture. base for this and many
    other models.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        """take in and process masked src and target sequences."""
        return self.decode(self.encode(src, src_mask), src_mask,
                           tgt, tgt_mask)

    def encode(self, src, src_mask, add_eps):
        return self.encoder(self.src_embed(src), src_mask, add_eps)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


class Generator(nn.Module):
    """define standard linear + softmax generation step."""

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return f.log_softmax(self.proj(x), dim=-1)


class Encoder(nn.Module):
    """core encoder is a stack of n layers"""

    def __init__(self, layer, n, in_shape, dropout):
        super(Encoder, self).__init__()
        self.layers = clones(layer, n)
        self.norm = LayerNorm(layer.size)
        self.dot_cnn1 = nn.Sequential(
            nn.Conv1d(in_shape, in_shape, kernel_size=1, stride=1),
            nn.Softplus()
        )
        self.dot_cnn2 = nn.Sequential(
            nn.Conv1d(2 * in_shape, in_shape, kernel_size=1, stride=1),
            nn.Softplus()
        )
        self.cnn_layer = nn.Sequential(
            nn.Conv1d(in_shape, 2*in_shape,
                      kernel_size=5, stride=1, padding=4),
            nn.Softplus()
        )

        self.eps = AddEps(in_shape)

        self.linear_o = nn.Linear(in_shape, in_shape)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask, add_eps):
        """pass the input (and mask) through each layer in turn."""
        for layer in self.layers:
            x = layer(x, mask)
        if add_eps:
            y = self.dot_cnn1(self.eps(x).transpose(1, 2)).transpose(1, 2)
            y = self.cnn_layer(self.eps(y).transpose(1, 2))[:, :, :-4]
        else:
            y = self.dot_cnn1(x.transpose(1, 2)).transpose(1, 2)
            y = self.cnn_layer(y.transpose(1, 2))[:, :, :-4]
        y = self.dot_cnn2(y).transpose(1, 2)
        y = x + y

        y = self.dropout(y)
        y = self.linear_o(y)
        return self.norm(y)


class LayerNorm(nn.Module):
    """construct a layernorm module (see citation for details)."""

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SubLayerConnection(nn.Module):
    """
    a residual connection followed by a layer norm.
    note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SubLayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """apply residual connection to any sublayer with the same size."""
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    """encoder is made up of self-attn and feed forward (defined below)"""

    def __init__(self, size, self_attn, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.sublayer = clones(SubLayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        """follow figure 1 (left) for connections."""
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return x


class Decoder(nn.Module):
    def __init__(self, hyper_params):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(
            hyper_params['d_model'], hyper_params['item_embed_size'])
        self.linear2 = nn.Linear(
            hyper_params['item_embed_size'], hyper_params['total_items'] + 1)
        nn.init.xavier_normal_(self.linear1.weight)
        nn.init.xavier_normal_(self.linear2.weight)
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.linear1(x)
        out_embed = x

        x = self.activation(x)
        x = self.linear2(x)
        return x, out_embed


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """implement the pe function."""

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)


class Model(nn.Module):
    def __init__(self, hyper_params):
        super(Model, self).__init__()

        c = copy.deepcopy
        attn = Attention(hyper_params['d_model'])
        position = PositionalEncoding(hyper_params['d_model'], hyper_params['dropout'])

        self.hyper_params = hyper_params

        self.decode = Decoder(hyper_params)

        self.embedding = nn.Sequential(Embeddings(hyper_params['d_model'],
                                                  hyper_params['total_items'] + 1), c(position))

        self.model = EncoderDecoder(
            Encoder(EncoderLayer(hyper_params['d_model'], c(attn), hyper_params['dropout']),
                    hyper_params['N_of_layers'], hyper_params['d_model'], hyper_params['dropout']),
            Decoder(hyper_params),
            nn.Sequential(Embeddings(hyper_params['d_model'], hyper_params['total_items'] + 1), c(position)),
            nn.Sequential(Embeddings(hyper_params['d_model'], hyper_params['total_items'] + 1), c(position)),
            Generator(hyper_params['d_model'], hyper_params['total_items'] + 1))

        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform(p)

    def forward(self, x, mask, add_eps):
        x_real = self.embedding(x)
        z_inferred = self.model.encode(x, mask, add_eps)
        # [batch_size x seq_len x total_items]
        dec_out, out_embed = self.decode(z_inferred)

        return dec_out, x_real, z_inferred, out_embed


class Adversary(nn.Module):
    def __init__(self, hyper_params):
        super(Adversary, self).__init__()
        self.hyper_params = hyper_params
        self.linear_test = nn.Linear(
            hyper_params['d_model'] + hyper_params['d_model'], hyper_params['d_model'])
        self.linear_i = nn.Linear(
            hyper_params['d_model'] + hyper_params['d_model'], 128)

        self.dnet_list = []
        self.net_list = []
        for _ in range(2):
            self.dnet_list.append(nn.Linear(128, 128))
            self.net_list.append(nn.Linear(128, 128))

        self.dnet_list = nn.ModuleList(self.dnet_list)
        self.net_list = nn.ModuleList(self.net_list)

        self.linear_o = nn.Linear(128, hyper_params['d_model'])
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)

    def forward(self, x, z, padding):
        # batch_size x seq_len x dim
        net = torch.cat((x, z), 2)
        net = self.linear_i(net)
        net = self.dropout1(net)

        for i in range(2):
            dnet = self.dnet_list[i](net)
            net = net + self.net_list[i](dnet)
            net = f.elu(net)

        # seq_len
        net = self.linear_o(net)
        net = self.dropout2(net)
        net = net + 0.5 * torch.square(z)

        net = net * (1.0 - padding.float().unsqueeze(2))
        # net = self.linear_test(net)
        return net


class GRUAdversary(nn.Module):
    def __init__(self, hyper_params):
        super(GRUAdversary, self).__init__()

        self.hyper_params = hyper_params
        self.gru = nn.GRU(
            input_size=hyper_params['d_model'] + hyper_params['d_model'],
            hidden_size=128,
            batch_first=True
        )
        self.linear = nn.Linear(128, 1)

    def forward(self, x, z, padding):
        x = f.gelu(self.gru(torch.cat([x, z], dim=-1))[0])
        x = self.linear(x).squeeze(2)
        return (1.0 - padding.float()) * x
