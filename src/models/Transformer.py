import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    # reference: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        truncated_div_term = div_term[:d_model//2]
        pe[:, 1::2] = torch.cos(position * truncated_div_term)
        # pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class Transformer(nn.Module):
    """Transformer Model

    Args:
        input_size (int): input size (# features)
        hidden_size (int): hidden size
        num_layers (int): number of transformer layers
        num_heads (int): number of heads in transformer
        dropout (float): dropout rate
        input_drop (float): input dropout for data augmentation
        noise_level (float): add gaussian noise to input for data augmentation
    """

    def __init__(self, configs):
        super().__init__()

        self.input_size = configs.enc_in
        self.output_size = configs.c_out
        self.hidden_size = configs.d_model
        self.d_ff = configs.d_ff
        self.num_layers = configs.e_layers
        self.noise_level = configs.noise_level
        self.dropout = configs.dropout

        self.input_drop = nn.Dropout(self.dropout)

        self.input_proj = nn.Linear(self.input_size, self.hidden_size)

        self.pe = PositionalEncoding(self.input_size, self.dropout)
        layer = nn.TransformerEncoderLayer(
            nhead=configs.n_heads, dropout=self.dropout, d_model=self.hidden_size, dim_feedforward=self.d_ff
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=self.num_layers)

        self.projection = nn.Sequential(
                nn.Linear(self.hidden_size, self.d_ff, bias=True),
                nn.GELU(),
                nn.Linear(self.d_ff, configs.c_out, bias=True),
        )


    def forward(self, x):
        # input [batch_size, seq_len, num_fea]
        x = self.input_drop(x)

        if self.training and self.noise_level > 0:
            noise = torch.randn_like(x).to(x)
            x = x + noise * self.noise_level
        # x [batch_size, seq_len, num_fea]

        x = x.permute(1, 0, 2).contiguous()  # the first dim need to be sequence
        # x [seq_len, batch_size, num_fea]
        x = self.pe(x)
        # x [seq_len, batch_size, num_fea]
        x = self.input_proj(x)
        # x [seq_len, batch_size, hidden_size]
        out = self.encoder(x)
        # out [seq_len, batch_size, hidden_size]
        # out[-1] [batch_size, hidden_size]
        out = self.projection(out[-1])

        return out