import torch
import torch.nn as nn
import torch.nn.functional as F



class GRU(nn.Module):
    def __init__(self, configs):
        super(GRU, self).__init__()
        self.input_size = configs.enc_in
        self.output_size = configs.c_out
        self.hidden_size = configs.d_model
        self.num_layers = configs.e_layers
        self.dropout = configs.dropout

        self.input_drop = nn.Dropout(self.dropout)
        self.rnn = nn.GRU(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout
        )

        self.projection = nn.Sequential(
            nn.Linear(self.hidden_size, configs.d_ff, bias=True),
            nn.GELU(),
            nn.Linear(configs.d_ff, configs.c_out, bias=True),
        )

    def forward(self, x):
        x = self.input_drop(x)
        r_out, _ = self.rnn(x)
        last_out = r_out[:, -1]
        last_out = self.projection(last_out)

        return last_out