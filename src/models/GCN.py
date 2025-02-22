import torch
import torch.nn as nn
import torch.nn.functional as F

class GCN(nn.Module):
    def __init__(self, configs):
        super().__init__()

        self.base_model = configs.base_model
        self.input_size = configs.enc_in
        self.output_size = configs.c_out
        self.hidden_size = configs.d_model
        self.d_ff = configs.d_ff
        self.dropout = configs.dropout
        self.num_layers = configs.e_layers

        if self.base_model == "GRU":
            self.rnn = nn.GRU(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                batch_first=True,
                dropout=self.dropout,
            )
        elif self.base_model == "LSTM":
            self.rnn = nn.LSTM(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                batch_first=True,
                dropout=self.dropout,
            )
        else:
            raise ValueError("Unknown base model name `%s`" % self.base_model)

        self.fc = nn.Linear(self.hidden_size, self.hidden_size)
        self.leaky_relu = nn.LeakyReLU()

        self.projection = nn.Sequential(
            nn.Linear(self.hidden_size, self.d_ff, bias=True),
            nn.GELU(),
            nn.Linear(self.d_ff, configs.c_out, bias=True),
        )

    def forward(self, x):
        out, _ = self.rnn(x)
        hidden = out[:, -1, :]  # (batch_size, hidden_size)
        
        batch_size = hidden.size(0)
        if batch_size > 0:
            # adj
            A = torch.ones(batch_size, batch_size, device=hidden.device)
            D = A.sum(dim=1).clamp(min=1e-6)
            D_sqrt_inv = torch.sqrt(1.0 / D)
            D_sqrt_inv = torch.diag(D_sqrt_inv).to(hidden.device)
            A_hat = torch.mm(torch.mm(D_sqrt_inv, A), D_sqrt_inv)

            hidden = torch.mm(A_hat, hidden)
        
        hidden = self.fc(hidden)
        hidden = self.leaky_relu(hidden)
        
        return self.projection(hidden).squeeze()