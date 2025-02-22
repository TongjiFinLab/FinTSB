import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTM(nn.Module):
    """LSTM Model

    Args:
        input_size (int): input size (# features)
        hidden_size (int): hidden size
        num_layers (int): number of hidden layers
        use_attn (bool): whether use attention layer.
            we use concat attention as https://github.com/fulifeng/Adv-ALSTM/
        dropout (float): dropout rate
        input_drop (float): input dropout for data augmentation
        noise_level (float): add gaussian noise to input for data augmentation
    """

    def __init__(self, configs):
        super().__init__()

        self.input_size = configs.enc_in
        self.output_size = configs.c_out
        self.hidden_size = configs.d_model
        self.num_layers = configs.e_layers
        self.use_attn = configs.use_attn
        self.noise_level = configs.noise_level
        self.dropout = configs.dropout

        self.input_drop = nn.Dropout(self.dropout)

        self.rnn = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout,
        )

        if self.use_attn:
            self.W = nn.Linear(self.hidden_size, self.hidden_size)
            self.u = nn.Linear(self.hidden_size, 1, bias=False)
            self.projection = nn.Sequential(
                nn.Linear(self.hidden_size * 2, configs.d_ff, bias=True),
                nn.GELU(),
                nn.Linear(configs.d_ff, configs.c_out, bias=True),
            )
        else:
            self.projection = nn.Sequential(
                nn.Linear(self.hidden_size, configs.d_ff, bias=True),
                nn.GELU(),
                nn.Linear(configs.d_ff, configs.c_out, bias=True),
            )

    def forward(self, x):

        x = self.input_drop(x)
        if self.training and self.noise_level > 0:
            noise = torch.randn_like(x).to(x)
            x = x + noise * self.noise_level

        rnn_out, _ = self.rnn(x)

        last_out = rnn_out[:, -1]

        if self.use_attn:
            laten = self.W(rnn_out).tanh()
            scores = self.u(laten).softmax(dim=1)
            att_out = (rnn_out * scores).sum(dim=1).squeeze()

            last_out = torch.cat([last_out, att_out], dim=1)
        
        last_out = self.projection(last_out)

        return last_out