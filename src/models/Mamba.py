from mamba_ssm import Mamba
import torch
import torch.nn as nn
import torch.nn.functional as F

class mamba(nn.Module):

    def __init__(self, configs):
        super().__init__()

        self.input_drop = nn.Dropout(configs.dropout)

        self.input_size = configs.enc_in
        self.output_size = configs.c_out
        self.num_layers = configs.e_layers
        self.noise_level = configs.noise_level

        self.mamba = nn.ModuleList([Mamba(
            d_model = configs.d_model, # Model dimension d_model
            d_state = configs.d_state, # SSM state expansion factor 16
            d_conv = configs.d_conv,   # Local convolution width 4
            expand = configs.expand,   # Block expansion factor 2
        ) for i in range(self.num_layers)])

        self.in_layer  = nn.Linear(self.input_size, configs.d_model)

        self.projection = nn.Sequential(
                nn.Linear(configs.d_model, configs.d_ff, bias=True),
                nn.GELU(),
                nn.Linear(configs.d_ff, configs.c_out, bias=True),
        )

    def forward(self, x):
        # input [batch_size, seq_len, num_fea]
        x = self.input_drop(x)

        if self.training and self.noise_level > 0:
            noise = torch.randn_like(x).to(x)
            x = x + noise * self.noise_level

        # x [batch_size, seq_len, num_fea]
        x = self.in_layer(x)
        for i in range(self.num_layers):
            x = self.mamba[i](x)
        # x [batch_size, seq_len, hidden_size]
        x = x[:,-1,:]
        out = self.projection(x)

        return out