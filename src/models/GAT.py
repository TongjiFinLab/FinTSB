import torch
import torch.nn as nn
import torch.nn.functional as F




class GAT(nn.Module):
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
            raise ValueError("unknown base model name `%s`" % self.base_model)

        self.transformation = nn.Linear(self.hidden_size, self.hidden_size)
        self.a = nn.Parameter(torch.randn(self.hidden_size * 2, 1))
        self.a.requires_grad = True
        self.fc = nn.Linear(self.hidden_size, self.hidden_size)
        self.leaky_relu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=1)

        self.projection = nn.Sequential(
                nn.Linear(self.hidden_size, self.d_ff, bias=True),
                nn.GELU(),
                nn.Linear(self.d_ff, configs.c_out, bias=True),
        )

    def cal_attention(self, x, y):
        x = self.transformation(x)
        y = self.transformation(y)

        sample_num = x.shape[0]
        dim = x.shape[1]
        e_x = x.expand(sample_num, sample_num, dim)
        e_y = torch.transpose(e_x, 0, 1)
        attention_in = torch.cat((e_x, e_y), 2).view(-1, dim * 2)
        self.a_t = torch.t(self.a)
        attention_out = self.a_t.mm(torch.t(attention_in)).view(sample_num, sample_num)
        attention_out = self.leaky_relu(attention_out)
        att_weight = self.softmax(attention_out)
        return att_weight

    def forward(self, x):
        out, _ = self.rnn(x)
        hidden = out[:, -1, :]
        att_weight = self.cal_attention(hidden, hidden)
        hidden = att_weight.mm(hidden) + hidden
        hidden = self.fc(hidden)
        hidden = self.leaky_relu(hidden)
        return self.projection(hidden).squeeze()