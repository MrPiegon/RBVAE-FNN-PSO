import torch
from torch import tanh
import torch.nn as nn
import torch.nn.functional as F

class JDnn(nn.Module):

    def __init__(self, n=6, hidden_units=None, latent_dim=512, prop_num=1, dropout=0.):
        super(JDnn, self).__init__()
        self.n = n
        self.latent_dim = latent_dim
        if hidden_units is None:
            self.hidden_units = [256, 128, 64, 32, 16, 8, 4, 8, 8, 8]
        self.input = nn.Linear(latent_dim, self.hidden_units[0])
        self.BatchNorm_input = nn.BatchNorm1d(self.hidden_units[0])
        # Dnn(former part)
        if n == 1 or n == 2:
            self.dnn_network_form = nn.Linear(self.hidden_units[0], self.hidden_units[1])
        else:
            self.dnn_network_form = nn.ModuleList(
                [nn.Linear(layer[0], layer[1]) for layer in list(zip(self.hidden_units[:n - 1], self.hidden_units[1:n]))])
        self.BatchNorm_form = nn.ModuleList([nn.BatchNorm1d(unit) for unit in self.hidden_units[1:n]])
        self.dnn_form_last = nn.Linear(self.hidden_units[n - 1], self.hidden_units[n])
        # concat
        self.BatchNorm_cont_form = nn.BatchNorm1d(self.hidden_units[n] + 2)
        if n == len(self.hidden_units) - 1:
            pass
        else:
            self.cont_layer = nn.Linear(self.hidden_units[n] + 2, self.hidden_units[n + 1])
            self.BatchNorm_cont_back = nn.BatchNorm1d(self.hidden_units[n + 1])
            # Dnn(latter part)
            if len(self.hidden_units[n + 1:-1]) == 0 or len(self.hidden_units[n + 1:-1]) == 1:
                self.dnn_network_back = nn.Linear(self.hidden_units[n + 1], self.hidden_units[-1])
                self.BatchNorm_back = nn.BatchNorm1d(self.hidden_units[-1])
            else:
                self.dnn_network_back = nn.ModuleList(
                    [nn.Linear(layer[0], layer[1]) for layer in
                     list(zip(self.hidden_units[n + 1:-1], self.hidden_units[n + 2:]))])
                self.BatchNorm_back = nn.ModuleList([nn.BatchNorm1d(unit) for unit in self.hidden_units[n + 2:]])
        # 插在非最后层
        self.output = nn.Linear(self.hidden_units[-1], prop_num)
        # 插在最后一层
        self.output_last = nn.Linear(self.hidden_units[-1] + 2, prop_num)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, z, t, p):
        if len(t.shape) == 1:
            t = t.unsqueeze(-1)
        if len(p.shape) == 1:
            p = p.unsqueeze(-1)
        if len(z.shape) == 1:
            z = z.unsqueeze(-1)
            z = torch.reshape(z, (1, -1))
        if len(t.shape) > 2 or len(p.shape) > 2:
            raise ValueError("Input tensor must be 1D or 2D")
        if self.n == 0:
            x = tanh(self.input(z))
        else:
            z = tanh(self.input(z))
            x = self.BatchNorm_input(z)
        # Dnn 前半部分
        if self.n == 0:
            pass
        elif self.n == 1:
            x = tanh(self.dnn_network_form(x))
        elif self.n == 2:
            x = self.BatchNorm_form[0](tanh(self.dnn_network_form(x)))
            x = tanh(self.dnn_form_last(x))
        else:
            for linear, BatchNorm in zip(self.dnn_network_form, self.BatchNorm_form):
                x = BatchNorm(tanh(linear(x)))
            x = tanh(self.dnn_form_last(x))
        # 插入t,p
        x = self.BatchNorm_cont_form(torch.hstack((x, t, p)))
        if self.n == len(self.hidden_units) - 1:
            x = tanh(self.output_last(x))
        else:
            x = self.BatchNorm_cont_back(self.cont_layer(x))
            # Dnn 后半部分
            if len(self.hidden_units[self.n + 1:-1]) == 0:
                pass
            elif len(self.hidden_units[self.n + 1:-1]) == 1:
                x = self.BatchNorm_back(tanh(self.dnn_network_back(x)))
            else:
                for linear, BatchNorm in zip(self.dnn_network_back, self.BatchNorm_back):
                    x = BatchNorm(tanh(linear(x)))
            x = tanh(self.output(x))
        x = self.dropout(x)
        return x

