import torch
import numpy as np
import torch.nn as nn

def init_params(m):
    # Initialize model's parameters. Input:self
    for module_name, module in m.named_modules():
        for param_name, param in module.named_parameters():
            if 'weight' in param_name:
                if 'conv' in param_name or 'lin' in param_name or 'ih' in param_name:
                    nn.init.xavier_uniform_(param)
                elif 'hh' in param_name:
                    nn.init.orthogonal_(param)
            elif param_name == 'bias':
                nn.init.constant_(param, 0.0)


class VAELoss(torch.autograd.Function):
    def __init__(self, loss_weight=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if loss_weight is not None:
            loss_weight = torch.FloatTensor(loss_weight).cuda()
        self.softmax_xentropy = nn.CrossEntropyLoss(weight=loss_weight, reduction='sum')

    def forward(self, x, x_decoded_mean, z_mean, z_log_var):
        x = x.contiguous().view(-1)
        x_decoded_mean = x_decoded_mean.contiguous().view(-1, x_decoded_mean.size(-1))
        xent_loss = self.softmax_xentropy(input=x_decoded_mean, target=x)
        kl_loss = - 0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())
        z_log_var_view = z_log_var.cpu().detach().numpy()
        z_mean_view = z_log_var.cpu().detach().numpy()
        return xent_loss, kl_loss

def kl_anneal_function(anneal_function, step, k1=0.1, k2=0.2, max_value=1.0, x0=100):
    assert anneal_function in ['logistic', 'linear', 'step', 'cyclical'], 'unknown anneal_function'
    if anneal_function == 'logistic':
        return float(1 / (1 + np.exp(- k1 * (step - x0))))
    elif anneal_function == 'step':
        cnt = step // x0
        step = step % x0
        if cnt > 0:
            max_value -= cnt * 0.1
            max_value = max(0.1, max_value)
        ma = min(k2 * cnt + k2, max_value)
        mi = 0.01 + k1 * cnt
        return min(ma, mi + 2 * step * (max(ma - mi, 0)) / x0)
    elif anneal_function == 'linear':
        return min(max_value, 0.01 + step / x0)
    elif anneal_function == 'cyclical':
        cnt = step // x0 // 5
        step = step % x0
        ma = min(k2 * cnt + k2, max_value)
        mi = k1
        return min(ma, ma * cnt + mi + 2 * step * (ma - mi) / x0)

def sample(z_mean, z_log_var, size, epsilon_std=1.0):
    epsilon = torch.FloatTensor(*size).normal_(0, epsilon_std).cuda()
    return z_mean + torch.exp(0.5 * z_log_var) * epsilon

def log_sum_exp(value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation
    value.exp().sum(dim, keepdim).log()
    """
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0), dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        return m + torch.log(sum_exp)