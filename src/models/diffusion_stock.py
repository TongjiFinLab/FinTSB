import torch
import torch.nn as nn
from denoising_diffusion_pytorch import Unet1D, GaussianDiffusion1D
import torch.nn.functional as F


class DiffStock(nn.Module):
    def __init__(self, configs):
        super(DiffStock, self).__init__()
        self.configs = configs
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.layer = configs.e_layers
        self.unet_model = Unet1D(
            dim=configs.d_model,
            dim_mults=(1, 2, 4),
            channels=configs.d_model,
        )
        self.diffusion_model = GaussianDiffusion1D(
            self.unet_model, 
            seq_length=self.seq_len,
            timesteps=200,
            sampling_timesteps=configs.num_samples,
            objective='pred_v'
        )
        self.in_proj = nn.Linear(configs.enc_in, configs.d_model, bias=True)

        # c_out = 1   daily return
        self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)

    def sample(self, x):
        B, L, C = x.shape
        n_steps = 4
        # noise = torch.randn(B, L, C).to(x.device)

        # noise = noise.unsqueeze(1).expand(-1, T, -1)  # [B, T, C]
        t = torch.randint(low=0, high=n_steps, size=(B // 2 + 1,)).cuda()
        t = torch.cat([t, n_steps - t - 1], dim=0)[:B]
        x = self.diffusion_model.q_sample(x, t)

        return x

    def forecast(self, x_enc):
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        x_enc = self.in_proj(x_enc)
        # embedding
        enc_out = self.sample(x_enc)


        # project back
        dec_out = self.projection(enc_out)
        return dec_out

    def forward(self, x):
        self.unet_model = self.unet_model.to(x.device)
        self.diffusion_model = self.diffusion_model.to(x.device)
        dec_out = self.forecast(x)
        return dec_out[:, -self.pred_len :, :]  # [B, L, D]
