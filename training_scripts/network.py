import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class GegEmbedding(nn.Module):
    def __init__(self, N_freqs, alpha=0.5):
        super(GegEmbedding, self).__init__()
        self.N_freqs = N_freqs
        self.alpha = alpha

    def forward(self, x):
        n = len(x)
        x_in = x.squeeze()
        c = torch.zeros(( n, self.N_freqs + 1 ), device=x.device)
        c[..., 0] = 1.0
        c[..., 1] = 2.0 * self.alpha * x_in
        for i in range(2, self.N_freqs + 1 ):
            c[..., i] = (  ( 2 * i - 2  + 2.0 * self.alpha ) * x_in * c[..., i-1]   +  (- i + 2  - 2.0 * self.alpha ) * c[..., i-2] ) / i

        return c[..., 1:].contiguous().view(n, -1)

class SineLayer(nn.Module):    
    def __init__(self, in_features, out_features, bias=False, is_first=False, is_res=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.is_res = is_res
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights(self.linear)
    
    def init_weights(self, layer):
        with torch.no_grad():
            if self.is_first:
                layer.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                layer.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,  np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        if self.is_res:
            return input + torch.sin(self.omega_0 * self.linear(input))
        else:
            return torch.sin(self.omega_0 * self.linear(input))

class SIGNET_static(nn.Module):
    def __init__(self, first_omega_0=30, hidden_omega_0=30., hidden_layers=8, in_feature_ratio=1, out_features=3, skips=[3], hidden_features=512, alpha=0.5, with_res=False, with_sigmoid=False, with_norm=False):
        super().__init__()

        in_features = int(in_feature_ratio * 512)
        
        self.with_res = with_res
        self.with_sigmoid = with_sigmoid
        self.D = hidden_layers + 2
        self.skips = skips

        for i in range(hidden_layers+1):
            if i == 0:
                layer = SineLayer(in_features, hidden_features, bias=True, is_first=True, is_res=False, omega_0=first_omega_0)
            elif i in skips:
                layer = SineLayer(hidden_features + in_features, hidden_features, bias=True, is_first=False, is_res=False, omega_0=hidden_omega_0)
            else:
                layer = SineLayer(hidden_features, hidden_features, is_first=False, bias=True, is_res=self.with_res, omega_0=hidden_omega_0)
            if with_norm:
                layer = nn.Sequential(layer, nn.LayerNorm(hidden_features, elementwise_affine=True))
            setattr(self, f"encoding_{i+1}", layer)

        final_linear = nn.Linear(hidden_features, out_features, bias=True)
        
        with torch.no_grad():
            final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,  np.sqrt(6 / hidden_features) / hidden_omega_0)
        setattr(self, f"encoding_{hidden_layers+2}", final_linear)
        
        self.N_xy = int(in_feature_ratio * 240)
        self.N_uv = int(in_feature_ratio * 16)

        self.xy_embedd = GegEmbedding(self.N_xy, alpha)
        self.uv_embedd = GegEmbedding(self.N_uv, alpha)

    def forward(self, x):
        # x: [B, 4]
        emb_x = torch.cat( [self.uv_embedd(x[:, 0]), self.uv_embedd(x[:, 1]), self.xy_embedd(x[:, 2]), self.xy_embedd(x[:, 3]) ], axis=1)
        out = emb_x
        for i in range(self.D):
            if i in self.skips:
                out = torch.cat([emb_x, out], -1)
            out = getattr(self, f"encoding_{i+1}")(out)
        return out
