import os
import argparse

import numpy as np
from PIL import Image

import torch
import torch.nn as nn

def carte_to_geg(x, N_freqs, alpha=0.5):
    n = x.shape[0]
    c = np.zeros(( n, N_freqs + 1 ))
    c[:, 0] = 1.0
    c[:, 1] = 2.0 * alpha * x
    for i in range(2, N_freqs + 1 ):
      c[:, i] = (  ( 2 * i - 2  + 2.0 * alpha ) * x * c[:, i-1]   +  (- i + 2  - 2.0 * alpha ) * c[:, i-2] ) / i 
    return c[:, 1:]

class Embedding(nn.Module):
    def __init__(self, N_freqs, n_size, alpha=0.5):
        super(Embedding, self).__init__()
        self.N_freqs = N_freqs
        self.alpha = alpha

        x = np.linspace(-0.5, 0.5, n_size)
        #x = np.linspace(-1, 1, n_size)
        self.cache_geg = nn.Parameter(torch.from_numpy(carte_to_geg(x, N_freqs, alpha)).float(), requires_grad=False)

    def forward(self, x):
        return self.cache_geg[x.long()]

class SineLayer(nn.Module):    
    def __init__(self, in_features, out_features, bias=True, is_first=False, is_res=False, omega_0=30):
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
                layer.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                layer.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        if self.is_res:
            return input + torch.sin(self.omega_0 * self.linear(input))
        else:
            return torch.sin(self.omega_0 * self.linear(input))

class SIGNET(nn.Module):
    def __init__(self, first_omega_0=30, hidden_omega_0=30., hidden_layers=8, in_feature_ratio=1, out_features=3,
               hidden_features=512, alpha=0.5, with_res=False, with_norm=False):
        super().__init__()
        
        in_features = int(in_feature_ratio * 512)
        self.with_res = with_res
        self.D = hidden_layers + 2

        for i in range(hidden_layers+1):
            if i == 0:
                layer = SineLayer(in_features, hidden_features, is_first=True, is_res=False, omega_0=first_omega_0)
            else:
                layer = SineLayer(hidden_features, hidden_features, is_first=False, is_res=self.with_res, omega_0=hidden_omega_0)
            if with_norm:
                layer = nn.Sequential(layer, nn.LayerNorm(hidden_features, elementwise_affine=False))
            setattr(self, f"encoding_{i+1}", layer)

        final_linear = nn.Linear(hidden_features, out_features)
        
        with torch.no_grad():
            final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,  np.sqrt(6 / hidden_features) / hidden_omega_0)
        setattr(self, f"encoding_{hidden_layers+2}", final_linear)
        
        self.N_xy = int(in_feature_ratio * (240))
        self.N_uv = int(in_feature_ratio * (16))

        self.xy_embedd = Embedding(self.N_xy, 1024, alpha)
        self.uv_embedd = Embedding(self.N_uv, 17, alpha)

    def forward(self, x):
        emb_x = torch.cat( [self.uv_embedd(x[:, 0]), self.uv_embedd(x[:, 1]), self.xy_embedd(x[:, 2]), self.xy_embedd(x[:, 3]) ], axis=1).to(x.device)
        out = emb_x
        for i in range(self.D):
            out = getattr(self, f"encoding_{i+1}")(out)
        return out

def get_LF_val(u, v, width=1024, height=1024):
    x = np.linspace(0, width-1, width)
    y = np.linspace(0, height-1, height)
    
    xv, yv = np.meshgrid(y, x)
    img_grid = torch.from_numpy(np.stack([yv, xv], axis=-1))
    
    uv_grid = torch.ones_like(img_grid)
    uv_grid[:, :, 0], uv_grid[:, :, 1] = u, v
    
    val_inp_t = torch.cat([uv_grid, img_grid], dim = -1).float()

    del img_grid, xv, yv
    return val_inp_t.view(-1, val_inp_t.shape[-1])

def eval_im(val_inp_t, batches, device):
    b_size = val_inp_t.shape[0] // batches
    with torch.no_grad():
        out = []
        for b in range(batches):
            out.append(model(val_inp_t[b_size*b:b_size*(b+1)].to(device)))
        out = torch.cat(out, dim = 0)
        out = torch.clamp(out, 0, 1)
        out_np = out.view(1024, 1024, 3).cpu().numpy() * 255
    return out_np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-u", type=int, default=0, help="angular dimension u")
    parser.add_argument("-v", type=int, default=0, help="angular dimension v")
    parser.add_argument("-b", type=int, default=4, help="batch size in inference")
    parser.add_argument("--scene", type=str, default="lego", help="lego or tarot")
    
    OUT_DIR = f'./decoded_images/{args.scene}'
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)
    
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SIGNET(hidden_layers=8, alpha=0.5, hidden_features=512, in_feature_ratio=1, with_norm=True, with_res=True)
    m_state_dict = torch.load(f'./encoded_weights/model_{args.scene}.pth')
    model.load_state_dict(m_state_dict, strict=False)
    model.eval()
    model = model.to(device)
    val_inp_t = get_LF_val(u=args.u, v=args.v).to(device)
    out_np = eval_im(val_inp_t, args.b, device)
    Image.fromarray(np.uint8(out_np)).save(f'{OUT_DIR}/%s_u%d_v%d.png' % (args.scene, args.u, args.v))
