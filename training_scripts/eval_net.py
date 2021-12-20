import os
import argparse

import numpy as np
from PIL import Image

import torch
import torch.nn as nn

from network import SIGNET_static

def get_LF_val(u, v, width=1024, height=1024):
    x = np.linspace(0, width-1, width)
    y = np.linspace(0, height-1, height)
    
    xv, yv = np.meshgrid(y, x)
    img_grid = torch.from_numpy(np.stack([yv, xv], axis=-1))
    
    uv_grid = torch.ones_like(img_grid)
    uv_grid[:, :, 0], uv_grid[:, :, 1] = u, v
    
    val_inp_t = torch.cat([uv_grid, img_grid], dim = -1).float()

    val_inp_t[..., :2] /= 17
    val_inp_t[..., 2] /= width
    val_inp_t[..., 3] /= height

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
    parser.add_argument("--exp_dir", type=str, help="directory to trained weights")
    args = parser.parse_args()

    OUT_DIR = f'./{args.exp_dir}/eval_output'
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SIGNET_static(hidden_layers=8, alpha=0.5, skips=[], hidden_features=512, with_norm=True, with_res=True)
    m_state_dict = torch.load(f'{args.exp_dir}/model.pth')
    model.load_state_dict(m_state_dict, strict=False)
    model.eval()
    model = model.to(device)
    val_inp_t = get_LF_val(u=args.u, v=args.v).to(device)
    out_np = eval_im(val_inp_t, args.b, device)
    Image.fromarray(np.uint8(out_np)).save(f'{OUT_DIR}/u{args.u}_v{args.v}.png')
