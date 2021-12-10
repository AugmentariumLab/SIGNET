import os
import random
import numpy as np
from PIL import Image
import scipy

import pickle
from tqdm import tqdm
from models import Embedding
import torch

import argparse

def save_dict(di_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)

def load_dict(filename_):
    with open(filename_, 'rb') as f:
        ret_di = pickle.load(f)
    return ret_di

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", type=str, help="Dataset directory")
    parser.add_argument("--save_dir", type=str, help="Directory to store the preprocessed training patches")
    args = parser.parse_args()

    img_dir = args.img_dir
    save_dir = args.save_dir

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
        os.makedirs(save_dir+'/train')

    if img_dir == './data/bracelet/':
        lf_arr = np.zeros((17, 17, 640, 1024, 3))
        width, height, u_dim, v_dim = 1024, 640, 17, 17
        val_img_path = './data/bracelet/out_01_01_-1512.372681_883.201477_.png'
    else:
        lf_arr = np.zeros((17, 17, 1024, 1024, 3))
        width, height, u_dim, v_dim = 1024, 1024, 17, 17
        if img_dir.endswith('tarot/'):
            val_img_path = './data/tarot/out_01_01_-863.245667_1019.015076.png'
        else:
            val_img_path = './data/lego/out_01_01_-390.668457_1095.314209.png'

    batch_size = width * 2
    i, j = 0, 0
    for file in sorted(os.listdir(img_dir)):
        if j == u_dim:
            i += 1
            j = 0
        if file.endswith(".png"):
            img_ = np.asarray(Image.open(img_dir + file).convert('RGB')) / 255.
            lf_arr[i, j] = img_
            j += 1
    print('Light field loaded')

    img_data = lf_arr.reshape(-1, 3)
    del lf_arr

    x = np.linspace(0, width-1, width)
    y = np.linspace(0, height-1, height)
    u = np.linspace(0, u_dim-1, u_dim)
    v = np.linspace(0, v_dim-1, v_dim)
    i = [u, v, x, y]
    uv, vv, xv, yv = np.meshgrid(*i)
    img_grid = np.stack([uv, vv, xv, yv], axis=-1)

    val_x = img_grid[1, 1]
    val_img_grid = val_x.reshape(-1, 4)

    img_ = np.asarray(Image.open(val_img_path).convert('RGB')) / 255.
    lf_arr = np.zeros((height, width, 3))
    lf_arr[:, :, :] = img_

    val_img_data = lf_arr.reshape(-1, 3).astype(np.float32)
    img_grid = img_grid.reshape(-1, 4)
    print('Linspace grid created')
    
    del uv, vv, xv, yv, x, y, u, v
    p_num = width * height * u_dim * v_dim
    batch_num = p_num // batch_size
    idx_list = np.split(np.random.permutation(np.arange(0, p_num)), batch_num)
    
    print(f'Writing patches to {save_dir}')
    val_data = {'x':val_img_grid,'y':val_img_data}
    save_dict(val_data, f'{save_dir}/patch_val.pkl')
    for p in tqdm(range(batch_num)):
        slt_idx = idx_list[p]
        x_p = img_grid[slt_idx]
        y_p = img_data[slt_idx]
        save_data = {'x':x_p,'y':y_p}
        save_dict(save_data, os.path.join(save_dir, 'train', f'patch_{p:05d}.pkl'))
