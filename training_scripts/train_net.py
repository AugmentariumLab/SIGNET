import os
import numpy as np
import tqdm
from PIL import Image
import argparse
import pickle

import torch
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from torch.utils.data import DataLoader

from network import SIGNET_static

class LFPatchDataset(torch.utils.data.Dataset):
    def __init__(self, is_train=True, file_dir = './patch_data/Lego/'):
        if is_train:
            self.file_dir = f'{file_dir}/train'
            self.file_list = []
            for f in sorted(os.listdir(self.file_dir)):
                self.file_list.append(f'{file_dir}/train/{f}')
            self.batch_num = len(self.file_list)
        else:
            self.batch_num = 1
            self.file_list = [f'{file_dir}/patch_val.pkl']*1
    def __len__(self):
        return self.batch_num

    def __getitem__(self, idx):
        filename_ = self.file_list[idx]
        with open(filename_, 'rb') as f:
            ret_di = pickle.load(f)

        lab_t = torch.from_numpy(ret_di['y']).float()
        inp_G_t = torch.from_numpy(ret_di['x']).float()
        
        return inp_G_t, lab_t

def compute_psnr(img1, img2):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    return 10 * np.log10(255**2 / mse)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, help="Root directory")
    parser.add_argument("--exp_name", type=str, default="test", help="Experiment name")
    parser.add_argument("--trainset_dir", type=str, default="lego")
    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--img_W", type=int, default=1024)
    parser.add_argument("--img_H", type=int, default=1024)

    args = parser.parse_args()

    device = ("cuda:0" if torch.cuda.is_available() else "cpu" )
    
    root_dir = args.root_dir
    exp_dir = f'{root_dir}/{args.exp_name}'
    print(f'Current experiment directory is: {exp_dir}')
    trainset_dir = f'{root_dir}/{args.trainset_dir}'

    num_epochs = args.num_epochs
    
    if not os.path.isdir(exp_dir):
        os.makedirs(exp_dir)
        os.makedirs(f'{exp_dir}/valout')

    val_im_shape = [1024, 1024]

    model = SIGNET_static(hidden_layers=8, alpha=0.5, skips=[], hidden_features=512, with_norm=True, with_res=True)
    model = model.to(device)

    trainset = LFPatchDataset(is_train=True, file_dir = trainset_dir)
    valset = LFPatchDataset(is_train=False, file_dir = trainset_dir)
    val_inp_t, _ = valset[0]

    bsize = 1
    train_loader = DataLoader(trainset, batch_size=bsize, drop_last=False, num_workers=8, pin_memory=True)
    iters = len(train_loader)

    # Frequency to save validation image
    val_freq = 200#iters * 2
    # Frequency to save the checkpoint
    save_freq = 5
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-7)
    
    print('Starts training')
    mse_losses, psnrs = [], []

    for epoch in range(num_epochs):
        e_psnr, e_loss, it = 0, 0, 0
        t = tqdm.tqdm(train_loader)

        for batch_idx, (inp_G_t, lab_t) in enumerate(t):
            optimizer.zero_grad()
            inp_G_t, lab_t = inp_G_t.view(-1, inp_G_t.shape[-1]).to(device), lab_t.view(-1, 3).to(device)
            
            # scale the input coordinates from integers to floats
            inp_G_t[..., :2] /= 17
            inp_G_t[..., 2] /= args.img_W
            inp_G_t[..., 3] /= args.img_H

            out = model(inp_G_t)
            mse_loss = torch.nn.functional.mse_loss(out, lab_t)
            loss = mse_loss
            loss.backward()
            optimizer.step()
            
            psnr = 10 * np.log10(1 / mse_loss.item())
            e_psnr += psnr
            e_loss += mse_loss.item()

            if it % val_freq == 0:
                val_inp_t = val_inp_t.view(-1, val_inp_t.shape[-1])
                b_size = val_inp_t.shape[0] // 16
                model.eval()
                with torch.no_grad():
                    out = []
                    for b in range(16):
                        out.append(model(val_inp_t[b_size*b:b_size*(b+1)].to(device)))
                    out = torch.cat(out, dim = 0)
                    out = torch.clamp(out, 0, 1)
                    out_np = out.view(val_im_shape[0], val_im_shape[1], 3).cpu().numpy() * 255
                    out_im = Image.fromarray(np.uint8(out_np))
                    out_name = f'valout/valout_e_{epoch}_it_{it}.png'
                    out_im.save(f'{exp_dir}/{out_name}')
                model.train()
            it += 1
            t.set_postfix(PSNR = psnr, EpochPSNR = e_psnr / it, EpochLoss = e_loss / it)
            
        scheduler.step()
        
        print('Epoch: %s Ave PSNR: %s Ave Loss: %s'%(epoch, e_psnr / it, e_loss / it))
        psnrs.append(e_psnr / it); mse_losses.append(e_loss / it)

        if epoch % save_freq == 0:
                torch.save(model.state_dict(), f'{exp_dir}/model.pth')

    torch.save(model.state_dict(), f'{exp_dir}/model.pth')

    np.savetxt(f'{exp_dir}/mse_stats.txt', mse_losses, delimiter=',')
    np.savetxt(f'{exp_dir}/psnr_stats.txt', psnrs, delimiter=',')
