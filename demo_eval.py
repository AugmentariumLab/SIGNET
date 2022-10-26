import os, tqdm, itertools, argparse
import torch
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim

from demo_decode import SIGNET, get_LF_val, eval_im

def compute_psnr(img1, img2):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    return 10 * np.log10(255**2 / mse)

def compute_ssim(img1, img2):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    score = ssim(img1, img2, multichannel=True, data_range=255)
    return score

def read_uv_view(u, v, img_dir):
    for file in sorted(os.listdir(img_dir)):
        _u, _v = file.split('_')[1], file.split('_')[2]
        if f'{u:02d}' == _u and f'{v:02d}' == _v:
            img = np.asarray(Image.open(f'{img_dir}/{file}')).astype(np.uint8)
            return img

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", type=int, default=4, help="batch size in inference")
    parser.add_argument("--scene", type=str, default="lego", help="lego or tarot")
    parser.add_argument("--img_dir", type=str, default="./data/lego", help="path to folder with all ground truth images")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SIGNET(hidden_layers=8, alpha=0.5, hidden_features=512, in_feature_ratio=1, with_norm=True, with_res=True)
    m_state_dict = torch.load(f'./encoded_weights/model_{args.scene}.pth')
    model.load_state_dict(m_state_dict)
    model.eval()
    model = model.to(device)
    val_inp_t = get_LF_val(u=0, v=0).to(device)

    uv_range = 17
    tbar = tqdm.tqdm(list(itertools.product(range(uv_range), range(uv_range))))
    p, s = 0, 0
    ct = 0
    for (u, v) in tbar:
        val_inp_t[..., 0] = v
        val_inp_t[..., 1] = u
        out_np = eval_im(model, val_inp_t, batches=args.b, device=device)
        img_gt = read_uv_view(u, v, args.img_dir)

        p_ = compute_psnr(img_gt, out_np)
        s_ = compute_ssim(img_gt, out_np)
        ct += 1
        p += p_
        s += s_
        tbar.set_postfix(PSNR = p_, AvePSNR = p / ct, SSIM = s_, AveSSIM = s / ct)

    print(f'PSNR: {p / ct} | SSIM: {s / ct}')
