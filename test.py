import numpy as np
import os
import argparse
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from skimage import img_as_ubyte
from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from skimage.metrics import structural_similarity as ssim_loss

from utils.loader import get_validation_data
import utils

parser = argparse.ArgumentParser(description='ShadowFormer Inference Without GT')
parser.add_argument('--input_dir', default='ISTD_Dataset/test',
    type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='/content/Result',
    type=str, help='Directory for results')
parser.add_argument('--weights', default='ISTD_model_latest.pth',
    type=str, help='Path to weights')
parser.add_argument('--gpus', default='0', type=str, help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--arch', default='ShadowFormer', type=str, help='arch')
parser.add_argument('--batch_size', default=1, type=int, help='Batch size for dataloader')
parser.add_argument('--save_images', action='store_true', help='Save output images')
parser.add_argument('--embed_dim', type=int, default=32)
parser.add_argument('--win_size', type=int, default=10)
parser.add_argument('--token_projection', type=str, default='linear')
parser.add_argument('--token_mlp', type=str,default='leff')
parser.add_argument('--vit_dim', type=int, default=256)
parser.add_argument('--vit_depth', type=int, default=12)
parser.add_argument('--vit_nheads', type=int, default=8)
parser.add_argument('--vit_mlp_dim', type=int, default=512)
parser.add_argument('--vit_patch_size', type=int, default=16)
parser.add_argument('--global_skip', action='store_true', default=False)
parser.add_argument('--local_skip', action='store_true', default=False)
parser.add_argument('--vit_share', action='store_true', default=False)
parser.add_argument('--train_ps', type=int, default=320)
parser.add_argument('--tile', type=int, default=None)
parser.add_argument('--tile_overlap', type=int, default=32)
parser.add_argument('--shadow_dir', default="/content/shadow/Shadow_A", type=str, help='Directory of input shadow images')
parser.add_argument('--mask_dir', default="/content/shadow/Shadow_B", type=str, help='Directory of binary masks')

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

device = torch.device("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available()
                      else "cpu")

utils.mkdir(args.result_dir)

test_dataset = get_validation_data(args.shadow_dir, args.mask_dir)

test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)

model_restoration = utils.get_arch(args)
model_restoration = torch.nn.DataParallel(model_restoration)
utils.load_checkpoint(model_restoration, args.weights)

model_restoration.to(device)
model_restoration.eval()

img_multiple_of = 8 * args.win_size

with torch.no_grad():
    for ii, data_test in enumerate(tqdm(test_loader), 0):
        rgb_noisy = data_test[0].to(device)
        mask = data_test[1].to(device)
        filenames = data_test[2]

        height, width = rgb_noisy.shape[2], rgb_noisy.shape[3]
        H, W = ((height + img_multiple_of) // img_multiple_of) * img_multiple_of, (
                    (width + img_multiple_of) // img_multiple_of) * img_multiple_of
        padh = H - height if height % img_multiple_of != 0 else 0
        padw = W - width if width % img_multiple_of != 0 else 0
        rgb_noisy = F.pad(rgb_noisy, (0, padw, 0, padh), 'reflect')
        mask = F.pad(mask, (0, padw, 0, padh), 'reflect')

        if args.tile is None:
            rgb_restored = model_restoration(rgb_noisy, mask)
        else:
            b, c, h, w = rgb_noisy.shape
            tile = min(args.tile, h, w)
            assert tile % 8 == 0, "tile size should be multiple of 8"
            tile_overlap = args.tile_overlap

            stride = tile - tile_overlap
            h_idx_list = list(range(0, h - tile, stride)) + [h - tile]
            w_idx_list = list(range(0, w - tile, stride)) + [w - tile]
            E = torch.zeros(b, c, h, w).type_as(rgb_noisy)
            W = torch.zeros_like(E)

            for h_idx in h_idx_list:
                for w_idx in w_idx_list:
                    in_patch = rgb_noisy[..., h_idx:h_idx + tile, w_idx:w_idx + tile]
                    mask_patch = mask[..., h_idx:h_idx + tile, w_idx:w_idx + tile]
                    out_patch = model_restoration(in_patch, mask_patch)
                    out_patch_mask = torch.ones_like(out_patch)

                    E[..., h_idx:(h_idx + tile), w_idx:(w_idx + tile)].add_(out_patch)
                    W[..., h_idx:(h_idx + tile), w_idx:(w_idx + tile)].add_(out_patch_mask)
            rgb_restored = E.div_(W)

        rgb_restored = torch.clamp(rgb_restored, 0, 1).cpu().numpy().squeeze().transpose((1, 2, 0))
        rgb_restored = rgb_restored[:height, :width, :]

        if args.save_images:
            save_path = os.path.join(args.result_dir, filenames[0])
            utils.save_img(rgb_restored*255.0, save_path)
