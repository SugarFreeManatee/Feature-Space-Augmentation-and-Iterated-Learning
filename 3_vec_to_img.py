import os
import argparse
import logging
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from diffusers import AutoencoderKL
from data_utils import FeatureDataset
from utils import decode_latents
from PIL import Image
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True, help="Path to pretrained model or model identifier from huggingface.co/models.")
    parser.add_argument("--labels_csv", type=str,default="data/fused_vecs.pkl" ,help="Path to csv file containing the labels for each image.")
    parser.add_argument("--output_dir", type=str, default="outputs", help="The output directory where latent vector dataframe will be written.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--dataloader_num_workers", type=int, default=1, help="Number of subprocesses to use for data loading.")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    data = FeatureDataset(args.labels_csv)
    loader = DataLoader(dataset=data, num_workers=args.dataloader_num_workers, batch_size=args.batch_size, shuffle=False,  pin_memory=True)
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae").to("cuda")
    vae.requires_grad_(False)
    with torch.no_grad():
        for step_t, (X, y) in tqdm(enumerate(loader), total=len(loader)):
            imgs = decode_latents(vae, X.cuda())
            for i in range(len(imgs)):
                img = Image.fromarray((imgs[i] * 255).astype(np.uint8))
                img.save(os.path.join(args.output_dir, f"img_{step_t*args.batch_size+i}.png"))
                data.data.at[step_t*args.batch_size+i, 'path'] = f"img_{step_t*args.batch_size+i}.png"
    data.data.to_pickle(os.path.join(args.output_dir, 'latent_vecs.pkl'))
if __name__ == "__main__":
    main()
