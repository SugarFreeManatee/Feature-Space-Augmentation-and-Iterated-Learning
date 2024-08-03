import os
import argparse
import logging
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from diffusers import AutoencoderKL
from data_utils import ImageDataset
import numpy as np
import pandas as pd
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True, help="Path to pretrained model or model identifier from huggingface.co/models.")
    parser.add_argument("--data_dir", type=str, required=True, help="A folder containing the training data. Folder contents must follow the structure described in the HuggingFace docs.")
    parser.add_argument("--labels_csv", type=str, help="Path to csv file containing the labels for each image.")
    parser.add_argument("--output_dir", type=str, default="data", help="The output directory where latent vector dataframe will be written.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--dataloader_num_workers", type=int, default=1, help="Number of subprocesses to use for data loading.")
    parser.add_argument("--save_every", type=int, default=500, help="Number of batches after which to save the latents.")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    data = ImageDataset(args.labels_csv, args.data_dir)
    loader = DataLoader(dataset=data, num_workers=args.dataloader_num_workers, batch_size=args.batch_size, shuffle=False, pin_memory=True)
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae").to("cuda")
    vae.requires_grad_(False)

    # Track the total number of processed images
    saved = 0
    latents_list = []

    with torch.no_grad():
        for step_t, (X, y) in tqdm(enumerate(loader), total=len(loader), miniters=50, maxinterval=float("inf")):
            X = vae.encode(X.cuda()).latent_dist.sample()
            X = X * vae.config.scaling_factor 
            latents_list.append(X.cpu().numpy())

            # Save latents every 'save_every' batches
            if (step_t + 1) % args.save_every == 0:
                latents_array = np.concatenate(latents_list, axis=0)
                output_file = os.path.join(args.output_dir, f'latent_vecs_part_{saved}.npy')
                np.save(output_file, latents_array)
                latents_list = []  # Clear the list for the next chunk
                saved += 1

        # Save any remaining latents
        if latents_list:
            latents_array = np.concatenate(latents_list, axis=0)
            output_file = os.path.join(args.output_dir, f'latent_vecs_part_{saved}.npy')
            np.save(output_file, latents_array)
    
    data.data.to_pickle(os.path.join(args.output_dir, 'labels.pkl'))

if __name__ == "__main__":
    main()