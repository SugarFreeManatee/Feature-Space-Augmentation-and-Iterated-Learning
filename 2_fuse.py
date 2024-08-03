import os
import argparse
import sys
import numpy as np
import pandas as pd
import random
from tqdm.auto import tqdm
from sklearn.neighbors import NearestNeighbors
import torch
from torch.utils.data import DataLoader
from diffusers import StableDiffusionPipeline
from pytorch_grad_cam import GradCAM, HiResCAM, AblationCAM, ScoreCAM, LayerCAM, EigenCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pathlib import Path
from utils import SmoteishFuse
from data_utils import FeatureDataset
from models import VGGClassifier, DensNetClassifier, Encoder, Decoder
from torch.profiler import profile, record_function, ProfilerActivity
import torch.nn as nn
import cProfile
from line_profiler import LineProfiler
import pickle
import pynndescent

parser = argparse.ArgumentParser(description="Script for data augmentation")

# Add arguments
parser.add_argument("--pkl_file", type=str, help="Path to pkl file containing the labels and features for each image.")
parser.add_argument('--cache_dir', type=str, default='cache/', help='Cache path')
parser.add_argument("--output_dir", type=str, default="data/", help="The output directory where the fused vectors will be written.")
parser.add_argument("--d_steps", default=0, type=int, help="Diffusion steps for inference.")
parser.add_argument('--model_path', type=str, default='/home/raelberg/storage/roent-lora/baseline/checkpoints', help='Path to the model checkpoints')
parser.add_argument("--pretrained_model_name_or_path", type=str, help="Path to pretrained model or model identifier from huggingface.co/models.")
parser.add_argument("--labels_csv", type=str, help="Path to csv file containing the labels for each image.")
parser.add_argument('--tail_labels', type=str, nargs='+', default=["Pneumothorax", "Fracture","Enlarged Cardiomediastinum" ,"Tortuous Aorta", "Pneumomediastinum"], help='List of tail labels')
parser.add_argument('--output_nc', type=int, default=64, help='Output channels of the encoder')
parser.add_argument('--classifier_weights', type=str, default=None, help='Path to the classifier weights')
parser.add_argument('--base_n', type=int, default=1000, help='Number of samples to generate for each tail class')
parser.add_argument('--load_maps', action="store_true", help='Load cached maps')
parser.add_argument('--save_maps', action="store_true", help='Save generated maps')
parser.add_argument('--load_knn', action="store_true", help='Load cached knn')
parser.add_argument('--save_knn', action="store_true", help='Save generated knn')
parser.add_argument('--verbose', action="store_true", help='Verbosity')
parser.add_argument('--debug', action="store_true", help='Measure run time')
parser.add_argument('--model', type=str, default='vgg', help='Model type')
parser.add_argument('--impression', action="store_true", help='Use impression')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
parser.add_argument('--dataloader_num_workers', type=int, default=8, help='Number of workers for dataloader')
parser.add_argument("--npy_dir", type=str, default ='', help="Path to npy files for image features.")
args = parser.parse_args()

def main():
    # Load data
    print("Loading data")
    dataset = FeatureDataset(args.pkl_file, args.npy_dir)
    df = dataset.data

    print("Loading labels")
    label_pd = pd.read_csv(args.labels_csv)
    labels = label_pd.columns[3:]
    tail_labels = args.tail_labels
    head_labels = [col for col in labels if col not in tail_labels]

    print("Loading models")
    models_ = torch.load(args.model_path)

    if args.model == 'vgg':
        classifier = VGGClassifier(args.output_nc, num_classes=len(df.label.iloc[0]), weights=args.classifier_weights).cuda()
    elif args.model == 'dense':
        classifier = DensNetClassifier(args.output_nc, num_classes=len(df.label.iloc[0]), weights=args.classifier_weights).cuda()

    classifier.load_state_dict(models_["classifier_state_dict"])
    decoder = Decoder(64, 4, 6).cuda().requires_grad_(True)
    decoder.load_state_dict(models_["decoder_state_dict"])
    encoder = Encoder(4, 64, 6).cuda().requires_grad_(True)
    encoder.load_state_dict(models_["encoder_state_dict"])
    target_layers = classifier.model.features[2:-1]

    # Initialize Stable Diffusion Pipeline
    pipe = None
    if args.d_steps > 0:
        print("Loading pipe")
        pipe = StableDiffusionPipeline.from_pretrained(args.pretrained_model_name_or_path, torch_dtype=torch.bfloat16, safety_checker=None, low_cpu_mem_usage=False)
        pipe.to("cuda")
        pipe.unet.to(memory_format=torch.channels_last)
        pipe.vae.to(memory_format=torch.channels_last)

    tail_classes = tail_labels
    tail_mask = np.zeros(len(dataset.data.label.iloc[0]))
    head_mask = np.zeros(len(dataset.data.label.iloc[0]))
    curr = 0
    for col in dataset.data.columns:
        if col in labels:
            if col in tail_classes:
                tail_mask[curr] = 1
            else:
                head_mask[curr] = 1
            curr += 1

    maps = []
    tail = []
    lossfn = nn.BCEWithLogitsLoss()
    df_map = pd.DataFrame()
    if not args.load_maps: 
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.dataloader_num_workers)
        print("Generating CAMs:")
        with EigenCAM(model=classifier, target_layers=target_layers) as cam:
            tail_mask = torch.tensor(tail_mask).cuda()
            head_mask = torch.tensor(head_mask).cuda()
            for latents_batch, labels_batch in tqdm(dataloader, total=len(dataloader), miniters=50, maxinterval=float("inf")):
                latents_batch = latents_batch.cuda()
                latents_batch.requires_grad = True
                labels_batch = labels_batch.cuda()
                # Forward pass through encoder
                X1_batch = encoder(latents_batch)
                # Get classifier outputs for the batch
                classifier_output = classifier(X1_batch.detach()).detach()
                # Compute predictions using masks
                tail_conditions = torch.max(tail_mask * labels_batch, axis=1)[0] > 1e-6
                head_conditions = ~tail_conditions
                predicted_tail_indices = torch.argmax(tail_mask * labels_batch * classifier_output, axis=1)
                predicted_head_indices = torch.argmax(head_mask * labels_batch * classifier_output, axis=1)
                # Create targets based on conditions
                targets_batch = [ClassifierOutputTarget(predicted_tail_indices[i]) if tail_conditions[i] 
                                else ClassifierOutputTarget(predicted_head_indices[i]) for i in range(len(labels_batch))]
                # Generate CAMs for the entire batch
                cam_map = cam(input_tensor=X1_batch, targets=targets_batch)
                # Append results
                maps.extend(cam_map)
                tail.extend(tail_conditions.detach().cpu().numpy().astype(int).tolist())

        df_map["maps"] = maps
        df_map["tail"] = tail    
    else:
        print("Loading CAMs:")
        df_map = pd.read_pickle(os.path.join(args.cache_dir, "maps.pkl"))
    
    if args.save_maps:
        df_map.to_pickle(os.path.join(args.cache_dir, "maps.pkl"))

    
    df = pd.concat([df, df_map], axis=1)
    X_head = df[df["tail"] == 0]
    X_tail = df[df["tail"] == 1]
    print("KNN")
    # if args.load_knn:
    #     with open(os.path.join(args.cache_dir, "knn.pkl"), 'rb') as file:
    #         nbs = pickle.load(file)
    # else:
    nbs = pynndescent.NNDescent(np.array([*X_head.maps.values]).reshape(len(X_head), -1), n_neighbors=3, metric='euclidean', n_jobs=-1)
    nbs.prepare()

    # if args.save_knn:
    #     with open(os.path.join(args.cache_dir, "knn.pkl"), 'wb') as file:
    #         pickle.dump(nbs, file)
    
    X, y = df.latents.values, np.stack(df.label.values)
    base_prob = args.base_n

    with torch.no_grad():
        df_aug = pd.DataFrame()
        for label in tail_labels:
            if args.verbose:
                print(label)
            new_label = np.array([1 if col == label else -1 for col in labels])
            df_curr = df[df[label] == 1]
            indices, _ = nbs.query(np.array([*df_curr.maps.values]).reshape(len(df_curr), -1), 3)
            to_generate = base_prob - len(df_curr)
            to_generate = base_prob
            X_res, y_res, imp_res = SmoteishFuse(X=X, tail_indices=df_curr.reset_index()["index"],
                                                 y=y, maps=df["maps"].values, 
                                                 indices=indices, n_sample=to_generate,
                                                 encoder=encoder, decoder=decoder, pipe=pipe, classes=labels, b_size=args.batch_size, 
                                                 d_steps=args.d_steps, verbose=args.verbose, num_workers=args.dataloader_num_workers)
            X_res_ = pd.DataFrame(X_res, columns=['latents'])
            y_res_ = pd.DataFrame(pd.Series([new_label] * len(X_res_)), columns=['label'])
            if args.impression:
                imp_res_ = pd.DataFrame(imp_res, columns=['impression'])
            new_df = pd.concat([y_res_, X_res_], axis=1)
            new_df["class"] = label
            df_aug = pd.concat([df_aug, new_df], axis=0)

            if args.debug:
                break

    if not args.debug:
        df_aug.to_pickle(os.path.join(args.output_dir, "fused_vecs.pkl"))

if __name__ == "__main__":
    if args.debug:
        lp = LineProfiler()
        lp_wrapper = lp(main)
        lp_wrapper()
        lp.print_stats()
    else:
        main() 