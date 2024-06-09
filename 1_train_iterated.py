import os
import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from diffusers import AutoencoderKL
from utils import calculate_weight, reset_weights
from data_utils import FeatureDataset, ImageDataset
from models import VGGClassifier, DensNetClassifier, Encoder, Decoder
from time import time

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--pretrained_model_name_or_path", type=str, help="Path to pretrained model or model identifier from huggingface.co/models.")
    parser.add_argument("--classifier_weights", type=str, help="Path to pretrained classifier state dict.")
    parser.add_argument("--train_data_dir", type=str, help="A folder containing the training data. Folder contents must follow the structure described in the HuggingFace docs.")
    parser.add_argument("--labels_csv", type=str, help="Path to csv file containing the labels for each image.")
    parser.add_argument("--output_dir", type=str, default="iterated-model", help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--seed", type=int, help="A seed for reproducible training.")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--in_epochs", type=int, default=1, help="Number of training epochs for the interaction phase.")
    parser.add_argument("--im_epochs", type=int, default=1, help="Number of training epochs for the imitation phase.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Initial learning rate.")
    parser.add_argument("--learning_rate_cls", type=float, default=1e-4, help="Initial learning rate for the classifier.")
    parser.add_argument("--pkl_file", type=str, help="Path to pkl file containing the labels and features for each image.")
    parser.add_argument("--val_pkl_file", type=str, default='', help="Path to pkl file containing the validation labels and features for each image.")
    parser.add_argument("--dataloader_num_workers", type=int, default=8, help="Number of subprocesses to use for data loading.")
    parser.add_argument("--checkpointing_steps", type=int, default=500, help="Save a checkpoint of the training state every X updates.")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer.")
    parser.add_argument("--input_nc", type=int, default=4, help="Number of input channels.")
    parser.add_argument("--output_nc", type=int, default=64, help="Number of output channels.")
    parser.add_argument("--ae_lambda", type=float, default=0.75, help="AE reconstruction loss proportion.")
    parser.add_argument("--model", type=str, default="vgg", help="Classifier model type.")
    parser.add_argument("--suffix", type=str, default="weighted", help="Suffix for model type.")
    parser.add_argument("--load", type=bool, default=False, help="Load model.")
    parser.add_argument("--data_type", type=str, default="latent_vec", help="Type of dataset (img or latent_vec)")

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    writer = SummaryWriter(os.path.join(args.output_dir, str(datetime.now()) + args.suffix))
    validation = args.val_pkl_file != ''
    if args.data_type == "latent_vec":
        data = FeatureDataset(args.pkl_file)
        if validation:
            val_data = FeatureDataset(args.val_pkl_file)
    else:
        data = ImageDataset(args.pkl_file)
        if validation:
            val_data = ImageDataset(args.val_pkl_file)
    if "weighted" in args.suffix:
        class_weights = calculate_weight(data.data)
        loss_fn_c = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(class_weights, device="cuda"))
    else:
        loss_fn_c = nn.BCEWithLogitsLoss()

    train_loader = DataLoader(dataset=data, num_workers=args.dataloader_num_workers, batch_size=args.batch_size, shuffle=True)
    
    if validation:
        val_loader = DataLoader(dataset=val_data, num_workers=args.dataloader_num_workers, batch_size=args.batch_size, shuffle=True)

    loss_fn_ae = nn.L1Loss()
    logits_loss = nn.BCELoss()

    decoder = Decoder(args.output_nc, args.input_nc, 6).to("cuda")
    encoder_prev = Encoder(args.input_nc, args.output_nc, 6).to("cuda")
    encoder_prev.requires_grad_(False)
    encoder = Encoder(args.input_nc, args.output_nc, 6).to("cuda")

    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae").to("cuda")
    vae.requires_grad_(False)

    if args.model == 'vgg':
        classifier = VGGClassifier(args.output_nc, num_classes=len(data.data.label.iloc[0]), weights=args.classifier_weights).cuda()
    elif args.model == 'dense':
        classifier = DensNetClassifier(args.output_nc, num_classes=len(data.data.label.iloc[0]), weights=args.classifier_weights).cuda()
    optimizer = torch.optim.AdamW
    optimizer_dc = optimizer(decoder.parameters(), lr=args.learning_rate, betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay, eps=args.adam_epsilon)
    optimizer_ec = optimizer(encoder.parameters(), lr=args.learning_rate, betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay, eps=args.adam_epsilon)
    optimizer_cls = optimizer(classifier.parameters(), lr=args.learning_rate_cls, betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay, eps=args.adam_epsilon)
    scheduler_dc = optim.lr_scheduler.StepLR(optimizer_dc, step_size=10, gamma=0.1)
    scheduler_cls = optim.lr_scheduler.StepLR(optimizer_cls, step_size=10, gamma=0.1)
    if args.load:
        params = torch.load(args.load_params)

        for model in [classifier, encoder, decoder]:
            model.load_state_dict(params[f"{model}_state_dict"])
        
        optimizer_cls.load_state_dict(params["optimizer_cls_state_dict"])
        optimizer_dc.load_state_dict(params["optimizer_dc_state_dict"])

    logging.info("Training started")
    max_step_im = 0
    max_step_in = 0
    for epoch in tqdm(range(args.epochs)):
        t = time()
        logging.info(f"Beginning interaction phase {epoch}...")
        
        train_loss_c = 0
        train_loss_ae = 0
        for epm in range(args.in_epochs):
            for step_t, (X, y) in enumerate(train_loader):
                if args.data_type == "img":
                    X = vae.encode(X.cuda()).latent_dist.sample()
                    X = X * vae.config.scaling_factor 
                y = y.to('cuda')
                X = X.to('cuda')
                X_s = encoder(X)
                X_c = decoder(X_s)
                prediction = classifier(X_s)
                loss_c = loss_fn_c(prediction.float(), y)
                train_loss_c += loss_c.item()

                loss_ae = loss_fn_ae(X_c, X)
                train_loss_ae += loss_ae.item()
                # Backpropagate
                loss = loss_ae + loss_c
                
                loss.backward()
                optimizer_cls.step()
                optimizer_dc.step()
                optimizer_ec.step()

                optimizer_cls.zero_grad()
                optimizer_dc.zero_grad()
                optimizer_ec.zero_grad() 
                curr_step = (epoch * args.in_epochs) * max_step_in + (epm * max_step_in)  + step_t
                writer.add_scalar('Loss_c/train', loss_c.item(), curr_step)
                writer.add_scalar('Loss_ae/train', loss_ae.item(), curr_step)
            scheduler_dc.step()
            scheduler_cls.step()
            if max_step_in == 0:
                max_step_in = step_t
        if validation:
            with torch.no_grad():
                v_loss_ae = 0
                v_loss_c  = 0
                v_map = 0

                for step_t, (X, y) in enumerate(val_loader):
                    X = X.to('cuda')
                    y = y.to('cuda')
                    X_new = encoder(X)
                    X_s = X_new
                    X_c = decoder(X_s)
                    # Convert images to latent space
                    # Sample noise that we'll add to the latents
                    prediction = classifier(X_s)
                    loss_c = loss_fn_c(prediction.float(), y)
                    v_loss_c += loss_c.item()
                    v_map += multilabel_auprc(input=prediction, target=y, num_labels = 19).item()
                    loss_ae = loss_fn_ae(X_c, X)
                    v_loss_ae += loss_ae.item()
                    
        
                writer.add_scalar('Loss_ae/val', v_loss_ae / (step_t + 1), epoch)
                writer.add_scalar('Loss_c/val', v_loss_c / (step_t + 1), epoch)
                writer.add_scalar('mAP/val', v_map / (step_t + 1), epoch)
        
        logging.info(f'Avg Reconstruction loss: {train_loss_ae / (args.in_epochs * (step_t + 1))}')
        logging.info(f'Avg Classification loss: {train_loss_c / (args.in_epochs * (step_t + 1))}')
        logging.info(f"Beginning imitation phase {epoch}...")
        if epoch %args.checkpointing_steps == 0:
            PATH = os.path.join(args.output_dir, f"classifier_{args.suffix}_{epoch}.pk")
            torch.save({
            'epoch': epoch,
            'classifier_state_dict': classifier.state_dict(),
            'optimizer_cls_state_dict': optimizer_cls.state_dict(),
            'optimizer_dc_state_dict': optimizer_dc.state_dict(),
            'decoder_state_dict': decoder.state_dict(),
            'encoder_state_dict': encoder.state_dict(),
            }, PATH)
        if epoch == (args.epochs - 1):
            break
        encoder_prev.load_state_dict(encoder.state_dict())
        encoder_prev.requires_grad_(False) 
        encoder_prev.eval()
        encoder.apply(reset_weights)
        optimizer_ec = optimizer(
                encoder.parameters(),
                lr=args.learning_rate,
                betas=(args.adam_beta1, args.adam_beta2),
                weight_decay=args.adam_weight_decay,
                eps=args.adam_epsilon,
            )
        train_loss_im = 0
        for ep in range(args.im_epochs):
            for step_t, (X, y) in enumerate(train_loader): 
                if args.data_type == "img":
                    X = vae.encode(X.cuda()).latent_dist.sample()
                    X = X * vae.config.scaling_factor 
                X = X.to('cuda')
                X_new = encoder(X)
                X_prev = encoder_prev(X.detach())
                b,c,h,w = X_prev.shape
                labels = torch.reshape(X_prev, (b*h*w, c))
                labels = torch.squeeze(torch.multinomial(labels, 1).long())
                labels = F.one_hot(labels, args.output_nc).float()
                labels = torch.reshape(labels, (b, c*h*w))
                logits = torch.reshape(X_new, (b, c*h*w))
                loss = logits_loss(logits, labels)
                loss.backward()    
                train_loss_im += loss.item()
                optimizer_ec.step()
                optimizer_ec.zero_grad()
                writer.add_scalar('Loss_im/train', loss.item(),  (epoch * args.im_epochs) * max_step_im + 
                   ep * max_step_im  + step_t)
            if max_step_im == 0:
                max_step_im = step_t

        logging.info(f'Avg Imitation loss: {train_loss_im / (args.im_epochs * (step_t + 1))}')
        eta = (time() - t) * (args.epochs - epoch - 1)
        logging.info(f"ETA: {str(timedelta(seconds=int(eta)))}")
    PATH = os.path.join(args.output_dir, f"classifier_{args.suffix}_{epoch}.pk")
    torch.save({
    'epoch': epoch,
    'classifier_state_dict': classifier.state_dict(),
    'optimizer_cls_state_dict': optimizer_cls.state_dict(),
    'optimizer_dc_state_dict': optimizer_dc.state_dict(),
    'decoder_state_dict': decoder.state_dict(),
    'encoder_state_dict': encoder.state_dict(),
    }, PATH)
    writer.close()
    logging.info("Training completed")


if __name__ == "__main__":
    main()
