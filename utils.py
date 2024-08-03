import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import random
from tqdm import tqdm
from data_utils import SmoteishDataset, InfiniteDataLoader
from torch.utils.data import Dataset, DataLoader

def calculate_weight(df):
    class_counts = df.label.sum(axis=0)
    total_samples = len(df)
    class_weights = total_samples / (class_counts * len(class_counts))
    return class_weights

def reset_weights(m):
    for name, layer in m.named_children():
        for n, l in layer.named_modules():
            if hasattr(l, 'reset_parameters'):
                l.reset_parameters()
                
def decode_latents(vae, latents):
        latents = 1 / vae.config.scaling_factor * latents
        image = vae.decode(latents, return_dict=False)[0]
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image

def Fuse(X1, X2, X1_map, X2_map, encoder, decoder, q1=0.5, q2=0.5):
    # Encode the inputs
    X1_encoded = encoder(X1).detach()
    X2_encoded = encoder(X2).detach()

    # Convert attention maps to GPU tensors if they aren't already
    X1_map = torch.tensor(X1_map, dtype=torch.float32).cuda()
    X2_map = torch.tensor(X2_map, dtype=torch.float32).cuda()

    # Create masks based on the maps
    N1 = X1_map.unsqueeze(1).expand_as(X1_encoded)
    N2 = X2_map.unsqueeze(1).expand_as(X2_encoded)

    top_mask = N1 > q1
    bot_mask = N2 < q2
    mask_nxor = ~(top_mask ^ bot_mask)
    mask_and = top_mask & bot_mask

    # View masking operations
    X_view1 = X1_encoded * (~mask_and & top_mask).float()
    X_view2 = X2_encoded * (~mask_and & bot_mask).float()

    # Random mask for the XOR region
    prob = torch.rand(1).item()
    mask = torch.rand(X_view1.shape[0], X_view1.shape[2], X_view1.shape[3]).cuda() > prob
    mask = mask.unsqueeze(1).expand_as(X_view1)

    # Combine features
    frank = (mask_nxor.float() * (mask * X1_encoded + (~mask) * X2_encoded)) + X_view1 + X_view2

    # Decode the combined features
    frank_decoded = decoder(frank).cpu()

    return frank_decoded
# https://github.com/niteshsukhwani/MLSMOTE/blob/master/mlsmote.py
def SmoteishFuse(
    X, tail_indices, y, 
    maps, indices, n_sample, 
    encoder, decoder, classes, 
    pipe=None, b_size=10, d_steps=0, 
    verbose=False, q1=None, q2=None,
    num_workers=1):
    if q1 is None:
        q1 = np.random.normal(loc=0.5, scale=0.1)
    if q2 is None:
        q2 = np.random.normal(loc=0.5, scale=0.1)
    
    dataset = SmoteishDataset(X, tail_indices, y, maps, indices)
    dataloader = InfiniteDataLoader(dataset, batch_size=b_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    
    new_X = []
    new_impr = []
    target = []
    disable = not verbose
    total_samples = 0

    for X1, X2, X1_map, X2_map, batch_target, references in tqdm(dataloader, total=1 + n_sample // b_size, disable=disable):
        X1 = X1.cuda(non_blocking=True).requires_grad_(False)
        X2 = X2.cuda(non_blocking=True).requires_grad_(False)
        X1_map = X1_map.numpy()
        X2_map = X2_map.numpy()

        inter_X = Fuse(X1, X2, X1_map, X2_map, encoder, decoder, q1, q2)

        for i in range(len(batch_target)):
            prompt = ' '.join([clas for e, clas in enumerate(classes) if batch_target[i][e] == 1])
            new_impr.append(prompt)

        if d_steps > 0:
            inter_X = diffuse(new_impr[total_samples:total_samples+len(inter_X)], pipe=pipe, inf_steps=d_steps, latents=inter_X.cuda().bfloat16()).cpu().float().numpy()

        new_X.extend(inter_X)  
        target.extend(batch_target)

        total_samples += len(batch_target)
        if total_samples >= n_sample:
            break

    new_X = pd.Series(new_X[:n_sample])
    target = pd.Series(target[:n_sample])
    new_impr = pd.Series(new_impr[:n_sample])
    return new_X, target, new_impr

def diffuse(prompt, pipe, steps=75, inf_steps=75, latents=None):
    with torch.no_grad():
        text_encoder, unet, scheduler, tokenizer = pipe.text_encoder, pipe.unet, pipe.scheduler, pipe.tokenizer
        torch_device = "cuda" if torch.cuda.is_available() else "cpu"
        height = 512  # default height of Stable Diffusion
        width = 512  # default width of Stable Diffusion
        num_inference_steps = steps  # Number of denoising steps
        guidance_scale = 3.5  # Scale for classifier-free guidance
        batch_size = len(prompt)
        
        # Tokenize the prompt
        text_input = tokenizer(
            prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt"
        )
        text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]
        
        # Unconditional embeddings
        max_length = text_input.input_ids.shape[-1]
        uncond_input = tokenizer([""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt")
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]
        
        # Concatenate to form the final text embeddings
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings], dim=0)
        assert text_embeddings.shape[0] == 2 * batch_size, "Mismatch in text_embeddings batch size."
        
        scheduler.set_timesteps(num_inference_steps)

        # Ensure latents have the correct shape
        if latents is None:
            latents = torch.randn((batch_size, 4, height, width), device=torch_device)
        elif latents.shape[0] != batch_size:
            raise ValueError(f"Latents batch size {latents.shape[0]} does not match prompt batch size {batch_size}.")
        
        for t in scheduler.timesteps[-inf_steps:]:
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)

            # predict the noise residual
            noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = scheduler.step(noise_pred, t, latents).prev_sample

        latents = latents.cpu()
        return latents