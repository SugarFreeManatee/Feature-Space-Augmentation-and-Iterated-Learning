import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import random
from tqdm import tqdm

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
    X1 = encoder(X1).detach().cpu().numpy()
    X2 = encoder(X2).detach().cpu().numpy()
    N1 = np.squeeze(X1_map)
    N1 = np.einsum('nwh,ncwh->ncwh', N1, np.ones(X1.shape, dtype=int))
    N2 = np.squeeze(X2_map)
    N2 = np.einsum('nwh,ncwh->ncwh', N2, np.ones(X1.shape, dtype=int))
    top_mask = N1 > q1
    bot_mask = N2 < q2
    mask_nxor = ~(top_mask ^ bot_mask)
    mask_and = (top_mask & bot_mask)
    X_view1 = X1 * np.logical_and(top_mask, ~mask_and)
    X_view2 = X2 * np.logical_and(bot_mask, ~mask_and)
    prob = random.random()
    mask = np.random.rand(*X_view1.shape[2:]) > prob
    frank = (mask_nxor * mask * X1) + (mask_nxor * (~mask) * X2) + X_view1 + X_view2
    frank = decoder(torch.tensor(frank).cuda().float()).cpu()
    return frank

def SmoteishFuse(X, tail_indices, y, maps, indices, n_sample, encoder, decoder, classes, pipe = None, b_size=10,
        d_steps=0, verbose=True, q1=None, q2=None):
    n = len(indices)
    if q1 == None:
        q1 = np.random.normal(loc=0.5, scale=0.1)
    if q2 == None:
        q2 = np.random.normal(loc=0.5, scale=0.1)
    new_X = np.zeros((n_sample, *X[0].shape))
    new_impr = []
    target = np.zeros((n_sample, y.shape[1]))
    choice_to_idx = {i: idx for i, idx in enumerate(tail_indices)}
    choices = np.random.randint(0, len(tail_indices), size=n_sample)
    X1 = np.zeros((b_size, *X[0].shape))
    X2 = np.zeros((b_size, *X[0].shape))
    X1_map = np.zeros((b_size, *maps[0].shape))
    X2_map = np.zeros((b_size, *maps[0].shape))
    curr = 0
    disable = not verbose
    for i in tqdm(range(0, n_sample), miniters=10, maxinterval=float("inf"), disable=disable):
        reference = choices[i]
        neighbour = random.choice(indices[reference])
        X1[curr, :, :, :] = X[choice_to_idx[reference]]
        X2[curr, :, :, :] = X[neighbour]
        X1_map[curr, :, :] = maps[reference]
        X2_map[curr, :, :] = maps[neighbour]
        target[i] = np.maximum(y[neighbour], y[choice_to_idx[reference]])
        prompt = ' '.join([clas for e, clas in enumerate(classes) if target[i][e] == 1])
        new_impr.append(prompt)
        curr += 1
        if curr == b_size or (n_sample - i) < b_size:
            inter_X = Fuse(torch.tensor(X1,dtype = torch.float).cuda().requires_grad_(False), 
                                           torch.tensor(X2,dtype = torch.float).cuda().requires_grad_(False), 
                                           X1_map, 
                                           X2_map, encoder, decoder, q1, q2)[:curr,:,:,:]
            if d_steps == 0:
                new_X[i-curr+1:i+1,:,:,:] = inter_X
            else:
                new_X[i-curr+1:i+1,:,:,:] = diffuse(new_impr[-curr:], 
                                                pipe = pipe,
                                                inf_steps = d_steps,
                                                latents = inter_X.cuda().bfloat16()).cpu().float().numpy()
            curr = 0
    new_X = pd.Series(list(new_X))
    target = pd.Series(list(target))
    new_impr = pd.Series(new_impr)
    return new_X, target, new_impr

def diffuse(prompt, pipe, steps = 75, inf_steps = 75, latents = None):
     with torch.no_grad():
        text_encoder, unet, scheduler, tokenizer = pipe.text_encoder, pipe.unet, pipe.scheduler, pipe.tokenizer
        torch_device = "cuda" if torch.cuda.is_available() else "cpu"
        height = 512  # default height of Stable Diffusion
        width = 512  # default width of Stable Diffusion
        num_inference_steps = steps  # Number of denoising steps
        guidance_scale = 3.5  # Scale for classifier-free guidance
        batch_size = len(prompt)
        text_input = tokenizer(
        prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt"
        )


        text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]
        max_length = text_input.input_ids.shape[-1]
        uncond_input = tokenizer([""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt")
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        scheduler.set_timesteps(num_inference_steps)

        for t in scheduler.timesteps[-inf_steps:]:
            latent_model_input = torch.cat([latents] * 2)

            latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)

            # predict the noise residual
            with torch.no_grad():
                noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = scheduler.step(noise_pred, t, latents).prev_sample
        latents = latents.cpu()
        return latents