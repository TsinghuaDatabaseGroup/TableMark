import os
import torch
import time
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from globals import *
from tabsyn.latent_utils import load_info_full
from tabsyn.tabwak.model import MLPDiffusion, DDIMScheduler, DDIMModel


def main():
    assert not os.path.exists(CFG_TABWAK.DM_PATH), 'Already Exists !'
    os.makedirs(CFG_TABWAK.DM_DIR, exist_ok=True)
    info = load_info_full(CFG_BASIC.DATA_DIR, CFG_BASIC.INFO_PATH, CFG_BASIC.TRANSFORM_LATENTS, CFG_VAE.EMBEDDING_PATH, CFG_VAE.PATH, CFG_BASIC.DEVICE, **get_embedding_module_param())
    train_data = (info['latents'] - info['mean']) / 2

    batch_size = 4096
    train_loader = DataLoader(train_data.cpu(), batch_size = batch_size, shuffle = True, num_workers = 4)

    num_epochs = 10000
    denoise_fn = MLPDiffusion(train_data.shape[1], 1024).to(CFG_BASIC.DEVICE)
    # DDIM training
    model = DDIMModel(denoise_fn).to(CFG_BASIC.DEVICE)
    noise_scheduler = DDIMScheduler(num_train_timesteps=1000)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=20, verbose=True)    
    model.train()

    best_loss = float('inf')
    patience = 0
    start_time = time.time()
    for epoch in range(num_epochs):
        
        pbar = tqdm(train_loader, total=len(train_loader))
        pbar.set_description(f"Epoch {epoch+1}/{num_epochs}")

        batch_loss = 0.0
        len_input = 0
        for batch in pbar:
            inputs = batch.float().to(CFG_BASIC.DEVICE)
            # DDIM training:
            noise = torch.randn(inputs.shape).to(CFG_BASIC.DEVICE)
            timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (inputs.shape[0],), device=CFG_BASIC.DEVICE).long()
            noisy_images = noise_scheduler.add_noise(inputs, noise, timesteps)
            loss = model(noise, noisy_images, timesteps)

            loss = loss.mean()
            batch_loss += loss.item() * len(inputs)
            len_input += len(inputs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix({"Loss": loss.item()})

        curr_loss = batch_loss/len_input
        scheduler.step(curr_loss)

        if curr_loss < best_loss:
            best_loss = loss.item()
            patience = 0
            torch.save(model.state_dict(), CFG_TABWAK.DM_PATH)
        else:
            patience += 1
            if patience == 500:
                print('Early stopping')
                break

    end_time = time.time()
    print('Time: ', end_time - start_time)
