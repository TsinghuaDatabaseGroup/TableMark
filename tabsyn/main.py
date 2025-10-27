import os
import torch
import numpy as np
import time
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from tabsyn.model import MLPDiffusion, Model
from tabsyn.latent_utils import load_info_full
from globals import *


def main():
    assert not os.path.exists(CFG_DM.PATH), 'Already Exists !'
    os.makedirs(CFG_DM.DIR, exist_ok=True)
    info = load_info_full(CFG_BASIC.DATA_DIR, CFG_BASIC.INFO_PATH, CFG_BASIC.TRANSFORM_LATENTS, CFG_EMBEDDING_MODEL.EMBEDDING_PATH, CFG_VAE.PATH, CFG_BASIC.DEVICE, **get_embedding_module_param())
    latents4dm = get_latents4dm_torch(info['latents'], CFG_CLUSTER.DIM_RATIO, CFG_CLUSTER.KEY)
    latents_normed = (latents4dm - latents4dm.mean(0)) / 2
    train_class_labels = torch.from_numpy(np.loadtxt(CFG_CLUSTER.ONEHOT_LABELS_PATH, dtype=np.float32))
    train_data = TensorDataset(latents_normed.clone().cpu().detach(), train_class_labels.clone().cpu().detach())
    train_loader = DataLoader(train_data, batch_size=CFG_DM.BATCH_SIZE, shuffle=True,num_workers=4)
    denoise_fn = MLPDiffusion(info['latent_dim'], 1024, CFG_CLUSTER.NUM_CLASSES, 0.1).to(CFG_BASIC.DEVICE)
    
    print(denoise_fn)
    num_params = sum(p.numel() for p in denoise_fn.parameters())
    print("the number of parameters", num_params)

    model = Model(denoise_fn=denoise_fn, hid_dim=info['latent_dim']).to(CFG_BASIC.DEVICE)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=20, verbose=True)
    best_loss = float('inf')
    patience = 0
    start_time = time.time()
    for epoch in range(CFG_DM.NUM_EPOCHS):
        pbar = tqdm(train_loader, total=len(train_loader))
        pbar.set_description(f"Epoch {epoch + 1}/{CFG_DM.NUM_EPOCHS}")
        batch_loss = 0.0
        len_input = 0
        for batch in pbar:
            x, class_labels = batch[0].to(torch.float32).to(CFG_BASIC.DEVICE), batch[1].to(torch.float32).to(CFG_BASIC.DEVICE)
            loss = model(x, class_labels)
            loss = loss.mean()
            batch_loss += loss.item() * len(x)
            len_input += len(x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix({"Loss": loss.item()})

        curr_loss = batch_loss / len_input
        scheduler.step(curr_loss)
        print(f'Current Loss: {curr_loss:.10f}, Best Loss: {best_loss:.10f}')

        if curr_loss < best_loss:
            best_loss = curr_loss
            patience = 0
            torch.save(model.state_dict(), CFG_DM.PATH)
        else:
            patience += 1
            if patience == 500:
                print('Early stopping')
                break

    end_time = time.time()
    print('Time: ', end_time - start_time)
