import pandas as pd
import numpy as np
import torch
import os
import sys
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from globals import *


def compute_loss(X_num, X_cat, Recon_X_num, Recon_X_cat, mask_col=None):
  ce_loss_fn = torch.nn.CrossEntropyLoss()
  if mask_col is None:
    mse_loss = (X_num - Recon_X_num).pow(2).mean()
  else:
    # remove the masked columns to calculate the mse loss
    non_mask_col = [i for i in range(X_num.size(1)) if i not in mask_col]
    mse_loss = (X_num[:, non_mask_col] - Recon_X_num[:, non_mask_col]).pow(2).mean()
  ce_loss = 0
  acc = 0
  total_num = 0

  for idx, x_cat in enumerate(Recon_X_cat):
    if x_cat is not None:
      ce_loss += ce_loss_fn(x_cat, X_cat[:, idx])
      x_hat = x_cat.argmax(dim=-1)
    acc += (x_hat == X_cat[:, idx]).float().sum()
    total_num += x_hat.shape[0]
  ce_loss /= (idx + 1)
  loss = mse_loss + ce_loss
  return loss


def table2latents(num: np.ndarray, cat: np.ndarray, info: dict, device: str, mask_col=None):
  with torch.no_grad():
    encoder = info['encoder']
    num = torch.from_numpy(num).to(device)
    cat = torch.from_numpy(cat).to(device)
    latents = encoder(num, cat)
    if info['transform_latents'] == 'none':
      latents = latents[:, 1:, :]
    else:
      assert info['transform_latents'] in ['vae', 'ae'], info['transform_latents']
  latents.requires_grad = True
  optimizer = torch.optim.Adam([latents], lr=1e-3)
  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
  with torch.enable_grad():
    decoder = info['decoder']
    pbar = tqdm(range(100), total=100)
    for i in pbar:
      optimizer.zero_grad()
      recon = decoder(latents)
      loss = compute_loss(num, cat, recon[0], recon[1], mask_col=mask_col)
      loss.backward()
      optimizer.step()
      scheduler.step(loss)
      pbar.set_postfix({'Loss': loss.item(), 'num_diff': ((num - recon[0]).abs().sum().item() / num.numel()) if num.numel() > 0 else 0})
  return latents.reshape(latents.shape[0], -1).cpu().detach().numpy()


def predict_labels(table: pd.DataFrame, **kwargs) -> np.ndarray:
  assert kwargs['classifier'] == 'nn'
  latents = table2latents(kwargs['num'], kwargs['cat'], kwargs['info'], kwargs['device'])
  latents4cluster = get_latents4cluster_numpy(latents, kwargs['dim_ratio'], kwargs['key'])
  centers = np.loadtxt(kwargs['centers_path'], dtype=np.float32)
  return ((latents4cluster - centers[:, None, :]) ** 2).sum(-1, keepdims=False).argmin(0, keepdims=False)
