import numpy as np
import os
import sys
from sklearn.decomposition import PCA

sys.path.append('.')
from tabsyn.latent_utils import load_info_full
from globals import get_embedding_module_param

threshold = 0.99
device = 'cuda:0'

for dataname in ['beijing', 'default', 'shoppers', 'phishing']:
  data_dir = f'data/{dataname}'
  info_path = f'{data_dir}/info.json'
  latents_dir = f'tabsyn/vae/ckpt/{dataname}/original-num_layers2-final'
  latents_path = f'{latents_dir}/train_z.npy'
  vae_path = f'{latents_dir}/model.pt'
  info = load_info_full(data_dir, info_path, 'none', latents_path, vae_path, device, **get_embedding_module_param())
  latents = info['latents'].cpu().detach().numpy()

  pca = PCA(n_components=None, random_state=665).fit(latents)
  lbds = pca.explained_variance_ratio_
  X = list(range(1, len(lbds) + 1))
  Y = lbds.cumsum()
  num_needed_projections = 1
  while num_needed_projections < len(Y):
    if Y[num_needed_projections - 1] > threshold:
      print(num_needed_projections, Y[num_needed_projections - 1])
      break
    num_needed_projections += 1
  assert sum(lbds[:num_needed_projections - 1]) < threshold and sum(lbds[:num_needed_projections]) > threshold, f'{sum(lbds[:num_needed_projections - 1])}-{sum(lbds[:num_needed_projections])}'

  W = pca.components_[:num_needed_projections, :]
  save_path = f'{latents_dir}/correct-pca-{threshold}.txt'
  if not os.path.exists(save_path):
    np.savetxt(save_path, W)
