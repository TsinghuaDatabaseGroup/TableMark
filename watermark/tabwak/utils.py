import torch
import pandas as pd
import numpy as np
from types import SimpleNamespace

from tabsyn.tabwak.model import MLPDiffusion, DDIMModel, DDIMScheduler


def noise2bit(noise: torch.Tensor, l: int):
  assert isinstance(noise, torch.Tensor) and isinstance(l, int)
  q = torch.distributions.Normal(0, 1).icdf(torch.Tensor([pivot for pivot in np.linspace(0, 1, l + 1, dtype=np.float32)]))
  mask = [((q[i] <= noise) & (noise < q[i + 1])) for i in range(0, l)]
  bit = noise.int()
  for i, m in enumerate(mask):
    bit[m] = i
  return bit


def bit2noise(bit: torch.Tensor, l: int, generator=None):
  assert isinstance(bit, torch.Tensor) and isinstance(l, int)
  assert 0 <= bit.min() and bit.max() < l
  uniform = torch.rand(size=bit.shape, device=bit.device, generator=generator)
  return torch.distributions.Normal(0, 1).icdf((uniform + bit) / l)


def tabwak_cal_match_rate(bit: torch.Tensor):
  assert isinstance(bit, torch.Tensor)
  len_all = len(bit[0])
  assert len_all % 2 == 0 and bit.dtype == torch.int32
  len_half = len_all // 2
  match_rate = (bit[:, : len_half] == bit[:, len_half:]).sum(1, keepdim=False) / len_half
  return match_rate.mean().item(), match_rate.std().item()


def noise2latent(noise: torch.Tensor, dataname='beijing'):
  assert isinstance(noise, torch.Tensor)
  device = noise.device
  steps = 1000
  args = SimpleNamespace()
  args.dataname = dataname
  train_z, _, _, ckpt_path, info, num_inverse, cat_inverse = get_input_generate(args)
  nums_dims = train_z.shape[1]
  mean = train_z.mean(0).to(device)

  denoise_fn = MLPDiffusion(nums_dims, 1024).to(device)
  model = DDIMModel(denoise_fn).to(device)
  model.load_state_dict(torch.load(f'{ckpt_path}/model.pt', weights_only=True))
  noise_scheduler = DDIMScheduler(num_train_timesteps=1000)
  latent = noise_scheduler.generate(model.noise_fn, noise, num_inference_steps=steps, eta=0.0)
  return latent


def noise2numcat(noise: torch.Tensor, dataname='beijing'):
  assert isinstance(noise, torch.Tensor)
  device = noise.device
  steps = 1000
  args = SimpleNamespace()
  args.dataname = dataname
  train_z, _, _, ckpt_path, info, num_inverse, cat_inverse = get_input_generate(args)
  nums_dims = train_z.shape[1]
  mean = train_z.mean(0).to(device)

  denoise_fn = MLPDiffusion(nums_dims, 1024).to(device)
  model = DDIMModel(denoise_fn).to(device)
  model.load_state_dict(torch.load(f'{ckpt_path}/model.pt', weights_only=True))
  noise_scheduler = DDIMScheduler(num_train_timesteps=1000)
  latent = noise_scheduler.generate(model.noise_fn, noise, num_inference_steps=steps, eta=0.0)
  latent4decoder = (2 * latent + mean).reshape(latent.shape[0], -1, D_TOKEN)
  decoder = info["pre_decoder"].cuda()
  syn_num, syn_cat_raw = decoder(latent4decoder)
  syn_cat_lst = []
  for pred in syn_cat_raw:
    syn_cat_lst.append(pred.argmax(dim=-1))
  syn_cat = torch.stack(syn_cat_lst).t()
  return syn_num, syn_cat


def noise2latent(noise: torch.Tensor, dataname='beijing') -> torch.Tensor:
  assert isinstance(noise, torch.Tensor)
  device = noise.device
  steps = 1000
  args = SimpleNamespace()
  args.dataname = dataname
  train_z, _, _, ckpt_path, info, num_inverse, cat_inverse = get_input_generate(args)
  nums_dims = train_z.shape[1]

  denoise_fn = MLPDiffusion(nums_dims, 1024).to(device)
  model = DDIMModel(denoise_fn).to(device)
  model.load_state_dict(torch.load(f'{ckpt_path}/model.pt', weights_only=True))
  noise_scheduler = DDIMScheduler(num_train_timesteps=1000)
  latent = noise_scheduler.generate(model.noise_fn, noise, num_inference_steps=steps, eta=0.0)
  return latent


def latent2numcat(latent: torch.Tensor, dataname='beijing') -> torch.Tensor:
  assert isinstance(latent, torch.Tensor)
  device = latent.device
  steps = 1000
  args = SimpleNamespace()
  args.dataname = dataname
  train_z, _, _, ckpt_path, info, num_inverse, cat_inverse = get_input_generate(args)
  nums_dims = train_z.shape[1]
  mean = train_z.mean(0).to(device)

  latent4decoder = (2 * latent + mean).reshape(latent.shape[0], -1, D_TOKEN)
  decoder = info["pre_decoder"].cuda()
  syn_num, syn_cat_raw = decoder(latent4decoder)
  syn_cat_lst = []
  for pred in syn_cat_raw:
    syn_cat_lst.append(pred.argmax(dim=-1))
  syn_cat = torch.stack(syn_cat_lst).t()
  return syn_num, syn_cat


def noise2data(noise: torch.Tensor, dataname='beijing') -> pd.DataFrame:
  assert isinstance(noise, torch.Tensor)
  device = noise.device
  steps = 1000
  args = SimpleNamespace()
  args.dataname = dataname
  train_z, _, _, ckpt_path, info, num_inverse, cat_inverse = get_input_generate(args)
  nums_dims = train_z.shape[1]
  mean = train_z.mean(0).to(device)

  denoise_fn = MLPDiffusion(nums_dims, 1024).to(device)
  model = DDIMModel(denoise_fn).to(device)
  model.load_state_dict(torch.load(f'{ckpt_path}/model.pt', weights_only=True))
  noise_scheduler = DDIMScheduler(num_train_timesteps=1000)
  latent = noise_scheduler.generate(model.noise_fn, noise, num_inference_steps=steps, eta=0.0)
  latent4decoder = (2 * latent + mean).reshape(latent.shape[0], -1, D_TOKEN)
  syn_num, syn_cat, syn_target = split_num_cat_target(latent4decoder, info, num_inverse, cat_inverse, device)
  syn_data = recover_data(syn_num, syn_cat, syn_target, info)
  idx_name_mapping = info['idx_name_mapping']
  idx_name_mapping = {int(key): value for key, value in idx_name_mapping.items()}
  syn_data.rename(columns = idx_name_mapping, inplace=True)
  return syn_data.dropna()


def latent4decoder2data(latent4decoder: torch.Tensor, dataname='beijing') -> pd.DataFrame:
  assert isinstance(latent4decoder, torch.Tensor)
  device = latent4decoder.device
  steps = 1000
  args = SimpleNamespace()
  args.dataname = dataname
  train_z, _, _, ckpt_path, info, num_inverse, cat_inverse = get_input_generate(args)

  latent4decoder = latent4decoder.reshape(latent4decoder.shape[0], -1, D_TOKEN)
  syn_num, syn_cat, syn_target = split_num_cat_target(latent4decoder, info, num_inverse, cat_inverse, device)
  syn_data = recover_data(syn_num, syn_cat, syn_target, info)
  idx_name_mapping = info['idx_name_mapping']
  idx_name_mapping = {int(key): value for key, value in idx_name_mapping.items()}
  syn_data.rename(columns = idx_name_mapping, inplace=True)
  return syn_data.dropna()



def numcat2latent(syn_num: torch.Tensor, syn_cat: torch.Tensor, dataname='beijing') -> torch.Tensor:
  assert isinstance(syn_num, torch.Tensor)
  assert isinstance(syn_cat, torch.Tensor)
  device = syn_num.device
  steps = 1000
  args = SimpleNamespace()
  args.dataname = dataname
  train_z, _, _, ckpt_path, info, _, _ = get_input_generate(args)
  num_dims = train_z.shape[1]
  mean = train_z.mean(0).to(device)

  denoise_fn = MLPDiffusion(num_dims, 1024).to(device)
  model = DDIMModel(denoise_fn).to(device)
  model.load_state_dict(torch.load(f'{ckpt_path}/model.pt', weights_only=True))
  biased_reversed_latent = get_decoder_latent(syn_num, syn_cat, info, device, mask_col=None)
  reversed_latent = (biased_reversed_latent - mean) / 2
  return reversed_latent


def numcat2noise(syn_num: torch.Tensor, syn_cat: torch.Tensor, dataname='beijing') -> torch.Tensor:
  assert isinstance(syn_num, torch.Tensor)
  assert isinstance(syn_cat, torch.Tensor)
  device = syn_num.device
  steps = 1000
  args = SimpleNamespace()
  args.dataname = dataname
  train_z, _, _, ckpt_path, info, _, _ = get_input_generate(args)
  num_dims = train_z.shape[1]
  mean = train_z.mean(0).to(device)

  denoise_fn = MLPDiffusion(num_dims, 1024).to(device)
  model = DDIMModel(denoise_fn).to(device)
  model.load_state_dict(torch.load(f'{ckpt_path}/model.pt', weights_only=True))
  biased_reversed_latent = get_decoder_latent(syn_num, syn_cat, info, device, mask_col=None)
  reversed_latent = (biased_reversed_latent - mean) / 2
  noise_scheduler = DDIMScheduler(num_train_timesteps=1000)
  reversed_noise = noise_scheduler.gen_reverse(model.noise_fn, reversed_latent, num_inference_steps=steps, eta=0.0)
  return reversed_noise


def latent2noise(latent: torch.Tensor, dataname='beijing') -> torch.Tensor:
  assert isinstance(latent, torch.Tensor)
  device = latent.device
  steps = 1000
  args = SimpleNamespace()
  args.dataname = dataname
  train_z, _, _, ckpt_path, _, _, _ = get_input_generate(args)
  nums_dims = train_z.shape[1]

  denoise_fn = MLPDiffusion(nums_dims, 1024).to(device)
  model = DDIMModel(denoise_fn).to(device)
  model.load_state_dict(torch.load(f'{ckpt_path}/model.pt', weights_only=True))
  noise_scheduler = DDIMScheduler(num_train_timesteps=1000)
  reversed_noise = noise_scheduler.gen_reverse(model.noise_fn, latent, num_inference_steps=steps, eta=0.0)
  return reversed_noise
