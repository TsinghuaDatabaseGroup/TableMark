import numpy as np
import torch
import json
import os

from globals import *
from eval.main import eval_all
from watermark.watermark import int2bit_str, check
from tabsyn.latent_utils import load_info_full, add_gauss_noise, add_uniform_noise, add_laplace_noise, alter_cat, alter_num
from tabsyn.tabwak.model import MLPDiffusion, DDIMModel, DDIMScheduler
from cluster.classify import table2latents
from tabsyn.latent_utils import recover_data, split_num_cat_target
from watermark.tabwak.utils import noise2bit, bit2noise
from watermark.regeneration_attack_vae.regeneration_attack_vae import attack
from process_dataset import process_data


def cal_match_rate(bit: torch.Tensor):
  len_all = len(bit[0])
  assert len_all % 2 == 0 and bit.dtype == torch.int32
  len_half = len_all // 2
  match_rate = (bit[:, : len_half] == bit[:, len_half:]).sum(1, keepdim=False) / len_half
  return match_rate.mean().item(), match_rate.std().item()


def embed(watermark_int: int, no_w_bit: torch.Tensor, in_dim: int, token_dim: int, num_watermark_bits: int) -> tuple[torch.Tensor, list[list[int]]]:
  n_column = in_dim // token_dim
  watermark_str = int2bit_str(watermark_int, num_watermark_bits)
  num_dims_per_watermark_bit = in_dim // num_watermark_bits // 2 * 2
  is_used = [False] * in_dim
  watermark_bit2dim_indices = [[] for _ in range(num_watermark_bits)]
  for i in range(num_watermark_bits):
    n_found = 0
    while n_found < num_dims_per_watermark_bit:
      for k in range(n_column):
        for l in range(token_dim):
          bit_idx = k * token_dim + l
          if not is_used[bit_idx]:
            is_used[bit_idx] = True
            watermark_bit2dim_indices[i].append(bit_idx)
            n_found += 1
            break
        if n_found == num_dims_per_watermark_bit:
          break
  w_bit = no_w_bit.clone().detach()
  for i in range(num_watermark_bits):
    w_bit[:, watermark_bit2dim_indices[i][num_dims_per_watermark_bit // 2:]] = (1 - int(watermark_str[i])) ^ w_bit[:, watermark_bit2dim_indices[i][:num_dims_per_watermark_bit // 2]]
  return w_bit, watermark_bit2dim_indices


def extract(w_syn_num: torch.Tensor, w_syn_cat: torch.Tensor, no_w_reversed_bit: torch.Tensor, watermark_bit2dim_indices: list[list[int]], inv_perm: torch.Tensor, codes: list[int], model, info: dict, num_watermark_bits: int) -> tuple[int, int]:
  assert len(w_syn_num) == len(w_syn_cat) == len(no_w_reversed_bit), f'{len(w_syn_num)}, {len(w_syn_cat)}, {len(no_w_reversed_bit)}'
  w_biased_reversed_latent = torch.from_numpy(table2latents(w_syn_num.cpu().detach().numpy(), w_syn_cat.cpu().detach().numpy(), info, CFG_BASIC.DEVICE, mask_col=None)).to(CFG_BASIC.DEVICE)
  w_reversed_latent = (w_biased_reversed_latent - info['mean']) / 2
  noise_scheduler = DDIMScheduler(num_train_timesteps=1000)
  w_reversed_noise = noise_scheduler.gen_reverse(model.noise_fn, w_reversed_latent, num_inference_steps=1000, eta=0.0, device=CFG_BASIC.DEVICE)
  w_reversed_permed_bit = noise2bit(w_reversed_noise, 2)
  w_reversed_bit = w_reversed_permed_bit[:, inv_perm]

  watermark_int_extracted = 0
  for i in range(num_watermark_bits):
    no_w_watermark_bit_reversed_bit = no_w_reversed_bit[:, watermark_bit2dim_indices[i]]
    w_watermark_bit_reversed_bit = w_reversed_bit[:, watermark_bit2dim_indices[i]]
    no_w_mean, no_w_std = cal_match_rate(no_w_watermark_bit_reversed_bit)
    w_mean, w_std = cal_match_rate(w_watermark_bit_reversed_bit)
    z_score = (w_mean - no_w_mean) / no_w_std * np.sqrt(len(w_syn_num))
    if z_score >= 0:
      watermark_int_extracted = watermark_int_extracted * 2 + 1
    else:
      watermark_int_extracted *= 2

  min_dist = num_watermark_bits
  watermark_int_pred = None
  for code in codes:
    dist = int2bit_str(code ^ watermark_int_extracted, num_watermark_bits).count('1')
    if dist < min_dist:
      min_dist = dist
      watermark_int_pred = code
  return watermark_int_extracted, watermark_int_pred


def main():
  info = load_info_full(CFG_BASIC.DATA_DIR, CFG_BASIC.INFO_PATH, CFG_BASIC.TRANSFORM_LATENTS, embedding_path=CFG_EMBEDDING_MODEL.EMBEDDING_PATH, vae_path=CFG_VAE.PATH, device=CFG_BASIC.DEVICE, **get_embedding_module_param())
  assert info['latents'].shape[1] >= CFG_WATERMARK.NUM_WATERMARK_BITS * 2
  
  denoise_fn = MLPDiffusion(info['latents'].shape[1], 1024).to(CFG_BASIC.DEVICE)
  model = DDIMModel(denoise_fn).to(CFG_BASIC.DEVICE)
  model.load_state_dict(torch.load(CFG_TABWAK.DM_PATH, weights_only=True, map_location=CFG_BASIC.DEVICE), strict=True)
  generator = torch.Generator()
  generator.manual_seed(CFG_CLUSTER.KEY)
  perm = torch.randperm(info['latents'].shape[1], generator=generator)
  inv_perm = torch.argsort(perm)
  codes = []
  with open(CFG_WATERMARK.CODE_PATH, 'r') as f:
    lines = f.readlines()
  for line in lines:
    codes.append(int(line))

  for _ in range(CFG_WATERMARK.NUM_WATERMARK_TRIALS):
    seed = get_random_seed()
    set_seed(seed)

    # no_w
    no_w_noise = torch.randn([CFG_SYN.NUM_SAMPLES, info['latents'].shape[1]], device=CFG_BASIC.DEVICE)
    noise_scheduler = DDIMScheduler(num_train_timesteps=1000)
    no_w_latent = noise_scheduler.generate(model.noise_fn, no_w_noise, num_inference_steps=1000, eta=0.0, device=CFG_BASIC.DEVICE)
    no_w_latent4decoder = (2 * no_w_latent + info['mean']).reshape(no_w_latent.shape[0], -1, CFG_VAE.TOKEN_DIM)
    decoder = info["decoder"]
    no_w_syn_num, no_w_syn_cat_raw = decoder(no_w_latent4decoder)
    no_w_syn_cat_lst = []
    for pred in no_w_syn_cat_raw:
      no_w_syn_cat_lst.append(pred.argmax(dim=-1))
    no_w_syn_cat = torch.stack(no_w_syn_cat_lst).t()
    no_w_biased_reversed_latent = torch.from_numpy(table2latents(no_w_syn_num.cpu().detach().numpy(), no_w_syn_cat.cpu().detach().numpy(), info, CFG_BASIC.DEVICE, mask_col=None)).to(CFG_BASIC.DEVICE)
    no_w_reversed_latent = (no_w_biased_reversed_latent - info['mean']) / 2
    noise_scheduler = DDIMScheduler(num_train_timesteps=1000)
    no_w_reversed_noise = noise_scheduler.gen_reverse(model.noise_fn, no_w_reversed_latent, num_inference_steps=1000, eta=0.0, device=CFG_BASIC.DEVICE)
    no_w_reversed_permed_bit = noise2bit(no_w_reversed_noise, 2)
    no_w_reversed_bit = no_w_reversed_permed_bit[:, inv_perm]

    # detection
    watermark_int_true = np.random.choice(codes)
    no_w_noise = torch.randn([CFG_SYN.NUM_SAMPLES, info['latents'].shape[1]], device=CFG_BASIC.DEVICE)
    no_w_bit = noise2bit(no_w_noise, 2)
    w_bit, watermark_bit2dim_indices = embed(watermark_int_true, no_w_bit, info['latents'].shape[1], CFG_VAE.TOKEN_DIM, CFG_WATERMARK.NUM_WATERMARK_BITS)
    permed_bit = w_bit[:, perm]
    w_noise = bit2noise(permed_bit, 2)
    noise_scheduler = DDIMScheduler(num_train_timesteps=1000)
    w_latent = noise_scheduler.generate(model.noise_fn, w_noise, num_inference_steps=1000, eta=0.0, device=CFG_BASIC.DEVICE)
    w_latent4decoder = (2 * w_latent + info['mean']).reshape(w_latent.shape[0], -1, CFG_VAE.TOKEN_DIM)
    w_syn_num, w_syn_cat, w_syn_target, w_syn_num4detection, w_syn_cat4detection = split_num_cat_target(w_latent4decoder, info, CFG_BASIC.DEVICE)
    w_syn_data = recover_data(w_syn_num, w_syn_cat, w_syn_target, info)
    watermark_int_extracted, watermark_int_pred = extract(torch.from_numpy(w_syn_num4detection).to(CFG_BASIC.DEVICE), torch.from_numpy(w_syn_cat4detection).to(CFG_BASIC.DEVICE), no_w_reversed_bit, watermark_bit2dim_indices, inv_perm, codes, model, info, CFG_WATERMARK.NUM_WATERMARK_BITS)
    r = {**get_res_header(seed), **check(watermark_int_true, watermark_int_extracted, watermark_int_pred, CFG_WATERMARK.NUM_WATERMARK_BITS, CFG_WATERMARK.MAX_NUM_ERROR_BITS)}
    with open(f'{CFG_BASIC.RESULTS_DIR}/tabwak_partition_detection.json', 'a') as f:
      f.write(json.dumps(r) + '\n')

    # quality
    r = {**get_res_header(seed), **eval_all(w_syn_data, info, CFG_SYN.REAL_DATA_PATH, CFG_SYN.TEST_DATA_PATH)}
    with open(f'{CFG_BASIC.RESULTS_DIR}/tabwak_partition_quality.json', 'a') as f:
      f.write(json.dumps(r) + '\n')

    # gauss_noise
    if CFG_BASIC.DATA_NAME != 'phishing':
      for ratio in [0.01]:
        print(f'ratio: {ratio}')
        cat = torch.from_numpy(w_syn_cat4detection).to(CFG_BASIC.DEVICE)
        _, num = add_gauss_noise(w_syn_data, info, ratio)
        num = torch.from_numpy(num).to(CFG_BASIC.DEVICE)
        watermark_int_extracted, watermark_int_pred = extract(num, cat, no_w_reversed_bit, watermark_bit2dim_indices, inv_perm, codes, model, info, CFG_WATERMARK.NUM_WATERMARK_BITS)
        r = {**get_res_header(seed), 'gauss_noise_ratio': ratio, **check(watermark_int_true, watermark_int_extracted, watermark_int_pred, CFG_WATERMARK.NUM_WATERMARK_BITS, CFG_WATERMARK.MAX_NUM_ERROR_BITS)}
        with open(f'{CFG_BASIC.RESULTS_DIR}/tabwak_partition_gauss_noise.json', 'a') as f:
          f.write(json.dumps(r) + '\n')
  
    # laplace_noise
    if CFG_BASIC.DATA_NAME != 'phishing':
      for ratio in [0.01]:
        print(f'ratio: {ratio}')
        cat = torch.from_numpy(w_syn_cat4detection).to(CFG_BASIC.DEVICE)
        _, num = add_laplace_noise(w_syn_data, info, ratio)
        num = torch.from_numpy(num).to(CFG_BASIC.DEVICE)
        watermark_int_extracted, watermark_int_pred = extract(num, cat, no_w_reversed_bit, watermark_bit2dim_indices, inv_perm, codes, model, info, CFG_WATERMARK.NUM_WATERMARK_BITS)
        r = {**get_res_header(seed), 'gauss_noise_ratio': ratio, **check(watermark_int_true, watermark_int_extracted, watermark_int_pred, CFG_WATERMARK.NUM_WATERMARK_BITS, CFG_WATERMARK.MAX_NUM_ERROR_BITS)}
        with open(f'{CFG_BASIC.RESULTS_DIR}/tabwak_partition_laplace_noise.json', 'a') as f:
          f.write(json.dumps(r) + '\n')

    # uniform_noise
    if CFG_BASIC.DATA_NAME != 'phishing':
      for ratio in [0.01]:
        print(f'ratio: {ratio}')
        cat = torch.from_numpy(w_syn_cat4detection).to(CFG_BASIC.DEVICE)
        _, num = add_uniform_noise(w_syn_data, info, ratio)
        num = torch.from_numpy(num).to(CFG_BASIC.DEVICE)
        watermark_int_extracted, watermark_int_pred = extract(num, cat, no_w_reversed_bit, watermark_bit2dim_indices, inv_perm, codes, model, info, CFG_WATERMARK.NUM_WATERMARK_BITS)
        r = {**get_res_header(seed), 'gauss_noise_ratio': ratio, **check(watermark_int_true, watermark_int_extracted, watermark_int_pred, CFG_WATERMARK.NUM_WATERMARK_BITS, CFG_WATERMARK.MAX_NUM_ERROR_BITS)}
        with open(f'{CFG_BASIC.RESULTS_DIR}/tabwak_partition_uniform_noise.json', 'a') as f:
          f.write(json.dumps(r) + '\n')

    # alteration
    for ratio in [0.01]:
      print(f'ratio: {ratio}')
      _, num = alter_num(w_syn_data, info, ratio)
      num = torch.from_numpy(num).to(CFG_BASIC.DEVICE)
      _, cat = alter_cat(w_syn_data, info, ratio)
      cat = torch.from_numpy(cat).to(CFG_BASIC.DEVICE)
      watermark_int_extracted, watermark_int_pred = extract(num, cat, no_w_reversed_bit, watermark_bit2dim_indices, inv_perm, codes, model, info, CFG_WATERMARK.NUM_WATERMARK_BITS)
      r = {**get_res_header(seed), 'alteration_ratio': ratio, **check(watermark_int_true, watermark_int_extracted, watermark_int_pred, CFG_WATERMARK.NUM_WATERMARK_BITS, CFG_WATERMARK.MAX_NUM_ERROR_BITS)}
      with open(f'{CFG_BASIC.RESULTS_DIR}/tabwak_partition_alteration.json', 'a') as f:
        f.write(json.dumps(r) + '\n')

    # deletion
    for ratio in [0.1]:
      print(f'ratio: {ratio}')
      selected_indices = np.random.choice(len(w_syn_data), [int(len(w_syn_data) * (1 - ratio))], replace=False)
      num = torch.from_numpy(w_syn_num4detection[selected_indices]).to(CFG_BASIC.DEVICE)
      cat = torch.from_numpy(w_syn_cat4detection[selected_indices]).to(CFG_BASIC.DEVICE)
      watermark_int_extracted, watermark_int_pred = extract(num, cat, no_w_reversed_bit[selected_indices], watermark_bit2dim_indices, inv_perm, codes, model, info, CFG_WATERMARK.NUM_WATERMARK_BITS)
      r = {**get_res_header(seed), 'sample_deletion_ratio': ratio, **check(watermark_int_true, watermark_int_extracted, watermark_int_pred, CFG_WATERMARK.NUM_WATERMARK_BITS, CFG_WATERMARK.MAX_NUM_ERROR_BITS)}
      with open(f'{CFG_BASIC.RESULTS_DIR}/tabwak_partition_sample_deletion.json', 'a') as f:
        f.write(json.dumps(r) + '\n')

    # insertion
    for ratio in [0.1]:
      print(f'ratio: {ratio}')
      original_indices = list(range(len(w_syn_data)))
      selected_indices = original_indices + np.random.choice(len(w_syn_data), [int(len(w_syn_data) * ratio)], replace=False).tolist()
      num = torch.from_numpy(w_syn_num4detection[selected_indices]).to(CFG_BASIC.DEVICE)
      cat = torch.from_numpy(w_syn_cat4detection[selected_indices]).to(CFG_BASIC.DEVICE)
      watermark_int_extracted, watermark_int_pred = extract(num, cat, no_w_reversed_bit[selected_indices], watermark_bit2dim_indices, inv_perm, codes, model, info, CFG_WATERMARK.NUM_WATERMARK_BITS)
      r = {**get_res_header(seed), 'sample_deletion_ratio': ratio, **check(watermark_int_true, watermark_int_extracted, watermark_int_pred, CFG_WATERMARK.NUM_WATERMARK_BITS, CFG_WATERMARK.MAX_NUM_ERROR_BITS)}
      with open(f'{CFG_BASIC.RESULTS_DIR}/tabwak_partition_sample_insertion.json', 'a') as f:
        f.write(json.dumps(r) + '\n')

    # regeneration
    token_dim = 4
    working_dir = f'{CFG_BASIC.ROOT_DIR}/watermark/regeneration_attack_vae/{CFG_WATERMARK.WATERMARK}-token_dim{token_dim}/{CFG_BASIC.DATA_NAME}/{CFG_CLUSTER.ALGORITHM}/{CFG_CLUSTER.NUM_CLASSES}/{CFG_EMBEDDING_MODEL.to_str()}/{watermark_int_true}'
    if os.path.exists(working_dir):
      continue
    os.makedirs(working_dir, exist_ok=True)
    w_syn_data.to_csv(f'{working_dir}/watermarked.csv', index=False)
    process_data(CFG_BASIC.DATA_NAME, f'{working_dir}/watermarked.csv')
    for ratio in [0.1]:
      reversed_table, reversed_num_norm, reversed_cat_norm, _, _ = attack(w_syn_data, info, token_dim, working_dir, ratio)
      reversed_table.to_csv(f'{working_dir}/attacked_{ratio}.csv', index=False)
      watermark_int_extracted, watermark_int_pred = extract(torch.from_numpy(reversed_num_norm).to(CFG_BASIC.DEVICE), torch.from_numpy(reversed_cat_norm).to(CFG_BASIC.DEVICE), no_w_reversed_bit, watermark_bit2dim_indices, inv_perm, codes, model, info, CFG_WATERMARK.NUM_WATERMARK_BITS)
      r = {**get_res_header(seed), **check(watermark_int_true, watermark_int_extracted, watermark_int_pred, CFG_WATERMARK.NUM_WATERMARK_BITS, CFG_WATERMARK.MAX_NUM_ERROR_BITS), 'attack_model': f'token_dim{token_dim}-ratio{ratio}'}
      with open(f'{CFG_BASIC.RESULTS_DIR}/tabwak_partition_regeneration_vae.json', 'a') as f:
        f.write(json.dumps(r) + '\n')
