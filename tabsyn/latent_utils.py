import numpy as np
import pandas as pd
import torch
import json

from utils_train import preprocess
from tabsyn.vae.model import create_encoder_decoder
from globals import *


def load_info_simple(info_path: str) -> dict:
    with open(info_path, 'r') as f:
        info = json.load(f)
    return info


def load_info_full(data_dir: str, info_path: str, transform_latents: str, embedding_path: str, vae_path: str, device: str, **kwargs) -> dict:
    info = load_info_simple(f'{data_dir}/info.json')
    _, _, categories, d_numerical, num_inverse, cat_inverse, num_transform, cat_transform = preprocess(data_dir, info_path, task_type=info['task_type'], inverse=True)
    latents = torch.tensor(np.load(embedding_path), device=device).float()
    if transform_latents == 'none':
        latents = latents[:, 1:, :]
    info['transform_latents'] = transform_latents
    B, num_tokens, token_dim = latents.size()
    latent_dim = num_tokens * token_dim
    latents = latents.view(B, latent_dim)
    kwargs['categories'] = categories
    kwargs['d_numerical'] = d_numerical
    encoder, decoder = create_encoder_decoder(transform_latents, vae_path, device, **kwargs)
    info['num_inverse'] = num_inverse
    info['cat_inverse'] = cat_inverse
    info['num_transform'] = num_transform
    info['cat_transform'] = cat_transform
    info['encoder'] = encoder
    info['decoder'] = decoder
    info['token_dim'] = latents
    info['latent_dim'] = latent_dim
    info['token_dim'] = token_dim
    info['latents'] = latents
    info['mean'] = latents.mean(0)
    return info

 
def add_gauss_noise(table: pd.DataFrame, info: dict, ratio: float, seed: int = None) -> tuple[pd.DataFrame, np.ndarray]:
    if info['num_transform'] is None:
        return table.copy(), np.zeros(shape=(len(table), 0), dtype=np.float32)
    if seed is None:
        seed = get_random_seed()
    table = table.copy()
    num_col_indices = info['num_col_idx']
    if info['task_type'] == 'regression':
        num_col_indices = info['target_col_idx'] + num_col_indices
    noise = ratio * np.random.default_rng(seed + 100).standard_normal(size=(len(table), len(num_col_indices))).astype(np.float32)
    table[table.columns[num_col_indices]] = table[table.columns[num_col_indices]].values * (1 + noise)
    num = info['num_transform'].transform(table[table.columns[num_col_indices]].values).astype(np.float32)
    return table, num


def add_uniform_noise(table: pd.DataFrame, info: dict, ratio: float, seed: int = None) -> tuple[pd.DataFrame, np.ndarray]:
    if info['num_transform'] is None:
        return table.copy(), np.zeros(shape=(len(table), 0), dtype=np.float32)
    if seed is None:
        seed = get_random_seed()
    table = table.copy()
    num_col_indices = info['num_col_idx']
    if info['task_type'] == 'regression':
        num_col_indices = info['target_col_idx'] + num_col_indices
    low = -np.sqrt(3)
    high = np.sqrt(3)
    noise = ratio * np.random.default_rng(seed + 100).uniform(low, high, size=(len(table), len(num_col_indices))).astype(np.float32)
    table[table.columns[num_col_indices]] = table[table.columns[num_col_indices]].values * (1 + noise)
    num = info['num_transform'].transform(table[table.columns[num_col_indices]].values).astype(np.float32)
    return table, num


def add_laplace_noise(table: pd.DataFrame, info: dict, ratio: float, seed: int = None) -> tuple[pd.DataFrame, np.ndarray]:
    if info['num_transform'] is None:
        return table.copy(), np.zeros(shape=(len(table), 0), dtype=np.float32)
    if seed is None:
        seed = get_random_seed()
    table = table.copy()
    num_col_indices = info['num_col_idx']
    if info['task_type'] == 'regression':
        num_col_indices = info['target_col_idx'] + num_col_indices
    scale = 1 / np.sqrt(2)
    noise = ratio * np.random.default_rng(seed + 100).laplace(0, scale, size=(len(table), len(num_col_indices))).astype(np.float32)
    table[table.columns[num_col_indices]] = table[table.columns[num_col_indices]].values * (1 + noise)
    num = info['num_transform'].transform(table[table.columns[num_col_indices]].values).astype(np.float32)
    return table, num


def alter_cat(table: pd.DataFrame, info: dict, ratio: float, seed: int = None) -> tuple[pd.DataFrame, np.ndarray]:
    if info['cat_transform'] is None:
        return table.copy(), np.zeros(shape=(len(table), 0), dtype=np.int64)
    if seed is None:
        seed = get_random_seed()
    table = table.copy()
    cat_col_indices = info['cat_col_idx']
    if info['task_type'] in ['binclass', 'multiclass']:
        cat_col_indices = info['target_col_idx'] + cat_col_indices
    mask = np.random.default_rng(seed + 200).binomial(1, ratio, size=(len(table), len(cat_col_indices)))
    cat = info['cat_transform'].transform(table[table.columns[cat_col_indices]].values.astype(object))
    cat = mask * np.random.default_rng(seed + 300).integers(cat.min(0), cat.max(0) + 1, size=cat.shape) + (1 - mask) * cat
    table.loc[:, table.columns[cat_col_indices]] = info['cat_inverse'](cat)
    return table, cat


def col_min_exclude_inf(arr: np.ndarray):
    mask = ~np.isinf(arr)
    masked_arr = np.ma.array(arr, mask=~mask)
    return masked_arr.min(axis=0).filled(np.nan)


def col_max_exclude_inf(arr: np.ndarray):
    mask = ~np.isinf(arr)
    masked_arr = np.ma.array(arr, mask=~mask)
    return masked_arr.max(axis=0).filled(np.nan)


def alter_num(table: pd.DataFrame, info: dict, ratio: float, seed: int = None) -> tuple[pd.DataFrame, np.ndarray]:
    if seed is None:
        seed = get_random_seed()
    table = table.copy()
    if info['num_transform'] is None:
        return table, np.zeros((len(table), 0), dtype=np.float32)
    num_col_indices = info['num_col_idx']
    if info['task_type'] == 'regression':
        num_col_indices = info['target_col_idx'] + num_col_indices
    table_num = table.loc[:, table.columns[num_col_indices]].values
    mask = np.random.default_rng(seed + 400).binomial(1, ratio, size=table_num.shape)
    table.loc[:, table.columns[num_col_indices]] = mask * np.random.default_rng(seed + 500).uniform(col_min_exclude_inf(table_num), col_max_exclude_inf(table_num) + 1e-6, size=table_num.shape) + (1 - mask) * table_num
    num = info['num_transform'].transform(table[table.columns[num_col_indices]].values).astype(np.float32)
    return table, num


def add_alter_num(table: pd.DataFrame, info: dict, add_ratio: float, alter_ratio: float, seed: int = None) -> tuple[pd.DataFrame, np.ndarray]:
    if seed is None:
        seed = get_random_seed()
    table = table.copy()
    if info['num_transform'] is None:
        return table, np.zeros((len(table), 0), dtype=np.float32)
    num_col_indices = info['num_col_idx']
    if info['task_type'] == 'regression':
        num_col_indices = info['target_col_idx'] + num_col_indices
    table_num = table.loc[:, table.columns[num_col_indices]].values
    mask = np.random.default_rng(seed + 600).binomial(1, alter_ratio, size=table_num.shape)
    noise = add_ratio * np.random.default_rng(seed + 700).standard_normal(size=table_num.shape).astype(np.float32)
    table.loc[:, table.columns[num_col_indices]] = mask * np.random.default_rng(seed + 800).uniform(col_min_exclude_inf(table_num), col_max_exclude_inf(table_num) + 1e-6, size=table_num.shape) + (1 - mask) * table_num * (1 + noise)
    num = info['num_transform'].transform(table[table.columns[num_col_indices]].values).astype(np.float32)
    return table, num


def delete(table: pd.DataFrame, info: dict, ratio: float, seed: int = None) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    if seed is None:
        seed = get_random_seed()
    table = table.copy()
    num_col_indices = info['num_col_idx']
    cat_col_indices = info['cat_col_idx']
    if info['task_type'] == 'regression':
        num_col_indices = info['target_col_idx'] + num_col_indices
    else:
        cat_col_indices = info['target_col_idx'] + cat_col_indices
    selected_indices = np.random.default_rng(seed + 900).choice(len(table), (int(len(table) * (1 - ratio)),), replace=False)
    table = table.iloc[selected_indices]
    if info['num_transform'] is not None:
        num = info['num_transform'].transform(table[table.columns[num_col_indices]].values).astype(np.float32)
    else:
        num = np.zeros((len(table), 0), dtype=np.float32)
    if info['cat_transform'] is not None:
        cat = info['cat_transform'].transform(table[table.columns[cat_col_indices]].values.astype(object))
    else:
        cat = np.zeros((len(table), 0), dtype=np.int64)
    return table.reset_index(drop=True), num, cat


def insert(table: pd.DataFrame, info: dict, ratio: float, seed: int = None) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    if seed is None:
        seed = get_random_seed()
    table = table.copy()
    num_col_indices = info['num_col_idx']
    cat_col_indices = info['cat_col_idx']
    if info['task_type'] == 'regression':
        num_col_indices = info['target_col_idx'] + num_col_indices
    else:
        cat_col_indices = info['target_col_idx'] + cat_col_indices
    selected_indices = np.concatenate([
        np.asarray(range(len(table))),
        np.random.default_rng(seed + 900).choice(len(table), (int(len(table) * ratio),), replace=False),
    ], axis=0)
    table = table.iloc[selected_indices]
    if info['num_transform'] is not None:
        num = info['num_transform'].transform(table[table.columns[num_col_indices]].values).astype(np.float32)
    else:
        num = np.zeros((len(table), 0), dtype=np.float32)
    cat = info['cat_transform'].transform(table[table.columns[cat_col_indices]].values.astype(object))
    return table.reset_index(drop=True), num, cat


@torch.no_grad()
def split_num_cat_target(syn_data, info: dict, device: str, decoder = None, num_transform = None, num_inverse = None, cat_inverse = None):
    if decoder is None:
        assert num_transform is None
        assert num_inverse is None
        assert cat_inverse is None
        decoder = info['decoder']
        num_transform = info['num_transform']
        num_inverse = info['num_inverse']
        cat_inverse = info['cat_inverse']

    task_type = info['task_type']
    num_col_idx = info['num_col_idx']
    cat_col_idx = info['cat_col_idx']
    target_col_idx = info['target_col_idx']

    n_num_feat = len(num_col_idx)
    n_cat_feat = len(cat_col_idx)

    if task_type == 'regression':
        n_num_feat += len(target_col_idx)
    else:
        n_cat_feat += len(target_col_idx)

    syn_data = syn_data.reshape(syn_data.shape[0], -1, info['token_dim'])
    norm_input = decoder(torch.tensor(syn_data, device=device))
    x_hat_num, x_hat_cat = norm_input

    syn_cat = []
    for pred in x_hat_cat:
        syn_cat.append(pred.argmax(dim = -1))

    syn_num = x_hat_num.cpu().numpy()
    if syn_cat:
        syn_cat = torch.stack(syn_cat).t().cpu().numpy()
    else:
        syn_cat = torch.zeros((len(syn_num), 0), dtype=torch.int64).cpu().numpy()

    if num_inverse is not None:
        num4detection = num_transform.transform(num_inverse(syn_num))  # consider transformation error
    else:
        num4detection = syn_num.copy()
    cat4detection = syn_cat  # no transformation error for cat

    if num_inverse is not None:
        syn_num = num_inverse(syn_num)
    if cat_inverse is not None:
        syn_cat = cat_inverse(syn_cat)

    if info['task_type'] == 'regression':
        syn_target = syn_num[:, :len(target_col_idx)]
        syn_num = syn_num[:, len(target_col_idx):]    
    else:
        print(syn_cat.shape)
        syn_target = syn_cat[:, :len(target_col_idx)]
        syn_cat = syn_cat[:, len(target_col_idx):]

    return syn_num, syn_cat, syn_target, num4detection, cat4detection


def recover_data(syn_num, syn_cat, syn_target, info):
    num_col_idx = info['num_col_idx']
    cat_col_idx = info['cat_col_idx']
    target_col_idx = info['target_col_idx']

    idx_mapping = info['idx_mapping']
    idx_mapping = {int(key): value for key, value in idx_mapping.items()}

    syn_df = pd.DataFrame()

    if info['task_type'] == 'regression':
        for i in range(len(num_col_idx) + len(cat_col_idx) + len(target_col_idx)):
            if i in set(num_col_idx):
                syn_df[i] = syn_num[:, idx_mapping[i]] 
            elif i in set(cat_col_idx):
                syn_df[i] = syn_cat[:, idx_mapping[i] - len(num_col_idx)]
            else:
                syn_df[i] = syn_target[:, idx_mapping[i] - len(num_col_idx) - len(cat_col_idx)]
    else:
        for i in range(len(num_col_idx) + len(cat_col_idx) + len(target_col_idx)):
            if i in set(num_col_idx):
                syn_df[i] = syn_num[:, idx_mapping[i]]
            elif i in set(cat_col_idx):
                syn_df[i] = syn_cat[:, idx_mapping[i] - len(num_col_idx)]
            else:
                syn_df[i] = syn_target[:, idx_mapping[i] - len(num_col_idx) - len(cat_col_idx)]

    return syn_df
    

def process_invalid_id(syn_cat, min_cat, max_cat):
    syn_cat = np.clip(syn_cat, min_cat, max_cat)
    return syn_cat
