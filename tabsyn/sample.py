import torch
import time
import os
import pandas as pd

from tabsyn.model import MLPDiffusion, Model
from tabsyn.latent_utils import load_info_full, recover_data, split_num_cat_target
from tabsyn.diffusion_utils import sample
from globals import *


def load_onehot_labels(num_samples: int, num_classes: int, num_samples_per_class: list, device: str) -> torch.Tensor:
    onehot_labels = torch.zeros([num_samples, num_classes], dtype=torch.float32, device=device)
    ptr = 0
    for i in range(num_classes):
        onehot_labels[ptr: ptr + num_samples_per_class[i], i] = 1
        ptr += num_samples_per_class[i]
    return onehot_labels


def load_model(latent_dim: int, num_classes: int, dm_path: str, device: str):
    denoise_fn = MLPDiffusion(latent_dim, 1024, num_classes).to(device)
    model = Model(denoise_fn=denoise_fn, hid_dim=latent_dim).to(device)
    model.load_state_dict(torch.load(dm_path, map_location=device))
    model.eval()
    return model


def sample_latents(model, train_data_info, onehot_labels, num_samples: str, num_sample_steps: int, device: str) -> torch.Tensor:
    x_next = sample(model.denoise_fn_D, num_samples, train_data_info['latent_dim'], onehot_labels, num_steps=num_sample_steps, device=device)
    x_next = x_next * 2 + train_data_info['mean'].to(device)
    return x_next


def latents2table(latents, info: dict, device: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    latents = latents.float().cpu().numpy()
    syn_num, syn_cat, syn_target, num4detection, cat4detection = split_num_cat_target(latents, info, device)
    table = recover_data(syn_num, syn_cat, syn_target, info)
    idx_name_mapping = info['idx_name_mapping']
    idx_name_mapping = {int(key): value for key, value in idx_name_mapping.items()}
    table.rename(columns = idx_name_mapping, inplace=True)
    table = process_table_dtypes(table, info)
    return table, num4detection, cat4detection


def profile(labels_true: torch.Tensor, info: dict, num_classes: int, dm_path: str, device: str, centers: np.ndarray, dim_ratio: str, key: int) -> np.ndarray:
    if isinstance(labels_true, list):
        labels_true = torch.tensor(labels_true, dtype=torch.int64, device=device)
    elif isinstance(labels_true, np.ndarray):
        labels_true = torch.from_numpy(labels_true).to(torch.int64).to(device)
    else:
        assert isinstance(labels_true, torch.Tensor), type(labels_true)
    model = load_model(info['latent_dim'], num_classes, dm_path, device)
    onehot_labels = torch.zeros(size=(len(labels_true), num_classes), dtype=torch.float32, device=device)
    for i in range(len(labels_true)):
        onehot_labels[i, labels_true[i]] = 1
    latents4dm = sample_latents(model, info, onehot_labels, len(labels_true), CFG_DM.NUM_SAMPLE_STEPS, device)
    latents = get_latents_from_latents4dm_torch(latents4dm, dim_ratio, key)
    latents4cluster = get_latents4cluster_torch(latents, dim_ratio, key)
    centers = torch.from_numpy(centers).to(device)
    labels_pred = torch.concat([
                ((latents4cluster[:len(latents4cluster) // 2] - centers[:, None, :]) ** 2).sum(-1, keepdim=False).argmin(0, keepdim=False),
                ((latents4cluster[len(latents4cluster) // 2:] - centers[:, None, :]) ** 2).sum(-1, keepdim=False).argmin(0, keepdim=False),
            ], dim=0)
    return labels_pred.cpu().detach().numpy()


def syn_table(info: dict, num_samples: int, num_classes: int, num_samples_per_class: list, num_sample_steps: int, dm_path: str, device: str, correct_guidance: bool, centers: np.ndarray, dim_ratio: float, key: int, shuffle=True, return_labels_pred=False, labels_true: np.ndarray = None):
    if num_samples_per_class is not None:
        assert labels_true is None
        assert num_samples == sum(num_samples_per_class), f'{num_samples} != {sum(num_samples_per_class)}'
    else:
        assert labels_true is not None
        assert num_samples == len(labels_true), f'{num_samples} != {len(labels_true)}'
        num_samples_per_class = [sum(labels_true == i) for i in range(num_classes)]
    if isinstance(dm_path, str):
        model = load_model(info['latent_dim'], num_classes, dm_path, device)
    else:
        model = dm_path.to(device)
    if not correct_guidance:
        assert False
        num_over_samples = int(1.001 * num_samples)
        over_sample_onehot_labels = torch.concat([onehot_labels, onehot_labels[torch.randint(0, num_samples, [num_over_samples - num_samples])]], dim=0)
        latents4dm = sample_latents(model, info, over_sample_onehot_labels, num_over_samples, num_sample_steps, device)
        latents = get_latents_from_latents4dm_torch(latents4dm, dim_ratio, key)
        table, num, cat = latents2table(latents, info, device)
        not_nan_indices = ~(pd.isna(table).sum(axis=1).astype(bool))
        nan_ratio = 1 - sum(not_nan_indices) / num_samples
        assert nan_ratio <= 0.0005, f'Too Many nans: {nan_ratio} !'
        latents = latents[not_nan_indices]
        table = table.loc[not_nan_indices, :]
        num = num[not_nan_indices]
        cat = cat[not_nan_indices]
        latents = latents[:num_samples]
        table = table.iloc[:num_samples, :]
        num = num[:num_samples]
        cat = cat[:num_samples]
    else:
        assert not return_labels_pred
        if os.path.exists(f'{CFG_DM.DIR}/accuracy.txt'):
            accuracies = np.loadtxt(f'{CFG_DM.DIR}/accuracy.txt', dtype=np.float32)
            over_sample_num_samples_per_class = [max(50, int(n * 1.55 / accuracies[i])) for i, n in enumerate(num_samples_per_class)]
            # over_sample_num_samples_per_class = [int((-6 * np.sqrt(1 - accuracies[i]) + np.sqrt(36 * (1 - accuracies[i]) + 4 * num_samples_per_class[i])) ** 2 / (4 * accuracies[i])) for i in range(len(num_samples_per_class))]
        else:
            over_sample_num_samples_per_class = [int(n * 3.5) for n in num_samples_per_class]
        num_over_samples = sum(over_sample_num_samples_per_class)
        over_sample_onehot_labels = load_onehot_labels(num_over_samples, num_classes, over_sample_num_samples_per_class, device)
        latents4dm = sample_latents(model, info, over_sample_onehot_labels, num_over_samples, num_sample_steps, device)
        latents = get_latents_from_latents4dm_torch(latents4dm, dim_ratio, key)
        latents4cluster = get_latents4cluster_torch(latents, dim_ratio, key)
        centers = torch.from_numpy(centers).to(device)
        labels = torch.concat([
            ((latents4cluster[:len(latents4cluster) // 2] - centers[:, None, :]) ** 2).sum(-1, keepdim=False).argmin(0, keepdim=False),
            ((latents4cluster[len(latents4cluster) // 2:] - centers[:, None, :]) ** 2).sum(-1, keepdim=False).argmin(0, keepdim=False),
            ], dim=0)
        valid_masks = (labels == over_sample_onehot_labels.argmax(dim=-1))
        onehot_labels = over_sample_onehot_labels[valid_masks]
        latents = latents[valid_masks]
        labels = labels[valid_masks]
        hist = labels.to(torch.float).histc(min=0, max=CFG_CLUSTER.NUM_CLASSES - 1, bins=CFG_CLUSTER.NUM_CLASSES).to(torch.int64).cpu().tolist()
        print(f'min: {min((hist[i] - num_samples_per_class[i]) / max(1, num_samples_per_class[i]) for i in range(num_classes))}', flush=True)
        assert all(hist[i] >= num_samples_per_class[i] for i in range(num_classes)), 'Poor Guidance !'

        table, num, cat = latents2table(latents, info, device)
        not_nan_indices = ~(pd.isna(table).sum(axis=1).astype(bool))
        nan_ratio = 1 - sum(not_nan_indices) / num_samples
        assert nan_ratio <= 0.0005, f'Too Many nans: {nan_ratio} !'
        labels = labels[not_nan_indices]
        hist = labels.to(torch.float).histc(min=0, max=CFG_CLUSTER.NUM_CLASSES - 1, bins=CFG_CLUSTER.NUM_CLASSES).to(torch.int64).cpu().tolist()
        assert all(hist[i] >= num_samples_per_class[i] for i in range(num_classes)), 'Poor Guidance !'        
        onehot_labels = onehot_labels[not_nan_indices]
        latents = latents[not_nan_indices]
        table = table.loc[not_nan_indices, :]
        num = num[not_nan_indices]
        cat = cat[not_nan_indices]

        final_indices = []
        if labels_true is not None:
            labels2indices = {i: [] for i in range(num_classes)}
            for index, label in enumerate(labels):
                labels2indices[label.item()].append(index)
            for label_true in labels_true:
                index = labels2indices[label_true].pop()
                final_indices.append(index)
        else:
            synthesized_labels_hist = [0] * num_classes
            for i in range(len(labels)):
                if synthesized_labels_hist[labels[i]] < num_samples_per_class[labels[i]]:
                    synthesized_labels_hist[labels[i]] += 1
                    final_indices.append(i)
        
        onehot_labels = onehot_labels[final_indices]
        latents = latents[final_indices]
        table = table.iloc[final_indices, :]
        num = num[final_indices]
        cat = cat[final_indices]

    table = process_table_dtypes(table, info).reset_index(drop=True)
    if shuffle and labels_true is None:
        perm = np.random.permutation(len(table))
        table = table.iloc[perm, :]
        table = table.reset_index(drop=True)
        num = num[perm]
        cat = cat[perm]
        perm = torch.from_numpy(perm).to(device)
        latents = latents[perm]
        onehot_labels = onehot_labels[perm]
    
    assert not pd.isna(table).any().any(), pd.isna(table).sum().sum()
    assert num_samples == len(onehot_labels) == len(latents) == len(table) == len(num) == len(cat), f'{num_samples} == {len(onehot_labels)} == {len(latents)} == {len(table)} == {len(num)} == {len(cat)}'
    return onehot_labels, latents, table, num, cat


def save_all(syn_latents: torch.Tensor, syn_table: pd.DataFrame, onehot_labels: torch.Tensor,
             latent_path: str, table_path: str, label_path: str):
    np.savetxt(latent_path, syn_latents.cpu().detach().numpy(), '%.16f')
    syn_table.to_csv(table_path, index=False)
    target_labels = onehot_labels.argmax(-1, keepdim=False)
    np.savetxt(label_path, target_labels.cpu().detach().numpy(), '%d')


def main():
    os.makedirs(CFG_SYN.DIR, exist_ok=True)
    start_time = time.time()
    info = load_info_full(CFG_BASIC.DATA_DIR, CFG_BASIC.INFO_PATH, CFG_BASIC.TRANSFORM_LATENTS, CFG_EMBEDDING_MODEL.EMBEDDING_PATH, CFG_VAE.PATH, CFG_BASIC.DEVICE, **get_embedding_module_param())
    onehot_labels, latents, table, _, _ = syn_table(info, CFG_SYN.NUM_SAMPLES, CFG_CLUSTER.NUM_CLASSES, CFG_SYN.NUM_SAMPLES_PER_CLASS, CFG_DM.NUM_SAMPLE_STEPS, CFG_DM.PATH, CFG_BASIC.DEVICE, CFG_DM.CORRECT_GUIDANCE, get_original_centers(CFG_CLUSTER.CENTERS_PATH), CFG_CLUSTER.DIM_RATIO, CFG_CLUSTER.KEY)
    save_all(latents, table, onehot_labels, f'{CFG_SYN.DIR}/latent.txt', f'{CFG_SYN.DIR}/tabsyn.csv', f'{CFG_SYN.DIR}/target_label.txt')
    end_time = time.time()
    print('Time:', end_time - start_time)
