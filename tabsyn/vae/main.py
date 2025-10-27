import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
from tqdm import tqdm
import time

from tabsyn.vae.model import VAE
from utils_train import preprocess, TabularDataset
from tabsyn.latent_utils import load_info_simple
from globals import *


LR = 1e-3
WD = 0

def compute_loss(X_num, X_cat, Recon_X_num, Recon_X_cat, mu_z, logvar_z):
    ce_loss_fn = nn.CrossEntropyLoss()
    if X_num.shape[1] == 0:
        mse_loss = torch.tensor(0)
    else:
        mse_loss = (X_num - Recon_X_num).pow(2).mean()
    ce_loss = 0
    acc = 0
    total_num = 0

    for idx, x_cat in enumerate(Recon_X_cat):
        if x_cat is not None:
            ce_loss += ce_loss_fn(x_cat, X_cat[:, idx])
            x_hat = x_cat.argmax(dim = -1)
        acc += (x_hat == X_cat[:,idx]).float().sum()
        total_num += x_hat.shape[0]
    
    ce_loss /= (idx + 1)
    acc /= total_num
    # loss = mse_loss + ce_loss

    temp = 1 + logvar_z - mu_z.pow(2) - logvar_z.exp()
    loss_kld = -0.5 * torch.mean(temp.mean(-1).mean())
    return mse_loss, ce_loss, loss_kld, acc


def get_noised_num_cat(num: torch.Tensor, cat: torch.Tensor, num_inverse, num_transform) -> tuple[torch.Tensor, torch.Tensor]:
    if CFG_VAE.ALTER_RATIO > 0:
        assert CFG_VAE.GAUSS_RATIO > 0
        if num.shape[1] > 0:
            table = num_inverse(num.cpu().detach().numpy())
            noise = CFG_VAE.GAUSS_RATIO * np.random.standard_normal(size=table.shape)
            mask = np.random.binomial(1, CFG_VAE.ALTER_RATIO, size=table.shape)
            num_noised = num_transform.transform(mask * np.random.uniform(table.min(0), table.max(0) + 1e-6, size=table.shape) + (1 - mask) * table * (1 + noise))
            num_noised = torch.from_numpy(num_noised).to(CFG_BASIC.DEVICE).float()
        else:
            num_noised = num.clone().detach()
        mask = np.random.binomial(1, CFG_VAE.ALTER_RATIO, size=cat.shape)
        cat_numpy = cat.cpu().detach().numpy()
        cat_noised = mask * np.random.randint(cat_numpy.min(0), cat_numpy.max(0) + 1, size=cat.shape) + (1 - mask) * cat_numpy
        cat_noised = torch.from_numpy(cat_noised).to(CFG_BASIC.DEVICE)
    elif CFG_VAE.GAUSS_RATIO > 0 and num.shape[1] > 0:
        table = num_inverse(num.cpu().detach().numpy())
        noise = CFG_VAE.GAUSS_RATIO * np.random.standard_normal(size=table.shape)
        num_noised = num_transform.transform(table * (1 + noise))
        num_noised = torch.from_numpy(num_noised).to(CFG_BASIC.DEVICE).float()
        cat_noised = cat.clone().detach()
    else:
        num_noised = num.clone().detach()
        cat_noised = cat.clone().detach()
    return num_noised, cat_noised


def main():
    max_beta = 1e-2
    min_beta = 1e-5
    lambd = 0.7

    os.makedirs(CFG_VAE.DIR, exist_ok=True)
    info = load_info_simple(CFG_BASIC.INFO_PATH)

    X_num, X_cat, categories, d_numerical, num_inverse, cat_inverse, num_transform, cat_transform = preprocess(CFG_BASIC.DATA_DIR, CFG_BASIC.INFO_PATH, task_type=info['task_type'], inverse=True)
    X_train_num, _ = X_num
    X_train_cat, _ = X_cat
    X_train_num, X_test_num = X_num
    X_train_cat, X_test_cat = X_cat

    # Fix the error of using testing set for validation from Tabsyn (@Tabsyn ICLR 2024), and use validation set instead
    print(f'test_num: {len(X_test_num)}, test_cat: {len(X_test_cat)}')
    X_full_num = torch.tensor(X_train_num).to(CFG_BASIC.DEVICE)
    X_full_cat = torch.tensor(X_train_cat).to(CFG_BASIC.DEVICE)
    rng = np.random.default_rng(CFG_BASIC.SEED)
    train_indices = rng.choice(len(X_train_num), size=(int(8 / 9 * len(X_train_num)),), replace=False)
    train_indices.sort()
    val_indices = np.asarray(list(set(range(len(X_train_num))) - set(train_indices)), dtype=train_indices.dtype)
    val_indices.sort()
    X_test_num = X_train_num[val_indices]
    X_train_num = X_train_num[train_indices]

    X_test_cat = X_train_cat[val_indices]
    X_train_cat = X_train_cat[train_indices]
    print(f'train_num: {len(X_train_num)}, train_cat: {len(X_train_cat)}, val_num: {len(X_test_num)}, val_cat: {len(X_test_cat)}')
    print(f'train_val total: num: {len(X_train_num) + len(X_test_num)}, cat: {len(X_train_cat) + len(X_test_cat)}, indices: {len(set(train_indices) | set(val_indices))}')
    print(f'full_num: {len(X_full_num)}, full_cat: {len(X_full_cat)}')
    print(f'train_test overlap: indices: {len(set(train_indices) & set(val_indices))}')
    ###################################################################################################################

    X_train_num, X_test_num = torch.tensor(X_train_num).float(), torch.tensor(X_test_num).float()
    X_train_cat, X_test_cat =  torch.tensor(X_train_cat), torch.tensor(X_test_cat)
    train_data = TabularDataset(X_train_num, X_train_cat)
    X_test_num = X_test_num.to(CFG_BASIC.DEVICE)
    X_test_cat = X_test_cat.to(CFG_BASIC.DEVICE)
    train_loader = DataLoader(train_data, batch_size=CFG_VAE.BATCH_SIZE, shuffle=True, num_workers=4)

    model = VAE(d_numerical, categories, CFG_VAE.NUM_LAYERS, CFG_VAE.TOKEN_DIM, CFG_VAE.NUM_HEADS, CFG_VAE.FACTOR, CFG_VAE.TOKEN_BIAS)
    model = model.to(CFG_BASIC.DEVICE)
    encoder = model.encode

    if not os.path.exists(CFG_VAE.PATH):
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.95, patience=10, verbose=True)
        best_train_loss = float('inf')
        current_lr = optimizer.param_groups[0]['lr']
        patience = 0

        beta = max_beta
        start_time = time.time()
        for epoch in range(CFG_VAE.NUM_EPOCHS):
            pbar = tqdm(train_loader, total=len(train_loader))
            pbar.set_description(f"Epoch {epoch+1}/{CFG_VAE.NUM_EPOCHS}")

            curr_loss_multi = 0.0
            curr_loss_gauss = 0.0
            curr_loss_kl = 0.0

            curr_count = 0

            for batch_num, batch_cat in pbar:
                model.train()
                optimizer.zero_grad()

                batch_num = batch_num.to(CFG_BASIC.DEVICE)
                batch_cat = batch_cat.to(CFG_BASIC.DEVICE)

                batch_num_noised, batch_cat_noised = get_noised_num_cat(batch_num, batch_cat, num_inverse, num_transform)

                Recon_X_num, Recon_X_cat, mu_z, std_z = model(batch_num_noised, batch_cat_noised)
                loss_mse, loss_ce, loss_kld, train_acc = compute_loss(batch_num, batch_cat, Recon_X_num, Recon_X_cat, mu_z, std_z)
                loss = loss_mse + loss_ce + beta * loss_kld
                loss.backward()
                optimizer.step()

                batch_length = batch_num.shape[0]
                curr_count += batch_length
                curr_loss_multi += loss_ce.item() * batch_length
                curr_loss_gauss += loss_mse.item() * batch_length
                curr_loss_kl    += loss_kld.item() * batch_length

            num_loss = curr_loss_gauss / curr_count
            cat_loss = curr_loss_multi / curr_count
            kl_loss = curr_loss_kl / curr_count

            model.eval()
            with torch.no_grad():
                X_test_num_noised, X_test_cat_noised = get_noised_num_cat(X_test_num, X_test_cat, num_inverse, num_transform)
                Recon_X_num, Recon_X_cat, mu_z, std_z = model(X_test_num_noised, X_test_cat_noised)
                val_mse_loss, val_ce_loss, val_kl_loss, val_acc = compute_loss(X_test_num, X_test_cat, Recon_X_num, Recon_X_cat, mu_z, std_z)
                val_loss = val_mse_loss.item() * X_train_num.shape[1] + val_ce_loss.item() * X_train_cat.shape[1]

                scheduler.step(val_loss)
                new_lr = optimizer.param_groups[0]['lr']

                if new_lr != current_lr:
                    current_lr = new_lr
                    print(f"Learning rate updated: {current_lr}")
                    
                train_loss = val_loss
                print(f'Current Loss: {train_loss:.10f}, Best Loss: {best_train_loss:.10f}')
                if train_loss < best_train_loss:
                    best_train_loss = train_loss
                    patience = 0
                    torch.save(model.state_dict(), CFG_VAE.PATH)
                else:
                    patience += 1
                    if patience == 10:
                        if beta > min_beta:
                            beta = beta * lambd

            print('epoch: {}, beta: {:.6f}, Train MSE: {:.6f}, Train CE: {:.6f}, Train KL: {:.6f}, Val MSE: {:.6f}, Val CE: {:.6f}, Train ACC: {:6f}, Val ACC: {:6f}'.format(epoch, beta, num_loss, cat_loss, kl_loss, val_mse_loss.item(), val_ce_loss.item(), train_acc.item(), val_acc.item() ))

        end_time = time.time()
        print('Training time: {:.4f} mins'.format((end_time - start_time)/60))
        

    if not os.path.exists(CFG_VAE.EMBEDDING_PATH):
        with torch.no_grad():
            model.load_state_dict(torch.load(CFG_VAE.PATH, weights_only=True), strict=True)
            model.eval()
            print('Successfully load and save the model!')
            train_z = encoder(X_full_num.float(), X_full_cat).detach().cpu().numpy()
            np.save(CFG_VAE.EMBEDDING_PATH, train_z)
            print('Successfully save pretrained embeddings in disk!')
