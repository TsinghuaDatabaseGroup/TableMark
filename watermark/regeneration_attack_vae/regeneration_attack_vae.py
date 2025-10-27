import json
import sys
import os
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

sys.path.append('.')
sys.path.append('../')
from globals import *
from process_dataset import process_data
from tabsyn.sample import syn_table
from tabsyn.latent_utils import load_info_full, preprocess
from watermark.watermark import create_watermark, check, embed_random_watermark
from utils_train import TabularDataset
from tabsyn.vae.model import VAE


LR = 1e-3
WD = 0


def compute_loss(X_num, X_cat, Recon_X_num, Recon_X_cat, mu_z, logvar_z):
  ce_loss_fn = torch.nn.CrossEntropyLoss()
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

  temp = 1 + logvar_z - mu_z.pow(2) - logvar_z.exp()
  loss_kld = -0.5 * torch.mean(temp.mean(-1).mean())
  return mse_loss, ce_loss, loss_kld, acc


def train_vae(token_dim: int, info: dict, info_path: str, working_dir: str, real_model_path: str, device: str):
  assert os.path.exists(f'{working_dir}/watermarked.csv'), 'No Watermarked Synthetic Table !'
  max_beta = 1e-2
  min_beta = 1e-5
  lambd = 0.7

  X_num, X_cat, categories, d_numerical, num_inverse, cat_inverse, num_transform, cat_transform = preprocess(working_dir, info_path, task_type=info['task_type'], inverse=True)
  X_train_num, _ = X_num
  X_train_cat, _ = X_cat
  X_train_num, X_test_num = X_num
  X_train_cat, X_test_cat = X_cat
  X_train_num, X_test_num = torch.tensor(X_train_num).float(), torch.tensor(X_test_num).float()
  X_train_cat, X_test_cat =  torch.tensor(X_train_cat), torch.tensor(X_test_cat)
  train_data = TabularDataset(X_train_num.float(), X_train_cat)
  X_test_num = X_test_num.float().to(device)
  X_test_cat = X_test_cat.to(device)
  train_loader = DataLoader(train_data, batch_size=CFG_VAE.BATCH_SIZE, shuffle=True, num_workers=4)

  model = VAE(d_numerical, categories, CFG_VAE.NUM_LAYERS, token_dim, CFG_VAE.NUM_HEADS, CFG_VAE.FACTOR, CFG_VAE.TOKEN_BIAS)
  model = model.to(device)
  encoder = model.encode
  if os.path.exists(f'{working_dir}/vae.pt'):
    model.load_state_dict(torch.load(f'{working_dir}/vae.pt'), strict=True)
  else:
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
        Recon_X_num, Recon_X_cat, mu_z, std_z = model(batch_num, batch_cat)
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
        Recon_X_num, Recon_X_cat, mu_z, std_z = model(X_test_num, X_test_cat)
        val_mse_loss, val_ce_loss, val_kl_loss, val_acc = compute_loss(X_test_num, X_test_cat, Recon_X_num, Recon_X_cat, mu_z, std_z)
        val_loss = val_mse_loss.item() * 0 + val_ce_loss.item()    

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
          torch.save(model.state_dict(), f'{working_dir}/vae.pt')
        else:
          patience += 1
          if patience == 10:
            if beta > min_beta:
              beta = beta * lambd
      print('epoch: {}, beta: {:.6f}, Train MSE: {:.6f}, Train CE: {:.6f}, Train KL: {:.6f}, Val MSE: {:.6f}, Val CE: {:.6f}, Train ACC: {:6f}, Val ACC: {:6f}'.format(epoch, beta, num_loss, cat_loss, kl_loss, val_mse_loss.item(), val_ce_loss.item(), train_acc.item(), val_acc.item()))

    end_time = time.time()
    print('Training time: {:.4f} mins'.format((end_time - start_time)/60))
  
  with torch.no_grad():
    model.eval()
    print('Successfully load and save the model!')
    train_idx = torch.from_numpy(np.load(f'{working_dir}/train_idx.npy')).to(device)
    test_idx = torch.from_numpy(np.load(f'{working_dir}/test_idx.npy')).to(device)
    full_num = torch.zeros((X_train_num.shape[0] + X_test_num.shape[0], X_train_num.shape[1]), dtype=torch.float32, device=device)
    full_num[train_idx] = X_train_num.to(device)
    full_num[test_idx] = X_test_num.to(device)
    full_cat = torch.zeros((X_train_cat.shape[0] + X_test_cat.shape[0], X_train_cat.shape[1]), dtype=torch.int64, device=device)
    full_cat[train_idx] = X_train_cat.to(device)
    full_cat[test_idx] = X_test_cat.to(device)
    train_z = encoder(full_num, full_cat).detach().cpu().numpy()
    if not os.path.exists(f'{working_dir}/train_z.npy'):
      np.save(f'{working_dir}/train_z.npy', train_z)
    print('Successfully save pretrained embeddings in disk!')
  return train_z[:, 1:], model.encode, model.decode, num_inverse, cat_inverse, num_transform, cat_transform


def attack(table, info: dict, token_dim: int, working_dir: str, ratio: float):
  reversed_latents, encoder, decoder, num_inverse, cat_inverse, num_transform, cat_transform = train_vae(token_dim, info, CFG_BASIC.INFO_PATH, working_dir, '', CFG_BASIC.DEVICE)

  if ratio > 1e-4:
    noise = ratio * np.random.default_rng(CFG_BASIC.SEED + 16985).standard_normal(size=reversed_latents.shape, dtype=np.float32)
    reversed_latents *= (1 + noise)
  reversed_num_norm, reversed_cat_norm = decoder(torch.from_numpy(reversed_latents).to(CFG_BASIC.DEVICE))
  reversed_cat_norm_list = []
  for pred in reversed_cat_norm:
    reversed_cat_norm_list.append(pred.argmax(dim = -1))
  reversed_cat_norm = torch.stack(reversed_cat_norm_list).t().detach().cpu().numpy()
  reversed_num_norm = reversed_num_norm.detach().cpu().numpy()
  if CFG_BASIC.DATA_NAME != 'phishing':
    reversed_num = num_inverse(reversed_num_norm)
  else:
    reversed_num = np.zeros(shape=(len(reversed_latents), 0), dtype=np.float32)
  reversed_cat = cat_inverse(reversed_cat_norm)

  num_col_indices = info['num_col_idx']
  cat_col_indices = info['cat_col_idx']
  if info['task_type'] == 'regression':
    num_col_indices = info['target_col_idx'] + num_col_indices
  else:
    cat_col_indices = info['target_col_idx'] + cat_col_indices
  reversed_table = table.copy()
  reversed_table[reversed_table.columns[num_col_indices]] = reversed_num
  reversed_table[reversed_table.columns[cat_col_indices]] = reversed_cat

  return reversed_table, reversed_num_norm, reversed_cat_norm, reversed_num, reversed_cat


def main():
  for _ in range(CFG_WATERMARK.NUM_WATERMARK_TRIALS):
    seed = get_random_seed()
    set_seed(seed)

    watermark = create_watermark(**get_create_watermark_params(seed))
    use_cached_watermark_int, watermark_int_true, num_samples_per_class, loss, quad_loss, bound, gap = embed_random_watermark(CFG_WATERMARK.NUM_SAMPLES_PER_CLASS_CACHE_PATH, get_res_header(seed), CFG_BASIC.MODE, watermark)

    info = load_info_full(CFG_BASIC.DATA_DIR, CFG_BASIC.INFO_PATH, CFG_BASIC.TRANSFORM_LATENTS, embedding_path=CFG_EMBEDDING_MODEL.EMBEDDING_PATH, vae_path=CFG_VAE.PATH, device=CFG_BASIC.DEVICE, **get_embedding_module_param())
    _, full_latents, full_table, full_num, full_cat = syn_table(info, CFG_SYN.NUM_SAMPLES, CFG_CLUSTER.NUM_CLASSES, num_samples_per_class, CFG_DM.NUM_SAMPLE_STEPS, CFG_DM.PATH, CFG_BASIC.DEVICE, CFG_DM.CORRECT_GUIDANCE, get_original_centers(CFG_CLUSTER.CENTERS_PATH), CFG_CLUSTER.DIM_RATIO, CFG_CLUSTER.KEY)
    
    token_dim = 4
    working_dir = f'{CFG_BASIC.ROOT_DIR}/watermark/regeneration_attack_vae/{CFG_WATERMARK.WATERMARK}-token_dim{token_dim}/{CFG_BASIC.DATA_NAME}/{CFG_CLUSTER.ALGORITHM}/{CFG_CLUSTER.NUM_CLASSES}/{CFG_EMBEDDING_MODEL.to_str()}/{watermark_int_true}'
    if os.path.exists(working_dir):
      continue
    os.makedirs(working_dir, exist_ok=True)
    full_table.to_csv(f'{working_dir}/watermarked.csv', index=False)
    process_data(CFG_BASIC.DATA_NAME, f'{working_dir}/watermarked.csv')
    for ratio in [0.1]:
      reversed_table, reversed_num_norm, reversed_cat_norm, _, _ = attack(full_table, info, token_dim, working_dir, ratio)
      reversed_table.to_csv(f'{working_dir}/attacked_{ratio}.csv', index=False)
      watermark_int_extracted, watermark_int_pred, _, _ = watermark.extract(reversed_table, classifier=CFG_CLUSTER.CLASSIFIER, classifier_path=CFG_CLUSTER.CLASSIFIER_PATH, info=info, num=reversed_num_norm, cat=reversed_cat_norm, latents=None, centers_path=CFG_CLUSTER.CENTERS_PATH, device=CFG_BASIC.DEVICE, dim_ratio=CFG_CLUSTER.DIM_RATIO, key=CFG_CLUSTER.KEY)
      r = {**get_res_header(seed), 'use_cached_watermark_int': use_cached_watermark_int, 'loss': loss, 'quad_loss': quad_loss, 'bound': bound, 'gap': gap, **check(watermark_int_true, watermark_int_extracted, watermark_int_pred, CFG_WATERMARK.NUM_WATERMARK_BITS, CFG_WATERMARK.MAX_NUM_ERROR_BITS), 'attack_model': f'token_dim{token_dim}-ratio{ratio}'}
      with open(f'{CFG_BASIC.RESULTS_DIR}/regeneration_vae.json', 'a') as f:
        f.write(json.dumps(r) + '\n')
