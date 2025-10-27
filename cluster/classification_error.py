import json
import sys
import os
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, silhouette_score, davies_bouldin_score, calinski_harabasz_score

sys.path.append('.')
sys.path.append('../')
from globals import *
from tabsyn.sample import syn_table, profile
from tabsyn.latent_utils import load_info_full, add_gauss_noise, alter_cat, alter_num, add_alter_num
from cluster.classify import predict_labels


def main():
  assert not os.path.exists(CFG_CLUSTER.CONFUSION_MAT_PATH), 'Already Exists !'
  BATCH_SIZE = {
    'beijing': 192000,
    'default': 81920,
    'shoppers': 81920,
    'phishing': 81920,
  }[CFG_BASIC.DATA_NAME]
  BATCH_SIZE = BATCH_SIZE // 4 * 3

  seed = get_random_seed()
  set_seed(seed)
  original_latents = load_train_latents_numpy()
  original_labels = load_train_labels_numpy()
  original_centers = get_original_centers(f'{CFG_CLUSTER.DIR}/center.txt')
  info = load_info_full(CFG_BASIC.DATA_DIR, CFG_BASIC.INFO_PATH, CFG_BASIC.TRANSFORM_LATENTS, embedding_path=CFG_EMBEDDING_MODEL.EMBEDDING_PATH, vae_path=CFG_VAE.PATH, device=CFG_BASIC.DEVICE, **get_embedding_module_param())

  # profile
  if not os.path.exists(f'{CFG_DM.DIR}/accuracy.txt'):
    labels_true = [i for _ in range(512) for i in range(CFG_CLUSTER.NUM_CLASSES)]
    labels_pred = []
    data_loader = DataLoader(labels_true, batch_size=BATCH_SIZE, num_workers=4, shuffle=False)
    for i, batch_labels_true in enumerate(data_loader):
      print(f'Batch {i + 1} / {len(data_loader)}', flush=True)
      batch_labels_pred = profile(batch_labels_true, info, CFG_CLUSTER.NUM_CLASSES, CFG_DM.PATH, CFG_BASIC.DEVICE, original_centers, CFG_CLUSTER.DIM_RATIO, CFG_CLUSTER.KEY).tolist()
      labels_pred += batch_labels_pred
    matrix = confusion_matrix(labels_true, labels_pred)
    accuracies = np.asarray([matrix[i, i] / sum(matrix[i]) for i in range(CFG_CLUSTER.NUM_CLASSES)])
    np.savetxt(f'{CFG_DM.DIR}/accuracy.txt', accuracies, '%5.3f')

  labels_true = [i for _ in range(CFG_WATERMARK.NUM_TESTED_SAMPLES_PER_CLASS) for i in range(CFG_CLUSTER.NUM_CLASSES)]
  labels_pred = []
  data_loader = DataLoader(labels_true, batch_size=BATCH_SIZE, num_workers=4, shuffle=False)
  for i, batch_labels_true in enumerate(data_loader):
    print(f'Batch {i + 1} / {len(data_loader)}', flush=True)
    _, latents, table, num, cat = syn_table(info, len(batch_labels_true), CFG_CLUSTER.NUM_CLASSES, None, CFG_DM.NUM_SAMPLE_STEPS, CFG_DM.PATH, CFG_BASIC.DEVICE, CFG_DM.CORRECT_GUIDANCE, original_centers, CFG_CLUSTER.DIM_RATIO, CFG_CLUSTER.KEY, False, False, batch_labels_true.cpu().detach().numpy())

    if CFG_WATERMARK.GAUSS_NOISE_RATIO > 0 and np.isclose(CFG_WATERMARK.ALTERATION_RATIO, 0):
      _, num = add_gauss_noise(table, info, CFG_WATERMARK.GAUSS_NOISE_RATIO)
    elif np.isclose(CFG_WATERMARK.GAUSS_NOISE_RATIO, 0) and CFG_WATERMARK.ALTERATION_RATIO > 0:
      _, num = alter_num(table, info, CFG_WATERMARK.ALTERATION_RATIO)
      _, cat = alter_cat(table, info, CFG_WATERMARK.ALTERATION_RATIO)
    elif CFG_WATERMARK.GAUSS_NOISE_RATIO > 0 and CFG_WATERMARK.ALTERATION_RATIO > 0:
      _, num = add_alter_num(table, info, CFG_WATERMARK.GAUSS_NOISE_RATIO, CFG_WATERMARK.ALTERATION_RATIO)
      _, cat = alter_cat(table, info, CFG_WATERMARK.ALTERATION_RATIO)
    batch_labels_pred = predict_labels(table, classifier=CFG_CLUSTER.CLASSIFIER, classifier_path=CFG_CLUSTER.CLASSIFIER_PATH, info=info, num=num, cat=cat, latents=latents.cpu().detach().numpy(), centers_path=CFG_CLUSTER.CENTERS_PATH, device=CFG_BASIC.DEVICE, key=CFG_CLUSTER.KEY, dim_ratio=CFG_CLUSTER.DIM_RATIO)
    labels_pred += batch_labels_pred.tolist()

  matrix = confusion_matrix(labels_true, labels_pred)
  np.savetxt(CFG_CLUSTER.CONFUSION_MAT_PATH, matrix, '%7d')
  accuracy = accuracy_score(labels_true, labels_pred)
  f1 = f1_score(labels_true, labels_pred, average='macro')
  sc = silhouette_score(original_latents, original_labels)
  dbs = davies_bouldin_score(original_latents, original_labels)
  chs = calinski_harabasz_score(original_latents, original_labels)

  r = {**get_res_header(seed),
        'accuracy': float(accuracy),
        'macro_f1': float(f1),
        'silhouette_score': float(sc), 'davies_bouldin_score': float(dbs), 'calinski_harabasz_score': float(chs),
        }
  with open(f'{CFG_CLUSTER.DIR}/accuracy.json', 'a') as f:
    f.write(json.dumps(r) + '\n')
  print('Finished !')
