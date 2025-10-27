import numpy as np
import os
import sys
import pandas as pd
import json
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

sys.path.append(os.path.join(os.getcwd(), './'))
sys.path.append(os.path.join(os.getcwd(), '../'))
sys.path.append(os.path.join(os.getcwd(), '../../'))
from globals import *


import numpy as np

def kmeans(latents: np.ndarray, num_classes: int, seed: int, num_inits: int = 1, max_iter: int = 300):
  km = KMeans(n_clusters=num_classes, random_state=seed, n_init=num_inits, max_iter=max_iter, verbose=True).fit(latents)
  labels = km.labels_
  score = silhouette_score(latents, labels)
  hist = pd.DataFrame(labels).value_counts().sort_index().to_list()
  var = np.var(hist)
  return labels.astype(np.int64), km.cluster_centers_, float(score), float(var)


def main():
  os.makedirs(CFG_CLUSTER.DIR, exist_ok=True)
  latents = load_train_latents_numpy()
  latents4cluster = get_latents4cluster_numpy(latents, CFG_CLUSTER.DIM_RATIO, CFG_CLUSTER.KEY)

  if os.path.exists(CFG_CLUSTER.LABELS_PATH):
    labels = np.loadtxt(CFG_CLUSTER.LABELS_PATH, dtype=np.int64)
    score = float(silhouette_score(latents4cluster, labels))
    var = float(np.var(pd.DataFrame(labels).value_counts().to_list()))
  else:
    if CFG_CLUSTER.ALGORITHM == 'kmeans':
      labels, centers, score, var = kmeans(latents4cluster, CFG_CLUSTER.NUM_CLASSES, CFG_BASIC.SEED)
    elif CFG_CLUSTER.ALGORITHM == 'kmeans10':
      labels, centers, score, var = kmeans(latents4cluster, CFG_CLUSTER.NUM_CLASSES, CFG_BASIC.SEED, 10)
    elif CFG_CLUSTER.ALGORITHM == 'kmeans100-1000':
      labels, centers, score, var = kmeans(latents4cluster, CFG_CLUSTER.NUM_CLASSES, CFG_BASIC.SEED, 100, 1000)
    else:
      assert False, f'Invalid Cluster Algorithm: {CFG_CLUSTER.ALGORITHM}'
    labels_hist = pd.DataFrame(labels).value_counts().sort_index().to_list()
    with open(CFG_CLUSTER.LABELS_HIST_PATH, 'w') as f:
      f.write(str(labels_hist))
    np.savetxt(CFG_CLUSTER.LABELS_PATH, labels, '%d')
    np.savetxt(CFG_CLUSTER.CENTERS_PATH, centers, '%.16f')
    labels_onehot = torch.nn.functional.one_hot(torch.from_numpy(labels), CFG_CLUSTER.NUM_CLASSES)
    np.savetxt(CFG_CLUSTER.ONEHOT_LABELS_PATH, labels_onehot, '%d')
  
  r = {**get_res_header(CFG_BASIC.SEED), 'silhouette_score': score, 'var': var}
  print('Finished !')

  with open(CFG_CLUSTER.METRIC_PATH, 'a') as f:
    f.write(json.dumps(r) + '\n')
