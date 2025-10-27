import random
import numpy as np
import pandas as pd
import torch
import argparse
import getpass
import time
import psutil
from scipy.special import comb
from scipy.optimize import brentq

FINAL = 'final'
QUERY = '1-all-sep-0.01-0.05-0.2'

class CFG_BASIC:
  USE_ARG = psutil.Process().parent().name().__contains__('systemd')
  SEED = 666
  GPU_ID = 3
  DEVICE = 'cuda'
  MODE = 'watermark_regeneration_vae'
  TRANSFORM_LATENTS = 'none'
  DATA_NAME = 'beijing'
  DATA_DIR = ''
  INFO_PATH = ''
  COLUMN_INFO_PATH = ''
  ROOT_DIR = f'/home/{getpass.getuser()}/TableMark'
  EVAL_CSV_PATH = ''
  RESULTS_DIR = f'{ROOT_DIR}/watermark/results_{FINAL}'

class CFG_CLUSTER:
  ALGORITHM = 'kmeans100-1000'
  NUM_CLASSES = 256
  CLASSIFIER = 'nn'
  NUM_GMM_CLUSTER_TRIALS = 10
  KEY = 985
  DIM_RATIO = 'correct-pca-0.99'
  NORM = False
  VAE_DIR = ''
  VAE_LABELS_PATH = ''
  VAE_ONEHOT_LABELS_PATH = ''
  VAE_CENTERS_PATH = ''
  VAE_CLASSIFIER_PATH = ''
  DIR = ''
  GROUP_PATH = ''
  CLASS_PATH = ''
  LABELS_PATH = ''
  LABELS_HIST_PATH = ''
  ONEHOT_LABELS_PATH = ''
  CENTERS_PATH = ''
  CLASSIFIER_PATH = ''
  REDUCED_LATENTS_2D_PATH = ''
  REDUCED_LATENTS_3D_PATH = ''
  METRIC_PATH = ''
  CONFUSION_MAT_PATH = ''
  EMD_PATH = ''
  DIM_INFO_DIR = ''

class CFG_SYN:
  NUM_SAMPLES = {
    'beijing': 37581,
    'default': 27000,
    'shoppers': 11097,
    'phishing': 9949,
  }[CFG_BASIC.DATA_NAME]
  NUM_SAMPLES_PER_CLASS = {
    2: [16280, 16281],
    3: [10853, 10853, 10855],
    5: [6512, 6512, 6512, 6512, 6513],
    10: [3256, 3256, 3256, 3256, 3256, 3256, 3256, 3256, 3256, 3257],
    15: [3462, 2992, 2679, 2408, 2346, 2079, 1711, 1611, 1532, 1353, 1199, 1151, 1075, 780, 622],
    25: None,
    20: [1628, 1628, 1628, 1628, 1628, 1628, 1628, 1628, 1628, 1628, 1628, 1628, 1628, 1628, 1628, 1628, 1628, 1628, 1628, 1629],
    25: None,
    30: None,
    64: None,
    128: [278, 538, 396, 455, 184, 99, 105, 305, 435, 209, 261, 174, 232, 74, 299, 350, 125, 263, 398, 366, 121, 351, 389, 242, 349, 226, 313, 332, 186, 146, 413, 314, 189, 348, 314, 390, 129, 226, 264, 107, 255, 189, 144, 404, 221, 380, 363, 323, 351, 272, 165, 251, 395, 284, 121, 172, 309, 227, 191, 245, 258, 196, 344, 194, 269, 174, 200, 333, 87, 104, 252, 622, 314, 378, 93, 241, 539, 200, 301, 195, 291, 72, 282, 530, 338, 134, 207, 290, 369, 199, 131, 242, 159, 127, 74, 191, 251, 111, 236, 129, 252, 139, 417, 276, 299, 511, 117, 222, 151, 205, 346, 185, 289, 282, 234, 125, 214, 140, 257, 273, 169, 106, 417, 118, 108, 150, 255, 490],
    256: None,
  }[CFG_CLUSTER.NUM_CLASSES]
  DIR = ''
  UNCONDITIONAL_DIR = ''
  SYN_DATA_PATH = ''
  SYN_LABELS_PATH = ''
  REAL_DATA_PATH = ''
  TEST_DATA_PATH = ''
  EVAL_RES_PATH = ''
  @staticmethod
  def to_str():
    return f'{CFG_SYN.NUM_SAMPLES}_{str(CFG_SYN.NUM_SAMPLES_PER_CLASS).replace("[", "").replace("]", "").replace(", ", "-")}'[:200]

class CFG_WATERMARK:
  NUM_USERS = 1000
  WATERMARK = 'pair_compare_one_pair'
  # WATERMARK = 'tabwak_partition'
  # WATERMARK = 'freqwm'
  # WATERMARK = 'tabular_mark'
  # WATERMARK = 'none'
  NUM_WATERMARK_BITS = 32
  MIN_HAMMING_DIST = 7
  MAX_NUM_ERROR_BITS = 1
  RATIO_NUM_SAMPLES_PER_CLASS_INTERVAL = -30000
  BIT_ERROR_RATE = 0.01
  FP_RATE = 0.0001
  ERROR_RATE = 0.001
  GAUSS_NOISE_RATIO = 0.01
  ALTERATION_RATIO = 0
  DELETION_RATE = 0.1
  QUALITY_LOSS = f'quad_random_init{CFG_CLUSTER.KEY}'
  TIME_LIMIT = 180
  MIN_GAP = 0.01
  CALLBACK_MODE = 'none'
  NUM_TESTED_SAMPLES_PER_CLASS = 25600
  TAO_APPROXIMATION = 0.0
  QUALITY_MODE = 'average'
  INIT_RATIO_NUM_SAMPLES_PER_CLASS_INTERVAL = 0
  # GEN_CODE_LOSS = 'none'
  GEN_CODE_LOSS = 'general_bfs'
  # NUM_SAMPLES_PER_CLASS_LOWER_BOUND = '1stage_final-0.01-0.01'
  NUM_SAMPLES_PER_CLASS_LOWER_BOUND = '6stage_splus-0.01-0.01'
  # NUM_SAMPLES_PER_CLASS_LOWER_BOUND = '-1'
  NUM_SAMPLES_PER_CLASS_UPPER_BOUND = NUM_SAMPLES_PER_CLASS_LOWER_BOUND
  NUM_WATERMARK_TRIALS = 1
  ATTACK_EMBEDDING_MODEL = f'original-num_layers2-{FINAL}'
  DIR = ''
  CODE_PATH = ''
  CODE_LOSS_PATH = ''
  NUM_SAMPLES_PER_CLASS_CACHE_PATH = ''
  NUM_SAMPLES_PER_CLASS_CACHE_USAGE_PATH = ''
  CLUSTER_ATTACK_SYN_DIR = ''
  REGENERATION_ATTACK_SYN_DIR = ''
  @staticmethod
  def to_str():
    return f'nu{CFG_WATERMARK.NUM_USERS}-w{CFG_WATERMARK.WATERMARK}-nwb{CFG_WATERMARK.NUM_WATERMARK_BITS}-mhd{CFG_WATERMARK.MIN_HAMMING_DIST}-mneb{CFG_WATERMARK.MAX_NUM_ERROR_BITS}-rnspci{CFG_WATERMARK.RATIO_NUM_SAMPLES_PER_CLASS_INTERVAL}-er{CFG_WATERMARK.ERROR_RATE}-ber{CFG_WATERMARK.BIT_ERROR_RATE}-dr{CFG_WATERMARK.DELETION_RATE}-ql{CFG_WATERMARK.QUALITY_LOSS}-tl{CFG_WATERMARK.TIME_LIMIT}-mg{CFG_WATERMARK.MIN_GAP}-cm{CFG_WATERMARK.CALLBACK_MODE}-ntspc{CFG_WATERMARK.NUM_TESTED_SAMPLES_PER_CLASS}-ta{CFG_WATERMARK.TAO_APPROXIMATION}-irnspci{CFG_WATERMARK.INIT_RATIO_NUM_SAMPLES_PER_CLASS_INTERVAL}-gcl{CFG_WATERMARK.GEN_CODE_LOSS}-lb{CFG_WATERMARK.NUM_SAMPLES_PER_CLASS_LOWER_BOUND}-ub{CFG_WATERMARK.NUM_SAMPLES_PER_CLASS_UPPER_BOUND}'

dataname2freqwm = {
  'beijing': (5, 'dyn', 53),
  'default': (5, 'dyn', 53),
  'shoppers': (5, 'dyn', 7),
  'phishing': (5, 'dyn', 8),
}
class CFG_FREQWM:
  B = 5
  T = 0
  Z = 53
  @staticmethod
  def to_str() -> str:
    return f'freqwm-{CFG_FREQWM.T}-{CFG_FREQWM.Z}-{CFG_FREQWM.B}'

class CFG_TABULAR_MARK:
  NUM_CELLS_RATIO = 0.15
  P_RATIO = 0.2
  NUM_UNITS = 2
  @staticmethod
  def to_str() -> str:
    return f'tabular_mark-{CFG_TABULAR_MARK.NUM_CELLS_RATIO}-{CFG_TABULAR_MARK.P_RATIO}-{CFG_TABULAR_MARK.NUM_UNITS}'

class CFG_TABWAK:
  DM_DIR = ''
  DM_PATH = ''

class CFG_VAE:
  TOKEN_DIM = 4
  NUM_LAYERS = 2
  NUM_HEADS = 1
  FACTOR = 32
  TOKEN_BIAS = True
  NUM_EPOCHS = 4000
  BATCH_SIZE = 4096
  GAUSS_RATIO = 0
  ALTER_RATIO = 0
  DIR = ''
  PATH = ''
  DECODER_PATH = ''
  ENCODER_PATH = ''
  EMBEDDING_PATH = ''
  @staticmethod
  def to_str():
    s = f'original-num_layers{CFG_VAE.NUM_LAYERS}-{FINAL}' + (f'-token_dim{CFG_VAE.TOKEN_DIM}' if CFG_VAE.TOKEN_DIM != 4 else '')
    if CFG_VAE.GAUSS_RATIO > 0 or CFG_VAE.ALTER_RATIO > 0:
      s += f'-gauss{CFG_VAE.GAUSS_RATIO}-alter{CFG_VAE.ALTER_RATIO}'
    return s
  
class CFG_DM:
  NUM_EPOCHS = 10000
  BATCH_SIZE = 4096
  NUM_SAMPLE_STEPS = 50
  CORRECT_GUIDANCE = True
  DIR = ''
  PATH = ''

# -------------------------------------------------------------------------------------------------------------------------------------------

if CFG_BASIC.USE_ARG:
  parser = argparse.ArgumentParser()
  parser.add_argument('--gpu_id', type=int)
  parser.add_argument('--mode', type=str)
  parser.add_argument('--dataname', type=str)
  parser.add_argument('--cluster_algorithm', type=str)
  parser.add_argument('--num_classes', type=int)
  parser.add_argument('--dim_ratio', type=str)
  parser.add_argument('--classifier', type=str)
  parser.add_argument('--num_samples', type=int)
  parser.add_argument('--num_samples_per_class', type=int, nargs='*')
  parser.add_argument('--num_users', type=int)
  parser.add_argument('--watermark', type=str)
  parser.add_argument('--num_watermark_bits', type=int)
  # parser.add_argument('--min_hamming_dist', type=int)
  parser.add_argument('--ratio_num_samples_per_class_interval', type=float)
  parser.add_argument('--num_watermark_trials', type=int)
  parser.add_argument('--quality_loss', type=str)
  parser.add_argument('--transform_latents', type=str)
  parser.add_argument('--weight_intra', type=float)
  parser.add_argument('--weight_inter', type=float)
  parser.add_argument('--center_decay', type=float)
  parser.add_argument('--loss_intra', type=str)
  parser.add_argument('--ratio_intra', type=float)
  parser.add_argument('--ratio_inter', type=float)
  parser.add_argument('--alpha', type=float)
  parser.add_argument('--weight_cluster', type=str)
  parser.add_argument('--weight_size', type=float)
  parser.add_argument('--weight_dist', type=float)
  parser.add_argument('--quantile_dist', type=float)
  parser.add_argument('--transform_latents_num_epochs', type=int)
  parser.add_argument('--loss_size', type=str)
  parser.add_argument('--loss_cluster', type=str)
  parser.add_argument('--transform_latents_batch_size', type=int)
  parser.add_argument('--max_mse', type=float)
  parser.add_argument('--max_ce', type=float)
  parser.add_argument('--aemlp_hidden_dim', type=int)
  parser.add_argument('--aemlp_out_dim', type=int)
  parser.add_argument('--aemlp_num_layers', type=int)
  parser.add_argument('--deletion_rate', type=float)
  parser.add_argument('--max_watermark_error_rate', type=float)
  parser.add_argument('--time_limit', type=int)
  parser.add_argument('--min_gap', type=float)
  parser.add_argument('--num_tested_samples_per_class', type=int)
  parser.add_argument('--tao_approximation', type=str)
  parser.add_argument('--init_ratio_num_samples_per_class_interval', type=float)
  parser.add_argument('--quality_mode', type=str)
  parser.add_argument('--correct_guidance', type=int)
  parser.add_argument('--gen_code_loss', type=str)
  parser.add_argument('--token_dim', type=int)
  parser.add_argument('--cluster_attack_embedding_model', type=str)
  parser.add_argument('--norm', type=int)
  parser.add_argument('--num_samples_per_class_lower_bound', type=str)
  parser.add_argument('--gauss', type=float)
  parser.add_argument('--alter', type=float)
  args = parser.parse_args()
  CFG_BASIC.GPU_ID = args.gpu_id
  CFG_BASIC.MODE = args.mode
  CFG_BASIC.DATA_NAME = args.dataname

  assert args.norm is not None
  CFG_CLUSTER.NORM = bool(args.norm)

  assert args.gauss is not None
  assert args.alter is not None
  CFG_WATERMARK.GAUSS_NOISE_RATIO = args.gauss
  CFG_WATERMARK.ALTERATION_RATIO = args.alter

  if CFG_BASIC.MODE == 'watermark_cluster':
    assert args.cluster_attack_embedding_model is not None
    CFG_WATERMARK.ATTACK_EMBEDDING_MODEL = args.cluster_attack_embedding_model

  if CFG_BASIC.MODE.__contains__('tabwak') or CFG_BASIC.MODE == 'vae_train' or (CFG_BASIC.MODE == 'watermark_detection' and args.watermark.__contains__('tabwak')):
    assert args.token_dim is not None
    CFG_VAE.TOKEN_DIM = args.token_dim

  if CFG_BASIC.MODE in ['gen_code', 'transform_latents', 'cluster', 'tabsyn_train', 'sample', 'eval', 'classify', 'watermark_detection', 'watermark_detection_tnr', 'watermark_quality', 'watermark_sample_deletion', 'watermark_gauss_noise', 'watermark_gauss_noise_label', 'watermark_noise_deletion', 'watermark_uniform_noise', 'watermark_laplace_noise', 'watermark_sample_insertion', 'watermark_regeneration_vae', 'watermark_alteration', 'watermark_alteration_label', 'watermark_cluster', 'classification_error', 'delta_approximation']:
    CFG_BASIC.TRANSFORM_LATENTS = args.transform_latents

  if CFG_BASIC.MODE in ['gen_code', 'transform_latents', 'cluster', 'classifier_train', 'tabsyn_train', 'sample', 'eval', 'classify', 'watermark_detection', 'watermark_detection_tnr', 'watermark_quality', 'watermark_sample_deletion', 'watermark_gauss_noise', 'watermark_gauss_noise_label', 'watermark_alteration', 'watermark_alteration_label', 'watermark_noise_deletion', 'watermark_uniform_noise', 'watermark_laplace_noise', 'watermark_sample_insertion', 'watermark_regeneration_vae', 'watermark_cluster', 'classification_error', 'delta_approximation', 'reducer_train_ae', 'reducer_train_mlp']:
    assert args.cluster_algorithm is not None
    assert args.num_classes is not None
    assert args.dim_ratio is not None
    CFG_CLUSTER.ALGORITHM = args.cluster_algorithm
    CFG_CLUSTER.NUM_CLASSES = args.num_classes
    CFG_CLUSTER.DIM_RATIO = args.dim_ratio
  
  if CFG_BASIC.MODE in ['gen_code', 'watermark_detection', 'watermark_detection_tnr', 'watermark_quality', 'watermark_sample_deletion', 'watermark_gauss_noise', 'watermark_gauss_noise_label', 'watermark_noise_deletion', 'watermark_uniform_noise', 'watermark_laplace_noise', 'watermark_sample_insertion', 'watermark_regeneration_vae', 'watermark_alteration', 'watermark_alteration_label', 'watermark_cluster']:
    assert args.gen_code_loss is not None
    CFG_WATERMARK.GEN_CODE_LOSS = args.gen_code_loss

  if CFG_BASIC.MODE in ['classifier_train', 'classify', 'watermark_detection', 'watermark_detection_tnr', 'watermark_sample_deletion', 'watermark_gauss_noise', 'watermark_gauss_noise_label', 'watermark_noise_deletion', 'watermark_uniform_noise', 'watermark_laplace_noise', 'watermark_sample_insertion', 'watermark_regeneration_vae', 'watermark_alteration', 'watermark_alteration_label', 'classification_error', 'delta_approximation', 'watermark_cluster']:
    assert args.classifier is not None
    CFG_CLUSTER.CLASSIFIER = args.classifier

  if CFG_BASIC.MODE in ['sample', 'eval', 'classify', 'watermark_detection', 'watermark_detection_tnr', 'watermark_quality', 'watermark_sample_deletion', 'watermark_gauss_noise', 'watermark_gauss_noise_label', 'watermark_noise_deletion', 'watermark_uniform_noise', 'watermark_laplace_noise', 'watermark_sample_insertion', 'watermark_regeneration_vae', 'watermark_alteration', 'watermark_alteration_label', 'watermark_cluster']:
    assert args.num_samples is not None
    CFG_SYN.NUM_SAMPLES = args.num_samples

  if CFG_BASIC.MODE in ['sample', 'eval', 'classify', 'watermark_detection', 'watermark_detection_tnr', 'watermark_quality', 'watermark_sample_deletion', 'watermark_gauss_noise', 'watermark_gauss_noise_label', 'watermark_noise_deletion', 'watermark_uniform_noise', 'watermark_laplace_noise', 'watermark_sample_insertion', 'watermark_regeneration_vae', 'watermark_alteration', 'watermark_alteration_label', 'watermark_cluster', 'classification_error', 'delta_approximation']:
    assert args.correct_guidance is not None
    CFG_DM.CORRECT_GUIDANCE = bool(args.correct_guidance)

  if CFG_BASIC.MODE in ['sample', 'eval', 'classify']:
    assert args.num_samples_per_class is not None
    CFG_SYN.NUM_SAMPLES_PER_CLASS = args.num_samples_per_class

  if CFG_BASIC.MODE in ['gen_code', 'watermark_detection', 'watermark_detection_tnr', 'watermark_quality', 'watermark_sample_deletion', 'watermark_gauss_noise', 'watermark_gauss_noise_label', 'watermark_noise_deletion', 'watermark_uniform_noise', 'watermark_laplace_noise', 'watermark_sample_insertion', 'watermark_regeneration_vae', 'watermark_alteration', 'watermark_alteration_label', 'watermark_cluster']:
    assert args.num_users is not None
    assert args.watermark is not None
    assert args.num_watermark_bits is not None
    # assert args.min_hamming_dist is not None
    assert args.quality_loss is not None
    CFG_WATERMARK.NUM_USERS = args.num_users
    CFG_WATERMARK.WATERMARK = args.watermark
    CFG_WATERMARK.NUM_WATERMARK_BITS = args.num_watermark_bits
    # CFG_WATERMARK.MIN_HAMMING_DIST = args.min_hamming_dist
    CFG_WATERMARK.QUALITY_LOSS = args.quality_loss

  if CFG_BASIC.MODE in ['classification_error', 'watermark_detection', 'watermark_detection_tnr', 'watermark_quality', 'watermark_sample_deletion', 'watermark_gauss_noise', 'watermark_gauss_noise_label', 'watermark_noise_deletion', 'watermark_uniform_noise', 'watermark_laplace_noise', 'watermark_sample_insertion', 'watermark_regeneration_vae', 'watermark_alteration', 'watermark_alteration_label', 'watermark_cluster']:
    assert args.num_tested_samples_per_class is not None
    CFG_WATERMARK.NUM_TESTED_SAMPLES_PER_CLASS = args.num_tested_samples_per_class

  if CFG_BASIC.MODE in ['watermark_detection', 'watermark_detection_tnr', 'watermark_quality', 'watermark_sample_deletion', 'watermark_gauss_noise', 'watermark_gauss_noise_label', 'watermark_noise_deletion', 'watermark_uniform_noise', 'watermark_laplace_noise', 'watermark_sample_insertion', 'watermark_regeneration_vae', 'watermark_alteration', 'watermark_alteration_label', 'watermark_cluster']:
    assert args.ratio_num_samples_per_class_interval is not None
    assert args.num_watermark_trials is not None
    assert args.max_watermark_error_rate is not None
    assert args.deletion_rate is not None
    assert args.time_limit is not None
    assert args.min_gap is not None
    assert args.tao_approximation is not None
    assert args.init_ratio_num_samples_per_class_interval is not None
    assert args.num_samples_per_class_lower_bound is not None
    CFG_WATERMARK.RATIO_NUM_SAMPLES_PER_CLASS_INTERVAL = args.ratio_num_samples_per_class_interval
    CFG_WATERMARK.NUM_WATERMARK_TRIALS = args.num_watermark_trials
    CFG_WATERMARK.ERROR_RATE = args.max_watermark_error_rate
    CFG_WATERMARK.DELETION_RATE = args.deletion_rate
    CFG_WATERMARK.TIME_LIMIT = args.time_limit
    CFG_WATERMARK.MIN_GAP = args.min_gap
    CFG_WATERMARK.TAO_APPROXIMATION = float(args.tao_approximation)
    CFG_WATERMARK.INIT_RATIO_NUM_SAMPLES_PER_CLASS_INTERVAL = args.init_ratio_num_samples_per_class_interval
    CFG_WATERMARK.NUM_SAMPLES_PER_CLASS_LOWER_BOUND = args.num_samples_per_class_lower_bound
    CFG_WATERMARK.NUM_SAMPLES_PER_CLASS_UPPER_BOUND = args.num_samples_per_class_lower_bound
    if CFG_BASIC.MODE == 'watermark_quality':
      assert args.quality_mode is not None
      CFG_WATERMARK.QUALITY_MODE = args.quality_mode

if CFG_BASIC.MODE == 'watermark_detection' and CFG_WATERMARK.WATERMARK == 'tabwak_partition' and CFG_WATERMARK.NUM_WATERMARK_BITS > 22:
  if CFG_BASIC.DATA_NAME in ['beijing']:
    CFG_VAE.TOKEN_DIM = 6
  else:
    CFG_VAE.TOKEN_DIM = 4
else:
  CFG_VAE.TOKEN_DIM = 4

if CFG_BASIC.MODE in ['gen_code', 'watermark_detection', 'watermark_detection_tnr', 'watermark_quality', 'watermark_sample_deletion', 'watermark_gauss_noise', 'watermark_gauss_noise_label', 'watermark_alteration', 'watermark_alteration_label', 'watermark_noise_deletion', 'watermark_uniform_noise', 'watermark_laplace_noise', 'watermark_sample_insertion', 'watermark_regeneration_vae']:
  assert 2 ** CFG_WATERMARK.NUM_WATERMARK_BITS >= CFG_WATERMARK.NUM_USERS

CFG_BASIC.DEVICE = f'cuda:{CFG_BASIC.GPU_ID}' if CFG_BASIC.DEVICE != 'cpu' else 'cpu'
CFG_BASIC.DATA_DIR = f'{CFG_BASIC.ROOT_DIR}/data/{CFG_BASIC.DATA_NAME}'
CFG_BASIC.INFO_PATH = f'{CFG_BASIC.DATA_DIR}/info.json'
CFG_BASIC.COLUMN_INFO_PATH = f'{CFG_BASIC.DATA_DIR}/{CFG_BASIC.DATA_NAME}.json'
CFG_BASIC.EVAL_CSV_PATH = f'{CFG_BASIC.ROOT_DIR}/out/{CFG_BASIC.DATA_NAME}_{CFG_FREQWM.to_str()}_{CFG_BASIC.GPU_ID}.csv' if CFG_WATERMARK.WATERMARK == 'freqwm' else f'{CFG_BASIC.ROOT_DIR}/out/{CFG_BASIC.DATA_NAME}_{CFG_TABULAR_MARK.to_str()}_{CFG_BASIC.GPU_ID}.csv' if CFG_WATERMARK.WATERMARK == 'tabular_mark' else f'{CFG_BASIC.ROOT_DIR}/out/{CFG_BASIC.DATA_NAME}_{CFG_WATERMARK.WATERMARK}_{CFG_BASIC.GPU_ID}.csv' if CFG_WATERMARK.WATERMARK.__contains__('tabwak') else f'{CFG_BASIC.ROOT_DIR}/out/{CFG_BASIC.DATA_NAME}_{CFG_WATERMARK.WATERMARK}_{CFG_WATERMARK.QUALITY_LOSS}_{CFG_WATERMARK.NUM_WATERMARK_BITS}_{CFG_WATERMARK.RATIO_NUM_SAMPLES_PER_CLASS_INTERVAL}_{CFG_WATERMARK.DELETION_RATE}_{CFG_WATERMARK.GAUSS_NOISE_RATIO}_{CFG_WATERMARK.ALTERATION_RATIO}_{CFG_BASIC.GPU_ID}_{CFG_WATERMARK.NUM_WATERMARK_TRIALS}.csv'

CFG_EMBEDDING_MODEL = {
  'none': CFG_VAE,
}[CFG_BASIC.TRANSFORM_LATENTS]
CFG_VAE.DIR = f'{CFG_BASIC.ROOT_DIR}/tabsyn/vae/ckpt/{CFG_BASIC.DATA_NAME}/{CFG_VAE.to_str()}'
CFG_VAE.PATH = f'{CFG_VAE.DIR}/model.pt'
CFG_VAE.ENCODER_PATH = f'{CFG_VAE.DIR}/encoder.pt'
CFG_VAE.DECODER_PATH = f'{CFG_VAE.DIR}/decoder.pt'
CFG_VAE.EMBEDDING_PATH = f'{CFG_VAE.DIR}/train_z.npy'

def first_int(s: str, sub: str) -> int:
  start = s.find(sub)
  assert start != -1
  start += len(sub)
  num_s = ''
  while start < len(s) and '0' <= s[start] <= '9':
    num_s += s[start]
    start += 1
  return int(num_s)

def first_float(s: str, sub: str) -> float:
  start = s.find(sub)
  assert start != -1
  start += len(sub)
  num_s = ''
  while start < len(s) and ('0' <= s[start] <= '9' or s[start] == '.'):
    num_s += s[start]
    start += 1
  return float(num_s)

CFG_CLUSTER.DIR = f'{CFG_BASIC.ROOT_DIR}/cluster/{CFG_BASIC.DATA_NAME}/{CFG_CLUSTER.ALGORITHM}/{CFG_CLUSTER.NUM_CLASSES}/{CFG_EMBEDDING_MODEL.to_str()}'
if CFG_CLUSTER.DIM_RATIO != '1':
  CFG_CLUSTER.DIR = f'{CFG_CLUSTER.DIR}/{CFG_CLUSTER.DIM_RATIO}-{CFG_CLUSTER.KEY}'
if CFG_CLUSTER.NORM:
  CFG_CLUSTER.DIR += '/norm'
CFG_CLUSTER.GROUP_PATH = f'{CFG_CLUSTER.DIR}/combined_cluster_{CFG_WATERMARK.NUM_WATERMARK_BITS * 2}.txt'
CFG_CLUSTER.CLASS_PATH = f'{CFG_CLUSTER.DIR}/class4one_pair_{CFG_WATERMARK.NUM_WATERMARK_BITS * 2}_average_num_std.txt'
CFG_CLUSTER.LABELS_PATH = f'{CFG_CLUSTER.DIR}/label.txt'
CFG_CLUSTER.LABELS_HIST_PATH = f'{CFG_CLUSTER.DIR}/label_hist.txt'
CFG_CLUSTER.ONEHOT_LABELS_PATH = f'{CFG_CLUSTER.DIR}/label_onehot.txt'
CFG_CLUSTER.CENTERS_PATH = f'{CFG_CLUSTER.DIR}/center.txt'
CFG_CLUSTER.CLASSIFIER_PATH = f'{CFG_CLUSTER.DIR}/{CFG_CLUSTER.CLASSIFIER}.joblib'
CFG_CLUSTER.VAE_DIR = f'{CFG_BASIC.ROOT_DIR}/cluster/{CFG_BASIC.DATA_NAME}/{CFG_CLUSTER.ALGORITHM}/{CFG_CLUSTER.NUM_CLASSES}/{CFG_VAE.to_str()}'
CFG_CLUSTER.VAE_LABELS_PATH = f'{CFG_CLUSTER.VAE_DIR}/label.txt'
CFG_CLUSTER.VAE_ONEHOT_LABELS_PATH = f'{CFG_CLUSTER.VAE_DIR}/label_onehot.txt'
CFG_CLUSTER.VAE_CENTERS_PATH = f'{CFG_CLUSTER.VAE_DIR}/center.txt'
CFG_CLUSTER.VAE_CLASSIFIER_PATH = f'{CFG_CLUSTER.VAE_DIR}/{CFG_CLUSTER.CLASSIFIER}.joblib'
CFG_CLUSTER.REDUCED_LATENTS_2D_PATH = f'{CFG_CLUSTER.DIR}/reduced_latent_2D.txt'
CFG_CLUSTER.REDUCED_LATENTS_3D_PATH = f'{CFG_CLUSTER.DIR}/reduced_latent_3D.txt'
CFG_CLUSTER.METRIC_PATH = f'{CFG_BASIC.ROOT_DIR}/cluster/{CFG_BASIC.DATA_NAME}/cluster.json'
CFG_CLUSTER.CONFUSION_MAT_PATH = f'{CFG_CLUSTER.DIR}/confusion_matrix_nn_{CFG_WATERMARK.NUM_TESTED_SAMPLES_PER_CLASS}_{CFG_DM.CORRECT_GUIDANCE}_{CFG_WATERMARK.GAUSS_NOISE_RATIO:g}_{CFG_WATERMARK.ALTERATION_RATIO:g}.txt'
CFG_CLUSTER.EMD_PATH = f'{CFG_CLUSTER.DIR}/emd.txt'
CFG_CLUSTER.DIM_INFO_DIR = f'{CFG_BASIC.ROOT_DIR}/cluster/{CFG_BASIC.DATA_NAME}/kmeans/{CFG_CLUSTER.NUM_CLASSES}/{CFG_EMBEDDING_MODEL.to_str()}'

CFG_DM.DIR = f'{CFG_BASIC.ROOT_DIR}/tabsyn/ckpt/{CFG_BASIC.DATA_NAME}/{CFG_CLUSTER.ALGORITHM}/{CFG_CLUSTER.NUM_CLASSES}/{CFG_EMBEDDING_MODEL.to_str()}'
if CFG_CLUSTER.DIM_RATIO!= '1':
  CFG_DM.DIR = f'{CFG_DM.DIR}/{CFG_CLUSTER.DIM_RATIO}-{CFG_CLUSTER.KEY}'
if CFG_CLUSTER.NORM:
  CFG_DM.DIR += '/norm'
CFG_DM.PATH = f'{CFG_DM.DIR}/model.pt'

if CFG_BASIC.MODE in ['sample', 'eval']:
  with open(f'{CFG_CLUSTER.DIR}/label_hist.txt', 'r') as f:
    CFG_SYN.NUM_SAMPLES_PER_CLASS = list(map(int, f.readline().replace('[', '').replace(']', '').replace(',', ' ').replace('  ', ' ').split()))
CFG_SYN.DIR = f'{CFG_BASIC.ROOT_DIR}/synthetic/{CFG_BASIC.DATA_NAME}/{CFG_CLUSTER.ALGORITHM}/{CFG_CLUSTER.NUM_CLASSES}/{CFG_EMBEDDING_MODEL.to_str()}'
CFG_SYN.UNCONDITIONAL_DIR = f'{CFG_BASIC.ROOT_DIR}/synthetic/{CFG_BASIC.DATA_NAME}/{CFG_CLUSTER.ALGORITHM}/1/{CFG_EMBEDDING_MODEL.to_str()}/{CFG_SYN.NUM_SAMPLES}_{CFG_SYN.NUM_SAMPLES}'
if CFG_CLUSTER.DIM_RATIO != '1':
  CFG_SYN.DIR = f'{CFG_SYN.DIR}/{CFG_CLUSTER.DIM_RATIO}-{CFG_CLUSTER.KEY}'
  CFG_SYN.UNCONDITIONAL_DIR = f'{CFG_SYN.UNCONDITIONAL_DIR}/{CFG_CLUSTER.DIM_RATIO}-{CFG_CLUSTER.KEY}'
if CFG_CLUSTER.NORM:
  CFG_SYN.DIR += '/norm'
  CFG_SYN.UNCONDITIONAL_DIR += '/norm'
CFG_SYN.DIR += f'/{CFG_SYN.to_str()}'
CFG_SYN.SYN_DATA_PATH = f'{CFG_SYN.DIR}/tabsyn.csv'
CFG_SYN.SYN_LABELS_PATH = f'{CFG_SYN.DIR}/label.csv'
CFG_SYN.REAL_DATA_PATH = f'{CFG_BASIC.ROOT_DIR}/synthetic/{CFG_BASIC.DATA_NAME}/real.csv'
CFG_SYN.TEST_DATA_PATH = f'{CFG_BASIC.ROOT_DIR}/synthetic/{CFG_BASIC.DATA_NAME}/test.csv'
CFG_SYN.EVAL_RES_PATH = f'{CFG_SYN.DIR}/quality.json'

if CFG_WATERMARK.QUALITY_LOSS.endswith('random_init'):
  CFG_WATERMARK.QUALITY_LOSS = f'{CFG_WATERMARK.QUALITY_LOSS}{CFG_CLUSTER.KEY}'
CFG_WATERMARK.MAX_NUM_ERROR_BITS = {
  (10, 32): 5,
  (100, 32): 4,
  (1000, 32): 3,
  (10000, 32): 2,
  (100000, 32): 1,
}[(CFG_WATERMARK.NUM_USERS, CFG_WATERMARK.NUM_WATERMARK_BITS)]
def get_fpr(k: int, tao: int, n: int) -> float:
  single = sum(comb(k, i) for i in range(tao, k + 1)) / (2 ** k)
  return single * n
CFG_WATERMARK.FP_RATE = get_fpr(CFG_WATERMARK.NUM_WATERMARK_BITS, CFG_WATERMARK.NUM_WATERMARK_BITS - CFG_WATERMARK.MAX_NUM_ERROR_BITS, CFG_WATERMARK.NUM_USERS)
CFG_WATERMARK.MIN_HAMMING_DIST = 2 * CFG_WATERMARK.MAX_NUM_ERROR_BITS + 1
# assert CFG_WATERMARK.MIN_HAMMING_DIST > 0 and CFG_WATERMARK.MIN_HAMMING_DIST % 2 == 1 and CFG_WATERMARK.MIN_HAMMING_DIST // 2 >= CFG_WATERMARK.MAX_NUM_ERROR_BITS, f'{CFG_WATERMARK.MIN_HAMMING_DIST}-{CFG_WATERMARK.MAX_NUM_ERROR_BITS}'
CFG_WATERMARK.DIR = f'{CFG_BASIC.ROOT_DIR}/watermark'
if CFG_CLUSTER.DIM_RATIO != '1':
  CFG_WATERMARK.CODE_PATH = f'{CFG_WATERMARK.DIR}/code/{CFG_BASIC.DATA_NAME}-{CFG_EMBEDDING_MODEL.to_str()}/{CFG_CLUSTER.DIM_RATIO}-{CFG_CLUSTER.KEY}/{CFG_CLUSTER.ALGORITHM}-{CFG_CLUSTER.NUM_CLASSES}/{CFG_WATERMARK.NUM_USERS}-{CFG_WATERMARK.WATERMARK}-{CFG_WATERMARK.NUM_WATERMARK_BITS}-{CFG_WATERMARK.MIN_HAMMING_DIST}-{CFG_WATERMARK.GEN_CODE_LOSS}.txt'\
    if CFG_WATERMARK.GEN_CODE_LOSS != 'none' and not CFG_WATERMARK.GEN_CODE_LOSS.__contains__('general') and CFG_WATERMARK.WATERMARK == 'pair_compare_one_pair' else f'{CFG_WATERMARK.DIR}/code/{CFG_WATERMARK.NUM_USERS}-{CFG_WATERMARK.NUM_WATERMARK_BITS}-{CFG_WATERMARK.MIN_HAMMING_DIST}.txt' if not CFG_WATERMARK.GEN_CODE_LOSS.__contains__('general') else f'{CFG_WATERMARK.DIR}/code/{CFG_WATERMARK.NUM_USERS}-{CFG_WATERMARK.NUM_WATERMARK_BITS}-{CFG_WATERMARK.MIN_HAMMING_DIST}_{CFG_WATERMARK.GEN_CODE_LOSS}.txt'
  CFG_WATERMARK.CODE_LOSS_PATH = f'{CFG_WATERMARK.DIR}/code/{CFG_BASIC.DATA_NAME}-{CFG_EMBEDDING_MODEL.to_str()}/{CFG_CLUSTER.DIM_RATIO}-{CFG_CLUSTER.KEY}/{CFG_CLUSTER.ALGORITHM}-{CFG_CLUSTER.NUM_CLASSES}/{CFG_WATERMARK.NUM_USERS}-{CFG_WATERMARK.WATERMARK}-{CFG_WATERMARK.NUM_WATERMARK_BITS}-{CFG_WATERMARK.MIN_HAMMING_DIST}-{CFG_WATERMARK.GEN_CODE_LOSS}-loss.txt' if not CFG_WATERMARK.GEN_CODE_LOSS.__contains__('general') else f'{CFG_WATERMARK.DIR}/code/{CFG_WATERMARK.NUM_USERS}-{CFG_WATERMARK.NUM_WATERMARK_BITS}-{CFG_WATERMARK.MIN_HAMMING_DIST}_{CFG_WATERMARK.GEN_CODE_LOSS}-loss.txt'
else:
  CFG_WATERMARK.CODE_PATH = f'{CFG_WATERMARK.DIR}/code/{CFG_BASIC.DATA_NAME}-{CFG_EMBEDDING_MODEL.to_str()}/{CFG_CLUSTER.ALGORITHM}-{CFG_CLUSTER.NUM_CLASSES}/{CFG_WATERMARK.NUM_USERS}-{CFG_WATERMARK.WATERMARK}-{CFG_WATERMARK.NUM_WATERMARK_BITS}-{CFG_WATERMARK.MIN_HAMMING_DIST}-{CFG_WATERMARK.GEN_CODE_LOSS}.txt'\
    if CFG_WATERMARK.GEN_CODE_LOSS != 'none' and not CFG_WATERMARK.GEN_CODE_LOSS.__contains__('general') and CFG_WATERMARK.WATERMARK == 'pair_compare_one_pair' else f'{CFG_WATERMARK.DIR}/code/{CFG_WATERMARK.NUM_USERS}-{CFG_WATERMARK.NUM_WATERMARK_BITS}-{CFG_WATERMARK.MIN_HAMMING_DIST}.txt' if not CFG_WATERMARK.GEN_CODE_LOSS.__contains__('general') else f'{CFG_WATERMARK.DIR}/code/{CFG_WATERMARK.NUM_USERS}-{CFG_WATERMARK.NUM_WATERMARK_BITS}-{CFG_WATERMARK.MIN_HAMMING_DIST}_{CFG_WATERMARK.GEN_CODE_LOSS}.txt'
  CFG_WATERMARK.CODE_LOSS_PATH = f'{CFG_WATERMARK.DIR}/code/{CFG_BASIC.DATA_NAME}-{CFG_EMBEDDING_MODEL.to_str()}/{CFG_CLUSTER.ALGORITHM}-{CFG_CLUSTER.NUM_CLASSES}/{CFG_WATERMARK.NUM_USERS}-{CFG_WATERMARK.WATERMARK}-{CFG_WATERMARK.NUM_WATERMARK_BITS}-{CFG_WATERMARK.MIN_HAMMING_DIST}-{CFG_WATERMARK.GEN_CODE_LOSS}-loss.txt' if not CFG_WATERMARK.GEN_CODE_LOSS.__contains__('general') else f'{CFG_WATERMARK.DIR}/code/{CFG_WATERMARK.NUM_USERS}-{CFG_WATERMARK.NUM_WATERMARK_BITS}-{CFG_WATERMARK.MIN_HAMMING_DIST}_{CFG_WATERMARK.GEN_CODE_LOSS}-loss.txt'
def cal_watermark_error_rate(e: float):
  p = 1
  for i in range(CFG_WATERMARK.NUM_WATERMARK_BITS - CFG_WATERMARK.MAX_NUM_ERROR_BITS, CFG_WATERMARK.NUM_WATERMARK_BITS + 1):
    p -= (1 - e) ** i * e ** (CFG_WATERMARK.NUM_WATERMARK_BITS - i) * comb(CFG_WATERMARK.NUM_WATERMARK_BITS, i)
  return p - CFG_WATERMARK.ERROR_RATE
CFG_WATERMARK.BIT_ERROR_RATE = brentq(cal_watermark_error_rate, 0, 1, maxiter=10000)
CFG_WATERMARK.NUM_SAMPLES_PER_CLASS_CACHE_PATH = f'{CFG_BASIC.ROOT_DIR}/watermark/results_{FINAL}/cache.json'
CFG_WATERMARK.NUM_SAMPLES_PER_CLASS_CACHE_USAGE_PATH = f'{CFG_BASIC.ROOT_DIR}/watermark/results_{FINAL}/cache_usage.json'
CFG_WATERMARK.CLUSTER_ATTACK_SYN_DIR = f'{CFG_WATERMARK.DIR}/cluster_attack/{CFG_BASIC.DATA_NAME}/{CFG_CLUSTER.ALGORITHM}/{CFG_CLUSTER.NUM_CLASSES}/{CFG_EMBEDDING_MODEL.to_str()}'
if CFG_CLUSTER.DIM_RATIO != '1':
  CFG_WATERMARK.CLUSTER_ATTACK_SYN_DIR += f'/{CFG_CLUSTER.DIM_RATIO}-{CFG_CLUSTER.KEY}'
CFG_WATERMARK.CLUSTER_ATTACK_SYN_DIR += f'/{CFG_WATERMARK.to_str()}/{CFG_WATERMARK.ATTACK_EMBEDDING_MODEL}'
if CFG_CLUSTER.NORM:
  CFG_WATERMARK.CLUSTER_ATTACK_SYN_DIR += '/norm'

CFG_WATERMARK.REGENERATION_ATTACK_SYN_DIR = f'{CFG_WATERMARK.DIR}/regeneration_attack/{CFG_BASIC.DATA_NAME}/{CFG_CLUSTER.ALGORITHM}/{CFG_CLUSTER.NUM_CLASSES}/{CFG_EMBEDDING_MODEL.to_str()}'
if CFG_CLUSTER.DIM_RATIO != '1':
  CFG_WATERMARK.REGENERATION_ATTACK_SYN_DIR += f'/{CFG_CLUSTER.DIM_RATIO}-{CFG_CLUSTER.KEY}'
CFG_WATERMARK.REGENERATION_ATTACK_SYN_DIR += f'/{CFG_WATERMARK.to_str()}/{CFG_WATERMARK.ATTACK_EMBEDDING_MODEL}'
if CFG_CLUSTER.NORM:
  CFG_WATERMARK.REGENERATION_ATTACK_SYN_DIR += '/norm'

CFG_FREQWM.B = dataname2freqwm[CFG_BASIC.DATA_NAME][0]
CFG_FREQWM.T = dataname2freqwm[CFG_BASIC.DATA_NAME][1]
CFG_FREQWM.Z = dataname2freqwm[CFG_BASIC.DATA_NAME][2]

CFG_TABWAK.DM_DIR = f'{CFG_BASIC.ROOT_DIR}/tabsyn/tabwak/ckpt/{CFG_BASIC.DATA_NAME}/{CFG_VAE.to_str()}-token_dim{CFG_VAE.TOKEN_DIM}'
CFG_TABWAK.DM_PATH = f'{CFG_TABWAK.DM_DIR}/model.pt'

np.set_printoptions(precision=4, linewidth=2000)
pd.options.display.max_rows = 100
pd.options.display.max_columns = 75
pd.options.display.max_colwidth = 2000
torch.set_printoptions(precision=4, linewidth=2000)
torch.manual_seed(CFG_BASIC.SEED)
torch.cuda.set_device(CFG_BASIC.GPU_ID)


def load_train_latents_numpy() -> np.ndarray:
  latents = np.load(CFG_EMBEDDING_MODEL.EMBEDDING_PATH)
  if CFG_BASIC.TRANSFORM_LATENTS in ['none']:
    latents = latents[:, 1:, :]
  return latents.reshape(latents.shape[0], -1)


def load_train_labels_numpy():
  return np.loadtxt(CFG_CLUSTER.LABELS_PATH, dtype=np.int64)


def get_embedding_module_param() -> dict:
  return {'vae_num_layers': CFG_VAE.NUM_LAYERS, 'num_vae_heads': CFG_VAE.NUM_HEADS, 'vae_factor': CFG_VAE.FACTOR, 'vae_bias': CFG_VAE.TOKEN_BIAS, 'token_dim': CFG_VAE.TOKEN_DIM}


def get_res_header(seed: int) -> dict:
  r = {'seed': seed, 'dataname': CFG_BASIC.DATA_NAME, 'transform_latents': CFG_BASIC.TRANSFORM_LATENTS,
       'token_dim': CFG_VAE.TOKEN_DIM, 'num_vae_epochs': CFG_VAE.NUM_EPOCHS, 'num_vae_layers': CFG_VAE.NUM_LAYERS, 'embedding_model': CFG_EMBEDDING_MODEL.to_str()}

  if CFG_BASIC.MODE in ['cluster', 'classification_error']:
    r.update({'cluster_algorithm': CFG_CLUSTER.ALGORITHM, 'num_classes': CFG_CLUSTER.NUM_CLASSES, 'key': CFG_CLUSTER.KEY, 'dim_ratio': CFG_CLUSTER.DIM_RATIO, 'norm': CFG_CLUSTER.NORM})
    if CFG_BASIC.MODE == 'classification_error':
      r.update({'classifier': CFG_CLUSTER.CLASSIFIER, 'num_tested_samples_per_class': CFG_WATERMARK.NUM_TESTED_SAMPLES_PER_CLASS, 'correct_guidance': CFG_DM.CORRECT_GUIDANCE})
  elif CFG_BASIC.MODE in ['gen_code', 'watermark_detection', 'watermark_detection_tnr', 'watermark_quality', 'watermark_sample_deletion', 'watermark_gauss_noise', 'watermark_gauss_noise_label', 'watermark_noise_deletion', 'watermark_uniform_noise', 'watermark_laplace_noise', 'watermark_sample_insertion', 'watermark_regeneration_vae', 'watermark_alteration', 'watermark_alteration_label', 'watermark_cluster']:
    r.update({'cluster_algorithm': CFG_CLUSTER.ALGORITHM, 'num_classes': CFG_CLUSTER.NUM_CLASSES, 'num_samples': CFG_SYN.NUM_SAMPLES, 'key': CFG_CLUSTER.KEY, 'dim_ratio': CFG_CLUSTER.DIM_RATIO, 'correct_guidance': CFG_DM.CORRECT_GUIDANCE, 'norm': CFG_CLUSTER.NORM})
    if CFG_WATERMARK.WATERMARK != 'none':
      r.update({'num_users': CFG_WATERMARK.NUM_USERS, 'watermark': CFG_WATERMARK.WATERMARK, 'num_watermark_bits': CFG_WATERMARK.NUM_WATERMARK_BITS,
                'min_hamming_dist': CFG_WATERMARK.MIN_HAMMING_DIST, 'time_limit': CFG_WATERMARK.TIME_LIMIT, 'min_gap': CFG_WATERMARK.MIN_GAP, 'callback_mode': CFG_WATERMARK.CALLBACK_MODE,
                'max_num_error_bits': CFG_WATERMARK.MAX_NUM_ERROR_BITS, 'bit_error_rate': CFG_WATERMARK.BIT_ERROR_RATE, 'error_rate': CFG_WATERMARK.ERROR_RATE, 'fp_rate': CFG_WATERMARK.FP_RATE})
      if CFG_WATERMARK.WATERMARK in ['pair_compare_pair', 'pair_compare_group_pair', 'pair_compare_one_pair']:
        r.update({'num_samples_per_class_interval': int(CFG_WATERMARK.RATIO_NUM_SAMPLES_PER_CLASS_INTERVAL * CFG_SYN.NUM_SAMPLES),
                  'ratio_num_samples_per_class_interval': CFG_WATERMARK.RATIO_NUM_SAMPLES_PER_CLASS_INTERVAL, 'quality_loss': CFG_WATERMARK.QUALITY_LOSS,
                  'num_tested_samples_per_class': CFG_WATERMARK.NUM_TESTED_SAMPLES_PER_CLASS, 'tao_approximation': CFG_WATERMARK.TAO_APPROXIMATION,
                  'gen_code_loss': CFG_WATERMARK.GEN_CODE_LOSS, 'deletion_rate': CFG_WATERMARK.DELETION_RATE,
                  'num_samples_per_class_lower_bound': CFG_WATERMARK.NUM_SAMPLES_PER_CLASS_LOWER_BOUND,
                  'num_samples_per_class_upper_bound': CFG_WATERMARK.NUM_SAMPLES_PER_CLASS_UPPER_BOUND,
                  'gauss': CFG_WATERMARK.GAUSS_NOISE_RATIO,
                  'alter': CFG_WATERMARK.ALTERATION_RATIO,
                  })
      elif CFG_WATERMARK.WATERMARK == 'freqwm':
        r.update({'t': CFG_FREQWM.T, 'z': CFG_FREQWM.Z, 'b': CFG_FREQWM.B})
      elif CFG_WATERMARK.WATERMARK == 'tabular_mark':
        r.update({'num_cells_ratio': CFG_TABULAR_MARK.NUM_CELLS_RATIO, 'p_ratio': CFG_TABULAR_MARK.P_RATIO, 'num_units': CFG_TABULAR_MARK.NUM_UNITS})
      elif CFG_WATERMARK.WATERMARK == 'tabwak_partition':
        pass
      else:
        assert False, CFG_WATERMARK.WATERMARK
    else:
      r.update({'watermark': 'none'})
    if CFG_BASIC.MODE != 'watermark_quality':
      r.update({'classifier': CFG_CLUSTER.CLASSIFIER})
      if CFG_BASIC.MODE == 'watermark_cluster':
        r.update({'attack_embedding_model': CFG_WATERMARK.ATTACK_EMBEDDING_MODEL})
    else:
      r.update({'quality_mode': CFG_WATERMARK.QUALITY_MODE})
  else:
    assert False, CFG_BASIC.MODE
  return r


def get_original_labels_hist(labels_path: str) -> list:
  return pd.DataFrame(np.loadtxt(labels_path), dtype=np.int64).value_counts().sort_index().to_list()


def get_original_centers(center_path: str) -> np.ndarray:
  return np.loadtxt(center_path, dtype=np.float32)


def is_number(s: str):
  if s[0] == '-':
    s = s[1:]
  return all(c in '0123456789.' for c in s) and s.count('.') <= 1


def get_latents4cluster_numpy(latents: np.ndarray, dim_ratio: str, key: int) -> np.ndarray:
  assert isinstance(latents, np.ndarray)
  assert not CFG_CLUSTER.NORM
  assert dim_ratio.startswith('correct')
  W = np.loadtxt(f'{CFG_VAE.DIR}/{dim_ratio}.txt', dtype=np.float32)
  assert not CFG_CLUSTER.NORM
  original_latents = np.load(CFG_VAE.EMBEDDING_PATH)[:, 1:]
  original_latents = original_latents.reshape(len(original_latents), -1)
  mean = original_latents.mean(0)
  latents = (latents - mean) @ W.T
  return latents


def get_latents4cluster_torch(latents: torch.Tensor, dim_ratio: str, key: int) -> torch.Tensor:
  assert isinstance(latents, torch.Tensor)
  return torch.from_numpy(get_latents4cluster_numpy(latents.cpu().detach().numpy(), dim_ratio, key)).to(latents.device)


def get_latents4dm_numpy(latents: np.ndarray, dim_ratio: str, key: int) -> np.ndarray:
  rng = np.random.default_rng(key)
  return latents.copy()


def get_latents4dm_torch(latents: torch.Tensor, dim_ratio: str, key: int) -> torch.Tensor:
  return torch.from_numpy(get_latents4dm_numpy(latents.cpu().detach().numpy(), dim_ratio, key)).to(latents.device)


def get_latents_from_latents4dm_numpy(latents4dm: np.ndarray, dim_ratio: str, key: int) -> np.ndarray:
  assert isinstance(latents4dm, np.ndarray)
  rng = np.random.default_rng(key)
  return latents4dm.copy()


def get_latents_from_latents4dm_torch(latents4dm: torch.Tensor, dim_ratio: str, key: int) -> torch.Tensor:
  assert isinstance(latents4dm, torch.Tensor)
  return torch.from_numpy(get_latents_from_latents4dm_numpy(latents4dm.cpu().detach().numpy(), dim_ratio, key)).to(latents4dm.device)


def get_create_watermark_params(seed: int) -> dict:
  original_labels_hist = get_original_labels_hist(CFG_CLUSTER.LABELS_PATH)
  original_centers = get_original_centers(f'{CFG_CLUSTER.DIR}/center.txt')
  confusion_mat = np.loadtxt(CFG_CLUSTER.CONFUSION_MAT_PATH, dtype=np.float32)
  original_latents = np.load(CFG_VAE.EMBEDDING_PATH)[:, 1:]
  original_latents = original_latents.reshape(len(original_latents), -1)
  original_latents = get_latents4cluster_numpy(original_latents, CFG_CLUSTER.DIM_RATIO, CFG_CLUSTER.KEY)
  original_labels = np.loadtxt(CFG_CLUSTER.LABELS_PATH, dtype=np.int64)
  original_clusters = [original_latents[original_labels == i] for i in range(CFG_CLUSTER.NUM_CLASSES)]
  return {'watermark_name': CFG_WATERMARK.WATERMARK, 'num_users': CFG_WATERMARK.NUM_USERS, 'num_watermark_bits': CFG_WATERMARK.NUM_WATERMARK_BITS,
          'codes_path': (CFG_WATERMARK.CODE_PATH if CFG_BASIC.MODE != 'gen_code' else ''), 'num_classes': CFG_CLUSTER.NUM_CLASSES,
          'ratio_num_samples_per_class_interval': CFG_WATERMARK.RATIO_NUM_SAMPLES_PER_CLASS_INTERVAL, 'quality_loss': CFG_WATERMARK.QUALITY_LOSS,
          'original_labels_hist': original_labels_hist, 'original_centers': original_centers, 'confusion_mat': confusion_mat,
          'tao_approximation': CFG_WATERMARK.TAO_APPROXIMATION,
          'init_ratio_num_samples_per_class_interval': CFG_WATERMARK.INIT_RATIO_NUM_SAMPLES_PER_CLASS_INTERVAL,
          'time_limit': CFG_WATERMARK.TIME_LIMIT,
          'min_gap': CFG_WATERMARK.MIN_GAP,
          'callback_mode': CFG_WATERMARK.CALLBACK_MODE,
          'max_bit_error_rate': CFG_WATERMARK.BIT_ERROR_RATE, 'deletion_rate': CFG_WATERMARK.DELETION_RATE,
          'num_samples_per_class_lower_bound': CFG_WATERMARK.NUM_SAMPLES_PER_CLASS_LOWER_BOUND, 'num_samples_per_class_upper_bound': CFG_WATERMARK.NUM_SAMPLES_PER_CLASS_UPPER_BOUND,
          'original_clusters': original_clusters, 'original_emd_path': CFG_CLUSTER.EMD_PATH, 'group_path': CFG_CLUSTER.GROUP_PATH, 'class_path': CFG_CLUSTER.CLASS_PATH, 'seed': seed}


def get_random_seed() -> int:
  return time.time_ns() % (2 ** 30)


def set_seed(seed):
  random.seed(seed)
  np.random.seed(seed + 1)
  torch.manual_seed(seed + 2)


def process_table_dtypes(table: pd.DataFrame, info: dict):
  table = table.copy()
  for i, col_name in enumerate(table.columns):
    if i in info['cat_col_idx'] or (i in info['target_col_idx'] and info['task_type'] != 'regression'):
      if not isinstance(table.iloc[0, i], str):
        table[col_name] = table[col_name].astype(np.int64)
      else:
        table[col_name] = table[col_name].astype(str)
    elif i in info['num_col_idx'] or (i in info['target_col_idx'] and info['task_type'] == 'regression'):
      table[col_name] = table[col_name].astype(np.float64)
    else:
      assert False, f'Column {i}: {col_name} is not in Info !'
  return table
