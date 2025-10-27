import json
import subprocess
import gpustat
import time
import random
import argparse


M_PER_G = 2 ** 10
MIN_MEM = 3 * M_PER_G
MAX_UTIL = 70
from globals import FINAL

DEFAULT_NUM_SAMPLES = {
  'beijing': 37581,
  'default': 27000,
  'phishing': 9949,
  'shoppers': 11097,
}

parser = argparse.ArgumentParser()
parser.add_argument('--auto', action='store_true')
args = parser.parse_args()

EXP_PATH = 'exp_config.json'
GPUS = {0: 2, 2: 2}

def get_gpu_id():
  stats = gpustat.GPUStatCollection.new_query()
  gpus = stats.gpus
  random.shuffle(gpus)
  for gpu in gpus:
    if gpu.memory_available >= MIN_MEM and gpu.utilization <= MAX_UTIL:
      return gpu.index
  return None


def run_exp(exp: dict, gpu_id: int = None):
  if gpu_id is None:
    if exp.get('gpu_id') is not None:
      gpu_id = exp['gpu_id']
    else:
      gpu_id = get_gpu_id()
      if gpu_id is None:
        print(f"Unable to Allocate GPU when Running {json.dumps(exp)}")
        exit(-1)

  exp['init_ratio_num_samples_per_class_interval'] = 0
  exp['tao_approximation'] = 0
  exp['min_gap'] = 0.01
  exp['ratio_num_samples_per_class_interval'] = -30000
  exp['classifier'] = 'nn'
  exp['correct_guidance'] = 1
  exp['transform_latents'] = 'none'
  exp['cluster_attack_embedding_model'] = 'original-num_layers2-final'
  exp['norm'] = 0
  exp['time_limit'] = 180
  exp['num_tested_samples_per_class'] = 25600
  exp['quality_loss'] = 'quad_random_init'
  exp['token_dim'] = 4
  exp['quality_mode'] = 'average'
  exp['dim_ratio'] = 'correct-pca-0.99'
  exp['cluster_algorithm'] = 'kmeans100-1000'

  mode = exp['mode']
  data_name = exp['dataname']
  norm = exp['norm']
  gauss = exp['gauss']
  alter = exp['alter']
  out_name_prefix = f'{mode}_{data_name}_{norm}_{FINAL}_{gauss}_{alter}'
  cmd = f'nohup python main.py --gpu_id {gpu_id} --mode {mode} --dataname {data_name} --norm {norm} --gauss {gauss} --alter {alter}'

  if mode.__contains__('tabwak') or mode == 'vae_train' or (mode == 'watermark_detection' and exp.get('watermark').__contains__('tabwak')):
    token_dim = exp['token_dim']
    out_name_prefix += f'_{token_dim}'
    cmd += f' --token_dim {token_dim}'

  if mode in ['gen_code', 'cluster', 'tabsyn_train', 'sample', 'eval', 'watermark_detection', 'watermark_quality', 'watermark_sample_deletion', 'watermark_noise_deletion', 'watermark_gauss_noise', 'watermark_uniform_noise', 'watermark_laplace_noise', 'watermark_sample_insertion', 'watermark_regeneration_vae', 'watermark_alteration', 'classification_error']:
    transform_latents = exp['transform_latents']
    cluster_algorithm = exp['cluster_algorithm']
    num_classes = exp['num_classes']
    dim_ratio = exp['dim_ratio']
    out_name_prefix += f'_{cluster_algorithm}_{num_classes}_{transform_latents}_{dim_ratio}'
    cmd += f' --cluster_algorithm {cluster_algorithm} --num_classes {num_classes} --transform_latents {transform_latents} --dim_ratio {dim_ratio}'

  if mode in ['watermark_detection', 'watermark_sample_deletion', 'watermark_noise_deletion', 'watermark_gauss_noise', 'watermark_uniform_noise', 'watermark_laplace_noise', 'watermark_sample_insertion', 'watermark_regeneration_vae', 'watermark_alteration', 'classification_error']:
    classifier = exp['classifier']
    out_name_prefix += f'_{classifier}'
    cmd += f' --classifier {classifier}'

  if mode in ['sample', 'eval', 'watermark_detection', 'watermark_quality', 'watermark_sample_deletion', 'watermark_noise_deletion', 'watermark_gauss_noise', 'watermark_uniform_noise', 'watermark_laplace_noise', 'watermark_sample_insertion', 'watermark_regeneration_vae', 'watermark_alteration']:
    num_samples = exp['num_samples'] if exp.get('num_samples') is not None else DEFAULT_NUM_SAMPLES[data_name]
    out_name_prefix += f'_{num_samples}'
    cmd += f' --num_samples {num_samples}'

  if mode in ['sample', 'eval', 'watermark_detection', 'watermark_quality', 'watermark_sample_deletion', 'watermark_noise_deletion', 'watermark_gauss_noise', 'watermark_uniform_noise', 'watermark_laplace_noise', 'watermark_sample_insertion', 'watermark_regeneration_vae', 'watermark_alteration', 'classification_error']:
    correct_guidance = exp['correct_guidance']
    out_name_prefix += f'_{correct_guidance}'
    cmd += f' --correct_guidance {correct_guidance}'

  if mode in ['sample', 'eval']:
    num_samples_per_class = ''
    for num in exp['num_samples_per_class']:
      num_samples_per_class += str(num) + ' '
    num_samples_per_class = num_samples_per_class[:-1]
    out_name_prefix += f'_{num_samples_per_class.replace(" ", "-")}'
    cmd += f' --num_samples_per_class {num_samples_per_class}'

  if mode in ['gen_code', 'watermark_detection', 'watermark_quality', 'watermark_sample_deletion', 'watermark_noise_deletion', 'watermark_gauss_noise', 'watermark_uniform_noise', 'watermark_laplace_noise', 'watermark_sample_insertion', 'watermark_regeneration_vae', 'watermark_alteration']:
    num_users = exp['num_users']
    watermark = exp['watermark']
    num_watermark_bits = exp['num_watermark_bits']
    quality_loss = exp['quality_loss']
    gen_code_loss = exp['gen_code_loss']
    out_name_prefix += f'_{num_users}_{watermark}_{num_watermark_bits}_{quality_loss}_{gen_code_loss}'
    cmd += f' --num_users {num_users} --watermark {watermark} --num_watermark_bits {num_watermark_bits} --quality_loss {quality_loss} --gen_code_loss {gen_code_loss}'

  if mode in ['classification_error', 'watermark_detection', 'watermark_quality', 'watermark_sample_deletion', 'watermark_noise_deletion', 'watermark_gauss_noise', 'watermark_uniform_noise', 'watermark_laplace_noise', 'watermark_sample_insertion', 'watermark_regeneration_vae', 'watermark_alteration']:
    num_tested_samples_per_class = exp['num_tested_samples_per_class']
    out_name_prefix += f'_{num_tested_samples_per_class}'
    cmd += f' --num_tested_samples_per_class {num_tested_samples_per_class}'

  if mode in ['watermark_detection', 'watermark_quality', 'watermark_sample_deletion', 'watermark_noise_deletion', 'watermark_gauss_noise', 'watermark_uniform_noise', 'watermark_laplace_noise', 'watermark_sample_insertion', 'watermark_regeneration_vae', 'watermark_alteration']:
    ratio_num_samples_per_class_interval = exp['ratio_num_samples_per_class_interval']
    num_watermark_trials = exp['num_watermark_trials']
    max_watermark_error_rate = exp['max_watermark_error_rate']
    deletion_rate = exp['deletion_rate']
    time_limit = exp['time_limit']
    min_gap = exp['min_gap']
    num_tested_samples_per_class = exp['num_tested_samples_per_class']
    tao_approximation = exp['tao_approximation']
    init_ratio_num_samples_per_class_interval = exp['init_ratio_num_samples_per_class_interval']
    num_samples_per_class_lower_bound = exp['num_samples_per_class_lower_bound']
    out_name_prefix += f'_{num_samples_per_class_lower_bound}_{ratio_num_samples_per_class_interval}_{num_watermark_trials}_{max_watermark_error_rate}_{deletion_rate}_{time_limit}_{min_gap}_{num_tested_samples_per_class}_{tao_approximation}_{init_ratio_num_samples_per_class_interval}'
    cmd += f' --num_samples_per_class_lower_bound {num_samples_per_class_lower_bound} --ratio_num_samples_per_class_interval {ratio_num_samples_per_class_interval} --num_watermark_trials {num_watermark_trials} --max_watermark_error_rate {max_watermark_error_rate} --deletion_rate {deletion_rate} --time_limit {time_limit} --min_gap {min_gap} --num_tested_samples_per_class {num_tested_samples_per_class} --tao_approximation {tao_approximation} --init_ratio_num_samples_per_class_interval {init_ratio_num_samples_per_class_interval}'
    if mode == 'watermark_quality':
      quality_mode = exp['quality_mode']
      out_name_prefix += f'_{quality_mode}'
      cmd += f' --quality_mode {quality_mode}'
    
  if mode in ['vae_train', 'reducer_train_ae', 'reducer_train_mlp']:
    cmd += f' > tabsyn/vae/ckpt/{exp["dataname"]}/{out_name_prefix[:200]}.out &'
  elif mode == 'tabsyn_train':
    cmd += f' > tabsyn/ckpt/{exp["dataname"]}/{exp["cluster_algorithm"]}/{exp["num_classes"]}/{out_name_prefix[:200]}.out &'
  else:
    cmd += f' > out/{out_name_prefix[:200]}.out &'

  print(f'GPU {gpu_id}: {cmd[len("nohup "):]}')
  subprocess.Popen(args=cmd, shell=True, text=True)
  return 0


def main():
  exps = []
  with open(EXP_PATH, 'r') as f:
    for line in f.readlines():
      if line.strip() == "" or line.startswith('//'):
        continue
      exps.append(json.loads(line))

  for i, exp in enumerate(exps):
    run_exp(exp)
    if i < len(exps) - 1:
      time.sleep(1)


main()
