import pandas as pd
import random
import hashlib
import gurobipy as grb
import os
import json
from tqdm import tqdm

from globals import *
from tabsyn.latent_utils import load_info_full, add_gauss_noise, add_uniform_noise, add_laplace_noise, alter_cat, alter_num, delete, insert
from tabsyn.sample import syn_table
from watermark.watermark import check, int2bit_list, bit_list2int
from eval.main import eval_all
from watermark.regeneration_attack_vae.regeneration_attack_vae import attack
from process_dataset import process_data


def int_hash(input_int1, input_int2):
  input_int1 = int(input_int1)
  input_int2 = int(input_int2)
  # 将整数转换为字节串
  input_bytes1 = input_int1.to_bytes((input_int1.bit_length() + 7) // 8, 'big') or b'\x00'
  input_bytes2 = input_int2.to_bytes((input_int2.bit_length() + 7) // 8, 'big') or b'\x00'
  
  # 拼接两个字节串
  concatenated_bytes = input_bytes1 + input_bytes2
  
  # 创建一个 sha256 哈希对象
  sha256_hash = hashlib.sha256()
  
  # 更新哈希对象，传入拼接后的字节串
  sha256_hash.update(concatenated_bytes)
  
  # 获取哈希值的字节串
  digest_bytes = sha256_hash.digest()
  
  # 将字节串转换为整数
  digest_int = int.from_bytes(digest_bytes, 'big')
  return digest_int


class FreqWM:
  def __init__(self, original_table: pd.DataFrame, labels_info: list[dict], t: int, z: int, b: int, num_watermark_bits: int, codes: list[int]):
    self.original_table = original_table.reset_index(drop=True)
    self.original_labels = None
    self.hist = None
    self.label_columns = None
    self.labels_info = labels_info
    self.eligible_label_pairs = []
    self.optimal_label_pairs = []
    self.s = []
    self.key = 985
    self.t = t
    self.z = z
    self.b = b
    self.num_watermark_bits = num_watermark_bits
    self.codes = codes
    self.gen_histogram()
    self.gen_bounds()
    self.get_eligible_label_pairs()
    self.select_optimal_label_pairs()
    self.original_label2indices = {}
    for i in range(len(self.original_labels)):
      self.original_label2indices.setdefault(tuple(v for v in self.original_labels.iloc[i, :]), []).append(i)
  
  def get_cell_label(self, value: str | int | float, info: dict) -> int | str:
    if info.get('num_bins') is None:
      return value
    return int((value - info['lower']) / info['bin_width'])

  def gen_histogram(self):
    self.original_labels = self.original_table.copy()
    self.label_columns = []
    for info in self.labels_info:
      self.label_columns.append(info['name'])
      num_bins = info.get('num_bins')
      if num_bins is not None:
        info['lower'] = min(self.original_labels[info['name']])
        info['upper'] = max(self.original_labels[info['name']])
        info['bin_width'] = (info['upper'] - info['lower']) / num_bins
        assert self.original_labels[info['name']].dtype not in [str, object]
        self.original_labels[info['name']] = self.original_labels[info['name']].apply(lambda x: int((x - info['lower']) / info['bin_width']))
    self.hist = self.original_labels.value_counts(self.label_columns, ascending=False).reset_index()
    self.original_labels = self.original_labels[self.label_columns]

  def gen_bounds(self):
    bounds = []
    for i in range(len(self.hist)):
      bounds.append((
        self.hist.loc[i, 'count'] - self.hist.loc[i + 1, 'count'] if i < len(self.hist) - 1 else self.hist.loc[len(self.hist) - 1, 'count'],
        self.hist.loc[i - 1, 'count'] - self.hist.loc[i, 'count'] if i > 0 else 9999999,
      ))
    self.hist['bound'] = bounds

  def get_eligible_label_pairs(self):
    self.s = np.zeros(shape=(len(self.hist), len(self.hist)), dtype=np.int64)
    bounds = self.hist['bound'].to_list()
    counts = self.hist['count'].to_list()
    for i in range(len(self.hist)):
      if bounds[i][1] - bounds[i][0] == 0:
        continue
      for j in range(i + 1, len(self.hist)):
        if bounds[j][1] - bounds[j][0] == 0:
          continue
        s = int_hash(counts[i], int_hash(self.key, counts[j])) % self.z
        self.s[i, j] = self.s[j, i] = s
        # if s >= 2 * (2 * self.t + 1) and (bounds[i][1] - bounds[i][0]) >= (s + 1) // 2 and (bounds[j][1] - bounds[j][0]) >= (s + 1) // 2:
        if s >= 2 and (bounds[i][1] - bounds[i][0]) >= (s + 1) // 2 and (bounds[j][1] - bounds[j][0]) >= (s + 1) // 2:
          self.eligible_label_pairs.append((i, j))

  def select_optimal_label_pairs(self):
    model = grb.Model()
    model.setParam('Presolve', 2)
    model.setParam('TimeLimit', 180)
    useful_node2original_node = sorted(set(p[0] for p in self.eligible_label_pairs) | set(p[1] for p in self.eligible_label_pairs))
    n = len(useful_node2original_node)
    x = model.addVars(n, n, vtype=grb.GRB.BINARY)
    model.setObjective(grb.quicksum([x[i, j] for i in range(n) for j in range(i + 1, n) if (useful_node2original_node[i], useful_node2original_node[j]) in self.eligible_label_pairs]), sense=grb.GRB.MAXIMIZE)
    model.addConstrs(x[i, j] == x[j, i] for i in range(n) for j in range(i + 1, n))
    model.addConstrs(x[i, i] == 0 for i in range(n))
    model.addConstrs(x[i, j] == 0 for i in range(n) for j in range(i + 1, n) if (useful_node2original_node[i], useful_node2original_node[j]) not in self.eligible_label_pairs)
    model.addConstrs(grb.quicksum([x[i, j] for i in range(n)]) <= 1 for j in range(n))
    model.addConstrs(grb.quicksum([x[j, i] for i in range(n)]) <= 1 for j in range(n))
    model.addConstr(grb.quicksum([x[i, j] * abs(self.hist.loc[useful_node2original_node[i], 'count'] - self.hist.loc[useful_node2original_node[j], 'count']) for i in range(n) for j in range(i + 1, n)]) <= self.b / 100 * len(self.original_table))
    model.optimize()
    assert model.Status == grb.GRB.OPTIMAL
    self.optimal_label_pairs = [(useful_node2original_node[i], useful_node2original_node[j]) for i in range(n) for j in range(i + 1, n) if x[i, j].x > 0]
    assert self.num_watermark_bits <= len(self.optimal_label_pairs), f'{self.num_watermark_bits} > {len(self.optimal_label_pairs)}'

  def embed(self, watermark_int: int):
    watermark_bits = int2bit_list(watermark_int, self.num_watermark_bits)
    w_hist = [h for h in self.hist['count']]
    for i, (lhs_class, rhs_class) in enumerate(self.optimal_label_pairs[:self.num_watermark_bits]):
      diff = (self.hist.loc[lhs_class, 'count'] - self.hist.loc[rhs_class, 'count']) % self.s[lhs_class, rhs_class]
      if watermark_bits[i] == 1:
        half_1 = diff // 2
        half_2 = diff - half_1
        w_hist[lhs_class] -= half_1
        w_hist[rhs_class] += half_2
      else:
        if diff > self.s[lhs_class, rhs_class] // 2:
          w_hist[lhs_class] -= diff - self.s[lhs_class, rhs_class] // 2
        elif diff < self.s[lhs_class, rhs_class] // 2:
          w_hist[rhs_class] -= self.s[lhs_class, rhs_class] // 2 - diff

    watermarked_table = self.original_table.reset_index(drop=True)
    watermarked_labels = watermarked_table.copy()
    for info in self.labels_info:
      if info.get('num_bins') is not None:
        watermarked_labels[info['name']] = watermarked_labels[info['name']].apply(lambda x: int((x - info['lower']) / info['bin_width']))
    watermarked_labels = watermarked_labels[self.label_columns]
    watermarked_label2indices = {}
    for i, key in enumerate(watermarked_labels.itertuples(index=False)):
      watermarked_label2indices.setdefault(tuple(v for v in key), []).append(i)
    added_indices = []
    deleted_indices = []
    for i in tqdm(range(len(w_hist)), total=len(w_hist)):
      key = tuple(v for v in self.hist.loc[i, self.label_columns])
      if w_hist[i] > self.hist.loc[i, 'count']:
        diff = w_hist[i] - self.hist.loc[i, 'count']
        added_indices += np.random.choice(self.original_label2indices[key], diff, replace=True).tolist()
      elif w_hist[i] < self.hist.loc[i, 'count']:
        diff = self.hist.loc[i, 'count'] - w_hist[i]
        indices = np.random.choice(watermarked_label2indices[key], diff, replace=False).tolist()
        watermarked_label2indices[key] = list(set(watermarked_label2indices[key]) - set(indices))
        deleted_indices += indices
    reversed_indices = list(set(range(len(watermarked_table))) - set(deleted_indices))
    watermarked_table = pd.concat([watermarked_table.iloc[reversed_indices, :], self.original_table.iloc[added_indices, :]], axis=0).reset_index(drop=True)
    watermarked_labels = pd.concat([watermarked_labels.iloc[reversed_indices, :], watermarked_labels.iloc[added_indices, :]], axis=0).reset_index(drop=True)
    return watermarked_table, w_hist, watermarked_labels
        
  def extract(self, table: pd.DataFrame) -> tuple[int, int, pd.DataFrame, pd.DataFrame]:
    table = table.copy()
    label_columns = []
    for info in self.labels_info:
      label_columns.append(info['name'])
      if info.get('num_bins') is not None:
        assert table[info['name']].dtype not in [str, object]
        table[info['name']] = table[info['name']].apply(lambda x: int((x - info['lower']) / info['bin_width']))
    labels_pred = table[label_columns]
    raw_hist = table.value_counts(label_columns, ascending=False)
    hist = self.hist.copy()
    for i in range(len(self.hist)):
      key = tuple(v for v in self.hist.loc[i, label_columns])
      if key in raw_hist.index:
        hist.loc[i, 'count'] = raw_hist[key]
      else:
        hist.loc[i, 'count'] = 0
    watermark_bits = []
    for i, (lhs_class, rhs_class) in enumerate(self.optimal_label_pairs[:self.num_watermark_bits]):
      diff = abs(hist.loc[lhs_class, 'count'] - hist.loc[rhs_class, 'count']) % self.s[lhs_class, rhs_class]
      # if diff <= self.t:
      if diff <= self.s[lhs_class, rhs_class] // 4:
        watermark_bits.append(1)
      else:
        watermark_bits.append(0)
    watermark_extracted = bit_list2int(watermark_bits)
    watermark_pred = None
    min_dist = self.num_watermark_bits
    for code in self.codes:
      dist = sum(int2bit_list(watermark_extracted ^ code, self.num_watermark_bits))
      if dist < min_dist:
        min_dist = dist
        watermark_pred = code
    return watermark_extracted, watermark_pred, hist, labels_pred

def main():
  labels_info = {
    'beijing': [
      {'name': 'year'},
      {'name': 'month'},
      {'name': 'TEMP', 'num_bins': 3},
      {'name': 'DEWP', 'num_bins': 5},
    ],
    'default': [
      {'name': 'SEX'},
      {'name': 'AGE', 'num_bins': 59},
      {'name': 'MARRIAGE'},
      {'name': 'default payment next month'},
    ],
    'phishing': [
      {'name': 'having_IP_Address'},
      {'name': 'URL_Length'},
      {'name': 'Shortining_Service'},
      {'name': 'having_At_Symbol'},
      # {'name': 'double_slash_redirecting'},
      {'name': 'Prefix_Suffix'},
      {'name': 'having_Sub_Domain'},
      {'name': 'SSLfinal_State'},
      {'name': 'Domain_registeration_length'},
      # {'name': 'Favicon'},
      {'name': 'port'},
      # {'name': 'Request_URL'},
      # {'name': 'Links_in_tags'},
      {'name': 'label'},
      # {'name': 'Statistical_report'}
      {'name': 'Redirect'}
    ],
    'shoppers': [
      {'name': 'ExitRates', 'num_bins': 12},
      {'name': 'VisitorType'},
      {'name': 'Weekend'},
      {'name': 'Region'},
      {'name': 'OperatingSystems'},
      {'name': 'Revenue'},
    ],
  }[CFG_BASIC.DATA_NAME]
  codes = []
  with open(CFG_WATERMARK.CODE_PATH, 'r') as f:
    lines = f.readlines()
  for line in lines:
    codes.append(int(line))
  codes = codes[:CFG_WATERMARK.NUM_USERS]
  original_labels_hist = get_original_labels_hist(CFG_CLUSTER.LABELS_PATH)

  with open(CFG_BASIC.COLUMN_INFO_PATH, 'r') as f:
    columns_info = json.load(f)['columns']
    columns_info = {col['name']: col for col in columns_info}

  for _ in range(CFG_WATERMARK.NUM_WATERMARK_TRIALS):
    seed = get_random_seed()
    set_seed(seed)

    info = load_info_full(CFG_BASIC.DATA_DIR, CFG_BASIC.INFO_PATH, CFG_BASIC.TRANSFORM_LATENTS, embedding_path=CFG_EMBEDDING_MODEL.EMBEDDING_PATH, vae_path=CFG_VAE.PATH, device=CFG_BASIC.DEVICE, **get_embedding_module_param())    
    while True:
      try:
        _, _, table_full, _, _ = syn_table(info, CFG_SYN.NUM_SAMPLES, CFG_CLUSTER.NUM_CLASSES, original_labels_hist, CFG_DM.NUM_SAMPLE_STEPS, CFG_DM.PATH, CFG_BASIC.DEVICE, CFG_DM.CORRECT_GUIDANCE, get_original_centers(CFG_CLUSTER.CENTERS_PATH), CFG_CLUSTER.DIM_RATIO, CFG_CLUSTER.KEY)
        freqwm = FreqWM(table_full, labels_info, CFG_FREQWM.T, CFG_FREQWM.Z, CFG_FREQWM.B, CFG_WATERMARK.NUM_WATERMARK_BITS, codes)
      except:
        print('Failed, Try angin !')
        continue
      break
    watermark_int_true = random.choice(codes)
    table_full, watermarked_hist, watermarked_labels = freqwm.embed(watermark_int_true)
    watermark_int_extracted, watermark_int_pred, hist_true, labels_pred = freqwm.extract(table_full)

    r = {**get_res_header(seed), 'labels_info': labels_info, **check(watermark_int_true, watermark_int_extracted, watermark_int_pred, CFG_WATERMARK.NUM_WATERMARK_BITS, CFG_WATERMARK.MAX_NUM_ERROR_BITS)}
    with open(f'{CFG_BASIC.RESULTS_DIR}/freqwm_detection.json', 'a') as f:
      f.write(json.dumps(r) + '\n')

    r = {**get_res_header(seed), 'labels_info': labels_info, 'quality_mode': 'average', **eval_all(table_full, info, CFG_SYN.REAL_DATA_PATH, CFG_SYN.TEST_DATA_PATH)}
    with open(f'{CFG_BASIC.RESULTS_DIR}/freqwm_quality.json', 'a') as f:
      f.write(json.dumps(r) + '\n')

    if CFG_BASIC.DATA_NAME != 'phishing':
      for ratio in [0.01]:
        print(f'ratio: {ratio}')
        table, _ = add_gauss_noise(table_full, info, ratio)
        table = table.sample(frac=1, replace=False).reset_index(drop=True)
        watermark_int_extracted, watermark_int_pred, hist_pred, labels_pred = freqwm.extract(table)
        label_accuracy = (watermarked_labels.values == labels_pred.values).all(1).sum() / len(watermarked_labels)
        r = {**get_res_header(seed), 'columns_info': columns_info, 'gauss_noise_ratio': ratio, **check(watermark_int_true, watermark_int_extracted, watermark_int_pred, CFG_WATERMARK.NUM_WATERMARK_BITS, CFG_WATERMARK.MAX_NUM_ERROR_BITS), 'label_accuracy': label_accuracy}
        with open(f'{CFG_BASIC.RESULTS_DIR}/freqwm_gauss_noise.json', 'a') as f:
          f.write(json.dumps(r) + '\n')

    if CFG_BASIC.DATA_NAME != 'phishing':
      for ratio in [0.01]:
        print(f'ratio: {ratio}')
        table, _ = add_uniform_noise(table_full, info, ratio)
        table = table.sample(frac=1, replace=False).reset_index(drop=True)
        watermark_int_extracted, watermark_int_pred, hist_pred, labels_pred = freqwm.extract(table)
        label_accuracy = (watermarked_labels.values == labels_pred.values).all(1).sum() / len(watermarked_labels)
        r = {**get_res_header(seed), 'columns_info': columns_info, 'gauss_noise_ratio': ratio, **check(watermark_int_true, watermark_int_extracted, watermark_int_pred, CFG_WATERMARK.NUM_WATERMARK_BITS, CFG_WATERMARK.MAX_NUM_ERROR_BITS), 'label_accuracy': label_accuracy}
        with open(f'{CFG_BASIC.RESULTS_DIR}/freqwm_uniform_noise.json', 'a') as f:
          f.write(json.dumps(r) + '\n')

    if CFG_BASIC.DATA_NAME != 'phishing':
      for ratio in [0.01]:
        print(f'ratio: {ratio}')
        table, _ = add_laplace_noise(table_full, info, ratio)
        table = table.sample(frac=1, replace=False).reset_index(drop=True)
        watermark_int_extracted, watermark_int_pred, hist_pred, labels_pred = freqwm.extract(table)
        label_accuracy = (watermarked_labels.values == labels_pred.values).all(1).sum() / len(watermarked_labels)
        r = {**get_res_header(seed), 'columns_info': columns_info, 'gauss_noise_ratio': ratio, **check(watermark_int_true, watermark_int_extracted, watermark_int_pred, CFG_WATERMARK.NUM_WATERMARK_BITS, CFG_WATERMARK.MAX_NUM_ERROR_BITS), 'label_accuracy': label_accuracy}
        with open(f'{CFG_BASIC.RESULTS_DIR}/freqwm_laplace_noise.json', 'a') as f:
          f.write(json.dumps(r) + '\n')

    for ratio in [0.1]:
      print(f'ratio: {ratio}')
      table, _, _ = insert(table_full, info, ratio)
      table = table.sample(frac=1, replace=False).reset_index(drop=True)
      watermark_int_extracted, watermark_int_pred, hist_pred, labels_pred = freqwm.extract(table)
      r = {**get_res_header(seed), 'columns_info': columns_info, 'sample_deletion_ratio': ratio, **check(watermark_int_true, watermark_int_extracted, watermark_int_pred, CFG_WATERMARK.NUM_WATERMARK_BITS, CFG_WATERMARK.MAX_NUM_ERROR_BITS)}
      with open(f'{CFG_BASIC.RESULTS_DIR}/freqwm_sample_insertion.json', 'a') as f:
        f.write(json.dumps(r) + '\n')

    for ratio in [0.01]:
      print(f'ratio: {ratio}')
      table, _ = alter_cat(table_full, info, ratio)
      table, _ = alter_num(table, info, ratio)
      watermark_int_extracted, watermark_int_pred, hist_pred, labels_pred = freqwm.extract(table)
      label_accuracy = (watermarked_labels.values == labels_pred.values).all(1).sum() / len(watermarked_labels)
      r = {**get_res_header(seed), 'labels_info': labels_info, 'alteration_ratio': ratio, **check(watermark_int_true, watermark_int_extracted, watermark_int_pred, CFG_WATERMARK.NUM_WATERMARK_BITS, CFG_WATERMARK.MAX_NUM_ERROR_BITS), 'label_accuracy': label_accuracy}
      with open(f'{CFG_BASIC.RESULTS_DIR}/freqwm_alteration.json', 'a') as f:
        f.write(json.dumps(r) + '\n')

    for ratio in [0.1]:
      print(f'ratio: {ratio}')
      table, _, _ = delete(table_full, info, ratio)
      watermark_int_extracted, watermark_int_pred, hist_pred, labels_pred = freqwm.extract(table)
      r = {**get_res_header(seed), 'labels_info': labels_info, 'sample_deletion_ratio': ratio, **check(watermark_int_true, watermark_int_extracted, watermark_int_pred, CFG_WATERMARK.NUM_WATERMARK_BITS, CFG_WATERMARK.MAX_NUM_ERROR_BITS)}
      with open(f'{CFG_BASIC.RESULTS_DIR}/freqwm_sample_deletion.json', 'a') as f:
        f.write(json.dumps(r) + '\n')

    token_dim = 4
    working_dir = f'{CFG_BASIC.ROOT_DIR}/watermark/regeneration_attack_vae/{CFG_WATERMARK.WATERMARK}-token_dim{token_dim}/{CFG_BASIC.DATA_NAME}/{CFG_CLUSTER.ALGORITHM}/{CFG_CLUSTER.NUM_CLASSES}/{CFG_EMBEDDING_MODEL.to_str()}/{watermark_int_true}'
    if os.path.exists(working_dir):
      continue
    os.makedirs(working_dir, exist_ok=True)
    table_full.to_csv(f'{working_dir}/watermarked.csv', index=False)
    process_data(CFG_BASIC.DATA_NAME, f'{working_dir}/watermarked.csv')
    for ratio in [0.1]:
      reversed_table, reversed_num_norm, reversed_cat_norm, _, _ = attack(table_full, info, token_dim, working_dir, ratio)
      reversed_table.to_csv(f'{working_dir}/attacked_{ratio}.csv', index=False)
      watermark_int_extracted, watermark_int_pred, hist_pred, labels_pred = freqwm.extract(reversed_table)
      label_accuracy = (watermarked_labels.values == labels_pred.values).all(1).sum() / len(watermarked_labels)
      r = {**get_res_header(seed), 'labels_info': labels_info, **check(watermark_int_true, watermark_int_extracted, watermark_int_pred, CFG_WATERMARK.NUM_WATERMARK_BITS, CFG_WATERMARK.MAX_NUM_ERROR_BITS), 'attack_model': f'token_dim{token_dim}-ratio{ratio}'}
      with open(f'{CFG_BASIC.RESULTS_DIR}/freqwm_regeneration_vae.json', 'a') as f:
        f.write(json.dumps(r) + '\n')
