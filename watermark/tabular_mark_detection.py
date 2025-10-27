import pandas as pd
import random
import os
import json
from sklearn.preprocessing import LabelEncoder

from globals import *
from tabsyn.latent_utils import load_info_full, add_gauss_noise, add_laplace_noise, add_uniform_noise, alter_cat, alter_num, delete, insert
from tabsyn.sample import syn_table
from watermark.watermark import check, int2bit_list, bit_list2int
from eval.main import eval_all
from watermark.regeneration_attack_vae.regeneration_attack_vae import attack
from process_dataset import process_data


def extract_logical_top_bits(arr: np.ndarray, num_msbs: int) -> list[int]:
  result = []
  for x in arr:
    x = int(x)
    if x < (1 << num_msbs):
      result.append(x)
      continue
    bit_length = x.bit_length()  # 最高位的索引 + 1
    shift = max(bit_length - num_msbs, 0)
    top = (x >> shift) % (1 << num_msbs)
    result.append(top)
  return result


class TabularMark:
  def __init__(self, original_table: pd.DataFrame, info: dict, columns_info: list[dict], keys_info: list[dict], num_cells_ratio: float, p_ratio: float, num_units: int, num_watermark_bits: int, codes: list[int]):
    # assert len(columns_info) == 1
    self.key = 985
    self.info = info.copy()
    self.cat_encoders = None
    self.original_table = original_table.reset_index(drop=True)
    self.columns_info = columns_info.copy()
    self.keys_info = keys_info.copy()
    self.num_cells_ratio = num_cells_ratio
    self.p_ratio = p_ratio
    assert num_units % 2 == 0
    self.num_units = num_units
    self.num_cells = int(num_cells_ratio * len(original_table))
    self.cells = [np.sort(np.random.default_rng(self.key + i * 10).choice(len(original_table), self.num_cells, replace=False)) for i in range(len(columns_info))]
    num_cells_per_watermark_bit = self.num_cells // num_watermark_bits
    self.watermark_bit2cells = [[self.cells[i][j * num_cells_per_watermark_bit: ((j + 1) * num_cells_per_watermark_bit) if j < num_watermark_bits - 1 else None] for j in range(num_watermark_bits)] for i in range(len(columns_info))]
    self.cells2watermark_bit = [{index: k for k, v in enumerate(self.watermark_bit2cells[i]) for index in v} for i in range(len(columns_info))]
    self.key2index = {key: i for i, key in enumerate(self.get_keys(original_table))}
    self.ps = [(p_ratio * np.std(original_table[column_info['name']].values)) if column_info['type'] == 'num' else None for column_info in columns_info]
    self.num_watermark_bits = num_watermark_bits
    self.codes = codes.copy()
    self.all_segments = [np.sort(np.random.default_rng(self.key).choice(num_units, num_units // 2, replace=False))]
    self.all_segments.append(np.asarray(sorted(set(range(num_units)) - set(self.all_segments[0].tolist())), dtype=np.int64))
    self.all_segments = [segments - num_units // 2 for segments in self.all_segments]
    self.len_units = [(2 * self.ps[i] / self.num_units) if self.ps[i] is not None else None for i in range(len(self.ps))]
    unique_cats = [original_table[col['name']].unique() if col['type'] == 'cat' else None for col in columns_info]
    self.all_cats = [np.random.default_rng(self.key).choice(unique_cats[i], len(unique_cats[i]) // 2, replace=False) if column_info['type'] == 'cat' else None for i, column_info in enumerate(columns_info)]
    self.all_cats = [[self.all_cats[i], list(set(unique_cats[i]) - set(self.all_cats[i]))] if column_info['type'] == 'cat' else None for i, column_info in enumerate(columns_info)]
    self.key2index = {}
    for i, key in enumerate(self.get_keys(original_table)):
      if self.key2index.get(key) is None:
        self.key2index[key] = i

  def encode_cat(self, table: pd.DataFrame) -> pd.DataFrame:
    cat_col_idx = self.info['cat_col_idx']
    if self.info['task_type'] == 'binclass':
      cat_col_idx = self.info['target_col_idx'] + cat_col_idx
    res = table.copy()
    if self.cat_encoders is None:
      original_table = pd.concat([pd.read_csv(f'{CFG_BASIC.DATA_DIR}/train.csv'), pd.read_csv(f'{CFG_BASIC.DATA_DIR}/test.csv')], axis=0)
      self.cat_encoders = [LabelEncoder().fit(original_table[original_table.columns[i]]) for i in cat_col_idx]
    for i in range(len(cat_col_idx)):
      res[res.columns[cat_col_idx[i]]] = self.cat_encoders[i].transform(res[res.columns[cat_col_idx[i]]])
    return res

  def get_keys(self, table: pd.DataFrame) -> list:
    keys = [0] * len(table)
    table = self.encode_cat(table)
    for key in self.keys_info:
      if key['type'] == 'num':
        k = extract_logical_top_bits(table[key['name']].values.astype(np.float64).view(np.uint64), key['num_msbs'])
      elif key['type'] == 'cat':
        k = extract_logical_top_bits(table[key['name']].values.astype(np.int64).view(np.uint64), key['num_msbs'])
      keys = [(1 << key['num_msbs']) * keys[i] + k[i] for i in range(len(keys))]
    return keys

  def sample_noise(self, candidate_segments: np.ndarray, length_unit: float, shape: int, seed: int) -> np.ndarray:
    selected_segments = np.random.default_rng(seed).choice(candidate_segments, shape)
    return selected_segments * length_unit + np.random.default_rng(seed + 1).uniform(0, length_unit + 1e-6, size=(shape,))

  def choice(self, candidata_values: np.ndarray, shape: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).choice(candidata_values, shape)

  def count_from_num(self, noise: np.ndarray, length_unit: float) -> np.ndarray:
    nan_mask = np.isnan(noise)
    noise[nan_mask] = 0
    segments = np.floor(noise / length_unit).astype(np.int64)
    res = [0, 0]
    for i, s in enumerate(segments):
      if nan_mask[i]:
        continue
      if s in self.all_segments[0]:
        res[0] += 1
      elif s in self.all_segments[1]:
        res[1] += 1
    return np.asarray(res, dtype=np.int64)
  
  def count_from_cat(self, cat: np.ndarray, col_idx: int) -> np.ndarray:
    res = [0, 0]
    for c in cat:
      if c in self.all_cats[col_idx][0]:
        res[0] += 1
      elif c in self.all_cats[col_idx][1]:
        res[1] += 1
    return np.asarray(res, dtype=np.int64)

  def embed(self, watermark_int: int, seed: int) -> pd.DataFrame:
    watermark_bits = int2bit_list(watermark_int, self.num_watermark_bits)
    watermarked_table = self.original_table.copy()
    for i, watermark_bit in enumerate(watermark_bits):
      for j, col in enumerate(self.columns_info):
        if col['type'] == 'num':
          watermarked_table.loc[self.watermark_bit2cells[j][i], col['name']] += self.sample_noise(self.all_segments[watermark_bit], self.len_units[j], len(self.watermark_bit2cells[j][i]), seed)
        elif col['type'] == 'cat':
          watermarked_table.loc[self.watermark_bit2cells[j][i], col['name']] = self.choice(self.all_cats[j][watermark_bit], len(self.watermark_bit2cells[j][i]), seed)
    return watermarked_table
        
  def extract(self, table: pd.DataFrame) -> tuple[int, int, list[np.ndarray]]:
    table = table.copy()
    watermark_bits = []
    keys = self.get_keys(table)
    watermark_bit2cells = [[[] for _ in range(self.num_watermark_bits)] for _ in range(len(self.columns_info))]
    original_watermark_bit2cells = [[[] for _ in range(self.num_watermark_bits)] for _ in range(len(self.columns_info))]
    used_keys = set()
    for i, key in enumerate(keys):
      if key in used_keys:
        continue
      used_keys.add(key)
      if self.key2index.get(key) is not None:
        for j in range(len(self.columns_info)):
          if self.cells2watermark_bit[j].get(self.key2index[key]) is not None:
            watermark_bit_index = self.cells2watermark_bit[j][self.key2index[key]]
            watermark_bit2cells[j][watermark_bit_index].append(i)
            original_watermark_bit2cells[j][watermark_bit_index].append(self.key2index[key])
    counts = []
    for i in range(self.num_watermark_bits):
      count = np.asarray([0, 0], dtype=np.int64)
      for j, col in enumerate(self.columns_info):
        if col['type'] == 'num':
          noise = table.loc[watermark_bit2cells[j][i], col['name']].values - self.original_table.loc[original_watermark_bit2cells[j][i], col['name']].values
          count += self.count_from_num(noise, self.len_units[j])
        elif col['type'] == 'cat':
          count += self.count_from_cat(table.loc[watermark_bit2cells[j][i], col['name']], j)
      if count[0] > count[1]:
        watermark_bits.append(0)
      else:
        watermark_bits.append(1)
      counts.append(count)
    watermark_extracted = bit_list2int(watermark_bits)
    watermark_pred = None
    min_dist = self.num_watermark_bits
    for code in self.codes:
      dist = sum(int2bit_list(watermark_extracted ^ code, self.num_watermark_bits))
      if dist < min_dist:
        min_dist = dist
        watermark_pred = code
    return watermark_extracted, watermark_pred, counts
  
  def cal_label_accuracy(self, watermark_int_true: int, counts: list[np.ndarray]) -> float:
    watermark_bits = int2bit_list(watermark_int_true, self.num_watermark_bits)
    num_corrects = 0
    for i, b in enumerate(watermark_bits):
      num_corrects += counts[i][b]
    return num_corrects / sum(count.sum() for count in counts)


def main():
  columns_info = {
    'beijing': [
      {'name': 'pm2.5', 'type': 'num'},
    ],
    'default': [
      {'name': 'LIMIT_BAL', 'type': 'num'},
    ],
    'phishing': [
      {'name': 'having_IP_Address', 'type': 'cat'},
    ],
    'shoppers': [
      {'name': 'Administrative', 'type': 'num'},
    ]
  }[CFG_BASIC.DATA_NAME]
  keys_info = {
    'beijing': [
      {'name': 'year', 'type': 'cat', 'num_msbs': 8},
      {'name': 'month', 'type': 'cat', 'num_msbs': 8},
      {'name': 'day', 'type': 'cat', 'num_msbs': 8},
      {'name': 'hour', 'type': 'cat', 'num_msbs': 8},
      {'name': 'cbwd', 'type': 'cat', 'num_msbs': 8},
    ],
    'default': [
      {'name': 'SEX', 'type': 'cat', 'num_msbs': 8},
      {'name': 'EDUCATION', 'type': 'cat', 'num_msbs': 8},
      {'name': 'MARRIAGE', 'type': 'cat', 'num_msbs': 8},
      {'name': 'PAY_0', 'type': 'cat', 'num_msbs': 8},
      {'name': 'PAY_2', 'type': 'cat', 'num_msbs': 8},
      {'name': 'PAY_3', 'type': 'cat', 'num_msbs': 8},
      {'name': 'PAY_4', 'type': 'cat', 'num_msbs': 8},
      {'name': 'PAY_5', 'type': 'cat', 'num_msbs': 8},
    ],
    'phishing': [
      {'name': 'URL_Length', 'type': 'cat', 'num_msbs': 4},
      {'name': 'Shortining_Service', 'type': 'cat', 'num_msbs': 4},
      {'name': 'having_At_Symbol', 'type': 'cat', 'num_msbs': 4},
      {'name': 'double_slash_redirecting', 'type': 'cat', 'num_msbs': 4},
      {'name': 'Prefix_Suffix', 'type': 'cat', 'num_msbs': 4},
      {'name': 'having_Sub_Domain', 'type': 'cat', 'num_msbs': 4},
      {'name': 'SSLfinal_State', 'type': 'cat', 'num_msbs': 4},
      {'name': 'Domain_registeration_length', 'type': 'cat', 'num_msbs': 4},
      {'name': 'Favicon', 'type': 'cat', 'num_msbs': 4},
      {'name': 'port', 'type': 'cat', 'num_msbs': 4},
      {'name': 'HTTPS_token', 'type': 'cat', 'num_msbs': 4},
      {'name': 'Request_URL', 'type': 'cat', 'num_msbs': 4},
      {'name': 'URL_of_Anchor', 'type': 'cat', 'num_msbs': 4},
      {'name': 'Links_in_tags', 'type': 'cat', 'num_msbs': 4},
      {'name': 'SFH', 'type': 'cat', 'num_msbs': 4},
      {'name': 'Submitting_to_email', 'type': 'cat', 'num_msbs': 4},
    ],
    'shoppers': [
      {'name': 'Month', 'type': 'cat', 'num_msbs': 8},
      {'name': 'OperatingSystems', 'type': 'cat', 'num_msbs': 8},
      {'name': 'Browser', 'type': 'cat', 'num_msbs': 8},
      {'name': 'Region', 'type': 'cat', 'num_msbs': 8},
      {'name': 'TrafficType', 'type': 'cat', 'num_msbs': 8},
      {'name': 'VisitorType', 'type': 'cat', 'num_msbs': 8},
      {'name': 'Weekend', 'type': 'cat', 'num_msbs': 8},
      {'name': 'Revenue', 'type': 'cat', 'num_msbs': 8},
    ],
  }[CFG_BASIC.DATA_NAME]
  info = load_info_full(CFG_BASIC.DATA_DIR, CFG_BASIC.INFO_PATH, CFG_BASIC.TRANSFORM_LATENTS, embedding_path=CFG_EMBEDDING_MODEL.EMBEDDING_PATH, vae_path=CFG_VAE.PATH, device=CFG_BASIC.DEVICE, **get_embedding_module_param())

  codes = []
  with open(CFG_WATERMARK.CODE_PATH, 'r') as f:
    lines = f.readlines()
  for line in lines:
    codes.append(int(line))
  codes = codes[:CFG_WATERMARK.NUM_USERS]
  original_labels_hist = get_original_labels_hist(CFG_CLUSTER.LABELS_PATH)

  for _ in range(CFG_WATERMARK.NUM_WATERMARK_TRIALS):
    seed = get_random_seed()
    set_seed(seed)
    
    _, _, table_full, _, _ = syn_table(info, CFG_SYN.NUM_SAMPLES, CFG_CLUSTER.NUM_CLASSES, original_labels_hist, CFG_DM.NUM_SAMPLE_STEPS, CFG_DM.PATH, CFG_BASIC.DEVICE, CFG_DM.CORRECT_GUIDANCE, get_original_centers(CFG_CLUSTER.CENTERS_PATH), CFG_CLUSTER.DIM_RATIO, CFG_CLUSTER.KEY)
    tabular_mark = TabularMark(table_full, info, columns_info, keys_info, CFG_TABULAR_MARK.NUM_CELLS_RATIO, CFG_TABULAR_MARK.P_RATIO, CFG_TABULAR_MARK.NUM_UNITS, CFG_WATERMARK.NUM_WATERMARK_BITS, codes)
    watermark_int_true = random.choice(codes)
    non_watermarked_table = table_full.copy().reset_index(drop=True)
    table_full = tabular_mark.embed(watermark_int_true, seed)
    watermark_int_extracted, watermark_int_pred, counts = tabular_mark.extract(table_full)
    label_accuracy = tabular_mark.cal_label_accuracy(watermark_int_true, counts)
    r = {**get_res_header(seed), 'columns_info': columns_info, **check(watermark_int_true, watermark_int_extracted, watermark_int_pred, CFG_WATERMARK.NUM_WATERMARK_BITS, CFG_WATERMARK.MAX_NUM_ERROR_BITS), 'label_accuracy': label_accuracy}
    with open(f'{CFG_BASIC.RESULTS_DIR}/tabular_mark_detection.json', 'a') as f:
      f.write(json.dumps(r) + '\n')

    r = {**get_res_header(seed), 'columns_info': columns_info, 'quality_mode': 'average', **eval_all(table_full, info, CFG_SYN.REAL_DATA_PATH, CFG_SYN.TEST_DATA_PATH)}
    with open(f'{CFG_BASIC.RESULTS_DIR}/tabular_mark_quality.json', 'a') as f:
      f.write(json.dumps(r) + '\n')

    if CFG_BASIC.DATA_NAME != 'phishing':
      for ratio in [0.01]:
        print(f'ratio: {ratio}')
        table, _ = add_gauss_noise(table_full, info, ratio)
        table = table.sample(frac=1, replace=False).reset_index(drop=True)
        label_accuracy = -4
        watermark_int_extracted, watermark_int_pred, counts = tabular_mark.extract(table)
        # label_accuracy = tabular_mark.cal_label_accuracy(watermark_int_true, counts)
        r = {**get_res_header(seed), 'columns_info': columns_info, 'gauss_noise_ratio': ratio, **check(watermark_int_true, watermark_int_extracted, watermark_int_pred, CFG_WATERMARK.NUM_WATERMARK_BITS, CFG_WATERMARK.MAX_NUM_ERROR_BITS), 'label_accuracy': label_accuracy}
        with open(f'{CFG_BASIC.RESULTS_DIR}/tabular_mark_gauss_noise.json', 'a') as f:
          f.write(json.dumps(r) + '\n')

    if CFG_BASIC.DATA_NAME != 'phishing':
      for ratio in [0.01]:
        print(f'ratio: {ratio}')
        table, _ = add_uniform_noise(table_full, info, ratio)
        table = table.sample(frac=1, replace=False).reset_index(drop=True)
        label_accuracy = -4
        watermark_int_extracted, watermark_int_pred, counts = tabular_mark.extract(table)
        # label_accuracy = tabular_mark.cal_label_accuracy(watermark_int_true, counts)
        r = {**get_res_header(seed), 'columns_info': columns_info, 'gauss_noise_ratio': ratio, **check(watermark_int_true, watermark_int_extracted, watermark_int_pred, CFG_WATERMARK.NUM_WATERMARK_BITS, CFG_WATERMARK.MAX_NUM_ERROR_BITS), 'label_accuracy': label_accuracy}
        with open(f'{CFG_BASIC.RESULTS_DIR}/tabular_mark_uniform_noise.json', 'a') as f:
          f.write(json.dumps(r) + '\n')

    if CFG_BASIC.DATA_NAME != 'phishing':
      for ratio in [0.01]:
        print(f'ratio: {ratio}')
        table, _ = add_laplace_noise(table_full, info, ratio)
        table = table.sample(frac=1, replace=False).reset_index(drop=True)
        label_accuracy = -4
        watermark_int_extracted, watermark_int_pred, counts = tabular_mark.extract(table)
        # label_accuracy = tabular_mark.cal_label_accuracy(watermark_int_true, counts)
        r = {**get_res_header(seed), 'columns_info': columns_info, 'gauss_noise_ratio': ratio, **check(watermark_int_true, watermark_int_extracted, watermark_int_pred, CFG_WATERMARK.NUM_WATERMARK_BITS, CFG_WATERMARK.MAX_NUM_ERROR_BITS), 'label_accuracy': label_accuracy}
        with open(f'{CFG_BASIC.RESULTS_DIR}/tabular_mark_laplace_noise.json', 'a') as f:
          f.write(json.dumps(r) + '\n')

    for ratio in [0.1]:
      print(f'ratio: {ratio}')
      table, _, _ = insert(table_full, info, ratio)
      table = table.sample(frac=1, replace=False).reset_index(drop=True)
      label_accuracy = -4
      watermark_int_extracted, watermark_int_pred, counts = tabular_mark.extract(table)
      # label_accuracy = tabular_mark.cal_label_accuracy(watermark_int_true, counts)
      r = {**get_res_header(seed), 'columns_info': columns_info, 'sample_deletion_ratio': ratio, **check(watermark_int_true, watermark_int_extracted, watermark_int_pred, CFG_WATERMARK.NUM_WATERMARK_BITS, CFG_WATERMARK.MAX_NUM_ERROR_BITS), 'label_accuracy': label_accuracy}
      with open(f'{CFG_BASIC.RESULTS_DIR}/tabular_mark_sample_insertion.json', 'a') as f:
        f.write(json.dumps(r) + '\n')

    for ratio in [0.01]:
      print(f'ratio: {ratio}')
      table, _ = alter_cat(table_full, info, ratio)
      table, _ = alter_num(table, info, ratio)
      table = table.sample(frac=1, replace=False).reset_index(drop=True)
      label_accuracy = -4
      watermark_int_extracted, watermark_int_pred, counts = tabular_mark.extract(table)
      # label_accuracy = tabular_mark.cal_label_accuracy(watermark_int_true, counts)
      r = {**get_res_header(seed), 'columns_info': columns_info, 'alteration_ratio': ratio, **check(watermark_int_true, watermark_int_extracted, watermark_int_pred, CFG_WATERMARK.NUM_WATERMARK_BITS, CFG_WATERMARK.MAX_NUM_ERROR_BITS), 'label_accuracy': label_accuracy}
      with open(f'{CFG_BASIC.RESULTS_DIR}/tabular_mark_alteration.json', 'a') as f:
        f.write(json.dumps(r) + '\n')

    for ratio in [0.1]:
      print(f'ratio: {ratio}')
      table, _, _ = delete(table_full, info, ratio)
      table = table.sample(frac=1, replace=False).reset_index(drop=True)
      label_accuracy = -4
      watermark_int_extracted, watermark_int_pred, counts = tabular_mark.extract(table)
      # label_accuracy = tabular_mark.cal_label_accuracy(watermark_int_true, counts)
      r = {**get_res_header(seed), 'columns_info': columns_info, 'sample_deletion_ratio': ratio, **check(watermark_int_true, watermark_int_extracted, watermark_int_pred, CFG_WATERMARK.NUM_WATERMARK_BITS, CFG_WATERMARK.MAX_NUM_ERROR_BITS), 'label_accuracy': label_accuracy}
      with open(f'{CFG_BASIC.RESULTS_DIR}/tabular_mark_sample_deletion.json', 'a') as f:
        f.write(json.dumps(r) + '\n')

    token_dim = 4
    working_dir = f'{CFG_BASIC.ROOT_DIR}/watermark/regeneration_attack_vae/{CFG_WATERMARK.WATERMARK}-token_dim{token_dim}/{CFG_BASIC.DATA_NAME}/{CFG_CLUSTER.ALGORITHM}/{CFG_CLUSTER.NUM_CLASSES}/{CFG_EMBEDDING_MODEL.to_str()}/{watermark_int_true}'
    if os.path.exists(working_dir):
      continue
    os.makedirs(working_dir, exist_ok=True)
    non_watermarked_table.to_csv(f'{working_dir}/non_watermarked.csv', index=False)
    table_full.to_csv(f'{working_dir}/watermarked.csv', index=False)
    process_data(CFG_BASIC.DATA_NAME, f'{working_dir}/watermarked.csv')
    for ratio in [0.1]:
      reversed_table, reversed_num_norm, reversed_cat_norm, _, _ = attack(table_full, info, token_dim, working_dir, ratio)
      reversed_table.to_csv(f'{working_dir}/attacked_{ratio}.csv', index=False)
      reversed_table = reversed_table.sample(frac=1, replace=False).reset_index(drop=True)
      watermark_int_extracted, watermark_int_pred, counts = tabular_mark.extract(reversed_table)
      label_accuracy = tabular_mark.cal_label_accuracy(watermark_int_true, counts)
      r = {**get_res_header(seed), 'columns_info': columns_info, **check(watermark_int_true, watermark_int_extracted, watermark_int_pred, CFG_WATERMARK.NUM_WATERMARK_BITS, CFG_WATERMARK.MAX_NUM_ERROR_BITS), 'label_accuracy': label_accuracy, 'attack_model': f'token_dim{token_dim}-ratio{ratio}'}
      with open(f'{CFG_BASIC.RESULTS_DIR}/tabular_mark_regeneration_vae.json', 'a') as f:
        f.write(json.dumps(r) + '\n')
