import pandas as pd
import numpy as np
import gurobipy as grb
import sys
import json
import random
import os
from abc import ABC, abstractmethod
from scipy.stats import norm

sys.path.append('.')
sys.path.append('../')
from cluster.classify import predict_labels


def sketch(x: list | np.ndarray) -> tuple[float, float, float, float]:
  return np.mean(x), np.std(x), np.min(x), np.max(x)


def int2bit_str(num: int, num_bits: int) -> str:
  s = bin(num).lstrip('0b')[-num_bits:]
  return '0' * (num_bits - len(s)) + s


def int2bit_list(num: int, num_bits: int) -> list:
  s = int2bit_str(num, num_bits)
  return [int(c) for c in s]


def bit_list2int(bits: list) -> int:
  return int(''.join([str(digit) for digit in bits]), base=2)


def int_int_match_count(a: int, b: int, num_bits: int):
  xor_bit_list = int2bit_list(a ^ b, num_bits)
  return num_bits - sum(xor_bit_list)


def check(watermark_int_true: int, watermark_int_extracted: int, watermark_int_pred: int, num_watermark_bits: int, max_num_error_bits: int):
  extracted_true_match_count = int_int_match_count(watermark_int_extracted, watermark_int_true, num_watermark_bits)
  extracted_true_match_rate = extracted_true_match_count / num_watermark_bits
  extracted_pred_match_count = int_int_match_count(watermark_int_extracted, watermark_int_pred, num_watermark_bits)
  extracted_pred_match_rate = extracted_pred_match_count / num_watermark_bits
  correct = int((watermark_int_true == watermark_int_pred) and (extracted_true_match_count >= num_watermark_bits - max_num_error_bits))
  return {
    'watermark_true': '0b' + int2bit_str(watermark_int_true, num_watermark_bits),
    'watermark_extracted': '0b' + int2bit_str(watermark_int_extracted, num_watermark_bits),
    'watermark_pred': '0b' + int2bit_str(watermark_int_pred, num_watermark_bits), 'correct': correct,
    'extracted_true_match_count': extracted_true_match_count, 'extracted_true_match_rate': extracted_true_match_rate,
    'extracted_pred_match_count': extracted_pred_match_count, 'extracted_pred_match_rate': extracted_pred_match_rate
  }


def handle_num_samples_mismatch(num_samples: int, original_lables_hist: list) -> list:
  adjusted_labels_hist = original_lables_hist.copy()
  num_mismatch = abs(num_samples - sum(original_lables_hist))
  if num_mismatch == 0:
    return adjusted_labels_hist
  if num_mismatch / num_samples <= 0.001:
    while num_mismatch > 0:
      num_mismatch -= 1
      adjusted_labels_hist[np.argmax(adjusted_labels_hist)] -= 1
  else:
    assert False
  assert all(adjusted_labels_hist[i] > 0 for i in range(len(adjusted_labels_hist)))
  assert num_samples == sum(adjusted_labels_hist)
  return adjusted_labels_hist


class Watermark(ABC):
  def __init__(self,
               num_users: int,
               num_watermark_bits: int,
               codes: list, num_classes: int,
               ratio_num_samples_per_class_interval: float,
               quality_loss: str,
               max_bit_error_rate: float,
               deletion_rate: float,
               original_labels_hist: list,
               original_centers: np.ndarray,
               original_clusters: list,
               original_emd_path: str,
               num_samples_per_class_lower_bound: float,
               num_samples_per_class_upper_bound: float,
               seed: int):
    super().__init__()
    assert 2 ** num_watermark_bits >= num_users
    self.num_watermark_bits = num_watermark_bits
    self.num_classes = num_classes
    self.ratio_num_samples_per_class_interval = ratio_num_samples_per_class_interval
    self.quality_loss = quality_loss
    self.max_bit_error_rate = max_bit_error_rate
    self.deletion_rate = deletion_rate
    self.seed = seed
    self.user2watermark_int = codes
    self.original_labels_hist = original_labels_hist
    self.original_centers = original_centers
    self.num_samples_per_class_lower_bound = num_samples_per_class_lower_bound
    self.num_samples_per_class_upper_bound = num_samples_per_class_upper_bound
    self.original_centers_dist = np.linalg.norm(original_centers - original_centers[:, None], axis=-1, ord=2)

  def get_user_watermark_int(self, user_id: int) -> int:
    return self.user2watermark_int[user_id]

  @abstractmethod
  def embed(self, watermark_int: int, num_samples: int, verbose=False) -> tuple[list, float, float, float, float]:
    pass

  @abstractmethod
  def extract_watermark_int(self, table: pd.DataFrame, **kwargs) -> tuple[int, list]:
    pass

  @abstractmethod
  def get_worst_case_watermark_int(self, use_codes: bool) -> tuple[int, float]:
    pass

  @abstractmethod
  def get_best_case_watermark_int(self, use_codes: bool) -> int:
    pass
 
  @abstractmethod
  def get_simple_abs_loss(self, watermark_int: int) -> int:
    pass

  @abstractmethod
  def get_simple_w1_loss(self, watermark_int: int) -> int:
    pass

  @abstractmethod
  def get_simple_quad_loss(self, watermark_int: int) -> int:
    pass

  def extract(self, table: pd.DataFrame, **kwargs) -> tuple[int, int, int, list]:
    watermark_int_extracted, num_samples_per_class_pred = self.extract_watermark_int(table, **kwargs)
    max_match_rate = -1
    watermark_int_pred = None
    user_id_pred = None
    for user_id, watermark_int in enumerate(self.user2watermark_int):
      match_rate = int_int_match_count(watermark_int_extracted, watermark_int, self.num_watermark_bits)
      if match_rate > max_match_rate:
        max_match_rate = match_rate
        watermark_int_pred = watermark_int
        user_id_pred = user_id
    return watermark_int_extracted, watermark_int_pred, user_id_pred, num_samples_per_class_pred


class PairCompare(Watermark):
  def __init__(self, num_users: int, num_watermark_bits: int, codes: list, num_classes: int, ratio_num_samples_per_class_interval: float, quality_loss: str, original_labels_hist: list, original_centers: np.ndarray, max_bit_error_rate: float, deletion_rate: float, init_ratio_num_samples_per_class_interval: float, confusion_mat: np.ndarray, tao_approximation: float, time_limit: float, min_gap: float, callback_mode: str, num_samples_per_class_lower_bound: float, num_samples_per_class_upper_bound: float, original_clusters: list, original_emd_path: str, seed: int):
    super().__init__(
      num_users=num_users,
      num_watermark_bits=num_watermark_bits,
      codes=codes,
      num_classes=num_classes,
      ratio_num_samples_per_class_interval=ratio_num_samples_per_class_interval,
      quality_loss=quality_loss,
      max_bit_error_rate=max_bit_error_rate,
      deletion_rate=deletion_rate,
      original_labels_hist=original_labels_hist,
      original_centers=original_centers,
      num_samples_per_class_lower_bound=num_samples_per_class_lower_bound,
      num_samples_per_class_upper_bound=num_samples_per_class_upper_bound,
      original_clusters=original_clusters,
      original_emd_path=original_emd_path,
      seed=seed)
    self.tao_approximation = tao_approximation
    self.time_limit = time_limit
    self.min_gap = min_gap
    self.callback_mode = callback_mode
    self.init_ratio_num_samples_per_class_interval = init_ratio_num_samples_per_class_interval
    self.confusion_mat = confusion_mat
    self.pairs = None

  @abstractmethod
  def get_worst_case_watermark_int(self, use_codes: bool) -> tuple[int, float]:
    pass

  def get_best_case_watermark_int(self, use_codes: bool) -> int:
    if use_codes:
      return self.user2watermark_int[0]
    watermark_int_bits = []
    for lhs_class, rhs_class in self.pairs:
      watermark_int_bits.append(int(self.original_labels_hist[lhs_class] > self.original_labels_hist[rhs_class]))
    return bit_list2int(watermark_int_bits)
  
  def get_simple_loss(self, watermark_int: int) -> int:
    if not hasattr(self, 'best_watermark_int'):
      setattr(self, 'best_watermark_int', self.get_best_case_watermark_int(False))
    return (watermark_int ^ getattr(self, 'best_watermark_int')).bit_count()

  def get_simple_abs_loss(self, watermark_int: int) -> int:
    cost = 0
    watermark_int_bits = int2bit_list(watermark_int, self.num_watermark_bits)
    for i, (lhs_class, rhs_class) in enumerate(self.pairs):
      cost += (int(self.original_labels_hist[lhs_class] > self.original_labels_hist[rhs_class]) ^ watermark_int_bits[i]) * abs(self.original_labels_hist[lhs_class] - self.original_labels_hist[rhs_class])
    return cost

  def get_simple_w1_loss(self, watermark_int: int) -> float:
    cost = 0
    watermark_int_bits = int2bit_list(watermark_int, self.num_watermark_bits)
    for i, (lhs_class, rhs_class) in enumerate(self.pairs):
      cost += (int(self.original_labels_hist[lhs_class] > self.original_labels_hist[rhs_class]) ^ watermark_int_bits[i]) * abs(self.original_labels_hist[lhs_class] - self.original_labels_hist[rhs_class]) * self.original_centers_dist[lhs_class, rhs_class]
    return cost

  def get_simple_quad_loss(self, watermark_int: int) -> int:
    if not hasattr(self, 'pair_diff_quad'):
      setattr(self, 'pair_diff_quad', np.asarray([(self.original_labels_hist[p[0]] - self.original_labels_hist[p[1]]) ** 2 for p in self.pairs]))
      setattr(self, 'best_watermark_int', self.get_best_case_watermark_int(False))
    return getattr(self, 'pair_diff_quad')[np.where(np.unpackbits(np.array([watermark_int ^ getattr(self, 'best_watermark_int')], dtype=np.uint32).view(np.uint8)[::-1]))[0]].sum()

  def get_real_w1_loss(self, num_samples_per_class: list[int]) -> float:
    return -1
    adjusted_original_labels_hist = handle_num_samples_mismatch(sum(num_samples_per_class), self.original_labels_hist)
    model = grb.Model()
    pi = model.addVars(self.num_classes, self.num_classes, vtype=grb.GRB.CONTINUOUS, name='pi', lb=0)
    model.setObjective(grb.quicksum(self.original_centers_dist[i, j] * pi[i, j] for i in range(self.num_classes) for j in range(self.num_classes)), sense=grb.GRB.MINIMIZE)
    model.addConstrs(grb.quicksum(pi[i, j] for j in range(self.num_classes)) == num_samples_per_class[i] for i in range(self.num_classes))
    model.addConstrs(grb.quicksum(pi[i, j] for i in range(self.num_classes)) == adjusted_original_labels_hist[j] for j in range(self.num_classes))
    model.optimize()
    assert model.Status == grb.GRB.OPTIMAL, model.Status
    obj = model.ObjVal
    model.dispose()
    return float(obj)

  def embed(self, watermark_int: int, num_samples: int, verbose=True) -> tuple[list, float, float, float, float]:
    assert False

  def extract_watermark_int(self, table: pd.DataFrame, **kwargs) -> tuple[int, list]:
    labels = predict_labels(table=table, **kwargs)
    hist = pd.DataFrame(labels).value_counts().sort_index()
    hist = [hist[i] if i in hist.index else 0 for i in range(self.num_classes)]
    watermark_bits = [0] * self.num_watermark_bits
    for i, (lhs_class, rhs_class) in enumerate(self.pairs):
      watermark_bits[i] = int(hist[lhs_class] >= hist[rhs_class])
    return bit_list2int(watermark_bits), labels


class PairCompareOnePair(PairCompare):
  def __init__(self, num_users, num_watermark_bits, codes, num_classes, ratio_num_samples_per_class_interval, quality_loss, original_labels_hist, original_centers, max_bit_error_rate, deletion_rate, init_ratio_num_samples_per_class_interval, confusion_mat, tao_approximation, time_limit, min_gap, callback_mode, num_samples_per_class_lower_bound, num_samples_per_class_upper_bound, original_clusters, original_emd_path, class_path, seed):
    super().__init__(num_users, num_watermark_bits, codes, num_classes, ratio_num_samples_per_class_interval, quality_loss, original_labels_hist, original_centers, max_bit_error_rate, deletion_rate, init_ratio_num_samples_per_class_interval, confusion_mat, tao_approximation, time_limit, min_gap, callback_mode, num_samples_per_class_lower_bound, num_samples_per_class_upper_bound, original_clusters, original_emd_path, seed)
    assert num_classes >= 2 * num_watermark_bits
    with open(class_path, 'r') as f:
      self.classes = list(map(int, f.readline().replace('[', '').replace(']', '').replace(',', ' ').replace('  ', ' ').split()))
    self.other_classes = sorted(set(range(num_classes)) - set(self.classes))
    self.pairs = []
    assert quality_loss.startswith('w1_random_init') or quality_loss.startswith('w0_random_init') or quality_loss.startswith('quad_random_init') or quality_loss.startswith('s1_random_init')
    key = int(quality_loss[quality_loss.find('random_init') + len('random_init'):])
    assert key == 985
    rng = np.random.default_rng(key)
    rng.shuffle(self.classes)
    self.pairs = [[self.classes[2 * i], self.classes[2 * i + 1]] for i in range(self.num_watermark_bits)]
    self.class2another = {l: r for l, r in self.pairs} | {r: l for l, r in self.pairs}

  def correct_code(self):
    best_code = self.get_best_case_watermark_int(False)
    for i in range(len(self.user2watermark_int)):
      self.user2watermark_int[i] ^= best_code

  def get_worst_case_watermark_int(self, use_codes) -> tuple[int, float]:
    if use_codes:
      return self.user2watermark_int[-1], -1
    watermark_int_bits = []
    for lhs_class, rhs_class in self.pairs:
      watermark_int_bits.append(int(self.original_labels_hist[lhs_class] < self.original_labels_hist[rhs_class]))
    return bit_list2int(watermark_int_bits), -1

  def embed_with_bound(self, watermark_bits: list[int], adjusted_original_labels_hist: list[int], num_samples_per_class_lower_bounds: list[int], num_samples_per_class_upper_bounds: list[int], diff_bound: int, time_limit: float, previous_init: list[int] = None, model: grb.Model = None) -> tuple[list[float], float, float, float, grb.Model, float]:
    if model is None:
      num_samples = sum(adjusted_original_labels_hist)
      print(f'New Model ! num_samples: {num_samples}, Using Bounds: watermark: -{adjusted_original_labels_hist[self.classes[0]] - num_samples_per_class_lower_bounds[self.classes[0]]} +{num_samples_per_class_upper_bounds[self.classes[0]] - adjusted_original_labels_hist[self.classes[0]]}, non-watermark: -{(adjusted_original_labels_hist[self.other_classes[0]] - num_samples_per_class_lower_bounds[self.other_classes[0]]) if len(self.other_classes) > 0 else "None"}, +{(num_samples_per_class_upper_bounds[self.other_classes[0]] - adjusted_original_labels_hist[self.other_classes[0]]) if len(self.other_classes) > 0 else "None"}, diff: {diff_bound}')
      less_thans = {}
      greater_thans = {}
      for i, (l, r) in enumerate(self.pairs):
        if watermark_bits[i] == 0:
          less_thans[l] = r
          greater_thans[r] = l
        else:
          less_thans[r] = l
          greater_thans[l] = r
      num_samples_per_class_lower_bounds = [h if greater_thans.get(i) is None else max(h, num_samples_per_class_lower_bounds[greater_thans[i]] + diff_bound) for i, h in enumerate(num_samples_per_class_lower_bounds)]
      num_samples_per_class_upper_bounds = [h if less_thans.get(i) is None else min(h, num_samples_per_class_upper_bounds[less_thans[i]] - diff_bound) for i, h in enumerate(num_samples_per_class_upper_bounds)]

      num_samples_per_class_lower_bounds = [round(h) for h in num_samples_per_class_lower_bounds]
      num_samples_per_class_upper_bounds = [round(h) for h in num_samples_per_class_upper_bounds]

      model = grb.Model()
      model.setParam('OutputFlag', 1)
      model.setParam('TimeLimit', time_limit)
      model.setParam('Presolve', 2)
      vtype = grb.GRB.CONTINUOUS if self.ratio_num_samples_per_class_interval >= 0 else grb.GRB.INTEGER
      x = model.addVars(self.num_classes, vtype=vtype, name='x', lb=0 if self.ratio_num_samples_per_class_interval == -10000 else num_samples_per_class_lower_bounds, ub=float('inf') if self.ratio_num_samples_per_class_interval == -10000 else num_samples_per_class_upper_bounds)
     
      if self.quality_loss == 'quad' or self.quality_loss.startswith('quad_random_init'):
        model.setObjective(grb.quicksum((x[i] - adjusted_original_labels_hist[i]) ** 2 for i in range(self.num_classes)), sense=grb.GRB.MINIMIZE)
      elif self.quality_loss in ['w1_mean_init', 'w1_weighted_init'] or self.quality_loss.startswith('w1_random_init'):
        pi = model.addVars(self.num_classes, self.num_classes, vtype=vtype, name='pi', lb=0)
        model.setObjective(grb.quicksum(self.original_centers_dist[i, j] * pi[i, j] for i in range(self.num_classes) for j in range(self.num_classes)), sense=grb.GRB.MINIMIZE)
        model.addConstrs(grb.quicksum(pi[i, j] for j in range(self.num_classes)) == x[i] for i in range(self.num_classes))
        model.addConstrs(grb.quicksum(pi[i, j] for i in range(self.num_classes)) == adjusted_original_labels_hist[j] for j in range(self.num_classes))
      elif self.quality_loss.startswith('w0_random_init'):
        pi = model.addVars(self.num_classes, self.num_classes, vtype=vtype, name='pi', lb=0)
        model.setObjective(grb.quicksum(pi[i, j] for i in range(self.num_classes) for j in range(self.num_classes)), sense=grb.GRB.MINIMIZE)
        model.addConstrs(grb.quicksum(pi[i, j] for j in range(self.num_classes)) == x[i] for i in range(self.num_classes))
        model.addConstrs(grb.quicksum(pi[i, j] for i in range(self.num_classes)) == adjusted_original_labels_hist[j] for j in range(self.num_classes))
      elif self.quality_loss.startswith('s1_random_init'):
        t = model.addVars(self.num_classes, self.num_classes, vtype=vtype, name='t')
        model.addConstrs(t[j, i] == 0 for i in range(self.num_classes) for j in range(i, self.num_classes))
        model.setObjective(grb.quicksum([self.original_centers_dist[i, j] * t[i, j] for i in range(self.num_classes) for j in range(i + 1, self.num_classes)]), sense=grb.GRB.MINIMIZE)
        model.addConstrs(t[i, j] >= ((x[i] - x[j]) - (adjusted_original_labels_hist[i] - adjusted_original_labels_hist[j])) for i in range(self.num_classes) for j in range(i + 1, self.num_classes))
        model.addConstrs(t[i, j] >= -((x[i] - x[j]) - (adjusted_original_labels_hist[i] - adjusted_original_labels_hist[j])) for i in range(self.num_classes) for j in range(i + 1, self.num_classes))
      else:
        assert False, f'Invalid Loss Function: {self.quality_loss}'

      if not (self.quality_loss.__contains__('w0') or self.quality_loss.__contains__('w1')):
        model.addConstr(x.sum() == num_samples)
      assert self.ratio_num_samples_per_class_interval == -30000
      inits = []
      if isinstance(previous_init, list):
        inits = [('from previous stage', previous_init)]
      elif False and previous_init != 'no_init':
        model.reset()
        init_constrs = []
        for i, (lhs_class, rhs_class) in enumerate(self.pairs):
          min_class, max_class = (lhs_class, rhs_class) if watermark_bits[i] == 0 else (rhs_class, lhs_class)
          constr = model.addConstr(x[min_class] - x[max_class] <= -diff_bound)
          init_constrs.append(constr)
        saved_time_limit = model.Params.TimeLimit
        model.setParam('TimeLimit', 1)
        model.optimize()
        model.setParam('TimeLimit', saved_time_limit)
        for constr in init_constrs:
          model.remove(constr)
        if hasattr(x[0], 'x'):
          inits.append((diff_bound, [x[i].x for i in range(self.num_classes)]))
          print(f'Successfully Get Init for Diff Bound {diff_bound}')
        else:
          print(f'Failed to Get Init for Diff Bound {diff_bound}')

      helps_ge0 = model.addVars(self.num_watermark_bits, vtype=grb.GRB.CONTINUOUS, lb=0)
      C: np.ndarray = self.confusion_mat / self.confusion_mat.sum(1, keepdims=True)
      assert self.tao_approximation == 0

      remaining_rate = 1 - self.deletion_rate
      zis_30000 = []
      mus_30000 = []
      for i, (lhs_class, rhs_class) in enumerate(self.pairs):
        min_class, max_class = (lhs_class, rhs_class) if watermark_bits[i] == 0 else (rhs_class, lhs_class)
        involved_classes = [min_class, max_class]
        other_classes = sorted(set(range(self.num_classes)) - set(involved_classes))

        coef_exp = [remaining_rate * (C[k, min_class] - C[k, max_class]) for k in range(self.num_classes)]
        coef_var = [remaining_rate * (C[k, min_class] * (1 - remaining_rate * C[k, min_class]) +
                                      C[k, max_class] * (1 - remaining_rate * C[k, max_class]) +
                                      2 * remaining_rate * C[k, min_class] * C[k, max_class]) for k in range(self.num_classes)]
        sub_model = grb.Model()
        sub_model.setParam('OutputFlag', 0)
        hist = sub_model.addVars(self.num_classes, vtype=grb.GRB.INTEGER, lb=num_samples_per_class_lower_bounds, ub=num_samples_per_class_upper_bounds)
        sub_model.addConstr(grb.quicksum(hist) == num_samples)
        for j, (l, r) in enumerate(self.pairs):
          l, r = (l, r) if watermark_bits[j] == 0 else (r, l)
          sub_model.addConstr(hist[l] - hist[r] <= -diff_bound)
        sub_model.setObjective(grb.quicksum([coef_exp[j] * hist[j] for j in range(self.num_classes) if j in other_classes]), sense=grb.GRB.MAXIMIZE)
        sub_model.optimize()
        assert sub_model.Status == grb.GRB.OPTIMAL
        exp_upper_bound = sub_model.ObjVal
        sub_model.setObjective(grb.quicksum([coef_var[j] * hist[j] for j in range(self.num_classes) if j in other_classes]), sense=grb.GRB.MAXIMIZE)
        sub_model.optimize()
        assert sub_model.Status == grb.GRB.OPTIMAL
        var_upper_bound = sub_model.ObjVal

        zi = exp_upper_bound + grb.quicksum([x[k] * remaining_rate * (C[k, min_class] - C[k, max_class]) for k in involved_classes])
        mu = var_upper_bound + grb.quicksum([x[k] * remaining_rate * (
          C[k, min_class] * (1 - remaining_rate * C[k, min_class]) +
          C[k, max_class] * (1 - remaining_rate * C[k, max_class]) +
          2 * remaining_rate * C[k, min_class] * C[k, max_class]) for k in involved_classes]
        )

        model.addConstr(x[min_class] - x[max_class] <= -diff_bound)
        zis_30000.append(zi)
        mus_30000.append(mu)
         
      for i, (zi, mu) in enumerate(zip(zis_30000, mus_30000)):
        model.addConstr(helps_ge0[i] == zi ** 2 - mu * norm.ppf(1 - self.max_bit_error_rate) ** 2)
        model.addConstr(zi <= 0)

      for diff_bound, init in inits:
        model.reset()
        for i in range(self.num_classes):
          x[i].Start = init[i]
        saved_time_limit = model.Params.TimeLimit
        model.setParam('TimeLimit', 1)
        model.optimize()
        model.setParam('TimeLimit', saved_time_limit)
        if hasattr(x[0], 'x'):
          print(f'Init Successfully with Diff Bound {diff_bound} !')
          break
        print(f"Init Failed with Diff Bound {diff_bound} !")
    else:
      print('Continuing with Previous Model !')
      model.setParam('TimeLimit', time_limit)
      x = [model.getVarByName(f'x[{i}]') for i in range(self.num_classes)]

    model.optimize()
    assert model.Status in [grb.GRB.OPTIMAL, grb.GRB.TIME_LIMIT], f'{model.Status} not in [{grb.GRB.OPTIMAL}, {grb.GRB.TIME_LIMIT}]'
    obj = model.ObjVal
    bound = model.ObjBound
    gap = (obj - bound + 1e-8) / (obj + 1e-8)
    x = [x[i].x for i in range(self.num_classes)]
    print(f'obj: {obj}, bound: {bound}, gap: {gap}', flush=True)
    remaining_time = time_limit - model.Runtime % time_limit
    return x, float(obj), float(bound), float(gap), model, remaining_time

  def embed_original(self, watermark_bits: list[int], adjusted_original_labels_hist: list[int], time_limit: float, previous_init: list[int] = None):
    num_samples = sum(adjusted_original_labels_hist)
    model = grb.Model()
    model.setParam('OutputFlag', 1)
    model.setParam('TimeLimit', time_limit)
    model.setParam('Presolve', 2)
    vtype = grb.GRB.CONTINUOUS if self.ratio_num_samples_per_class_interval >= 0 else grb.GRB.INTEGER
    x = model.addVars(self.num_classes, vtype=vtype, name='x', lb=0, ub=float('inf'))
    if self.quality_loss == 'quad' or self.quality_loss.startswith('quad_random_init'):
      model.setObjective(grb.quicksum((x[i] - adjusted_original_labels_hist[i]) ** 2 for i in range(self.num_classes)), sense=grb.GRB.MINIMIZE)
    elif self.quality_loss in ['w1_mean_init', 'w1_weighted_init'] or self.quality_loss.startswith('w1_random_init'):
      pi = model.addVars(self.num_classes, self.num_classes, vtype=vtype, name='pi', lb=0)
      model.setObjective(grb.quicksum(self.original_centers_dist[i, j] * pi[i, j] for i in range(self.num_classes) for j in range(self.num_classes)), sense=grb.GRB.MINIMIZE)
      model.addConstrs(grb.quicksum(pi[i, j] for j in range(self.num_classes)) == x[i] for i in range(self.num_classes))
      model.addConstrs(grb.quicksum(pi[i, j] for i in range(self.num_classes)) == adjusted_original_labels_hist[j] for j in range(self.num_classes))
    elif self.quality_loss.startswith('w0_random_init'):
      pi = model.addVars(self.num_classes, self.num_classes, vtype=vtype, name='pi', lb=0)
      model.setObjective(grb.quicksum(pi[i, j] for i in range(self.num_classes) for j in range(self.num_classes)), sense=grb.GRB.MINIMIZE)
      model.addConstrs(grb.quicksum(pi[i, j] for j in range(self.num_classes)) == x[i] for i in range(self.num_classes))
      model.addConstrs(grb.quicksum(pi[i, j] for i in range(self.num_classes)) == adjusted_original_labels_hist[j] for j in range(self.num_classes))
    elif self.quality_loss.startswith('s1_random_init'):
      t = model.addVars(self.num_classes, self.num_classes, vtype=vtype, name='t')
      model.addConstrs(t[j, i] == 0 for i in range(self.num_classes) for j in range(i, self.num_classes))
      model.setObjective(grb.quicksum([self.original_centers_dist[i, j] * t[i, j] for i in range(self.num_classes) for j in range(i + 1, self.num_classes)]), sense=grb.GRB.MINIMIZE)
      model.addConstrs(t[i, j] >= ((x[i] - x[j]) - (adjusted_original_labels_hist[i] - adjusted_original_labels_hist[j])) for i in range(self.num_classes) for j in range(i + 1, self.num_classes))
      model.addConstrs(t[i, j] >= -((x[i] - x[j]) - (adjusted_original_labels_hist[i] - adjusted_original_labels_hist[j])) for i in range(self.num_classes) for j in range(i + 1, self.num_classes))
    else:
      assert False, f'Invalid Loss Function: {self.quality_loss}'

    if not (self.quality_loss.__contains__('w0') or self.quality_loss.__contains__('w1')):
      model.addConstr(x.sum() == num_samples)

    helps_ge0 = model.addVars(self.num_watermark_bits, vtype=grb.GRB.CONTINUOUS, lb=0)
    C: np.ndarray = self.confusion_mat / self.confusion_mat.sum(1, keepdims=True)
    remaining_rate = 1 - self.deletion_rate

    for i, (lhs_class, rhs_class) in enumerate(self.pairs):
      min_class, max_class = (lhs_class, rhs_class) if watermark_bits[i] == 0 else (rhs_class, lhs_class)
      zi = remaining_rate * (
        grb.quicksum([C[k, min_class] * x[k] for k in range(self.num_classes)])
        - grb.quicksum([C[k, max_class] * x[k] for k in range(self.num_classes)]))
      mu = remaining_rate * (
        grb.quicksum([C[k, l] * (1 - remaining_rate * C[k, l]) * x[k] for k in range(self.num_classes) for l in [min_class, max_class]])
      ) + 2 * remaining_rate ** 2 * (
        grb.quicksum([C[k, min_class] * C[k, max_class] * x[k] for k in range(self.num_classes)])
      )
      model.addConstr(helps_ge0[i] == zi ** 2 - mu * norm.ppf(1 - self.max_bit_error_rate) ** 2)
      model.addConstr(zi <= 0)
   
    if previous_init is not None:
      for i in range(self.num_classes):
        x[i].Start = previous_init[i]
    model.optimize()
    assert model.Status in [grb.GRB.OPTIMAL, grb.GRB.TIME_LIMIT], f'{model.Status} not in [{grb.GRB.OPTIMAL}, {grb.GRB.TIME_LIMIT}]'
    obj = model.ObjVal
    bound = model.ObjBound
    gap = (obj - bound + 1e-8) / (obj + 1e-8)
    x = [x[i].x for i in range(self.num_classes)]
    print(f'obj: {obj}, bound: {bound}, gap: {gap}', flush=True)
    return x, float(obj), float(bound), float(gap), model

  def embed_simple(self, watermark_bits: list[int], adjusted_original_labels_hist: list[int], ratio: float, time_limit: float):
    num_samples = sum(adjusted_original_labels_hist)
    model = grb.Model()
    model.setParam('OutputFlag', 1)
    model.setParam('TimeLimit', time_limit)
    model.setParam('Presolve', 2)
    vtype = grb.GRB.CONTINUOUS if self.ratio_num_samples_per_class_interval >= 0 else grb.GRB.INTEGER
    x = model.addVars(self.num_classes, vtype=vtype, name='x', lb=0, ub=float('inf'))
    if self.quality_loss == 'quad' or self.quality_loss.startswith('quad_random_init'):
      model.setObjective(grb.quicksum((x[i] - adjusted_original_labels_hist[i]) ** 2 for i in range(self.num_classes)), sense=grb.GRB.MINIMIZE)
    elif self.quality_loss in ['w1_mean_init', 'w1_weighted_init'] or self.quality_loss.startswith('w1_random_init'):
      pi = model.addVars(self.num_classes, self.num_classes, vtype=vtype, name='pi', lb=0)
      model.setObjective(grb.quicksum(self.original_centers_dist[i, j] * pi[i, j] for i in range(self.num_classes) for j in range(self.num_classes)), sense=grb.GRB.MINIMIZE)
      model.addConstrs(grb.quicksum(pi[i, j] for j in range(self.num_classes)) == x[i] for i in range(self.num_classes))
      model.addConstrs(grb.quicksum(pi[i, j] for i in range(self.num_classes)) == adjusted_original_labels_hist[j] for j in range(self.num_classes))
    elif self.quality_loss.startswith('w0_random_init'):
      pi = model.addVars(self.num_classes, self.num_classes, vtype=vtype, name='pi', lb=0)
      model.setObjective(grb.quicksum(pi[i, j] for i in range(self.num_classes) for j in range(self.num_classes)), sense=grb.GRB.MINIMIZE)
      model.addConstrs(grb.quicksum(pi[i, j] for j in range(self.num_classes)) == x[i] for i in range(self.num_classes))
      model.addConstrs(grb.quicksum(pi[i, j] for i in range(self.num_classes)) == adjusted_original_labels_hist[j] for j in range(self.num_classes))
    elif self.quality_loss.startswith('s1_random_init'):
      t = model.addVars(self.num_classes, self.num_classes, vtype=vtype, name='t')
      model.addConstrs(t[j, i] == 0 for i in range(self.num_classes) for j in range(i, self.num_classes))
      model.setObjective(grb.quicksum([self.original_centers_dist[i, j] * t[i, j] for i in range(self.num_classes) for j in range(i + 1, self.num_classes)]), sense=grb.GRB.MINIMIZE)
      model.addConstrs(t[i, j] >= ((x[i] - x[j]) - (adjusted_original_labels_hist[i] - adjusted_original_labels_hist[j])) for i in range(self.num_classes) for j in range(i + 1, self.num_classes))
      model.addConstrs(t[i, j] >= -((x[i] - x[j]) - (adjusted_original_labels_hist[i] - adjusted_original_labels_hist[j])) for i in range(self.num_classes) for j in range(i + 1, self.num_classes))
    else:
      assert False, f'Invalid Loss Function: {self.quality_loss}'

    if not (self.quality_loss.__contains__('w0') or self.quality_loss.__contains__('w1')):
      model.addConstr(x.sum() == num_samples)

    for i, (lhs_class, rhs_class) in enumerate(self.pairs):
      min_class, max_class = (lhs_class, rhs_class) if watermark_bits[i] == 0 else (rhs_class, lhs_class)
      model.addConstr(x[min_class] - x[max_class] <= -max(1, ratio * num_samples))
   
    model.optimize()
    assert model.Status in [grb.GRB.OPTIMAL, grb.GRB.TIME_LIMIT], f'{model.Status} not in [{grb.GRB.OPTIMAL}, {grb.GRB.TIME_LIMIT}]'
    obj = model.ObjVal
    bound = model.ObjBound
    gap = (obj - bound + 1e-8) / (obj + 1e-8)
    x = [x[i].x for i in range(self.num_classes)]
    print(f'obj: {obj}, bound: {bound}, gap: {gap}', flush=True)
    return x, float(obj), float(bound), float(gap), model

  def embed(self, watermark_int: int, num_samples: int) -> tuple[list, float, float, float, float]:
    print(f'seed = {self.seed}, watermark_int = {watermark_int}')
    print(self.num_samples_per_class_lower_bound, self.num_samples_per_class_upper_bound)
    adjusted_original_labels_hist = handle_num_samples_mismatch(num_samples, self.original_labels_hist)
    watermark_bits = int2bit_list(watermark_int, self.num_watermark_bits)

    assert self.ratio_num_samples_per_class_interval == -30000
    num_samples_per_class_lower_bounds = [max(0, h - int(0.01 * num_samples)) if i not in self.classes else h - int(0.01 * num_samples) for i, h in enumerate(adjusted_original_labels_hist)]
    num_samples_per_class_upper_bounds = [h + int(0.01 * num_samples) if i not in self.classes else h + int(0.01 * num_samples) for i, h in enumerate(adjusted_original_labels_hist)]
    diff_bound = int(0.999999999 + 0.0001 * num_samples)
    if self.num_samples_per_class_lower_bound.startswith('0'):
      x, obj, bound, gap, _ = self.embed_simple(watermark_bits, adjusted_original_labels_hist, float(self.num_samples_per_class_lower_bound), self.time_limit)
    elif self.num_samples_per_class_lower_bound == '-1':
      x, obj, bound, gap, _ = self.embed_original(watermark_bits, adjusted_original_labels_hist, self.time_limit)
    elif self.num_samples_per_class_lower_bound.__contains__('stage_final') and len(self.num_samples_per_class_lower_bound) == len(self.num_samples_per_class_upper_bound):
      # assert False
      num_stages = int(self.num_samples_per_class_lower_bound[0])
      non_watermark_ratio, watermark_ratio = map(float, self.num_samples_per_class_lower_bound.split('-')[-2:])
      assert non_watermark_ratio == watermark_ratio
      num_samples_per_class_lower_bounds = [max(1, h - int((watermark_ratio if i in self.classes else non_watermark_ratio) * num_samples)) for i, h in enumerate(adjusted_original_labels_hist)]
      num_samples_per_class_upper_bounds = [h + int((watermark_ratio if i in self.classes else non_watermark_ratio) * num_samples) for i, h in enumerate(adjusted_original_labels_hist)]
      diff_bound = 1
      print('stage: 0')

      while True:
        try:
          x, obj, bound, gap, model, _ = self.embed_with_bound(watermark_bits, adjusted_original_labels_hist, num_samples_per_class_lower_bounds, num_samples_per_class_upper_bounds, diff_bound, self.time_limit / num_stages)
          break
        except AssertionError:
          print(f'Failed, non_watermark_ratio: {non_watermark_ratio} -> {non_watermark_ratio * 2}, watermark_ratio: {watermark_ratio} -> {watermark_ratio * 2}')
          non_watermark_ratio *= 2
          watermark_ratio *= 2
          num_samples_per_class_lower_bounds = [max(1, h - int((watermark_ratio if i in self.classes else non_watermark_ratio) * num_samples)) for i, h in enumerate(adjusted_original_labels_hist)]
          num_samples_per_class_upper_bounds = [h + int((watermark_ratio if i in self.classes else non_watermark_ratio) * num_samples) for i, h in enumerate(adjusted_original_labels_hist)]

      prev_non_watermark_class_bound = int(non_watermark_ratio * num_samples)
      prev_watermark_class_bound = int(watermark_ratio * num_samples)
      for stage_id in range(1, num_stages):
        print(f'stage: {stage_id}')
        non_watermark_class_bound = max(abs(x[i] - adjusted_original_labels_hist[i]) for i in self.other_classes)
        watermark_class_bound = max(abs(x[i] - adjusted_original_labels_hist[i]) for i in self.classes)
        num_samples_per_class_lower_bounds = [max(1, adjusted_original_labels_hist[i] - (watermark_class_bound if i in self.classes else non_watermark_class_bound)) for i in range(self.num_classes)]
        num_samples_per_class_upper_bounds = [adjusted_original_labels_hist[i] + (watermark_class_bound if i in self.classes else non_watermark_class_bound) for i in range(self.num_classes)]
        if np.isclose(prev_non_watermark_class_bound, non_watermark_class_bound) and np.isclose(prev_watermark_class_bound, watermark_class_bound):
          x, obj, bound, gap, model, remaining_time = self.embed_with_bound(watermark_bits, adjusted_original_labels_hist, num_samples_per_class_lower_bounds, num_samples_per_class_upper_bounds, diff_bound, self.time_limit / num_stages, x, model)
        else:
          x, obj, bound, gap, model, remaining_time = self.embed_with_bound(watermark_bits, adjusted_original_labels_hist, num_samples_per_class_lower_bounds, num_samples_per_class_upper_bounds, diff_bound, self.time_limit / num_stages, x)
        prev_non_watermark_class_bound = non_watermark_class_bound
        prev_watermark_class_bound = watermark_class_bound
      model.dispose()
    elif self.num_samples_per_class_lower_bound.__contains__('stage_splus') and len(self.num_samples_per_class_lower_bound) == len(self.num_samples_per_class_upper_bound):
      num_stages = int(self.num_samples_per_class_lower_bound[0])
      non_watermark_ratio, watermark_ratio = map(float, self.num_samples_per_class_lower_bound.split('-')[-2:])
      assert non_watermark_ratio == watermark_ratio
      ratio = non_watermark_ratio
      num_samples_per_class_lower_bounds = [max(1, h - int(ratio * num_samples)) for i, h in enumerate(adjusted_original_labels_hist)]
      num_samples_per_class_upper_bounds = [h + int(ratio * num_samples) for i, h in enumerate(adjusted_original_labels_hist)]
      diff_bound = 1
      print('stage: 0')
      while True:
        try:
          x, obj, bound, gap, model, _ = self.embed_with_bound(watermark_bits, adjusted_original_labels_hist, num_samples_per_class_lower_bounds, num_samples_per_class_upper_bounds, diff_bound, self.time_limit / num_stages)
          break
        except AssertionError:
          print(f'Failed, ratio: {ratio} -> {ratio * 2}')
          ratio *= 2
          num_samples_per_class_lower_bounds = [max(1, h - int(ratio * num_samples)) for i, h in enumerate(adjusted_original_labels_hist)]
          num_samples_per_class_upper_bounds = [h + int(ratio * num_samples) for i, h in enumerate(adjusted_original_labels_hist)]
      prev_bound = int(ratio * num_samples)
      for stage_id in range(1, num_stages - 1):
        print(f'stage: {stage_id}')
        all_bound = max(abs(x[i] - adjusted_original_labels_hist[i]) for i in range(self.num_classes))
        num_samples_per_class_lower_bounds = [max(1, adjusted_original_labels_hist[i] - all_bound) for i in range(self.num_classes)]
        num_samples_per_class_upper_bounds = [adjusted_original_labels_hist[i] + all_bound for i in range(self.num_classes)]
        if np.isclose(prev_bound, all_bound):
          x, obj, bound, gap, model, remaining_time = self.embed_with_bound(watermark_bits, adjusted_original_labels_hist, num_samples_per_class_lower_bounds, num_samples_per_class_upper_bounds, diff_bound, self.time_limit / num_stages, x, model)
        else:
          x, obj, bound, gap, model, remaining_time = self.embed_with_bound(watermark_bits, adjusted_original_labels_hist, num_samples_per_class_lower_bounds, num_samples_per_class_upper_bounds, diff_bound, self.time_limit / num_stages, x)
        prev_bound = all_bound
      print(f'stage: {num_stages - 1}')
      num_samples_per_class_lower_bounds = [min(x[i], adjusted_original_labels_hist[i]) - 1e-6 for i in range(self.num_classes)]
      num_samples_per_class_upper_bounds = [max(x[i], adjusted_original_labels_hist[i]) + 1e-6 for i in range(self.num_classes)]
      x, obj, bound, gap, model, _ = self.embed_with_bound(watermark_bits, adjusted_original_labels_hist, num_samples_per_class_lower_bounds, num_samples_per_class_upper_bounds, diff_bound, self.time_limit / num_stages, x)
      model.dispose()
    else:
      assert False
   
    num_samples_per_class = [round(x[i]) for i in range(self.num_classes)]
    total = sum(num_samples_per_class)
    assert abs(total - num_samples) <= 0.005 * num_samples, f'Maybe Inaccurate: {abs(total - num_samples)} !'
    if total < num_samples:
      num_samples_per_class[np.argmin(num_samples_per_class)] += (num_samples - total)
    else:
      num_samples_per_class[np.argmax(num_samples_per_class)] -= (total - num_samples)
    quad_loss = np.var([n - o for n, o in zip(num_samples_per_class, adjusted_original_labels_hist)])
    mae = sum([abs(n - o) for n, o in zip(num_samples_per_class, adjusted_original_labels_hist)]) / self.num_classes
    print(f'quad_loss: {quad_loss:.0f}', flush=True)
    return num_samples_per_class, float(mae), float(quad_loss), float(bound), float(gap)


def create_watermark(watermark_name: str, num_users: int, num_watermark_bits: int, codes_path: str, num_classes: int, ratio_num_samples_per_class_interval: float, quality_loss: str, original_labels_hist: list, original_centers: np.ndarray, max_bit_error_rate: float, deletion_rate: float, confusion_mat: np.ndarray, tao_approximation: float, init_ratio_num_samples_per_class_interval: float, time_limit: int, min_gap: float, callback_mode: str, num_samples_per_class_lower_bound: float, num_samples_per_class_upper_bound: float, original_clusters: list, original_emd_path: str, group_path: str, class_path: str, seed: int) -> Watermark:
  codes = []
  if codes_path != '':
    with open(codes_path, 'r') as f:
      lines = f.readlines()[:num_users]
    for line in lines:
      codes.append(int(line))
  if watermark_name == 'pair_compare_one_pair':
    watermark = PairCompareOnePair(
      num_users=num_users,
      num_watermark_bits=num_watermark_bits,
      codes=codes,
      num_classes=num_classes,
      ratio_num_samples_per_class_interval=ratio_num_samples_per_class_interval,
      quality_loss=quality_loss,
      original_labels_hist=original_labels_hist,
      original_centers=original_centers,
      max_bit_error_rate=max_bit_error_rate,
      deletion_rate=deletion_rate,
      confusion_mat=confusion_mat,
      tao_approximation=tao_approximation,
      init_ratio_num_samples_per_class_interval=init_ratio_num_samples_per_class_interval,
      time_limit=time_limit,
      min_gap=min_gap,
      callback_mode=callback_mode,
      num_samples_per_class_lower_bound=num_samples_per_class_lower_bound,
      num_samples_per_class_upper_bound=num_samples_per_class_upper_bound,
      original_clusters=original_clusters,
      original_emd_path=original_emd_path,
      class_path=class_path,
      seed=seed
    )
  else:
    assert False, f'Invalid Watermark: {watermark_name}'
  if codes_path.__contains__('general'):
    watermark.correct_code()
  return watermark


def cache_num_samples_per_class(cache_path: str, key: dict, mode: str, watermark_int_true: int, num_samples_per_class: list, loss: float, quad_loss: float, bound: float, gap: float) -> None:
  cache_entry = {**key, 'watermark_int_true': watermark_int_true, 'num_samples_per_class': num_samples_per_class, 'loss': loss, 'quad_loss': quad_loss, 'bound': bound, 'gap': gap, 'mode': mode}
  with open(cache_path, 'a') as f:
    f.write(json.dumps(cache_entry) + '\n')


def key_matches_entry(key: dict, entry: dict) -> bool:
  if entry.get('enable') == False:
    return False
  for k, v in key.items():
    if k not in ['enable', 'seed', 'quality_mode', 'classifier'] and entry.get(k) != v:
      return False
  return True


def load_num_samples_per_class(cache_path: str, key: dict, mode: str) -> tuple[list, float, float, float, float] | None:
  cache_entries = []
  if os.path.exists(cache_path):
    with open(cache_path, 'r') as f:
      cache_entries += [json.loads(line) for line in f.readlines()]
  
  matched_entries = {}
  for cache_entry in cache_entries:
    if key_matches_entry(key, cache_entry):
      matched_entries.setdefault(cache_entry['watermark_int_true'], []).append(cache_entry)
  if not matched_entries:
    print(f'Key not Match !')
    return None
  for watermark_int_true, entries in matched_entries.items():
    used_modes = [entry['mode'] for entry in entries]
    if mode not in used_modes:
      print(f'Key Matched, and no Conflict Mode for {mode} !')
      return entries[0]['watermark_int_true'], entries[0]['num_samples_per_class'], entries[0]['loss'], entries[0]['quad_loss'], entries[0]['bound'], entries[0]['gap']
  print(f'Key Matched, but Conflict Mode {mode} !')
  return None


def embed_random_watermark(cache_path: str, key: dict, mode: str, watermark: Watermark) -> tuple[bool, int, list, float, float, float, float]:
  cache_entry = load_num_samples_per_class(cache_path, key, mode)
  if cache_entry is not None:
    print(f"Use Cached Watermark Int for {mode} !")
    watermark_int_true, num_samples_per_class, loss, quad_loss, bound, gap = cache_entry
    cache_num_samples_per_class(cache_path, key, mode, watermark_int_true, num_samples_per_class, loss, quad_loss, bound, gap)
    print(f"Successfully Cached Watermark Int for {mode} !")
    use_cached_watermark_int = True
  else:
    print(f"Try Generating New Watermark Int for {mode} !")
    user_id_true = random.randint(0, key['num_users'] - 1)
    watermark_int_true = watermark.get_user_watermark_int(user_id_true)
    num_samples_per_class, loss, quad_loss, bound, gap = watermark.embed(watermark_int_true, key['num_samples'])
    cache_num_samples_per_class(cache_path, key, mode, watermark_int_true, num_samples_per_class, loss, quad_loss, bound, gap)
    print(f"Successfully Cached Watermark Int for {mode} !")
    use_cached_watermark_int = False
  return use_cached_watermark_int, watermark_int_true, num_samples_per_class, loss, quad_loss, bound, gap
