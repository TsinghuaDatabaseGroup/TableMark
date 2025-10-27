import pandas as pd
import numpy as np
import json
from copy import deepcopy
from tqdm import tqdm

from globals import *


def cal_query(df: pd.DataFrame, query: dict):
  if query['filters']:
    mask = np.ones(len(df), dtype=bool)
    for flt in query['filters']:
      col, op, val = flt['col'], flt['op'], flt['val']
      if op == '<=':
        mask &= df[col].values <= val
      elif op == '>=':
        mask &= df[col].values >= val
      else:
        mask &= df[col].values == val
      filtered_df = df[mask]
  else:
    filtered_df = df
  selectivity = len(filtered_df) / len(df)
  if query['agg_col'] != '*':
    return selectivity, filtered_df.agg({query['agg_col']: {'count': 'count', 'sum': 'sum', 'avg': 'mean', 'var': 'var'}[query['agg']]}).fillna(0).values.item()
  return selectivity, len(filtered_df)


def check(real_data: pd.DataFrame, query: dict, selectivity: float) -> tuple[float, float] | None:
  query_selectivity, query_res = cal_query(real_data, query)
  if (not np.isclose(query_selectivity, selectivity, rtol=0.05)) or np.isclose(query_res, 0, atol=1e-2):
    return None
  return query_selectivity, query_res


def prepare_query(num_samples: int = 1000):
  with open(CFG_BASIC.INFO_PATH, 'r') as num_filters:
    info = json.load(num_filters)
  real_data = pd.read_csv(CFG_SYN.REAL_DATA_PATH)
  real_data = process_table_dtypes(real_data, info).reset_index(drop=True)
  queries = {}
  discrete_cols_set = set()
  discrete_cols_list = []
  discrete_col_value_dict = {}
  continuous_cols = []
  with open(f'{CFG_BASIC.DATA_DIR}/{info["name"]}.json', 'r') as num_filters:
    info = json.load(num_filters)
  for col in info['columns']:
    if col['type'] == 'categorical':
      discrete_cols_set.add(col['name'])
      discrete_cols_list.append(col['name'])
      discrete_col_value_dict[col['name']] = col['i2s']
      if CFG_BASIC.DATA_NAME == 'phishing':
        discrete_col_value_dict[col['name']] = [int(tmp) for tmp in discrete_col_value_dict[col['name']]]
    else:
      continuous_cols.append(col['name'])

  np.random.seed(CFG_BASIC.SEED + 333)
  for agg in ['count',
              'avg',
              ]:
    queries[agg] = {}
    for selectivity in [0.01, 0.05, 0.2]:
      bar = tqdm(desc=f'{agg}{selectivity}', total=num_samples)
      queries[agg][selectivity] = []
      while len(queries[agg][selectivity]) < num_samples:
        query = {'agg': agg, 'filters': []}
        query['agg_col'] = np.random.choice(continuous_cols) if CFG_BASIC.DATA_NAME != 'phishing' else np.random.choice(discrete_cols_list)
        num_filters = np.random.randint(1, real_data.shape[1] + 1)
        filter_cols = np.random.choice(real_data.columns, num_filters, replace=False)
        tp = real_data.sample(1)
        discrete_filter_indices = []
        for i, col in enumerate(filter_cols):
          if col in discrete_cols_set:
            query['filters'].append({'col': col, 'op': '==', 'val': tp[col].values.item()})
            discrete_filter_indices.append(i)
          else:
            query['filters'].append({'col': col, 'op': np.random.choice(['<=', '>=']), 'val': tp[col].values.item()})
        if discrete_filter_indices and np.random.randint(0, 10) == 0:
          group_by_predicate_index = np.random.choice(discrete_filter_indices)
          group_by_col = query['filters'][group_by_predicate_index]['col']
          for val in discrete_col_value_dict[group_by_col]:
            new_query = deepcopy(query)
            new_query['filters'][group_by_predicate_index]['val'] = val
            r = check(real_data, new_query, selectivity)
            if r is not None:
              new_query['selectivity'], new_query['ground_truth'] = r
              queries[agg][selectivity].append(new_query)
              bar.update(1)
        else:
          r = check(real_data, query, selectivity)
          if r is not None:
            query['selectivity'], query['ground_truth'] = r
            queries[agg][selectivity].append(query)
            bar.update(1)
  with open(f'{CFG_BASIC.DATA_DIR}/query{QUERY}.json', 'w') as f:
    json.dump(queries, f, indent=2)
  print('finish prepare_query')


def eval_query(syn_data: pd.DataFrame, info: dict):
  print('eval_query', flush=True)
  syn_data = syn_data.reset_index(drop=True)
  info = deepcopy(info)

  with open(f'{CFG_BASIC.ROOT_DIR}/data/{info["name"]}/query{QUERY}.json', 'r') as f:
    queries = json.load(f)
    
  res = {'query': QUERY}
  for agg, selectivities_queries in queries.items():
    for selectivity, queries in selectivities_queries.items():
      errors = []
      selectivities = []
      for query in queries:
        query_selectivity, query_res = cal_query(syn_data, query)
        selectivities.append(query_selectivity)
        error = abs((query_res - query['ground_truth']) / query['ground_truth'])
        errors.append(error)
      res[agg + str(selectivity) + '_95th'] = np.quantile(errors, 0.95)

  print('finish eval_query_adv', flush=True)
  print(res)
  return res
