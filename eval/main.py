import sys
import os
import json
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), './'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from globals import *
from tabsyn.latent_utils import load_info_simple
from eval_density import eval_density
from eval_detection import eval_detection
from eval_mle import eval_mle
from eval_query import eval_query


def eval_all(syn_data: pd.DataFrame, info: dict, real_data_path: str, test_data_path: str):
  syn_data.to_csv(CFG_BASIC.EVAL_CSV_PATH, index=False)
  syn_data = process_table_dtypes(pd.read_csv(CFG_BASIC.EVAL_CSV_PATH), info)  # in case of some synthcity lib bugs
  real_data = pd.read_csv(real_data_path)
  test_data = pd.read_csv(test_data_path)
  real_data = process_table_dtypes(real_data, info)
  test_data = process_table_dtypes(test_data, info)
  syn_data.columns = real_data.columns
  original_num_samples = len(syn_data)
  syn_data = syn_data.dropna(how='any').reset_index(drop=True)
  print(f'NULL: {1 - len(syn_data) / original_num_samples}')
  print(f'Syn Data: {len(syn_data)}, Real Data: {len(real_data)}, Test Data: {len(test_data)}')
  
  return {
    **eval_density(syn_data, real_data, info),
    **eval_detection(syn_data, real_data, info),
    **eval_mle(syn_data, test_data, info),
    **eval_query(syn_data, info),
  }


def main():
  info = load_info_simple(CFG_BASIC.INFO_PATH)
  syn_data = pd.read_csv(CFG_SYN.SYN_DATA_PATH)
  syn_data = process_table_dtypes(syn_data, info)
  res = eval_all(syn_data, info, CFG_SYN.REAL_DATA_PATH, CFG_SYN.TEST_DATA_PATH)
  with open(CFG_SYN.EVAL_RES_PATH, 'a') as f:
    f.write(json.dumps(res) + '\n')
