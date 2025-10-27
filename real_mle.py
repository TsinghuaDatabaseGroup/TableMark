import sys
import pandas as pd
import json
import warnings
warnings.filterwarnings('ignore')
sys.path.append('.')
sys.path.append('./eval/')
from eval.eval_mle import eval_mle

for dataname in [
  'beijing',
  'default',
  'phishing',
  'shoppers',
  ]:
  train_path = f'synthetic/{dataname}/real.csv'
  test_path = f'synthetic/{dataname}/test.csv'
  train_data = pd.read_csv(train_path)
  test_data = pd.read_csv(test_path)
  with open(f'data/{dataname}/info.json', 'r') as f:
    info = json.load(f)
  for _ in range(100):
    r = eval_mle(train_data, test_data, info)
    with open('watermark/results_final/real_mle.json', 'a') as f:
      f.write(json.dumps({'dataname': dataname, **r}) + '\n')
