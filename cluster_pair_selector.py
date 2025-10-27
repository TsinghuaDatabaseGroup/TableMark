import numpy as np
import os


def choose(v: list, m: int) -> list[int]:
    sorted_with_indices = sorted(((val, idx) for idx, val in enumerate(v)), key=lambda x: x[0])
    n = len(sorted_with_indices)
    
    if m <= 0:
        return []
    if m >= n:
        return sorted(idx for val, idx in sorted_with_indices)
    
    min_spread = float('inf')
    best_start = 0
    
    # Slide the window of size m to find the minimal spread
    for i in range(n - m + 1):
        current_spread = np.std([sorted_with_indices[l][0] for l in range(i, i + m)])
        if current_spread < min_spread:
            min_spread = current_spread
            best_start = i
    
    # Extract the indices from the best window and sort them
    chosen_indices = [sorted_with_indices[i][1] for i in range(best_start, best_start + m)]
    chosen_indices.sort()
    
    return chosen_indices

num_classes = 256
datanames = [
    'beijing',
    'default',
    'phishing',
    'shoppers',
    ]
dim_ratio = 'correct-pca-0.99'
cluster_alg = 'kmeans100-1000'

import sys
sys.path.append('.')

for dataname in datanames:
    hist_path = f'cluster/{dataname}/{cluster_alg}/{num_classes}/original-num_layers2-final/{dim_ratio}-985/label_hist.txt'
    with open(hist_path, 'r') as f:
        hist = list(map(int, f.readline().replace('[', '').replace(']', '').replace(',', ' ').replace('  ', ' ').split()))
        assert len(hist) == num_classes
    print(dataname)
    for m in [64]:
        path = f'cluster/{dataname}/{cluster_alg}/{len(hist)}/original-num_layers2-final/{dim_ratio}-985/class4one_pair_{m}_average_num_std.txt'
        res = choose(hist, m)
        print(res)
        sorted_choosen = sorted(np.asarray(hist)[res])
        print(f'choose: {m:3d}: {np.std(np.asarray(hist)[res])}, {np.mean([abs(sorted_choosen[i] - sorted_choosen[m - i - 1]) for i in range(m // 2)])}', flush=True)
        if not os.path.exists(path):
            with open(path, 'w') as f:
              f.write(str(res))

    print()
