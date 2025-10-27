import pandas as pd
import json
from sdmetrics.reports.single_table import QualityReport, DiagnosticReport
from copy import deepcopy

from globals import *


def reorder(real_data, syn_data, info):
    num_col_idx = info['num_col_idx']
    cat_col_idx = info['cat_col_idx']
    target_col_idx = info['target_col_idx']

    task_type = info['task_type']
    if task_type == 'regression':
        num_col_idx += target_col_idx
    else:
        cat_col_idx += target_col_idx

    real_num_data = real_data[num_col_idx]
    real_cat_data = real_data[cat_col_idx]

    new_real_data = pd.concat([real_num_data, real_cat_data], axis=1)
    new_real_data.columns = range(len(new_real_data.columns))

    syn_num_data = syn_data[num_col_idx]
    syn_cat_data = syn_data[cat_col_idx]
    
    new_syn_data = pd.concat([syn_num_data, syn_cat_data], axis=1)
    new_syn_data.columns = range(len(new_syn_data.columns))

    
    metadata = info['metadata']

    columns = metadata['columns']
    metadata['columns'] = {}

    inverse_idx_mapping = info['inverse_idx_mapping']

    for i in range(len(new_real_data.columns)):
        if i < len(num_col_idx):
            metadata['columns'][i] = columns[num_col_idx[i]]
        else:
            metadata['columns'][i] = columns[cat_col_idx[i-len(num_col_idx)]]
    

    return new_real_data, new_syn_data, metadata


def eval_density(syn_data: pd.DataFrame, real_data: pd.DataFrame, info: dict):
    syn_data = syn_data.copy()
    real_data = real_data.copy()
    info = deepcopy(info)

    real_data.columns = range(len(real_data.columns))
    syn_data.columns = range(len(syn_data.columns))

    metadata = info['metadata']
    metadata['columns'] = {int(key): value for key, value in metadata['columns'].items()}

    new_real_data, new_syn_data, metadata = reorder(real_data, syn_data, info)

    qual_report = QualityReport()
    qual_report.generate(new_real_data, new_syn_data, metadata)

    diag_report = DiagnosticReport()
    diag_report.generate(new_real_data, new_syn_data, metadata)

    quality =  qual_report.get_properties()
    Shape = quality['Score'][0]
    Trend = quality['Score'][1]
    print(f'Shape: {Shape}, Trend: {Trend}')

    return {'shape': Shape, 'trend': Trend}
