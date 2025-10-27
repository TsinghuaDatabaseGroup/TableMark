import pandas as pd
from copy import deepcopy

from mle.mle import get_evaluator


def eval_mle(syn_data: pd.DataFrame, test_data: pd.DataFrame, info: dict):
    print('eval_mle', flush=True)
    syn_data = syn_data.copy()
    test_data = test_data.copy()
    info = deepcopy(info)

    syn_data = syn_data.to_numpy()
    test_data = test_data.to_numpy()
    task_type = info['task_type']

    evaluator = get_evaluator(task_type)

    failed = False

    try:
        if task_type == 'regression':
            best_r2_scores, best_rmse_scores = evaluator(syn_data, test_data, info)
            
            overall_scores = {}
            for score_name in [
                # 'best_r2_scores',
                'best_rmse_scores'
                ]:
                overall_scores[score_name] = {}
                
                scores = eval(score_name)
                for method in scores:
                    name = method['name']  
                    method.pop('name')
                    overall_scores[score_name][name] = method 
        else:
            best_f1_scores, best_weighted_scores, best_auroc_scores, best_acc_scores, best_avg_scores = evaluator(syn_data, test_data, info)

            overall_scores = {}
            for score_name in [
                'best_auroc_scores'
                ]:
                overall_scores[score_name] = {}
                
                scores = eval(score_name)
                for method in scores:
                    name = method['name']  
                    method.pop('name')
                    overall_scores[score_name][name] = method 
        print(f'Overall MLE Score: {overall_scores}')
    except (ValueError, KeyError, ZeroDivisionError):
        failed = True

    if task_type == 'regression':
        return {
            'rmse': (overall_scores['best_rmse_scores']['XGBRegressor']['RMSE'] if not failed else -1)
        }
    else:
        return {
            'binary_f1': (overall_scores['best_auroc_scores']['XGBClassifier']['binary_f1'] if not failed else -1),
            'auroc': (overall_scores['best_auroc_scores']['XGBClassifier']['roc_auc'] if not failed else -1),
        }
