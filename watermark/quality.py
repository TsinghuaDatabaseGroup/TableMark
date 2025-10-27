import json
import sys

sys.path.append('.')
sys.path.append('../')
from globals import *
from tabsyn.sample import syn_table
from tabsyn.latent_utils import load_info_full
from watermark.watermark import create_watermark, int2bit_str, embed_random_watermark
from eval.main import eval_all


def main():
  for _ in range(CFG_WATERMARK.NUM_WATERMARK_TRIALS):
    seed = get_random_seed()
    set_seed(seed)
    original_labels_hist = get_original_labels_hist(CFG_CLUSTER.LABELS_PATH)
    info = load_info_full(CFG_BASIC.DATA_DIR, CFG_BASIC.INFO_PATH, CFG_BASIC.TRANSFORM_LATENTS, embedding_path=CFG_EMBEDDING_MODEL.EMBEDDING_PATH, vae_path=CFG_VAE.PATH, device=CFG_BASIC.DEVICE, **get_embedding_module_param())

    if CFG_WATERMARK.WATERMARK != 'none':
      watermark = create_watermark(**get_create_watermark_params(seed))

      if CFG_WATERMARK.QUALITY_MODE == 'average':
        try:
          use_cached_watermark_int, watermark_int_true, num_samples_per_class, loss, quad_loss, bound, gap = embed_random_watermark(CFG_WATERMARK.NUM_SAMPLES_PER_CLASS_CACHE_PATH, get_res_header(seed), CFG_BASIC.MODE, watermark)
        except (AttributeError, AssertionError):
          print('Failed !')
          r = {**get_res_header(seed), 'use_cached_watermark_int': False,
              'loss': -1, 'quad_loss': -1, 'bound': -1, 'gap': -1, 'watermark_true': 'none'}
          with open(f'{CFG_BASIC.RESULTS_DIR}/quality.json', 'a') as f:
            f.write(json.dumps(r) + '\n')
          continue
      else:
        if CFG_WATERMARK.QUALITY_MODE == 'worst':
          watermark_int_true, _ = watermark.get_worst_case_watermark_int(CFG_WATERMARK.GEN_CODE_LOSS != 'none')
        elif CFG_WATERMARK.QUALITY_MODE == 'best':
          watermark_int_true = watermark.get_best_case_watermark_int(CFG_WATERMARK.GEN_CODE_LOSS != 'none')
        elif CFG_WATERMARK.QUALITY_MODE.isdigit():
          watermark_int_true = int(CFG_WATERMARK.QUALITY_MODE)
        else:
          assert False, CFG_WATERMARK.QUALITY_MODE
        use_cached_watermark_int = False
        num_samples_per_class, loss, quad_loss, bound, gap = watermark.embed(watermark_int_true, CFG_SYN.NUM_SAMPLES)

      _, latents, table, num, cat = syn_table(info, CFG_SYN.NUM_SAMPLES, CFG_CLUSTER.NUM_CLASSES, num_samples_per_class, CFG_DM.NUM_SAMPLE_STEPS, CFG_DM.PATH, CFG_BASIC.DEVICE, CFG_DM.CORRECT_GUIDANCE, get_original_centers(CFG_CLUSTER.CENTERS_PATH), CFG_CLUSTER.DIM_RATIO, CFG_CLUSTER.KEY)
      
      r = {**get_res_header(seed), 'use_cached_watermark_int': use_cached_watermark_int,
           'loss': loss, 'quad_loss': quad_loss, 'bound': bound, 'gap': gap, 'watermark_true': '0b' + int2bit_str(watermark_int_true, CFG_WATERMARK.NUM_WATERMARK_BITS),
           **eval_all(table, info, CFG_SYN.REAL_DATA_PATH, CFG_SYN.TEST_DATA_PATH)}
    else:
      _, _, table, _, _ = syn_table(info, CFG_SYN.NUM_SAMPLES, CFG_CLUSTER.NUM_CLASSES, original_labels_hist, CFG_DM.NUM_SAMPLE_STEPS, CFG_DM.PATH, CFG_BASIC.DEVICE, CFG_DM.CORRECT_GUIDANCE, get_original_centers(CFG_CLUSTER.CENTERS_PATH), CFG_CLUSTER.DIM_RATIO, CFG_CLUSTER.KEY)
      r = {**get_res_header(seed), **eval_all(table, info, CFG_SYN.REAL_DATA_PATH, CFG_SYN.TEST_DATA_PATH)}
    
    with open(f'{CFG_BASIC.RESULTS_DIR}/quality.json', 'a') as f:
      f.write(json.dumps(r) + '\n')
