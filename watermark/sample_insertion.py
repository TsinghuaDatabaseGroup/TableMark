import json
import sys

sys.path.append('.')
sys.path.append('../')
from globals import *
from tabsyn.sample import syn_table
from tabsyn.latent_utils import load_info_full, insert
from watermark.watermark import create_watermark, check, embed_random_watermark


def main():
  for _ in range(CFG_WATERMARK.NUM_WATERMARK_TRIALS):
    seed = get_random_seed()
    set_seed(seed)

    watermark = create_watermark(**get_create_watermark_params(seed))
    use_cached_watermark_int, watermark_int_true, num_samples_per_class, loss, quad_loss, bound, gap = embed_random_watermark(CFG_WATERMARK.NUM_SAMPLES_PER_CLASS_CACHE_PATH, get_res_header(seed), CFG_BASIC.MODE, watermark)

    info = load_info_full(CFG_BASIC.DATA_DIR, CFG_BASIC.INFO_PATH, CFG_BASIC.TRANSFORM_LATENTS, embedding_path=CFG_EMBEDDING_MODEL.EMBEDDING_PATH, vae_path=CFG_VAE.PATH, device=CFG_BASIC.DEVICE, **get_embedding_module_param())
    _, full_latents, full_table, full_num, full_cat = syn_table(info, CFG_SYN.NUM_SAMPLES, CFG_CLUSTER.NUM_CLASSES, num_samples_per_class, CFG_DM.NUM_SAMPLE_STEPS, CFG_DM.PATH, CFG_BASIC.DEVICE, CFG_DM.CORRECT_GUIDANCE, get_original_centers(CFG_CLUSTER.CENTERS_PATH), CFG_CLUSTER.DIM_RATIO, CFG_CLUSTER.KEY)
    
    for ratio in [0, 0.1, 0.2, 0.3, 0.4, 0.5]:
      print(f'ratio: {ratio}')
      table, num, cat = insert(full_table, info, ratio)

      watermark_int_extracted, watermark_int_pred, _, _ = watermark.extract(table, classifier=CFG_CLUSTER.CLASSIFIER, classifier_path=CFG_CLUSTER.CLASSIFIER_PATH, info=info, num=num, cat=cat, latents=None, centers_path=CFG_CLUSTER.CENTERS_PATH, device=CFG_BASIC.DEVICE, dim_ratio=CFG_CLUSTER.DIM_RATIO, key=CFG_CLUSTER.KEY)
      r = {**get_res_header(seed), 'use_cached_watermark_int': use_cached_watermark_int, 'loss': loss, 'quad_loss': quad_loss, 'bound': bound, 'gap': gap, **check(watermark_int_true, watermark_int_extracted, watermark_int_pred, CFG_WATERMARK.NUM_WATERMARK_BITS, CFG_WATERMARK.MAX_NUM_ERROR_BITS), 'sample_deletion_ratio': ratio}
      with open(f'{CFG_BASIC.RESULTS_DIR}/sample_insertion.json', 'a') as f:
        f.write(json.dumps(r) + '\n')
