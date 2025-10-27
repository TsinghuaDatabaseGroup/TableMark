import json
import sys

sys.path.append('.')
sys.path.append('../')
from globals import *
from tabsyn.sample import syn_table
from tabsyn.latent_utils import load_info_full, add_gauss_noise
from watermark.watermark import create_watermark, embed_random_watermark, check


def main():
  if CFG_BASIC.DATA_NAME == 'phishing':
    return
  for exp_id in range(CFG_WATERMARK.NUM_WATERMARK_TRIALS):
    seed = get_random_seed()
    set_seed(seed)

    watermark = create_watermark(**get_create_watermark_params(seed))
    use_cached_watermark_int, watermark_int_true, num_samples_per_class, loss, quad_loss, bound, gap = embed_random_watermark(CFG_WATERMARK.NUM_SAMPLES_PER_CLASS_CACHE_PATH, get_res_header(seed), CFG_BASIC.MODE, watermark)

    info = load_info_full(CFG_BASIC.DATA_DIR, CFG_BASIC.INFO_PATH, CFG_BASIC.TRANSFORM_LATENTS, embedding_path=CFG_EMBEDDING_MODEL.EMBEDDING_PATH, vae_path=CFG_VAE.PATH, device=CFG_BASIC.DEVICE, **get_embedding_module_param())
    _, latents_full, table_full, num_full, cat_full = syn_table(info, CFG_SYN.NUM_SAMPLES, CFG_CLUSTER.NUM_CLASSES, num_samples_per_class, CFG_DM.NUM_SAMPLE_STEPS, CFG_DM.PATH, CFG_BASIC.DEVICE, CFG_DM.CORRECT_GUIDANCE, get_original_centers(CFG_CLUSTER.CENTERS_PATH), CFG_CLUSTER.DIM_RATIO, CFG_CLUSTER.KEY, shuffle=False)
    
    for ratio in [0, 0.01, 0.02, 0.03, 0.04, 0.05]:
      print(f'ratio: {ratio}')
      latents = latents_full.clone().detach()
      cat = cat_full.copy()
      table, num = add_gauss_noise(table_full, info, ratio)

      watermark_int_extracted, watermark_int_pred, _, _ = watermark.extract(table, classifier=CFG_CLUSTER.CLASSIFIER, classifier_path=CFG_CLUSTER.CLASSIFIER_PATH, info=info, num=num, cat=cat, latents=latents.cpu().detach().numpy(), centers_path=CFG_CLUSTER.CENTERS_PATH, device=CFG_BASIC.DEVICE, dim_ratio=CFG_CLUSTER.DIM_RATIO, key=CFG_CLUSTER.KEY)
      r = {**get_res_header(seed), 'use_cached_watermark_int': use_cached_watermark_int, 'loss': loss, 'quad_loss': quad_loss, 'bound': bound, 'gap': gap, 'gauss_noise_ratio': ratio, **check(watermark_int_true, watermark_int_extracted, watermark_int_pred, CFG_WATERMARK.NUM_WATERMARK_BITS, CFG_WATERMARK.MAX_NUM_ERROR_BITS)}
      with open(f'{CFG_BASIC.RESULTS_DIR}/gauss_noise.json', 'a') as f:
        f.write(json.dumps(r) + '\n')
