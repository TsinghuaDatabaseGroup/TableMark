import cluster.cluster
import eval.eval_query
import tabsyn.sample
import tabsyn.vae.main
import tabsyn.main
import eval.main
import watermark.gen_code
import watermark.detection
import watermark.freqwm_detection
import watermark.laplace_noise
import watermark.noise_deletion
import watermark.quality
import watermark.regeneration_attack_vae.regeneration_attack_vae
import watermark.sample_deletion
import watermark.gauss_noise
import watermark.alteration
import cluster.classification_error
import tabsyn.tabwak.main
import watermark.sample_insertion
import watermark.tabular_mark_detection
import watermark.tabwak.partition
from globals import *
import watermark.uniform_noise


if __name__ == '__main__':
	if CFG_BASIC.MODE == 'prepare_query':
		eval.eval_query.prepare_query()
		exit(0)
		
	if CFG_BASIC.MODE == 'cluster':
		module = cluster.cluster
	elif CFG_BASIC.MODE == 'vae_train':
		module = tabsyn.vae.main
	elif CFG_BASIC.MODE == 'tabsyn_train':
		module = tabsyn.main
	elif CFG_BASIC.MODE == 'sample':
		module = tabsyn.sample
	elif CFG_BASIC.MODE == 'eval':
		module = eval.main
	elif CFG_BASIC.MODE == 'gen_code':
		module = watermark.gen_code
	elif CFG_BASIC.MODE == 'watermark_detection':
		if CFG_WATERMARK.WATERMARK == 'freqwm':
			module = watermark.freqwm_detection
		elif CFG_WATERMARK.WATERMARK == 'tabular_mark':
			module = watermark.tabular_mark_detection
		elif CFG_WATERMARK.WATERMARK == 'tabwak_partition':
			module = watermark.tabwak.partition
		elif CFG_WATERMARK.WATERMARK == 'pair_compare_one_pair':
			module = watermark.detection
		else:
			assert False
	elif CFG_BASIC.MODE == 'watermark_quality':
		module = watermark.quality
	elif CFG_BASIC.MODE == 'watermark_sample_deletion':
		module = watermark.sample_deletion
	elif CFG_BASIC.MODE == 'watermark_sample_insertion':
		module = watermark.sample_insertion
	elif CFG_BASIC.MODE == 'watermark_gauss_noise':
		module = watermark.gauss_noise
	elif CFG_BASIC.MODE == 'watermark_laplace_noise':
		module = watermark.laplace_noise
	elif CFG_BASIC.MODE == 'watermark_uniform_noise':
		module = watermark.uniform_noise
	elif CFG_BASIC.MODE == 'watermark_noise_deletion':
		module = watermark.noise_deletion
	elif CFG_BASIC.MODE == 'watermark_alteration':
		module = watermark.alteration
	elif CFG_BASIC.MODE == 'watermark_regeneration_vae':
		module = watermark.regeneration_attack_vae.regeneration_attack_vae
	elif CFG_BASIC.MODE == 'classification_error':
		module = cluster.classification_error
	elif CFG_BASIC.MODE == 'tabwak_dm_train':
		module = tabsyn.tabwak.main
	else:
		raise RuntimeError(f'Unknown CFG_BASIC.MODE: {CFG_BASIC.MODE}')
    
	module.main()
