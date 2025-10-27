# TableMark: A Multi-bit Watermark for Synthetic Tabular Data

### Environment

#### Python

```
conda create -n TableMark python=3.10
conda activate TableMark
pip install -r requirements.txt
```

#### Gurobi

You need to apply for a Gurobi license from [here](https://license.gurobi.com/manager/licenses/) and put the lisence file (i.e., gurobi.lic) in your home directory.


### Reproducing our Experimental Results

#### Step 0

We have uploaded the split datasets, clustering results, selected cluster pairs, generated watermark database, trained models, the estimated transition matrix, and aggregation queries that are used to evaluate synthetic tables' performance on statistical analyses, so there's no need for data preparation or model training. 

As the only prerequisite, you just need to create the following folders to store results.

```
mkdir out
mkdir watermark/results_final
```

#### Step 1

Execute the following command to obtain the machine learning performance of the original table (i.e., train and val sets) from each dataset. This command needs to be run **only once**.

```
python real_mle.py
```

#### Step 2

We have provided the experimental configurations in the ```experiments/``` folder. You can simply copy and paste them to ```exp_config.json```, and then execute the following command.

```
python driver.py
```

This command will silently run the experiments configured in ```exp_config.json``` using the ```nohup``` tool.

#### Step 3

We have provided Jupyter notebook files in the ```watermark/ipynb/``` folder to help you analyze the complicated results over multiple trials (i.e., average results and visualize them) for reproducing our experimental results.

- ```detection.ipynb```: Traceability accuracy without attacks.
- ```quality.ipynb```: Utility metrics and ablation studies on optimization mechanisms.
- ```gauss_noise.ipynb```: Traceability accuracy under the Gaussian perturbation attack.
- ```laplace_noise.ipynb```: Traceability accuracy under the Laplace perturbation attack.
- ```uniform_noise.ipynb```: Traceability accuracy under the uniform perturbation attack.
- ```alteration.ipynb```: Traceability accuracy under the alteration attack.
- ```sample_deletion.ipynb```: Traceability accuracy under the tuple-deletion attack.
- ```sample_insertion.ipynb```: Traceability accuracy under the tuple-insertion attack.
- ```regeneration_vae.ipynb```: Traceability accuracy under the regeneration attack.
- ```theory.ipynb```: Traceability accuracy under attack intensity thresholds (i.e., $I_{per}$, $I_{alt}$, and $I_{del}$) and actual attack intensities, and RMSE under different attack intensity thresholds, which is used for the second experiment in our paper.

### Notes

- We strongly suggest you reserve all parameters in each line of exp_config.json, even though certain experiments do not need all parameters.

- To evaluate baselines, set ```"mode"``` to ```"watermark_detection"``` and ```"watermark"``` to baseline name (i.e., ```"tabular_mark"```, ```"freqwm"```, and ```"tabwak_partition"```).

- To evaluate utility of w/o WM, set ```"mode"``` to ```"watermark_quality"``` and ```"watermark"``` to ```"none"```.

- If you have trained a new tuple embedding model and procuded new tuple latents of the original table, please execute the following command to perform PCA on these tuple latents for clustering (you may need to edit pca.py).

```
python pca.py
```

- If you have produced new clustering results, before conduct watermarking experiments, please execute the following command to generate cluster pairs (you may need to edit ```cluster_pair_selector.py```).

```
python cluster_pair_selector.py
```

- If you add your own datasets, you need to add the corresponding data names to the dataset list in ```pca.py```, ```cluster_pair_selector.py```, and ```real_mle.py```.

### Parameter Specifications of ```exp_config.json```

To run TableMark, you just need to config ```exp_config.json```, each line corresponding to an experiment. For each experiment, parameters you need to config are as follows.

| Parameters | Type | Descriptions |
|--------|--------|--------|
| gpu_id  | Integer | Virtual GPU ID in your machine for running the experiment. |
| mode  | String  | Experiment type, possible values are as follows.<ul><li>"vae_train": Train a tuple embedding model.</li><li>"cluster": Cluster the latent space into a specified number of clusters.</li><li>"tabsyn_train": Train a conditional latent diffusion model.</li><li>"classification_error": Estimate a transition matrix.</li><li>"sample": Generate a non-watermarked table according to the original histogram. The results will be saved in the "syntheic" folder.</li><li>"eval": Test utility for the synthetic non-watermarked table in the "sample" experiment.</li><li>"gen_code": Generate a watermark database.</li><li>"prepare_query": Prepare 1000 aggregation queries per aggregation function (COUNT and AVG) per selectivity (0.01, 0.05, and 0.2), for utility evaluation.</li><li>"tabwak_dm_train": Train a DDIM for the Tabwak baseline.</li><li>"watermark_detection": Test traceability accuracy without attacks. Also used for baseline evaluation.</li><li>"watermark_quality": Test all utility metrics.</li><li>"watermark_gauss_noise": Testing traceability accuracy under the Gaussian perturbation attacks.</li><li>"watermark_laplace_noise": Testing traceability accuracy under the Laplace perturbation attacks.</li><li>"watermark_uniform_noise": Testing traceability accuracy under the uniform perturbation attacks.</li><li>"watermark_alteration": Testing traceability accuracy under the alteration attack.</li><li>"watermark_sample_deletion": Testing traceability accuracy under the tuple-deletion attack.</li><li>"watermark_sample_insertion": Testing traceability accuracy under the tuple-insertion attack.</li><li>"watermark_regeneration_vae": Testing traceability accuracy under the regeneration attack.</li></ul> |
| num_samples_per_class_lower_bound | String | Possible values are as follows.<ul><li>-1: Disable the constraint simplification mechanism.</li><li>1stage-final-%f-%f: Enable the constraint simplification mechanism, but disable the multi-stage optimization mechanism. Both %f shoule be substitued by the same $\tau$ value (e.g., 0.01).</li><li>6stage-splus-%f-%f: Enable the constraint simplification and the multi-stage optimization mechanisms. Both %f shoule be substitued by the same $\tau$ value (e.g., 0.01).</li></ul></ul> Note: This parameter has a strange name for historcal reasons. We plan to make it more reader-friendly in the near future. |
| dataname | String | Possible values: {"adult", "beijing", "default", "magic", "phishing", "shoppers"}, but you are allowed to add your own datasets. |
| num_watermark_trials | Integer | Number of trails for mesuring traceability accuracy or utility metrics. For experimental results presented in our paper, this value is set to 100. |
| watermark | String | Possible values: {"none", "pair_compare_one_pair", "tabular_mark", "freqwm", "tabwak_partition"}, corresponding to w/o WM (for "watermark_quality" only, otherwise error), TableMark, TabularMark, FreqyWM, and TabWak, respectively.|
| deletion_rate | Float | Corresponds to $I_{del}$. |
| gauss | Float | Corresponds to $I_{per}$. |
| alter | Float | Corresponds to $I_{alt}$. |
| gen_code_loss | String | "none" corresponds to randomly generated watermark database, "general_bfs" corresponds to the watermark-database generation mechanism presented in out paper.|
| max_watermark_error_rate | Float | Corresponds to $FNR$. |
| num_classes | Integer | # clusters $M$ |
| num_watermark_bits | Interger | Watermark length $L$. |
| num_users | Interger | # maximum supported data buyers $N$. |

### Acknowledgement

- All four datasets are open source and are from [UCI maching learning library](https://archive.ics.uci.edu/datasets).

- Data preprocessing, tuple embedding model, diffusion model, and evaluation codes are modified from [Tabsyn](https://github.com/amazon-science/tabsyn).

- Query error evaluation codes are modified from [SynMeter](https://github.com/zealscott/SynMeter/tree/main).

- We acknowledge [Gurobi](https://www.gurobi.com) for providing the student license.
