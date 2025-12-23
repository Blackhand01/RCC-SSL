auto_configs/exp_20251122_215057_dino_v3-all_ABL__exp_dino_v3_abl01.yaml codice <<
experiment:
  name: eval_exp_dino_v3_abl01
  seed: 1337
  outputs_root: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251122_215057_dino_v3-all_ABL/exp_dino_v3_abl01/eval
data:
  backend: webdataset
  img_size: 224
  imagenet_norm: false
  num_workers: 8
  batch_size: 256
  webdataset:
    test_dir: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/data/processed/rcc_webdataset_final/test
    pattern: shard-*.tar
    image_key: img.jpg;jpg;jpeg;png
    meta_key: meta.json;json
labels:
  class_order:
  - ccRCC
  - pRCC
  - CHROMO
  - ONCO
  - NOT_TUMOR
model:
  name: dino_v3_ssl_linear_best
  arch_hint: ssl_linear
  backbone_name: vit_small_patch16_224
  ssl_backbone_ckpt: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251122_215057_dino_v3-all_ABL/exp_dino_v3_abl01/checkpoints/dino_v3__ssl_best.pt
  ssl_head_ckpt: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251122_215057_dino_v3-all_ABL/exp_dino_v3_abl01/checkpoints/dino_v3__ssl_linear_best.pt
  strict_load: false
  allow_arch_autoswap: false
evaluation:
  save_logits: true
  save_embeddings: true
  save_preds_csv: true
  umap:
    enabled: true
    source: features
    n_neighbors: 15
    min_dist: 0.1
    random_state: 1337
runtime:
  device: cuda
  precision: fp32
>>

auto_configs/exp_20251122_215057_dino_v3-all_ABL__exp_dino_v3_abl02.yaml codice <<
experiment:
  name: eval_exp_dino_v3_abl02
  seed: 1337
  outputs_root: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251122_215057_dino_v3-all_ABL/exp_dino_v3_abl02/eval
data:
  backend: webdataset
  img_size: 224
  imagenet_norm: false
  num_workers: 8
  batch_size: 256
  webdataset:
    test_dir: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/data/processed/rcc_webdataset_final/test
    pattern: shard-*.tar
    image_key: img.jpg;jpg;jpeg;png
    meta_key: meta.json;json
labels:
  class_order:
  - ccRCC
  - pRCC
  - CHROMO
  - ONCO
  - NOT_TUMOR
model:
  name: dino_v3_ssl_linear_best
  arch_hint: ssl_linear
  backbone_name: vit_small_patch16_224
  ssl_backbone_ckpt: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251122_215057_dino_v3-all_ABL/exp_dino_v3_abl02/checkpoints/dino_v3__ssl_best.pt
  ssl_head_ckpt: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251122_215057_dino_v3-all_ABL/exp_dino_v3_abl02/checkpoints/dino_v3__ssl_linear_best.pt
  strict_load: false
  allow_arch_autoswap: false
evaluation:
  save_logits: true
  save_embeddings: true
  save_preds_csv: true
  umap:
    enabled: true
    source: features
    n_neighbors: 15
    min_dist: 0.1
    random_state: 1337
runtime:
  device: cuda
  precision: fp32
>>

auto_configs/exp_20251122_215057_dino_v3-all_ABL__exp_dino_v3_abl03.yaml codice <<
experiment:
  name: eval_exp_dino_v3_abl03
  seed: 1337
  outputs_root: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251122_215057_dino_v3-all_ABL/exp_dino_v3_abl03/eval
data:
  backend: webdataset
  img_size: 224
  imagenet_norm: false
  num_workers: 8
  batch_size: 256
  webdataset:
    test_dir: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/data/processed/rcc_webdataset_final/test
    pattern: shard-*.tar
    image_key: img.jpg;jpg;jpeg;png
    meta_key: meta.json;json
labels:
  class_order:
  - ccRCC
  - pRCC
  - CHROMO
  - ONCO
  - NOT_TUMOR
model:
  name: dino_v3_ssl_linear_best
  arch_hint: ssl_linear
  backbone_name: vit_small_patch16_224
  ssl_backbone_ckpt: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251122_215057_dino_v3-all_ABL/exp_dino_v3_abl03/checkpoints/dino_v3__ssl_best.pt
  ssl_head_ckpt: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251122_215057_dino_v3-all_ABL/exp_dino_v3_abl03/checkpoints/dino_v3__ssl_linear_best.pt
  strict_load: false
  allow_arch_autoswap: false
evaluation:
  save_logits: true
  save_embeddings: true
  save_preds_csv: true
  umap:
    enabled: true
    source: features
    n_neighbors: 15
    min_dist: 0.1
    random_state: 1337
runtime:
  device: cuda
  precision: fp32
>>

auto_configs/exp_20251122_215057_dino_v3-all_ABL__exp_dino_v3_abl04.yaml codice <<
experiment:
  name: eval_exp_dino_v3_abl04
  seed: 1337
  outputs_root: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251122_215057_dino_v3-all_ABL/exp_dino_v3_abl04/eval
data:
  backend: webdataset
  img_size: 224
  imagenet_norm: false
  num_workers: 8
  batch_size: 256
  webdataset:
    test_dir: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/data/processed/rcc_webdataset_final/test
    pattern: shard-*.tar
    image_key: img.jpg;jpg;jpeg;png
    meta_key: meta.json;json
labels:
  class_order:
  - ccRCC
  - pRCC
  - CHROMO
  - ONCO
  - NOT_TUMOR
model:
  name: dino_v3_ssl_linear_best
  arch_hint: ssl_linear
  backbone_name: vit_small_patch16_224
  ssl_backbone_ckpt: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251122_215057_dino_v3-all_ABL/exp_dino_v3_abl04/checkpoints/dino_v3__ssl_best.pt
  ssl_head_ckpt: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251122_215057_dino_v3-all_ABL/exp_dino_v3_abl04/checkpoints/dino_v3__ssl_linear_best.pt
  strict_load: false
  allow_arch_autoswap: false
evaluation:
  save_logits: true
  save_embeddings: true
  save_preds_csv: true
  umap:
    enabled: true
    source: features
    n_neighbors: 15
    min_dist: 0.1
    random_state: 1337
runtime:
  device: cuda
  precision: fp32
>>

auto_configs/exp_20251122_215057_dino_v3-all_ABL__exp_dino_v3_abl05.yaml codice <<
experiment:
  name: eval_exp_dino_v3_abl05
  seed: 1337
  outputs_root: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251122_215057_dino_v3-all_ABL/exp_dino_v3_abl05/eval
data:
  backend: webdataset
  img_size: 224
  imagenet_norm: false
  num_workers: 8
  batch_size: 256
  webdataset:
    test_dir: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/data/processed/rcc_webdataset_final/test
    pattern: shard-*.tar
    image_key: img.jpg;jpg;jpeg;png
    meta_key: meta.json;json
labels:
  class_order:
  - ccRCC
  - pRCC
  - CHROMO
  - ONCO
  - NOT_TUMOR
model:
  name: dino_v3_ssl_linear_best
  arch_hint: ssl_linear
  backbone_name: vit_small_patch16_224
  ssl_backbone_ckpt: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251122_215057_dino_v3-all_ABL/exp_dino_v3_abl05/checkpoints/dino_v3__ssl_best.pt
  ssl_head_ckpt: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251122_215057_dino_v3-all_ABL/exp_dino_v3_abl05/checkpoints/dino_v3__ssl_linear_best.pt
  strict_load: false
  allow_arch_autoswap: false
evaluation:
  save_logits: true
  save_embeddings: true
  save_preds_csv: true
  umap:
    enabled: true
    source: features
    n_neighbors: 15
    min_dist: 0.1
    random_state: 1337
runtime:
  device: cuda
  precision: fp32
>>

auto_configs/exp_20251122_215057_dino_v3-all_ABL__exp_dino_v3_abl06.yaml codice <<
experiment:
  name: eval_exp_dino_v3_abl06
  seed: 1337
  outputs_root: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251122_215057_dino_v3-all_ABL/exp_dino_v3_abl06/eval
data:
  backend: webdataset
  img_size: 224
  imagenet_norm: false
  num_workers: 8
  batch_size: 256
  webdataset:
    test_dir: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/data/processed/rcc_webdataset_final/test
    pattern: shard-*.tar
    image_key: img.jpg;jpg;jpeg;png
    meta_key: meta.json;json
labels:
  class_order:
  - ccRCC
  - pRCC
  - CHROMO
  - ONCO
  - NOT_TUMOR
model:
  name: dino_v3_ssl_linear_best
  arch_hint: ssl_linear
  backbone_name: vit_small_patch16_224
  ssl_backbone_ckpt: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251122_215057_dino_v3-all_ABL/exp_dino_v3_abl06/checkpoints/dino_v3__ssl_best.pt
  ssl_head_ckpt: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251122_215057_dino_v3-all_ABL/exp_dino_v3_abl06/checkpoints/dino_v3__ssl_linear_best.pt
  strict_load: false
  allow_arch_autoswap: false
evaluation:
  save_logits: true
  save_embeddings: true
  save_preds_csv: true
  umap:
    enabled: true
    source: features
    n_neighbors: 15
    min_dist: 0.1
    random_state: 1337
runtime:
  device: cuda
  precision: fp32
>>

auto_configs/exp_20251122_215057_dino_v3-all_ABL__exp_dino_v3_abl07.yaml codice <<
experiment:
  name: eval_exp_dino_v3_abl07
  seed: 1337
  outputs_root: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251122_215057_dino_v3-all_ABL/exp_dino_v3_abl07/eval
data:
  backend: webdataset
  img_size: 224
  imagenet_norm: false
  num_workers: 8
  batch_size: 256
  webdataset:
    test_dir: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/data/processed/rcc_webdataset_final/test
    pattern: shard-*.tar
    image_key: img.jpg;jpg;jpeg;png
    meta_key: meta.json;json
labels:
  class_order:
  - ccRCC
  - pRCC
  - CHROMO
  - ONCO
  - NOT_TUMOR
model:
  name: dino_v3_ssl_linear_best
  arch_hint: ssl_linear
  backbone_name: vit_small_patch16_224
  ssl_backbone_ckpt: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251122_215057_dino_v3-all_ABL/exp_dino_v3_abl07/checkpoints/dino_v3__ssl_best.pt
  ssl_head_ckpt: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251122_215057_dino_v3-all_ABL/exp_dino_v3_abl07/checkpoints/dino_v3__ssl_linear_best.pt
  strict_load: false
  allow_arch_autoswap: false
evaluation:
  save_logits: true
  save_embeddings: true
  save_preds_csv: true
  umap:
    enabled: true
    source: features
    n_neighbors: 15
    min_dist: 0.1
    random_state: 1337
runtime:
  device: cuda
  precision: fp32
>>

auto_configs/exp_20251122_215057_dino_v3-all_ABL__exp_dino_v3_abl08.yaml codice <<
experiment:
  name: eval_exp_dino_v3_abl08
  seed: 1337
  outputs_root: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251122_215057_dino_v3-all_ABL/exp_dino_v3_abl08/eval
data:
  backend: webdataset
  img_size: 224
  imagenet_norm: false
  num_workers: 8
  batch_size: 256
  webdataset:
    test_dir: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/data/processed/rcc_webdataset_final/test
    pattern: shard-*.tar
    image_key: img.jpg;jpg;jpeg;png
    meta_key: meta.json;json
labels:
  class_order:
  - ccRCC
  - pRCC
  - CHROMO
  - ONCO
  - NOT_TUMOR
model:
  name: dino_v3_ssl_linear_best
  arch_hint: ssl_linear
  backbone_name: vit_small_patch16_224
  ssl_backbone_ckpt: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251122_215057_dino_v3-all_ABL/exp_dino_v3_abl08/checkpoints/dino_v3__ssl_best.pt
  ssl_head_ckpt: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251122_215057_dino_v3-all_ABL/exp_dino_v3_abl08/checkpoints/dino_v3__ssl_linear_best.pt
  strict_load: false
  allow_arch_autoswap: false
evaluation:
  save_logits: true
  save_embeddings: true
  save_preds_csv: true
  umap:
    enabled: true
    source: features
    n_neighbors: 15
    min_dist: 0.1
    random_state: 1337
runtime:
  device: cuda
  precision: fp32
>>

auto_configs/exp_20251122_215057_dino_v3-all_ABL__exp_dino_v3_abl13.yaml codice <<
experiment:
  name: eval_exp_dino_v3_abl13
  seed: 1337
  outputs_root: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251122_215057_dino_v3-all_ABL/exp_dino_v3_abl13/eval
data:
  backend: webdataset
  img_size: 224
  imagenet_norm: false
  num_workers: 8
  batch_size: 256
  webdataset:
    test_dir: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/data/processed/rcc_webdataset_final/test
    pattern: shard-*.tar
    image_key: img.jpg;jpg;jpeg;png
    meta_key: meta.json;json
labels:
  class_order:
  - ccRCC
  - pRCC
  - CHROMO
  - ONCO
  - NOT_TUMOR
model:
  name: dino_v3_ssl_linear_best
  arch_hint: ssl_linear
  backbone_name: vit_small_patch16_224
  ssl_backbone_ckpt: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251122_215057_dino_v3-all_ABL/exp_dino_v3_abl13/checkpoints/dino_v3__ssl_best.pt
  ssl_head_ckpt: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251122_215057_dino_v3-all_ABL/exp_dino_v3_abl13/checkpoints/dino_v3__ssl_linear_best.pt
  strict_load: false
  allow_arch_autoswap: false
evaluation:
  save_logits: true
  save_embeddings: true
  save_preds_csv: true
  umap:
    enabled: true
    source: features
    n_neighbors: 15
    min_dist: 0.1
    random_state: 1337
runtime:
  device: cuda
  precision: fp32
>>

auto_configs/exp_20251122_215057_dino_v3-all_ABL__exp_dino_v3_abl14.yaml codice <<
experiment:
  name: eval_exp_dino_v3_abl14
  seed: 1337
  outputs_root: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251122_215057_dino_v3-all_ABL/exp_dino_v3_abl14/eval
data:
  backend: webdataset
  img_size: 224
  imagenet_norm: false
  num_workers: 8
  batch_size: 256
  webdataset:
    test_dir: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/data/processed/rcc_webdataset_final/test
    pattern: shard-*.tar
    image_key: img.jpg;jpg;jpeg;png
    meta_key: meta.json;json
labels:
  class_order:
  - ccRCC
  - pRCC
  - CHROMO
  - ONCO
  - NOT_TUMOR
model:
  name: dino_v3_ssl_linear_best
  arch_hint: ssl_linear
  backbone_name: vit_small_patch16_224
  ssl_backbone_ckpt: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251122_215057_dino_v3-all_ABL/exp_dino_v3_abl14/checkpoints/dino_v3__ssl_best.pt
  ssl_head_ckpt: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251122_215057_dino_v3-all_ABL/exp_dino_v3_abl14/checkpoints/dino_v3__ssl_linear_best.pt
  strict_load: false
  allow_arch_autoswap: false
evaluation:
  save_logits: true
  save_embeddings: true
  save_preds_csv: true
  umap:
    enabled: true
    source: features
    n_neighbors: 15
    min_dist: 0.1
    random_state: 1337
runtime:
  device: cuda
  precision: fp32
>>

auto_configs/exp_20251122_215057_dino_v3-all_ABL__exp_dino_v3_abl19.yaml codice <<
experiment:
  name: eval_exp_dino_v3_abl19
  seed: 1337
  outputs_root: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251122_215057_dino_v3-all_ABL/exp_dino_v3_abl19/eval
data:
  backend: webdataset
  img_size: 224
  imagenet_norm: false
  num_workers: 8
  batch_size: 256
  webdataset:
    test_dir: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/data/processed/rcc_webdataset_final/test
    pattern: shard-*.tar
    image_key: img.jpg;jpg;jpeg;png
    meta_key: meta.json;json
labels:
  class_order:
  - ccRCC
  - pRCC
  - CHROMO
  - ONCO
  - NOT_TUMOR
model:
  name: dino_v3_ssl_linear_best
  arch_hint: ssl_linear
  backbone_name: vit_small_patch16_224
  ssl_backbone_ckpt: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251122_215057_dino_v3-all_ABL/exp_dino_v3_abl19/checkpoints/dino_v3__ssl_best.pt
  ssl_head_ckpt: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251122_215057_dino_v3-all_ABL/exp_dino_v3_abl19/checkpoints/dino_v3__ssl_linear_best.pt
  strict_load: false
  allow_arch_autoswap: false
evaluation:
  save_logits: true
  save_embeddings: true
  save_preds_csv: true
  umap:
    enabled: true
    source: features
    n_neighbors: 15
    min_dist: 0.1
    random_state: 1337
runtime:
  device: cuda
  precision: fp32
>>

auto_configs/exp_20251122_215057_dino_v3-all_ABL__exp_dino_v3_abl20.yaml codice <<
experiment:
  name: eval_exp_dino_v3_abl20
  seed: 1337
  outputs_root: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251122_215057_dino_v3-all_ABL/exp_dino_v3_abl20/eval
data:
  backend: webdataset
  img_size: 224
  imagenet_norm: false
  num_workers: 8
  batch_size: 256
  webdataset:
    test_dir: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/data/processed/rcc_webdataset_final/test
    pattern: shard-*.tar
    image_key: img.jpg;jpg;jpeg;png
    meta_key: meta.json;json
labels:
  class_order:
  - ccRCC
  - pRCC
  - CHROMO
  - ONCO
  - NOT_TUMOR
model:
  name: dino_v3_ssl_linear_best
  arch_hint: ssl_linear
  backbone_name: vit_small_patch16_224
  ssl_backbone_ckpt: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251122_215057_dino_v3-all_ABL/exp_dino_v3_abl20/checkpoints/dino_v3__ssl_best.pt
  ssl_head_ckpt: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251122_215057_dino_v3-all_ABL/exp_dino_v3_abl20/checkpoints/dino_v3__ssl_linear_best.pt
  strict_load: false
  allow_arch_autoswap: false
evaluation:
  save_logits: true
  save_embeddings: true
  save_preds_csv: true
  umap:
    enabled: true
    source: features
    n_neighbors: 15
    min_dist: 0.1
    random_state: 1337
runtime:
  device: cuda
  precision: fp32
>>

auto_configs/exp_20251122_215057_dino_v3-all_ABL__exp_dino_v3_abl21.yaml codice <<
experiment:
  name: eval_exp_dino_v3_abl21
  seed: 1337
  outputs_root: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251122_215057_dino_v3-all_ABL/exp_dino_v3_abl21/eval
data:
  backend: webdataset
  img_size: 224
  imagenet_norm: false
  num_workers: 8
  batch_size: 256
  webdataset:
    test_dir: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/data/processed/rcc_webdataset_final/test
    pattern: shard-*.tar
    image_key: img.jpg;jpg;jpeg;png
    meta_key: meta.json;json
labels:
  class_order:
  - ccRCC
  - pRCC
  - CHROMO
  - ONCO
  - NOT_TUMOR
model:
  name: dino_v3_ssl_linear_best
  arch_hint: ssl_linear
  backbone_name: vit_small_patch16_224
  ssl_backbone_ckpt: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251122_215057_dino_v3-all_ABL/exp_dino_v3_abl21/checkpoints/dino_v3__ssl_best.pt
  ssl_head_ckpt: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251122_215057_dino_v3-all_ABL/exp_dino_v3_abl21/checkpoints/dino_v3__ssl_linear_best.pt
  strict_load: false
  allow_arch_autoswap: false
evaluation:
  save_logits: true
  save_embeddings: true
  save_preds_csv: true
  umap:
    enabled: true
    source: features
    n_neighbors: 15
    min_dist: 0.1
    random_state: 1337
runtime:
  device: cuda
  precision: fp32
>>

auto_configs/exp_20251122_215057_dino_v3-all_ABL__exp_dino_v3_abl26.yaml codice <<
experiment:
  name: eval_exp_dino_v3_abl26
  seed: 1337
  outputs_root: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251122_215057_dino_v3-all_ABL/exp_dino_v3_abl26/eval
data:
  backend: webdataset
  img_size: 224
  imagenet_norm: false
  num_workers: 8
  batch_size: 256
  webdataset:
    test_dir: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/data/processed/rcc_webdataset_final/test
    pattern: shard-*.tar
    image_key: img.jpg;jpg;jpeg;png
    meta_key: meta.json;json
labels:
  class_order:
  - ccRCC
  - pRCC
  - CHROMO
  - ONCO
  - NOT_TUMOR
model:
  name: dino_v3_ssl_linear_best
  arch_hint: ssl_linear
  backbone_name: vit_small_patch16_224
  ssl_backbone_ckpt: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251122_215057_dino_v3-all_ABL/exp_dino_v3_abl26/checkpoints/dino_v3__ssl_best.pt
  ssl_head_ckpt: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251122_215057_dino_v3-all_ABL/exp_dino_v3_abl26/checkpoints/dino_v3__ssl_linear_best.pt
  strict_load: false
  allow_arch_autoswap: false
evaluation:
  save_logits: true
  save_embeddings: true
  save_preds_csv: true
  umap:
    enabled: true
    source: features
    n_neighbors: 15
    min_dist: 0.1
    random_state: 1337
runtime:
  device: cuda
  precision: fp32
>>

auto_configs/exp_20251122_215057_dino_v3-all_ABL__exp_dino_v3_abl27.yaml codice <<
experiment:
  name: eval_exp_dino_v3_abl27
  seed: 1337
  outputs_root: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251122_215057_dino_v3-all_ABL/exp_dino_v3_abl27/eval
data:
  backend: webdataset
  img_size: 224
  imagenet_norm: false
  num_workers: 8
  batch_size: 256
  webdataset:
    test_dir: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/data/processed/rcc_webdataset_final/test
    pattern: shard-*.tar
    image_key: img.jpg;jpg;jpeg;png
    meta_key: meta.json;json
labels:
  class_order:
  - ccRCC
  - pRCC
  - CHROMO
  - ONCO
  - NOT_TUMOR
model:
  name: dino_v3_ssl_linear_best
  arch_hint: ssl_linear
  backbone_name: vit_small_patch16_224
  ssl_backbone_ckpt: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251122_215057_dino_v3-all_ABL/exp_dino_v3_abl27/checkpoints/dino_v3__ssl_best.pt
  ssl_head_ckpt: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251122_215057_dino_v3-all_ABL/exp_dino_v3_abl27/checkpoints/dino_v3__ssl_linear_best.pt
  strict_load: false
  allow_arch_autoswap: false
evaluation:
  save_logits: true
  save_embeddings: true
  save_preds_csv: true
  umap:
    enabled: true
    source: features
    n_neighbors: 15
    min_dist: 0.1
    random_state: 1337
runtime:
  device: cuda
  precision: fp32
>>

auto_configs/exp_20251123_213900_i_jepa__exp_i_jepa_abl01.yaml codice <<
experiment:
  name: eval_exp_i_jepa_abl01
  seed: 1337
  outputs_root: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251123_213900_i_jepa/exp_i_jepa_abl01/eval
data:
  backend: webdataset
  img_size: 224
  imagenet_norm: false
  num_workers: 8
  batch_size: 256
  webdataset:
    test_dir: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/data/processed/rcc_webdataset_final/test
    pattern: shard-*.tar
    image_key: img.jpg;jpg;jpeg;png
    meta_key: meta.json;json
labels:
  class_order:
  - ccRCC
  - pRCC
  - CHROMO
  - ONCO
  - NOT_TUMOR
model:
  name: i_jepa_ssl_linear_best
  arch_hint: ssl_linear
  backbone_name: vit_small_patch16_224
  ssl_backbone_ckpt: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251123_213900_i_jepa/exp_i_jepa_abl01/checkpoints/i_jepa__ssl_best.pt
  ssl_head_ckpt: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251123_213900_i_jepa/exp_i_jepa_abl01/checkpoints/i_jepa__ssl_linear_best.pt
  strict_load: false
  allow_arch_autoswap: false
evaluation:
  save_logits: true
  save_embeddings: true
  save_preds_csv: true
  umap:
    enabled: true
    source: features
    n_neighbors: 15
    min_dist: 0.1
    random_state: 1337
runtime:
  device: cuda
  precision: fp32
>>

auto_configs/exp_20251123_213900_i_jepa__exp_i_jepa_abl02.yaml codice <<
experiment:
  name: eval_exp_i_jepa_abl02
  seed: 1337
  outputs_root: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251123_213900_i_jepa/exp_i_jepa_abl02/eval
data:
  backend: webdataset
  img_size: 224
  imagenet_norm: false
  num_workers: 8
  batch_size: 256
  webdataset:
    test_dir: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/data/processed/rcc_webdataset_final/test
    pattern: shard-*.tar
    image_key: img.jpg;jpg;jpeg;png
    meta_key: meta.json;json
labels:
  class_order:
  - ccRCC
  - pRCC
  - CHROMO
  - ONCO
  - NOT_TUMOR
model:
  name: i_jepa_ssl_linear_best
  arch_hint: ssl_linear
  backbone_name: vit_small_patch16_224
  ssl_backbone_ckpt: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251123_213900_i_jepa/exp_i_jepa_abl02/checkpoints/i_jepa__ssl_best.pt
  ssl_head_ckpt: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251123_213900_i_jepa/exp_i_jepa_abl02/checkpoints/i_jepa__ssl_linear_best.pt
  strict_load: false
  allow_arch_autoswap: false
evaluation:
  save_logits: true
  save_embeddings: true
  save_preds_csv: true
  umap:
    enabled: true
    source: features
    n_neighbors: 15
    min_dist: 0.1
    random_state: 1337
runtime:
  device: cuda
  precision: fp32
>>

auto_configs/exp_20251123_213900_i_jepa__exp_i_jepa_abl03.yaml codice <<
experiment:
  name: eval_exp_i_jepa_abl03
  seed: 1337
  outputs_root: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251123_213900_i_jepa/exp_i_jepa_abl03/eval
data:
  backend: webdataset
  img_size: 224
  imagenet_norm: false
  num_workers: 8
  batch_size: 256
  webdataset:
    test_dir: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/data/processed/rcc_webdataset_final/test
    pattern: shard-*.tar
    image_key: img.jpg;jpg;jpeg;png
    meta_key: meta.json;json
labels:
  class_order:
  - ccRCC
  - pRCC
  - CHROMO
  - ONCO
  - NOT_TUMOR
model:
  name: i_jepa_ssl_linear_best
  arch_hint: ssl_linear
  backbone_name: vit_small_patch16_224
  ssl_backbone_ckpt: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251123_213900_i_jepa/exp_i_jepa_abl03/checkpoints/i_jepa__ssl_best.pt
  ssl_head_ckpt: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251123_213900_i_jepa/exp_i_jepa_abl03/checkpoints/i_jepa__ssl_linear_best.pt
  strict_load: false
  allow_arch_autoswap: false
evaluation:
  save_logits: true
  save_embeddings: true
  save_preds_csv: true
  umap:
    enabled: true
    source: features
    n_neighbors: 15
    min_dist: 0.1
    random_state: 1337
runtime:
  device: cuda
  precision: fp32
>>

auto_configs/exp_20251123_213900_i_jepa__exp_i_jepa_abl04.yaml codice <<
experiment:
  name: eval_exp_i_jepa_abl04
  seed: 1337
  outputs_root: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251123_213900_i_jepa/exp_i_jepa_abl04/eval
data:
  backend: webdataset
  img_size: 224
  imagenet_norm: false
  num_workers: 8
  batch_size: 256
  webdataset:
    test_dir: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/data/processed/rcc_webdataset_final/test
    pattern: shard-*.tar
    image_key: img.jpg;jpg;jpeg;png
    meta_key: meta.json;json
labels:
  class_order:
  - ccRCC
  - pRCC
  - CHROMO
  - ONCO
  - NOT_TUMOR
model:
  name: i_jepa_ssl_linear_best
  arch_hint: ssl_linear
  backbone_name: vit_small_patch16_224
  ssl_backbone_ckpt: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251123_213900_i_jepa/exp_i_jepa_abl04/checkpoints/i_jepa__ssl_best.pt
  ssl_head_ckpt: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251123_213900_i_jepa/exp_i_jepa_abl04/checkpoints/i_jepa__ssl_linear_best.pt
  strict_load: false
  allow_arch_autoswap: false
evaluation:
  save_logits: true
  save_embeddings: true
  save_preds_csv: true
  umap:
    enabled: true
    source: features
    n_neighbors: 15
    min_dist: 0.1
    random_state: 1337
runtime:
  device: cuda
  precision: fp32
>>

auto_configs/exp_20251123_213900_i_jepa__exp_i_jepa_abl05.yaml codice <<
experiment:
  name: eval_exp_i_jepa_abl05
  seed: 1337
  outputs_root: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251123_213900_i_jepa/exp_i_jepa_abl05/eval
data:
  backend: webdataset
  img_size: 224
  imagenet_norm: false
  num_workers: 8
  batch_size: 256
  webdataset:
    test_dir: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/data/processed/rcc_webdataset_final/test
    pattern: shard-*.tar
    image_key: img.jpg;jpg;jpeg;png
    meta_key: meta.json;json
labels:
  class_order:
  - ccRCC
  - pRCC
  - CHROMO
  - ONCO
  - NOT_TUMOR
model:
  name: i_jepa_ssl_linear_best
  arch_hint: ssl_linear
  backbone_name: vit_small_patch16_224
  ssl_backbone_ckpt: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251123_213900_i_jepa/exp_i_jepa_abl05/checkpoints/i_jepa__ssl_best.pt
  ssl_head_ckpt: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251123_213900_i_jepa/exp_i_jepa_abl05/checkpoints/i_jepa__ssl_linear_best.pt
  strict_load: false
  allow_arch_autoswap: false
evaluation:
  save_logits: true
  save_embeddings: true
  save_preds_csv: true
  umap:
    enabled: true
    source: features
    n_neighbors: 15
    min_dist: 0.1
    random_state: 1337
runtime:
  device: cuda
  precision: fp32
>>

auto_configs/exp_20251123_213900_i_jepa__exp_i_jepa_abl06.yaml codice <<
experiment:
  name: eval_exp_i_jepa_abl06
  seed: 1337
  outputs_root: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251123_213900_i_jepa/exp_i_jepa_abl06/eval
data:
  backend: webdataset
  img_size: 224
  imagenet_norm: false
  num_workers: 8
  batch_size: 256
  webdataset:
    test_dir: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/data/processed/rcc_webdataset_final/test
    pattern: shard-*.tar
    image_key: img.jpg;jpg;jpeg;png
    meta_key: meta.json;json
labels:
  class_order:
  - ccRCC
  - pRCC
  - CHROMO
  - ONCO
  - NOT_TUMOR
model:
  name: i_jepa_ssl_linear_best
  arch_hint: ssl_linear
  backbone_name: vit_small_patch16_224
  ssl_backbone_ckpt: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251123_213900_i_jepa/exp_i_jepa_abl06/checkpoints/i_jepa__ssl_best.pt
  ssl_head_ckpt: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251123_213900_i_jepa/exp_i_jepa_abl06/checkpoints/i_jepa__ssl_linear_best.pt
  strict_load: false
  allow_arch_autoswap: false
evaluation:
  save_logits: true
  save_embeddings: true
  save_preds_csv: true
  umap:
    enabled: true
    source: features
    n_neighbors: 15
    min_dist: 0.1
    random_state: 1337
runtime:
  device: cuda
  precision: fp32
>>

auto_configs/exp_20251123_213900_i_jepa__exp_i_jepa_abl07.yaml codice <<
experiment:
  name: eval_exp_i_jepa_abl07
  seed: 1337
  outputs_root: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251123_213900_i_jepa/exp_i_jepa_abl07/eval
data:
  backend: webdataset
  img_size: 224
  imagenet_norm: false
  num_workers: 8
  batch_size: 256
  webdataset:
    test_dir: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/data/processed/rcc_webdataset_final/test
    pattern: shard-*.tar
    image_key: img.jpg;jpg;jpeg;png
    meta_key: meta.json;json
labels:
  class_order:
  - ccRCC
  - pRCC
  - CHROMO
  - ONCO
  - NOT_TUMOR
model:
  name: i_jepa_ssl_linear_best
  arch_hint: ssl_linear
  backbone_name: vit_small_patch16_224
  ssl_backbone_ckpt: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251123_213900_i_jepa/exp_i_jepa_abl07/checkpoints/i_jepa__ssl_best.pt
  ssl_head_ckpt: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251123_213900_i_jepa/exp_i_jepa_abl07/checkpoints/i_jepa__ssl_linear_best.pt
  strict_load: false
  allow_arch_autoswap: false
evaluation:
  save_logits: true
  save_embeddings: true
  save_preds_csv: true
  umap:
    enabled: true
    source: features
    n_neighbors: 15
    min_dist: 0.1
    random_state: 1337
runtime:
  device: cuda
  precision: fp32
>>

auto_configs/exp_20251123_213900_i_jepa__exp_i_jepa_abl08.yaml codice <<
experiment:
  name: eval_exp_i_jepa_abl08
  seed: 1337
  outputs_root: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251123_213900_i_jepa/exp_i_jepa_abl08/eval
data:
  backend: webdataset
  img_size: 224
  imagenet_norm: false
  num_workers: 8
  batch_size: 256
  webdataset:
    test_dir: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/data/processed/rcc_webdataset_final/test
    pattern: shard-*.tar
    image_key: img.jpg;jpg;jpeg;png
    meta_key: meta.json;json
labels:
  class_order:
  - ccRCC
  - pRCC
  - CHROMO
  - ONCO
  - NOT_TUMOR
model:
  name: i_jepa_ssl_linear_best
  arch_hint: ssl_linear
  backbone_name: vit_small_patch16_224
  ssl_backbone_ckpt: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251123_213900_i_jepa/exp_i_jepa_abl08/checkpoints/i_jepa__ssl_best.pt
  ssl_head_ckpt: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251123_213900_i_jepa/exp_i_jepa_abl08/checkpoints/i_jepa__ssl_linear_best.pt
  strict_load: false
  allow_arch_autoswap: false
evaluation:
  save_logits: true
  save_embeddings: true
  save_preds_csv: true
  umap:
    enabled: true
    source: features
    n_neighbors: 15
    min_dist: 0.1
    random_state: 1337
runtime:
  device: cuda
  precision: fp32
>>

auto_configs/exp_20251123_213900_i_jepa__exp_i_jepa_abl09.yaml codice <<
experiment:
  name: eval_exp_i_jepa_abl09
  seed: 1337
  outputs_root: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251123_213900_i_jepa/exp_i_jepa_abl09/eval
data:
  backend: webdataset
  img_size: 224
  imagenet_norm: false
  num_workers: 8
  batch_size: 256
  webdataset:
    test_dir: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/data/processed/rcc_webdataset_final/test
    pattern: shard-*.tar
    image_key: img.jpg;jpg;jpeg;png
    meta_key: meta.json;json
labels:
  class_order:
  - ccRCC
  - pRCC
  - CHROMO
  - ONCO
  - NOT_TUMOR
model:
  name: i_jepa_ssl_linear_best
  arch_hint: ssl_linear
  backbone_name: vit_small_patch16_224
  ssl_backbone_ckpt: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251123_213900_i_jepa/exp_i_jepa_abl09/checkpoints/i_jepa__ssl_best.pt
  ssl_head_ckpt: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251123_213900_i_jepa/exp_i_jepa_abl09/checkpoints/i_jepa__ssl_linear_best.pt
  strict_load: false
  allow_arch_autoswap: false
evaluation:
  save_logits: true
  save_embeddings: true
  save_preds_csv: true
  umap:
    enabled: true
    source: features
    n_neighbors: 15
    min_dist: 0.1
    random_state: 1337
runtime:
  device: cuda
  precision: fp32
>>

auto_configs/exp_20251123_213900_i_jepa__exp_i_jepa_abl10.yaml codice <<
experiment:
  name: eval_exp_i_jepa_abl10
  seed: 1337
  outputs_root: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251123_213900_i_jepa/exp_i_jepa_abl10/eval
data:
  backend: webdataset
  img_size: 224
  imagenet_norm: false
  num_workers: 8
  batch_size: 256
  webdataset:
    test_dir: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/data/processed/rcc_webdataset_final/test
    pattern: shard-*.tar
    image_key: img.jpg;jpg;jpeg;png
    meta_key: meta.json;json
labels:
  class_order:
  - ccRCC
  - pRCC
  - CHROMO
  - ONCO
  - NOT_TUMOR
model:
  name: i_jepa_ssl_linear_best
  arch_hint: ssl_linear
  backbone_name: vit_small_patch16_224
  ssl_backbone_ckpt: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251123_213900_i_jepa/exp_i_jepa_abl10/checkpoints/i_jepa__ssl_best.pt
  ssl_head_ckpt: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251123_213900_i_jepa/exp_i_jepa_abl10/checkpoints/i_jepa__ssl_linear_best.pt
  strict_load: false
  allow_arch_autoswap: false
evaluation:
  save_logits: true
  save_embeddings: true
  save_preds_csv: true
  umap:
    enabled: true
    source: features
    n_neighbors: 15
    min_dist: 0.1
    random_state: 1337
runtime:
  device: cuda
  precision: fp32
>>

auto_configs/exp_20251123_213900_i_jepa__exp_i_jepa_abl11.yaml codice <<
experiment:
  name: eval_exp_i_jepa_abl11
  seed: 1337
  outputs_root: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251123_213900_i_jepa/exp_i_jepa_abl11/eval
data:
  backend: webdataset
  img_size: 224
  imagenet_norm: false
  num_workers: 8
  batch_size: 256
  webdataset:
    test_dir: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/data/processed/rcc_webdataset_final/test
    pattern: shard-*.tar
    image_key: img.jpg;jpg;jpeg;png
    meta_key: meta.json;json
labels:
  class_order:
  - ccRCC
  - pRCC
  - CHROMO
  - ONCO
  - NOT_TUMOR
model:
  name: i_jepa_ssl_linear_best
  arch_hint: ssl_linear
  backbone_name: vit_small_patch16_224
  ssl_backbone_ckpt: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251123_213900_i_jepa/exp_i_jepa_abl11/checkpoints/i_jepa__ssl_best.pt
  ssl_head_ckpt: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251123_213900_i_jepa/exp_i_jepa_abl11/checkpoints/i_jepa__ssl_linear_best.pt
  strict_load: false
  allow_arch_autoswap: false
evaluation:
  save_logits: true
  save_embeddings: true
  save_preds_csv: true
  umap:
    enabled: true
    source: features
    n_neighbors: 15
    min_dist: 0.1
    random_state: 1337
runtime:
  device: cuda
  precision: fp32
>>

auto_configs/exp_20251123_213900_i_jepa__exp_i_jepa_abl12.yaml codice <<
experiment:
  name: eval_exp_i_jepa_abl12
  seed: 1337
  outputs_root: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251123_213900_i_jepa/exp_i_jepa_abl12/eval
data:
  backend: webdataset
  img_size: 224
  imagenet_norm: false
  num_workers: 8
  batch_size: 256
  webdataset:
    test_dir: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/data/processed/rcc_webdataset_final/test
    pattern: shard-*.tar
    image_key: img.jpg;jpg;jpeg;png
    meta_key: meta.json;json
labels:
  class_order:
  - ccRCC
  - pRCC
  - CHROMO
  - ONCO
  - NOT_TUMOR
model:
  name: i_jepa_ssl_linear_best
  arch_hint: ssl_linear
  backbone_name: vit_small_patch16_224
  ssl_backbone_ckpt: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251123_213900_i_jepa/exp_i_jepa_abl12/checkpoints/i_jepa__ssl_best.pt
  ssl_head_ckpt: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251123_213900_i_jepa/exp_i_jepa_abl12/checkpoints/i_jepa__ssl_linear_best.pt
  strict_load: false
  allow_arch_autoswap: false
evaluation:
  save_logits: true
  save_embeddings: true
  save_preds_csv: true
  umap:
    enabled: true
    source: features
    n_neighbors: 15
    min_dist: 0.1
    random_state: 1337
runtime:
  device: cuda
  precision: fp32
>>

auto_configs/exp_20251123_213900_i_jepa__exp_i_jepa_abl13.yaml codice <<
experiment:
  name: eval_exp_i_jepa_abl13
  seed: 1337
  outputs_root: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251123_213900_i_jepa/exp_i_jepa_abl13/eval
data:
  backend: webdataset
  img_size: 224
  imagenet_norm: false
  num_workers: 8
  batch_size: 256
  webdataset:
    test_dir: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/data/processed/rcc_webdataset_final/test
    pattern: shard-*.tar
    image_key: img.jpg;jpg;jpeg;png
    meta_key: meta.json;json
labels:
  class_order:
  - ccRCC
  - pRCC
  - CHROMO
  - ONCO
  - NOT_TUMOR
model:
  name: i_jepa_ssl_linear_best
  arch_hint: ssl_linear
  backbone_name: vit_small_patch16_224
  ssl_backbone_ckpt: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251123_213900_i_jepa/exp_i_jepa_abl13/checkpoints/i_jepa__ssl_best.pt
  ssl_head_ckpt: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251123_213900_i_jepa/exp_i_jepa_abl13/checkpoints/i_jepa__ssl_linear_best.pt
  strict_load: false
  allow_arch_autoswap: false
evaluation:
  save_logits: true
  save_embeddings: true
  save_preds_csv: true
  umap:
    enabled: true
    source: features
    n_neighbors: 15
    min_dist: 0.1
    random_state: 1337
runtime:
  device: cuda
  precision: fp32
>>

auto_configs/exp_20251123_213900_i_jepa__exp_i_jepa_abl14.yaml codice <<
experiment:
  name: eval_exp_i_jepa_abl14
  seed: 1337
  outputs_root: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251123_213900_i_jepa/exp_i_jepa_abl14/eval
data:
  backend: webdataset
  img_size: 224
  imagenet_norm: false
  num_workers: 8
  batch_size: 256
  webdataset:
    test_dir: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/data/processed/rcc_webdataset_final/test
    pattern: shard-*.tar
    image_key: img.jpg;jpg;jpeg;png
    meta_key: meta.json;json
labels:
  class_order:
  - ccRCC
  - pRCC
  - CHROMO
  - ONCO
  - NOT_TUMOR
model:
  name: i_jepa_ssl_linear_best
  arch_hint: ssl_linear
  backbone_name: vit_small_patch16_224
  ssl_backbone_ckpt: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251123_213900_i_jepa/exp_i_jepa_abl14/checkpoints/i_jepa__ssl_best.pt
  ssl_head_ckpt: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251123_213900_i_jepa/exp_i_jepa_abl14/checkpoints/i_jepa__ssl_linear_best.pt
  strict_load: false
  allow_arch_autoswap: false
evaluation:
  save_logits: true
  save_embeddings: true
  save_preds_csv: true
  umap:
    enabled: true
    source: features
    n_neighbors: 15
    min_dist: 0.1
    random_state: 1337
runtime:
  device: cuda
  precision: fp32
>>

auto_configs/exp_20251123_213900_i_jepa__exp_i_jepa_abl16.yaml codice <<
experiment:
  name: eval_exp_i_jepa_abl16
  seed: 1337
  outputs_root: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251123_213900_i_jepa/exp_i_jepa_abl16/eval
data:
  backend: webdataset
  img_size: 224
  imagenet_norm: false
  num_workers: 8
  batch_size: 256
  webdataset:
    test_dir: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/data/processed/rcc_webdataset_final/test
    pattern: shard-*.tar
    image_key: img.jpg;jpg;jpeg;png
    meta_key: meta.json;json
labels:
  class_order:
  - ccRCC
  - pRCC
  - CHROMO
  - ONCO
  - NOT_TUMOR
model:
  name: i_jepa_ssl_linear_best
  arch_hint: ssl_linear
  backbone_name: vit_small_patch16_224
  ssl_backbone_ckpt: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251123_213900_i_jepa/exp_i_jepa_abl16/checkpoints/i_jepa__ssl_best.pt
  ssl_head_ckpt: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251123_213900_i_jepa/exp_i_jepa_abl16/checkpoints/i_jepa__ssl_linear_best.pt
  strict_load: false
  allow_arch_autoswap: false
evaluation:
  save_logits: true
  save_embeddings: true
  save_preds_csv: true
  umap:
    enabled: true
    source: features
    n_neighbors: 15
    min_dist: 0.1
    random_state: 1337
runtime:
  device: cuda
  precision: fp32
>>

auto_configs/exp_20251123_213900_i_jepa__exp_i_jepa_abl17.yaml codice <<
experiment:
  name: eval_exp_i_jepa_abl17
  seed: 1337
  outputs_root: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251123_213900_i_jepa/exp_i_jepa_abl17/eval
data:
  backend: webdataset
  img_size: 224
  imagenet_norm: false
  num_workers: 8
  batch_size: 256
  webdataset:
    test_dir: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/data/processed/rcc_webdataset_final/test
    pattern: shard-*.tar
    image_key: img.jpg;jpg;jpeg;png
    meta_key: meta.json;json
labels:
  class_order:
  - ccRCC
  - pRCC
  - CHROMO
  - ONCO
  - NOT_TUMOR
model:
  name: i_jepa_ssl_linear_best
  arch_hint: ssl_linear
  backbone_name: vit_small_patch16_224
  ssl_backbone_ckpt: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251123_213900_i_jepa/exp_i_jepa_abl17/checkpoints/i_jepa__ssl_best.pt
  ssl_head_ckpt: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251123_213900_i_jepa/exp_i_jepa_abl17/checkpoints/i_jepa__ssl_linear_best.pt
  strict_load: false
  allow_arch_autoswap: false
evaluation:
  save_logits: true
  save_embeddings: true
  save_preds_csv: true
  umap:
    enabled: true
    source: features
    n_neighbors: 15
    min_dist: 0.1
    random_state: 1337
runtime:
  device: cuda
  precision: fp32
>>

auto_configs/exp_20251123_213900_i_jepa__exp_i_jepa_abl18.yaml codice <<
experiment:
  name: eval_exp_i_jepa_abl18
  seed: 1337
  outputs_root: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251123_213900_i_jepa/exp_i_jepa_abl18/eval
data:
  backend: webdataset
  img_size: 224
  imagenet_norm: false
  num_workers: 8
  batch_size: 256
  webdataset:
    test_dir: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/data/processed/rcc_webdataset_final/test
    pattern: shard-*.tar
    image_key: img.jpg;jpg;jpeg;png
    meta_key: meta.json;json
labels:
  class_order:
  - ccRCC
  - pRCC
  - CHROMO
  - ONCO
  - NOT_TUMOR
model:
  name: i_jepa_ssl_linear_best
  arch_hint: ssl_linear
  backbone_name: vit_small_patch16_224
  ssl_backbone_ckpt: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251123_213900_i_jepa/exp_i_jepa_abl18/checkpoints/i_jepa__ssl_best.pt
  ssl_head_ckpt: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251123_213900_i_jepa/exp_i_jepa_abl18/checkpoints/i_jepa__ssl_linear_best.pt
  strict_load: false
  allow_arch_autoswap: false
evaluation:
  save_logits: true
  save_embeddings: true
  save_preds_csv: true
  umap:
    enabled: true
    source: features
    n_neighbors: 15
    min_dist: 0.1
    random_state: 1337
runtime:
  device: cuda
  precision: fp32
>>

auto_configs/exp_20251123_213900_i_jepa__exp_i_jepa_abl19.yaml codice <<
experiment:
  name: eval_exp_i_jepa_abl19
  seed: 1337
  outputs_root: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251123_213900_i_jepa/exp_i_jepa_abl19/eval
data:
  backend: webdataset
  img_size: 224
  imagenet_norm: false
  num_workers: 8
  batch_size: 256
  webdataset:
    test_dir: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/data/processed/rcc_webdataset_final/test
    pattern: shard-*.tar
    image_key: img.jpg;jpg;jpeg;png
    meta_key: meta.json;json
labels:
  class_order:
  - ccRCC
  - pRCC
  - CHROMO
  - ONCO
  - NOT_TUMOR
model:
  name: i_jepa_ssl_linear_best
  arch_hint: ssl_linear
  backbone_name: vit_small_patch16_224
  ssl_backbone_ckpt: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251123_213900_i_jepa/exp_i_jepa_abl19/checkpoints/i_jepa__ssl_best.pt
  ssl_head_ckpt: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251123_213900_i_jepa/exp_i_jepa_abl19/checkpoints/i_jepa__ssl_linear_best.pt
  strict_load: false
  allow_arch_autoswap: false
evaluation:
  save_logits: true
  save_embeddings: true
  save_preds_csv: true
  umap:
    enabled: true
    source: features
    n_neighbors: 15
    min_dist: 0.1
    random_state: 1337
runtime:
  device: cuda
  precision: fp32
>>

auto_configs/exp_20251123_213900_i_jepa__exp_i_jepa_abl20.yaml codice <<
experiment:
  name: eval_exp_i_jepa_abl20
  seed: 1337
  outputs_root: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251123_213900_i_jepa/exp_i_jepa_abl20/eval
data:
  backend: webdataset
  img_size: 224
  imagenet_norm: false
  num_workers: 8
  batch_size: 256
  webdataset:
    test_dir: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/data/processed/rcc_webdataset_final/test
    pattern: shard-*.tar
    image_key: img.jpg;jpg;jpeg;png
    meta_key: meta.json;json
labels:
  class_order:
  - ccRCC
  - pRCC
  - CHROMO
  - ONCO
  - NOT_TUMOR
model:
  name: i_jepa_ssl_linear_best
  arch_hint: ssl_linear
  backbone_name: vit_small_patch16_224
  ssl_backbone_ckpt: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251123_213900_i_jepa/exp_i_jepa_abl20/checkpoints/i_jepa__ssl_best.pt
  ssl_head_ckpt: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251123_213900_i_jepa/exp_i_jepa_abl20/checkpoints/i_jepa__ssl_linear_best.pt
  strict_load: false
  allow_arch_autoswap: false
evaluation:
  save_logits: true
  save_embeddings: true
  save_preds_csv: true
  umap:
    enabled: true
    source: features
    n_neighbors: 15
    min_dist: 0.1
    random_state: 1337
runtime:
  device: cuda
  precision: fp32
>>

auto_configs/exp_20251123_213900_i_jepa__exp_i_jepa_abl21.yaml codice <<
experiment:
  name: eval_exp_i_jepa_abl21
  seed: 1337
  outputs_root: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251123_213900_i_jepa/exp_i_jepa_abl21/eval
data:
  backend: webdataset
  img_size: 224
  imagenet_norm: false
  num_workers: 8
  batch_size: 256
  webdataset:
    test_dir: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/data/processed/rcc_webdataset_final/test
    pattern: shard-*.tar
    image_key: img.jpg;jpg;jpeg;png
    meta_key: meta.json;json
labels:
  class_order:
  - ccRCC
  - pRCC
  - CHROMO
  - ONCO
  - NOT_TUMOR
model:
  name: i_jepa_ssl_linear_best
  arch_hint: ssl_linear
  backbone_name: vit_small_patch16_224
  ssl_backbone_ckpt: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251123_213900_i_jepa/exp_i_jepa_abl21/checkpoints/i_jepa__ssl_best.pt
  ssl_head_ckpt: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251123_213900_i_jepa/exp_i_jepa_abl21/checkpoints/i_jepa__ssl_linear_best.pt
  strict_load: false
  allow_arch_autoswap: false
evaluation:
  save_logits: true
  save_embeddings: true
  save_preds_csv: true
  umap:
    enabled: true
    source: features
    n_neighbors: 15
    min_dist: 0.1
    random_state: 1337
runtime:
  device: cuda
  precision: fp32
>>

auto_configs/exp_20251123_213900_i_jepa__exp_i_jepa_abl22.yaml codice <<
experiment:
  name: eval_exp_i_jepa_abl22
  seed: 1337
  outputs_root: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251123_213900_i_jepa/exp_i_jepa_abl22/eval
data:
  backend: webdataset
  img_size: 224
  imagenet_norm: false
  num_workers: 8
  batch_size: 256
  webdataset:
    test_dir: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/data/processed/rcc_webdataset_final/test
    pattern: shard-*.tar
    image_key: img.jpg;jpg;jpeg;png
    meta_key: meta.json;json
labels:
  class_order:
  - ccRCC
  - pRCC
  - CHROMO
  - ONCO
  - NOT_TUMOR
model:
  name: i_jepa_ssl_linear_best
  arch_hint: ssl_linear
  backbone_name: vit_small_patch16_224
  ssl_backbone_ckpt: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251123_213900_i_jepa/exp_i_jepa_abl22/checkpoints/i_jepa__ssl_best.pt
  ssl_head_ckpt: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251123_213900_i_jepa/exp_i_jepa_abl22/checkpoints/i_jepa__ssl_linear_best.pt
  strict_load: false
  allow_arch_autoswap: false
evaluation:
  save_logits: true
  save_embeddings: true
  save_preds_csv: true
  umap:
    enabled: true
    source: features
    n_neighbors: 15
    min_dist: 0.1
    random_state: 1337
runtime:
  device: cuda
  precision: fp32
>>

auto_configs/exp_20251123_213900_i_jepa__exp_i_jepa_abl24.yaml codice <<
experiment:
  name: eval_exp_i_jepa_abl24
  seed: 1337
  outputs_root: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251123_213900_i_jepa/exp_i_jepa_abl24/eval
data:
  backend: webdataset
  img_size: 224
  imagenet_norm: false
  num_workers: 8
  batch_size: 256
  webdataset:
    test_dir: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/data/processed/rcc_webdataset_final/test
    pattern: shard-*.tar
    image_key: img.jpg;jpg;jpeg;png
    meta_key: meta.json;json
labels:
  class_order:
  - ccRCC
  - pRCC
  - CHROMO
  - ONCO
  - NOT_TUMOR
model:
  name: i_jepa_ssl_linear_best
  arch_hint: ssl_linear
  backbone_name: vit_small_patch16_224
  ssl_backbone_ckpt: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251123_213900_i_jepa/exp_i_jepa_abl24/checkpoints/i_jepa__ssl_best.pt
  ssl_head_ckpt: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251123_213900_i_jepa/exp_i_jepa_abl24/checkpoints/i_jepa__ssl_linear_best.pt
  strict_load: false
  allow_arch_autoswap: false
evaluation:
  save_logits: true
  save_embeddings: true
  save_preds_csv: true
  umap:
    enabled: true
    source: features
    n_neighbors: 15
    min_dist: 0.1
    random_state: 1337
runtime:
  device: cuda
  precision: fp32
>>

auto_configs/exp_20251123_213900_i_jepa__exp_i_jepa_abl25.yaml codice <<
experiment:
  name: eval_exp_i_jepa_abl25
  seed: 1337
  outputs_root: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251123_213900_i_jepa/exp_i_jepa_abl25/eval
data:
  backend: webdataset
  img_size: 224
  imagenet_norm: false
  num_workers: 8
  batch_size: 256
  webdataset:
    test_dir: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/data/processed/rcc_webdataset_final/test
    pattern: shard-*.tar
    image_key: img.jpg;jpg;jpeg;png
    meta_key: meta.json;json
labels:
  class_order:
  - ccRCC
  - pRCC
  - CHROMO
  - ONCO
  - NOT_TUMOR
model:
  name: i_jepa_ssl_linear_best
  arch_hint: ssl_linear
  backbone_name: vit_small_patch16_224
  ssl_backbone_ckpt: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251123_213900_i_jepa/exp_i_jepa_abl25/checkpoints/i_jepa__ssl_best.pt
  ssl_head_ckpt: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251123_213900_i_jepa/exp_i_jepa_abl25/checkpoints/i_jepa__ssl_linear_best.pt
  strict_load: false
  allow_arch_autoswap: false
evaluation:
  save_logits: true
  save_embeddings: true
  save_preds_csv: true
  umap:
    enabled: true
    source: features
    n_neighbors: 15
    min_dist: 0.1
    random_state: 1337
runtime:
  device: cuda
  precision: fp32
>>

auto_configs/exp_20251123_213900_i_jepa__exp_i_jepa_abl26.yaml codice <<
experiment:
  name: eval_exp_i_jepa_abl26
  seed: 1337
  outputs_root: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251123_213900_i_jepa/exp_i_jepa_abl26/eval
data:
  backend: webdataset
  img_size: 224
  imagenet_norm: false
  num_workers: 8
  batch_size: 256
  webdataset:
    test_dir: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/data/processed/rcc_webdataset_final/test
    pattern: shard-*.tar
    image_key: img.jpg;jpg;jpeg;png
    meta_key: meta.json;json
labels:
  class_order:
  - ccRCC
  - pRCC
  - CHROMO
  - ONCO
  - NOT_TUMOR
model:
  name: i_jepa_ssl_linear_best
  arch_hint: ssl_linear
  backbone_name: vit_small_patch16_224
  ssl_backbone_ckpt: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251123_213900_i_jepa/exp_i_jepa_abl26/checkpoints/i_jepa__ssl_best.pt
  ssl_head_ckpt: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251123_213900_i_jepa/exp_i_jepa_abl26/checkpoints/i_jepa__ssl_linear_best.pt
  strict_load: false
  allow_arch_autoswap: false
evaluation:
  save_logits: true
  save_embeddings: true
  save_preds_csv: true
  umap:
    enabled: true
    source: features
    n_neighbors: 15
    min_dist: 0.1
    random_state: 1337
runtime:
  device: cuda
  precision: fp32
>>

auto_configs/exp_20251123_213900_i_jepa__exp_i_jepa_abl27.yaml codice <<
experiment:
  name: eval_exp_i_jepa_abl27
  seed: 1337
  outputs_root: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251123_213900_i_jepa/exp_i_jepa_abl27/eval
data:
  backend: webdataset
  img_size: 224
  imagenet_norm: false
  num_workers: 8
  batch_size: 256
  webdataset:
    test_dir: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/data/processed/rcc_webdataset_final/test
    pattern: shard-*.tar
    image_key: img.jpg;jpg;jpeg;png
    meta_key: meta.json;json
labels:
  class_order:
  - ccRCC
  - pRCC
  - CHROMO
  - ONCO
  - NOT_TUMOR
model:
  name: i_jepa_ssl_linear_best
  arch_hint: ssl_linear
  backbone_name: vit_small_patch16_224
  ssl_backbone_ckpt: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251123_213900_i_jepa/exp_i_jepa_abl27/checkpoints/i_jepa__ssl_best.pt
  ssl_head_ckpt: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251123_213900_i_jepa/exp_i_jepa_abl27/checkpoints/i_jepa__ssl_linear_best.pt
  strict_load: false
  allow_arch_autoswap: false
evaluation:
  save_logits: true
  save_embeddings: true
  save_preds_csv: true
  umap:
    enabled: true
    source: features
    n_neighbors: 15
    min_dist: 0.1
    random_state: 1337
runtime:
  device: cuda
  precision: fp32
>>

auto_configs/exp_20251123_220420_moco_v3__exp_moco_v3_abl01.yaml codice <<
experiment:
  name: eval_exp_moco_v3_abl01
  seed: 1337
  outputs_root: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251123_220420_moco_v3/exp_moco_v3_abl01/eval
data:
  backend: webdataset
  img_size: 224
  imagenet_norm: false
  num_workers: 8
  batch_size: 256
  webdataset:
    test_dir: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/data/processed/rcc_webdataset_final/test
    pattern: shard-*.tar
    image_key: img.jpg;jpg;jpeg;png
    meta_key: meta.json;json
labels:
  class_order:
  - ccRCC
  - pRCC
  - CHROMO
  - ONCO
  - NOT_TUMOR
model:
  name: moco_v3_ssl_linear_best
  arch_hint: ssl_linear
  backbone_name: vit_small_patch16_224
  ssl_backbone_ckpt: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251123_220420_moco_v3/exp_moco_v3_abl01/checkpoints/moco_v3__ssl_best.pt
  ssl_head_ckpt: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251123_220420_moco_v3/exp_moco_v3_abl01/checkpoints/moco_v3__ssl_linear_best.pt
  strict_load: false
  allow_arch_autoswap: false
evaluation:
  save_logits: true
  save_embeddings: true
  save_preds_csv: true
  umap:
    enabled: true
    source: features
    n_neighbors: 15
    min_dist: 0.1
    random_state: 1337
runtime:
  device: cuda
  precision: fp32
>>

auto_configs/exp_20251123_220420_moco_v3__exp_moco_v3_abl07.yaml codice <<
experiment:
  name: eval_exp_moco_v3_abl07
  seed: 1337
  outputs_root: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251123_220420_moco_v3/exp_moco_v3_abl07/eval
data:
  backend: webdataset
  img_size: 224
  imagenet_norm: false
  num_workers: 8
  batch_size: 256
  webdataset:
    test_dir: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/data/processed/rcc_webdataset_final/test
    pattern: shard-*.tar
    image_key: img.jpg;jpg;jpeg;png
    meta_key: meta.json;json
labels:
  class_order:
  - ccRCC
  - pRCC
  - CHROMO
  - ONCO
  - NOT_TUMOR
model:
  name: moco_v3_ssl_linear_best
  arch_hint: ssl_linear
  backbone_name: vit_small_patch16_224
  ssl_backbone_ckpt: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251123_220420_moco_v3/exp_moco_v3_abl07/checkpoints/moco_v3__ssl_best.pt
  ssl_head_ckpt: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251123_220420_moco_v3/exp_moco_v3_abl07/checkpoints/moco_v3__ssl_linear_best.pt
  strict_load: false
  allow_arch_autoswap: false
evaluation:
  save_logits: true
  save_embeddings: true
  save_preds_csv: true
  umap:
    enabled: true
    source: features
    n_neighbors: 15
    min_dist: 0.1
    random_state: 1337
runtime:
  device: cuda
  precision: fp32
>>

auto_configs/exp_20251123_220420_moco_v3__exp_moco_v3_abl08.yaml codice <<
experiment:
  name: eval_exp_moco_v3_abl08
  seed: 1337
  outputs_root: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251123_220420_moco_v3/exp_moco_v3_abl08/eval
data:
  backend: webdataset
  img_size: 224
  imagenet_norm: false
  num_workers: 8
  batch_size: 256
  webdataset:
    test_dir: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/data/processed/rcc_webdataset_final/test
    pattern: shard-*.tar
    image_key: img.jpg;jpg;jpeg;png
    meta_key: meta.json;json
labels:
  class_order:
  - ccRCC
  - pRCC
  - CHROMO
  - ONCO
  - NOT_TUMOR
model:
  name: moco_v3_ssl_linear_best
  arch_hint: ssl_linear
  backbone_name: vit_small_patch16_224
  ssl_backbone_ckpt: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251123_220420_moco_v3/exp_moco_v3_abl08/checkpoints/moco_v3__ssl_best.pt
  ssl_head_ckpt: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251123_220420_moco_v3/exp_moco_v3_abl08/checkpoints/moco_v3__ssl_linear_best.pt
  strict_load: false
  allow_arch_autoswap: false
evaluation:
  save_logits: true
  save_embeddings: true
  save_preds_csv: true
  umap:
    enabled: true
    source: features
    n_neighbors: 15
    min_dist: 0.1
    random_state: 1337
runtime:
  device: cuda
  precision: fp32
>>

auto_configs/exp_20251123_220420_moco_v3__exp_moco_v3_abl09.yaml codice <<
experiment:
  name: eval_exp_moco_v3_abl09
  seed: 1337
  outputs_root: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251123_220420_moco_v3/exp_moco_v3_abl09/eval
data:
  backend: webdataset
  img_size: 224
  imagenet_norm: false
  num_workers: 8
  batch_size: 256
  webdataset:
    test_dir: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/data/processed/rcc_webdataset_final/test
    pattern: shard-*.tar
    image_key: img.jpg;jpg;jpeg;png
    meta_key: meta.json;json
labels:
  class_order:
  - ccRCC
  - pRCC
  - CHROMO
  - ONCO
  - NOT_TUMOR
model:
  name: moco_v3_ssl_linear_best
  arch_hint: ssl_linear
  backbone_name: vit_small_patch16_224
  ssl_backbone_ckpt: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251123_220420_moco_v3/exp_moco_v3_abl09/checkpoints/moco_v3__ssl_best.pt
  ssl_head_ckpt: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251123_220420_moco_v3/exp_moco_v3_abl09/checkpoints/moco_v3__ssl_linear_best.pt
  strict_load: false
  allow_arch_autoswap: false
evaluation:
  save_logits: true
  save_embeddings: true
  save_preds_csv: true
  umap:
    enabled: true
    source: features
    n_neighbors: 15
    min_dist: 0.1
    random_state: 1337
runtime:
  device: cuda
  precision: fp32
>>

auto_configs/exp_20251123_220420_moco_v3__exp_moco_v3_abl11.yaml codice <<
experiment:
  name: eval_exp_moco_v3_abl11
  seed: 1337
  outputs_root: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251123_220420_moco_v3/exp_moco_v3_abl11/eval
data:
  backend: webdataset
  img_size: 224
  imagenet_norm: false
  num_workers: 8
  batch_size: 256
  webdataset:
    test_dir: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/data/processed/rcc_webdataset_final/test
    pattern: shard-*.tar
    image_key: img.jpg;jpg;jpeg;png
    meta_key: meta.json;json
labels:
  class_order:
  - ccRCC
  - pRCC
  - CHROMO
  - ONCO
  - NOT_TUMOR
model:
  name: moco_v3_ssl_linear_best
  arch_hint: ssl_linear
  backbone_name: vit_small_patch16_224
  ssl_backbone_ckpt: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251123_220420_moco_v3/exp_moco_v3_abl11/checkpoints/moco_v3__ssl_best.pt
  ssl_head_ckpt: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251123_220420_moco_v3/exp_moco_v3_abl11/checkpoints/moco_v3__ssl_linear_best.pt
  strict_load: false
  allow_arch_autoswap: false
evaluation:
  save_logits: true
  save_embeddings: true
  save_preds_csv: true
  umap:
    enabled: true
    source: features
    n_neighbors: 15
    min_dist: 0.1
    random_state: 1337
runtime:
  device: cuda
  precision: fp32
>>

auto_configs/exp_20251123_220420_moco_v3__exp_moco_v3_abl12.yaml codice <<
experiment:
  name: eval_exp_moco_v3_abl12
  seed: 1337
  outputs_root: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251123_220420_moco_v3/exp_moco_v3_abl12/eval
data:
  backend: webdataset
  img_size: 224
  imagenet_norm: false
  num_workers: 8
  batch_size: 256
  webdataset:
    test_dir: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/data/processed/rcc_webdataset_final/test
    pattern: shard-*.tar
    image_key: img.jpg;jpg;jpeg;png
    meta_key: meta.json;json
labels:
  class_order:
  - ccRCC
  - pRCC
  - CHROMO
  - ONCO
  - NOT_TUMOR
model:
  name: moco_v3_ssl_linear_best
  arch_hint: ssl_linear
  backbone_name: vit_small_patch16_224
  ssl_backbone_ckpt: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251123_220420_moco_v3/exp_moco_v3_abl12/checkpoints/moco_v3__ssl_best.pt
  ssl_head_ckpt: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251123_220420_moco_v3/exp_moco_v3_abl12/checkpoints/moco_v3__ssl_linear_best.pt
  strict_load: false
  allow_arch_autoswap: false
evaluation:
  save_logits: true
  save_embeddings: true
  save_preds_csv: true
  umap:
    enabled: true
    source: features
    n_neighbors: 15
    min_dist: 0.1
    random_state: 1337
runtime:
  device: cuda
  precision: fp32
>>

auto_configs/exp_20251123_220420_moco_v3__exp_moco_v3_abl13.yaml codice <<
experiment:
  name: eval_exp_moco_v3_abl13
  seed: 1337
  outputs_root: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251123_220420_moco_v3/exp_moco_v3_abl13/eval
data:
  backend: webdataset
  img_size: 224
  imagenet_norm: false
  num_workers: 8
  batch_size: 256
  webdataset:
    test_dir: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/data/processed/rcc_webdataset_final/test
    pattern: shard-*.tar
    image_key: img.jpg;jpg;jpeg;png
    meta_key: meta.json;json
labels:
  class_order:
  - ccRCC
  - pRCC
  - CHROMO
  - ONCO
  - NOT_TUMOR
model:
  name: moco_v3_ssl_linear_best
  arch_hint: ssl_linear
  backbone_name: vit_small_patch16_224
  ssl_backbone_ckpt: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251123_220420_moco_v3/exp_moco_v3_abl13/checkpoints/moco_v3__ssl_best.pt
  ssl_head_ckpt: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251123_220420_moco_v3/exp_moco_v3_abl13/checkpoints/moco_v3__ssl_linear_best.pt
  strict_load: false
  allow_arch_autoswap: false
evaluation:
  save_logits: true
  save_embeddings: true
  save_preds_csv: true
  umap:
    enabled: true
    source: features
    n_neighbors: 15
    min_dist: 0.1
    random_state: 1337
runtime:
  device: cuda
  precision: fp32
>>

auto_configs/exp_20251123_220420_moco_v3__exp_moco_v3_abl14.yaml codice <<
experiment:
  name: eval_exp_moco_v3_abl14
  seed: 1337
  outputs_root: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251123_220420_moco_v3/exp_moco_v3_abl14/eval
data:
  backend: webdataset
  img_size: 224
  imagenet_norm: false
  num_workers: 8
  batch_size: 256
  webdataset:
    test_dir: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/data/processed/rcc_webdataset_final/test
    pattern: shard-*.tar
    image_key: img.jpg;jpg;jpeg;png
    meta_key: meta.json;json
labels:
  class_order:
  - ccRCC
  - pRCC
  - CHROMO
  - ONCO
  - NOT_TUMOR
model:
  name: moco_v3_ssl_linear_best
  arch_hint: ssl_linear
  backbone_name: vit_small_patch16_224
  ssl_backbone_ckpt: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251123_220420_moco_v3/exp_moco_v3_abl14/checkpoints/moco_v3__ssl_best.pt
  ssl_head_ckpt: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251123_220420_moco_v3/exp_moco_v3_abl14/checkpoints/moco_v3__ssl_linear_best.pt
  strict_load: false
  allow_arch_autoswap: false
evaluation:
  save_logits: true
  save_embeddings: true
  save_preds_csv: true
  umap:
    enabled: true
    source: features
    n_neighbors: 15
    min_dist: 0.1
    random_state: 1337
runtime:
  device: cuda
  precision: fp32
>>

auto_configs/exp_20251123_220420_moco_v3__exp_moco_v3_abl15.yaml codice <<
experiment:
  name: eval_exp_moco_v3_abl15
  seed: 1337
  outputs_root: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251123_220420_moco_v3/exp_moco_v3_abl15/eval
data:
  backend: webdataset
  img_size: 224
  imagenet_norm: false
  num_workers: 8
  batch_size: 256
  webdataset:
    test_dir: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/data/processed/rcc_webdataset_final/test
    pattern: shard-*.tar
    image_key: img.jpg;jpg;jpeg;png
    meta_key: meta.json;json
labels:
  class_order:
  - ccRCC
  - pRCC
  - CHROMO
  - ONCO
  - NOT_TUMOR
model:
  name: moco_v3_ssl_linear_best
  arch_hint: ssl_linear
  backbone_name: vit_small_patch16_224
  ssl_backbone_ckpt: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251123_220420_moco_v3/exp_moco_v3_abl15/checkpoints/moco_v3__ssl_best.pt
  ssl_head_ckpt: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251123_220420_moco_v3/exp_moco_v3_abl15/checkpoints/moco_v3__ssl_linear_best.pt
  strict_load: false
  allow_arch_autoswap: false
evaluation:
  save_logits: true
  save_embeddings: true
  save_preds_csv: true
  umap:
    enabled: true
    source: features
    n_neighbors: 15
    min_dist: 0.1
    random_state: 1337
runtime:
  device: cuda
  precision: fp32
>>

eval_models.sbatch codice <<
#!/usr/bin/env bash
#SBATCH -J eval
#SBATCH -A mla_group_01
#SBATCH -p gpu_a40
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH -o /home/mla_group_01/rcc-ssrl/logs/05_eval/eval.%j.out
#SBATCH -e /home/mla_group_01/rcc-ssrl/logs/05_eval/eval.%j.err

set -euo pipefail

WORKDIR="/home/mla_group_01/rcc-ssrl/src/evaluation"
VENV="/home/mla_group_01/rcc-ssrl/.venvs/eval"
PYTHON="$VENV/bin/python"

mkdir -p /home/mla_group_01/rcc-ssrl/logs/05_eval
cd "$WORKDIR"

# venv bootstrap (idempotente)
[[ -f "$VENV/bin/activate" ]] || python3 -m venv "$VENV"
source "$VENV/bin/activate"
pip -q install --upgrade pip
pip -q install -r /home/mla_group_01/rcc-ssrl/src/evaluation/requirements_eval.txt

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=1

# === Config obbligatoria: deve arrivare da auto_eval (CFG_PATH) ===
CFG_PATH="${CFG_PATH:-}"

if [[ -z "$CFG_PATH" ]]; then
  echo "[ERROR] CFG_PATH non impostato." >&2
  echo "[HINT] Lancia tramite auto_eval.py --submit oppure esporta CFG_PATH a mano." >&2
  exit 1
fi

CFG_USE="$CFG_PATH"

ONLY_ONE_SHARD="${ONLY_ONE_SHARD:-0}"
SHARD_EXAMPLE="${SHARD_EXAMPLE:-shard-000000.tar}"

TMP_CFG=""
cleanup() { [[ -n "${TMP_CFG}" && -f "${TMP_CFG}" ]] && rm -f "${TMP_CFG}"; }
trap cleanup EXIT

echo "[INFO] Host=$(hostname)  CFG_USE=${CFG_USE}  SMOKE=${ONLY_ONE_SHARD:-0}"

exec "$PYTHON" eval.py --config "$CFG_USE"
>>

eval.py codice <<
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test-only evaluation per modelli già addestrati (encoder + classifier).
Patch: salva predictions.csv ARRICCHITO con wds_key + metadati per XAI alignment.
"""
import os, sys, json, argparse, logging, random, csv
from pathlib import Path
from datetime import datetime
import numpy as np
from ssl_linear_loader import SSLLinearClassifier  # local import

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torchvision import transforms, datasets
import torchvision

from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix, roc_auc_score,
    classification_report, average_precision_score
)
from sklearn.preprocessing import label_binarize

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# opzionali
try:
    import webdataset as wds
    HAVE_WDS = True
except Exception:
    HAVE_WDS = False

try:
    import umap
    HAVE_UMAP = True
except Exception:
    HAVE_UMAP = False


# ------------------------ utils base ------------------------
def set_seed(s: int):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.benchmark = True

def setup_logger():
    log = logging.getLogger("eval")
    log.setLevel(logging.INFO)
    h = logging.StreamHandler(sys.stdout); h.setLevel(logging.INFO)
    h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", "%H:%M:%S"))
    if not log.handlers:
        log.addHandler(h)
    return log

def default_preprocess(img_size, imagenet_norm=False):
    from torchvision import transforms
    tfm = [
        transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),  # [0,1]
    ]
    if imagenet_norm:
        tfm.append(transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]))
    return transforms.Compose(tfm)


# ------------------------ plotting ------------------------
def plot_confmat(cm, class_names, out_png):
    plt.figure(figsize=(7,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel("True"); plt.xlabel("Predicted"); plt.tight_layout()
    plt.savefig(out_png, dpi=150); plt.close()

def plot_umap_logits(logits, labels, class_names, out_png, umap_cfg):
    reducer = umap.UMAP(
        n_neighbors=umap_cfg["n_neighbors"],
        min_dist=umap_cfg["min_dist"],
        random_state=umap_cfg["random_state"]
    )
    X2 = reducer.fit_transform(logits)
    plt.figure(figsize=(7,6))
    palette = plt.cm.tab10.colors
    for cid, cname in enumerate(class_names):
        idx = labels == cid
        plt.scatter(X2[idx,0], X2[idx,1], s=3, alpha=0.7, label=cname, color=palette[cid % len(palette)])
    plt.legend(markerscale=4, fontsize=8, bbox_to_anchor=(1.05,1), loc="upper left")
    plt.tight_layout(); plt.savefig(out_png, dpi=150, bbox_inches="tight"); plt.close()


# ------------------------ model loaders ------------------------
def load_model_from_repo(repo_root, module_name, class_name, checkpoint, strict, log):
    sys.path.insert(0, os.path.join(repo_root, "src"))
    mod = __import__(module_name, fromlist=[class_name])
    ModelClass = getattr(mod, class_name)
    model = ModelClass()
    if checkpoint and os.path.isfile(checkpoint):
        sd = torch.load(checkpoint, map_location="cpu")
        for k in ["state_dict", "model", "module", "net"]:
            if isinstance(sd, dict) and k in sd and isinstance(sd[k], dict):
                sd = sd[k]; break
        sd = {k.replace("module.", ""): v for k, v in sd.items()} if isinstance(sd, dict) else sd
        missing, unexpected = model.load_state_dict(sd, strict=strict)
        log.info(f"Checkpoint caricato. Missing:{len(missing)} Unexpected:{len(unexpected)}")
    else:
        log.warning("Checkpoint mancante o non leggibile.")
    return model

def load_fallback_resnet50(checkpoint, strict, num_classes=5, log=None):
    model = torchvision.models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    if checkpoint and os.path.isfile(checkpoint):
        sd = torch.load(checkpoint, map_location="cpu")
        if "state_dict" in sd: sd = sd["state_dict"]
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        missing, unexpected = model.load_state_dict(sd, strict=strict)
        if log: log.info(f"Checkpoint (fallback). Missing:{len(missing)} Unexpected:{len(unexpected)}")
    return model


# ------------------------ helpers ------------------------
def softmax_logits(x):
    x = x - x.max(dim=1, keepdim=True).values
    return torch.exp(x) / torch.exp(x).sum(dim=1, keepdim=True)

def _parse_meta(meta_any):
    if isinstance(meta_any, (bytes, bytearray)):
        return json.loads(meta_any.decode("utf-8"))
    if isinstance(meta_any, str):
        return json.loads(meta_any)
    if isinstance(meta_any, dict):
        return meta_any
    if isinstance(meta_any, (list, tuple)) and len(meta_any) == 1:
        return _parse_meta(meta_any[0])
    return {}

# ------------------------ WebDataset loader (grezzo, con __key__) ------------------------
def make_wds_loader(test_dir, pattern, image_key, meta_key, class_order, preprocess, batch_size, num_workers):
    import os, glob
    if not HAVE_WDS:
        raise RuntimeError("webdataset non disponibile")

    shard_glob = os.path.join(test_dir, pattern)
    shards = sorted(glob.glob(shard_glob))
    if len(shards) == 0:
        raise FileNotFoundError(f"Nessun shard trovato con pattern: {shard_glob}")

    seen_keys = set()
    def _is_new(sample):
        # sample è un dict in questa fase
        k = sample.get("__key__")
        if k in seen_keys:
            return False
        seen_keys.add(k)
        return True

    def _is_valid_tuple(t):
        # dopo to_tuple: vogliamo (img, meta, key) tutti non-null
        return (
            isinstance(t, (tuple, list)) and len(t) >= 3
            and (t[0] is not None) and (t[1] is not None) and (t[2] is not None)
        )

    ds = (
        wds.WebDataset(
            shards,
            shardshuffle=False,
            handler=wds.warn_and_continue,
            empty_check=False
        )
        .decode("pil")
        .select(_is_new)
        .to_tuple(image_key, meta_key, "__key__", handler=wds.warn_and_continue)  # -> (img, meta, key)
        .select(_is_valid_tuple)                                                 # filtra qualsiasi anomalia
        .map_tuple(preprocess, lambda m: m, lambda k: k)                          # (img_t, meta_raw, key)
        .shuffle(0)
        .repeat(0)
    )

    eff_workers = min(num_workers, max(1, len(shards)))

    # batch_size=1 + collate_fn robusto: ritorna direttamente la tupla o None
    def _collate_first(batch):
        if not batch:
            return None
        item = batch[0]  # (img, meta, key)
        if not (isinstance(item, (tuple, list)) and len(item) >= 3):
            return None
        img, meta, key = item
        # Se l'immagine è 3D, aggiungi la batch dim
        if isinstance(img, torch.Tensor) and img.ndim == 3:
            img = img.unsqueeze(0)  # [1,C,H,W]
        return (img, meta, key)


    return DataLoader(
        ds,
        batch_size=1,  # <<< importante per avere una tupla (img, meta, key)
        num_workers=eff_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=_collate_first,
    )





# ------------------------ main ------------------------
def main():
    import yaml
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    log = setup_logger()
    set_seed(cfg["experiment"]["seed"])

    device = torch.device(cfg["runtime"].get("device","cuda") if torch.cuda.is_available() else "cpu")
    img_size = int(cfg["data"]["img_size"])
    imagenet_norm = bool(cfg["data"].get("imagenet_norm", False))
    preprocess = default_preprocess(img_size, imagenet_norm)

    # output dir
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_tag = cfg["model"]["name"]
    out_root = Path(cfg["experiment"]["outputs_root"]) / model_tag / ts
    out_root.mkdir(parents=True, exist_ok=True)
    log.info(f"Output dir: {str(out_root)}")

    # classi
    class_names = cfg.get("labels", {}).get("class_order", ["ccRCC","pRCC","CHROMO","ONCO","NOT_TUMOR"])
    class_to_id = {c:i for i,c in enumerate(class_names)}

    # loader TEST
    backend = cfg["data"]["backend"].lower()
    batch_size = int(cfg["data"]["batch_size"])
    num_workers = int(cfg["data"]["num_workers"])

    if backend == "webdataset":
        w = cfg["data"]["webdataset"]
        test_loader = make_wds_loader(
        test_dir=w["test_dir"],
        pattern=w["pattern"],
        image_key=w["image_key"],
        meta_key=w["meta_key"],
        class_order=class_names,
        preprocess=preprocess,
        batch_size=batch_size,
        num_workers=num_workers
    )

    else:
        ds = datasets.ImageFolder(cfg["data"]["imagefolder"]["test_dir"], transform=preprocess)
        test_loader = DataLoader(ds, batch_size=batch_size, num_workers=num_workers, pin_memory=torch.cuda.is_available())

    # modello
    model_cfg = cfg["model"]
    model = None
    arch_hint = model_cfg.get("arch_hint", "cnn").lower()
    if arch_hint == "ssl_linear":
        num_classes = len(class_names)
        model = SSLLinearClassifier(backbone_name=model_cfg.get("backbone_name", "resnet50"),
                                    num_classes=num_classes)
        allow_swap = bool(model_cfg.get("allow_arch_autoswap", True))
        mb, ub = model.load_backbone_from_ssl(model_cfg["ssl_backbone_ckpt"], allow_autoswap=allow_swap)
        mh, uh = model.load_head_from_probe(model_cfg["ssl_head_ckpt"])
        log.info(f"SSL backbone loaded (missing={mb}, unexpected={ub}); head loaded (missing={mh}, unexpected={uh})")
    else:
        if model_cfg.get("module") and model_cfg.get("class_name"):
            try:
                model = load_model_from_repo(
                    model_cfg["repo_root"], model_cfg["module"], model_cfg["class_name"],
                    model_cfg.get("checkpoint", ""), model_cfg.get("strict_load", False), log
                )
            except Exception as e:
                log.warning(f"Import repo fallito ({e}). Fallback ResNet50.")
        if model is None:
            model = load_fallback_resnet50(
                model_cfg.get("checkpoint",""),
                model_cfg.get("strict_load", False),
                num_classes=len(class_names),
                log=log
            )
    model = model.to(device).eval()

    # === EVALUATION LOOP con salvataggi allineati ===
    y_true, y_pred = [], []
    logits_list = []
    rows = []

    with torch.no_grad():
        if backend == "webdataset":
            for sample in test_loader:
                if sample is None:
                    continue
                img, meta_any, key = sample
                meta = _parse_meta(meta_any)
                lab_txt = meta.get("class_label", None)
                if lab_txt is None:
                    continue
                if lab_txt not in class_to_id:
                    continue
                lab = class_to_id[lab_txt]

                x = img.to(device, non_blocking=True)
                out = model(x)
                logits = out[0] if isinstance(out, (list, tuple)) else (out["logits"] if isinstance(out, dict) and "logits" in out else out)
                pred = int(torch.argmax(logits, dim=1).item())

                y_true.append(lab); y_pred.append(pred)
                logits_list.append(logits.detach().cpu().numpy())

                rows.append({
                    "wds_key": key,
                    "patient_id": meta.get("patient_id"),
                    "slide_id": meta.get("wsi_or_roi") or meta.get("slide_id"),
                    "coords": meta.get("coords"),
                    "y_true": int(lab),
                    "y_pred": int(pred),
                })
        else:
            # imagefolder: niente meta/keys → CSV minimale
            for x, lab in test_loader:
                x = x.to(device, non_blocking=True)
                logits = model(x)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                y_true.append(lab.numpy()); y_pred.append(preds)
                logits_list.append(logits.detach().cpu().numpy())

    y_true = np.concatenate([np.atleast_1d(t) for t in y_true]).astype(int)
    y_pred = np.concatenate([np.atleast_1d(p) for p in y_pred]).astype(int)
    logits_np = np.concatenate(logits_list, axis=0) if len(logits_list)>0 else None

    # === metriche ===
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }
    try:
        if logits_np is not None:
            probs = torch.from_numpy(logits_np)
            probs = softmax_logits(probs).numpy()
            y_bin = label_binarize(y_true, classes=np.arange(len(class_names)))
            metrics["macro_auc_ovr"] = float(roc_auc_score(y_bin, probs, average="macro", multi_class="ovr"))
            metrics["macro_auprc"] = float(average_precision_score(y_bin, probs, average="macro"))
            # ECE
            confidences = probs.max(axis=1); predictions = probs.argmax(axis=1)
            accuracies = (predictions == y_true)
            bins = np.linspace(0.0, 1.0, 16)
            ece = 0.0
            for i in range(15):
                in_bin = (confidences > bins[i]) & (confidences <= bins[i+1])
                prop = in_bin.mean()
                if prop > 0:
                    ece += abs(accuracies[in_bin].mean() - confidences[in_bin].mean()) * prop
            metrics["ece"] = float(ece)
    except Exception as e:
        log.warning(f"AUC/AUPRC/ECE failed: {e}")

    # confusion matrix + report
    cm = confusion_matrix(y_true, y_pred)
    plot_confmat(cm, class_names, out_root / f"cm_{model_tag}.png")
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    with open(out_root / f"report_per_class.json", "w") as f:
        json.dump(report, f, indent=2)

    # salvataggi
    with open(out_root / f"metrics_{model_tag}.json", "w") as f:
        json.dump({
            "experiment": cfg["experiment"],
            "model": cfg["model"],
            "metrics": metrics,
            "class_names": class_names
        }, f, indent=2)

    if logits_np is not None:
        np.save(out_root / "logits_test.npy", logits_np)

    # >>>>>>> CSV con lo STESSO NOME di prima, ma ARRICCHITO <<<<<<<
    with open(out_root / "predictions.csv", "w", newline="") as f:
        if backend == "webdataset":
            fieldnames = ["wds_key", "patient_id", "slide_id", "coords", "y_true", "y_pred"]
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        else:
            w = csv.writer(f)
            w.writerow(["y_true","y_pred"])
            for t,p in zip(y_true, y_pred):
                w.writerow([int(t), int(p)])

    log.info(f"[TEST] Acc={metrics['accuracy']:.4f}  BalAcc={metrics['balanced_accuracy']:.4f}  MacroF1={metrics['macro_f1']:.4f}")
    if "macro_auc_ovr" in metrics:
        log.info(f"[TEST] MacroAUC(OvR)={metrics['macro_auc_ovr']:.4f}")
    log.info("FINITO.")

    # UMAP opzionale
    if cfg["evaluation"]["umap"]["enabled"] and HAVE_UMAP and logits_np is not None:
        source = cfg["evaluation"]["umap"].get("source","logits")
        data_umap = logits_np
        plot_umap_logits(
            data_umap, y_true, class_names,
            out_root / f"embedding_{model_tag}_{source}.png",
            cfg["evaluation"]["umap"]
        )

if __name__ == "__main__":
    main()
>>

requirements_eval.txt codice <<
# core
numpy>=1.23
pyyaml>=6.0
tqdm>=4.66.0

# data + image
Pillow>=9.5
webdataset>=0.2.57

# metrics & viz
scikit-learn>=1.3
umap-learn>=0.5.5
matplotlib>=3.7
seaborn>=0.13

# opzionale per viT da timm (se vuoi alternative a torchvision vit)
timm>=0.9.8
>>

ssl_linear_loader.py codice <<
# file: /home/mla_group_01/rcc-ssrl/src/evaluation/ssl_linear_loader.py
#!/usr/bin/env python3
from __future__ import annotations
from typing import Dict, Tuple
import torch, torch.nn as nn
import os

try:
    import timm
    HAVE_TIMM = True
except Exception:
    HAVE_TIMM = False

# ---- ResNet backbone (fallback / compat) ----
try:
    from src.training.trainer.backbones import ResNetBackbone as _ResNetBackbone
except Exception:
    from torchvision import models
    import torch.nn.functional as F
    class _ResNetBackbone(nn.Module):
        def __init__(self, name: str="resnet50"):
            super().__init__()
            m = models.resnet50(weights=None) if "50" in name else models.resnet34(weights=None)
            self.stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool)
            self.layer1, self.layer2, self.layer3, self.layer4 = m.layer1, m.layer2, m.layer3, m.layer4
            self.out_dim = m.fc.in_features
        def _fwd(self, x):
            x = self.stem(x); x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x); return x
        def forward_global(self, x):
            x = self._fwd(x)
            return torch.flatten(F.adaptive_avg_pool2d(x, 1), 1)

# ---- ViT backbone (timm) ----
class _VitBackbone(nn.Module):
    def __init__(self, name: str="vit_small_patch16_224"):
        super().__init__()
        if not HAVE_TIMM:
            raise RuntimeError("timm is required for ViT backbones.")
        self.model = timm.create_model(name, pretrained=False, num_classes=0, dynamic_img_size=True)
        self.out_dim = self.model.num_features

    def forward_global(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.model.forward_features(x)
        if feats.dim() == 3:  # [B, T, C] tipico ViT
            return feats.mean(dim=1)
        # per sicurezza: se è [B, C, H, W], pool 2D
        return torch.flatten(torch.nn.functional.adaptive_avg_pool2d(feats, 1), 1)


_PREFIXES = ("stu.", "student.", "backbone_q.", "backbone.", "module.stu.", "module.backbone_q.")
def _safe_torch_load(path: str):
    import torch
    try:
        # torch>=2.4
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")

def _load_state(path: str) -> Dict[str, torch.Tensor]:
    sd = _safe_torch_load(path)
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    return sd if isinstance(sd, dict) else {}

def _best_substate(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    cands = [{k[len(p):]: v for k, v in sd.items() if k.startswith(p)} for p in _PREFIXES]
    best = max(cands, key=lambda x: len(x))
    return best if len(best) else sd

def _looks_like_vit(sd: Dict[str, torch.Tensor]) -> bool:
    ks = list(sd.keys())
    return any(k.startswith(("pos_embed","cls_token","blocks.","patch_embed.","norm.","fc_norm")) for k in ks)

def _strip_prefix(d: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    plen = len(prefix)
    return {k[plen:]: v for k, v in d.items() if k.startswith(prefix)}

def _maybe_strip(d: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    return _strip_prefix(d, prefix) if any(k.startswith(prefix) for k in d.keys()) else d

class SSLLinearClassifier(nn.Module):
    """Compose an SSL backbone (ResNet or ViT) with a linear head."""
    def __init__(self, backbone_name: str="resnet50", num_classes: int=5):
        super().__init__()
        if backbone_name.startswith("vit"):
            self.backbone = _VitBackbone(backbone_name)
        else:
            self.backbone = _ResNetBackbone(backbone_name)
        self.head = nn.Linear(self.backbone.out_dim, num_classes)
    @staticmethod
    def _try_load(module: nn.Module, state: Dict[str, torch.Tensor]) -> Tuple[int, int, Tuple[list, list]]:
        missing, unexpected = module.load_state_dict(state, strict=False)
        miss_l, unexp_l = list(missing), list(unexpected)
        if os.environ.get("EVAL_DEBUG", "0") == "1":
            print(f"[DEBUG] backbone try_load: missing={len(miss_l)} unexpected={len(unexp_l)}")
            if len(miss_l) <= 15:  print("[DEBUG]  missing keys:", miss_l)
            if len(unexp_l) <= 15: print("[DEBUG]  unexpected keys:", unexp_l)
        return len(miss_l), len(unexp_l), (miss_l, unexp_l)
    @staticmethod
    def _swap_key_prefix(d: Dict[str, torch.Tensor], old: str, new: str) -> Dict[str, torch.Tensor]:
        # Esempio: fc_norm.xxx -> norm.xxx (alcuni ViT timm differiscono qui)
        out = {}
        for k, v in d.items():
            if k.startswith(old):
                out[new + k[len(old):]] = v
            else:
                out[k] = v
        return out

    def load_backbone_from_ssl(self, ssl_backbone_ckpt: str, allow_autoswap: bool = True) -> Tuple[int, int]:
        raw = _load_state(ssl_backbone_ckpt)
        sub = _best_substate(raw)

        # Rileva tipo checkpoint
        want_vit = _looks_like_vit(sub)
        have_vit = isinstance(self.backbone, _VitBackbone)

        # Autoswap arch se necessario
        if allow_autoswap and want_vit and not have_vit:
            self.backbone = _VitBackbone("vit_small_patch16_224")
            self.head = nn.Linear(self.backbone.out_dim, self.head.out_features)
            have_vit = True
        elif allow_autoswap and (not want_vit) and have_vit:
            self.backbone = _ResNetBackbone("resnet50")
            self.head = nn.Linear(self.backbone.out_dim, self.head.out_features)
            have_vit = False

        # Rimuovi 'vit.' se presente
        sub = _maybe_strip(sub, "vit.")

        target_module = self.backbone.model if have_vit else self.backbone

        # --- Primo tentativo (quello che già fai) ---
        m, u, _ = self._try_load(target_module, sub)
        best = (m + u, m, u, sub)

        # --- Se non è perfetto, prova rimappi equivalenti comuni ---
        if m + u > 0 and have_vit:
            candidates = []

            # 1) strip 'model.' (alcuni export hanno 'model.' già incluso)
            if any(k.startswith("model.") for k in sub.keys()):
                candidates.append({k[6:]: v for k, v in sub.items() if k.startswith("model.")})

            # 2) aggiungi 'model.' su tutte le chiavi (in caso inverso)
            candidates.append({f"model.{k}": v for k, v in sub.items()})

            # 3) fc_norm <-> norm (varianti timm)
            candidates.append(_swap_key_prefix(sub, "fc_norm.", "norm."))
            candidates.append(_swap_key_prefix(sub, "norm.", "fc_norm."))

            for cand in candidates:
                m2, u2, _ = self._try_load(target_module, cand)
                score = m2 + u2
                if score < best[0]:
                    best = (score, m2, u2, cand)
                    if score == 0:
                        break  # perfetto, basta così

            # Ricarica il migliore
            if best[3] is not sub:
                target_module.load_state_dict(best[3], strict=False)

        # Ritorna i missing/unexpected del best
        return best[1], best[2]


    def load_head_from_probe(self, ssl_head_ckpt: str) -> Tuple[int, int]:
        hd = torch.load(ssl_head_ckpt, map_location="cpu", weights_only=True)
        if isinstance(hd, dict) and "state_dict" in hd:
            hd = hd["state_dict"]
        missing, unexpected = self.head.load_state_dict(hd, strict=False)
        if (missing or unexpected) and os.environ.get("EVAL_DEBUG", "0") == "1":
            print(f"[DEBUG] head load: missing={missing} unexpected={unexpected}")
        return len(missing), len(unexpected)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone.forward_global(x)
        return self.head(feats)
>>

tools/auto_eval.py codice <<
# file: /home/mla_group_01/rcc-ssrl/src/evaluation/tools/auto_eval.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Auto-discover SSL runs from mlruns, build eval YAMLs, and (optionally) submit SLURM jobs.

- Accepts either the experiment folder (containing exp_*/ runs) or a single run folder.
- Fixed defaults for WebDataset test set (override via env RCC_WDS_TEST_DIR or CLI).
- Prefers class_order and backbone from experiment_snapshot.yaml.
- Outputs eval results inside the *same run folder*: <run_dir>/eval.
- Writes generated eval configs to /home/mla_group_01/rcc-ssrl/src/evaluation/auto_configs.
"""

import os
import json
import yaml
import argparse
import subprocess
from pathlib import Path
from typing import Optional, Tuple

# ---------- Paths anchored to this file (cwd-agnostic) ----------
SCRIPT_DIR = Path(__file__).resolve().parent              # .../scripts/05_evaluation/tools
EVAL_DIR   = SCRIPT_DIR.parent                            # .../scripts/05_evaluation
CFG_OUTPUT_DIR = (EVAL_DIR / "auto_configs").resolve()    # where .yaml will be written
SBATCH_DEFAULT = (EVAL_DIR / "eval_models.sbatch").resolve()
# print("script dir:", SCRIPT_DIR)
# print("eval dir:", EVAL_DIR)
# print("cfg output dir:", CFG_OUTPUT_DIR)
# print("sbatch default:", SBATCH_DEFAULT)


# ---------- Defaults (overridable by env) ----------
DEFAULT_WDS_TEST_DIR = os.environ.get(
    "RCC_WDS_TEST_DIR",
    "/beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/data/processed/rcc_webdataset_final/test",
)
DEFAULT_WDS_PATTERN = "shard-*.tar"
DEFAULT_IMAGE_KEY = "img.jpg;jpg;jpeg;png"
DEFAULT_META_KEY = "meta.json;json"

# Optional dataset registry keyed by dataset_key in snapshot
DATASET_REGISTRY = {
    "rcc_final_ablation": {
        "test_dir": DEFAULT_WDS_TEST_DIR,
        "pattern": DEFAULT_WDS_PATTERN,
        "image_key": DEFAULT_IMAGE_KEY,
        "meta_key": DEFAULT_META_KEY,
    },
    "rcc_v2": {
        "test_dir": DEFAULT_WDS_TEST_DIR,
        "pattern": DEFAULT_WDS_PATTERN,
        "image_key": DEFAULT_IMAGE_KEY,
        "meta_key": DEFAULT_META_KEY,
    },
}

# ---------- I/O ----------
def read_json(p: Path):
    return json.load(open(p)) if p.is_file() else {}

def read_yaml(p: Path):
    import yaml as _y
    return _y.safe_load(open(p)) if p.is_file() else {}

# ---------- Helpers ----------
def guess_model_name(run_dir: Path) -> str:
    """Infer model short name from checkpoint file or run folder name."""
    for ck in (run_dir / "checkpoints").glob("*__ssl_best.pt"):
        return ck.name.split("__", 1)[0]
    parts = run_dir.name.split("_")
    return parts[1] if len(parts) >= 2 else "ssl_model"

def _first_ckpt_match(run_dir: Path, pat: str) -> Path:
    """Resolve checkpoint path; support wildcards and fallback to any in checkpoints/."""
    cand = list(run_dir.glob(pat))
    if not cand and run_dir.joinpath(pat).exists():
        cand = [run_dir.joinpath(pat)]
    if not cand:
        cand = list(run_dir.glob("checkpoints/*"))
    if not cand:
        raise FileNotFoundError(f"Cannot resolve: {pat} in {run_dir}")
    return cand[0]

def detect_backbone_name(snapshot: Path, ssl_ckpt: Path) -> str:
    """
    Decide the backbone name for eval:
    1) Infer from checkpoint keys (source of truth).
    2) If snapshot agrees, keep snapshot's name; otherwise force a safe default of that family.
    """
    snap_name: Optional[str] = None
    try:
        snap = read_yaml(snapshot) or {}
        snap_name = str(snap.get("model", {}).get("backbone", {}).get("name", "")).strip() or None
    except Exception:
        snap_name = None

    want_vit: Optional[bool] = None
    try:
        import torch
        sd = torch.load(ssl_ckpt, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
            sd = sd["state_dict"]
        ks = list(sd.keys()) if isinstance(sd, dict) else []
        want_vit = any(k.startswith(("pos_embed", "blocks.", "patch_embed.")) for k in ks)
    except Exception:
        want_vit = None

    if want_vit is True:
        # ViT family
        return snap_name if (snap_name and snap_name.startswith("vit")) else "vit_small_patch16_224"
    if want_vit is False:
        # ResNet family
        return snap_name if (snap_name and not snap_name.startswith("vit")) else "resnet50"
    # Unknown → fallback: prefer snapshot, else resnet50
    return snap_name or "resnet50"

def _torch_load_weights(path: str):
    import torch
    try:
        return torch.load(path, map_location="cpu", weights_only=True)  # torch>=2.4
    except TypeError:
        return torch.load(path, map_location="cpu")

def _read_head_in_features(head_ckpt: Path) -> Optional[int]:
    """Return in_features of linear head from its checkpoint, if detectable."""
    import torch
    hd = _torch_load_weights(str(head_ckpt))
    if isinstance(hd, dict) and "state_dict" in hd and isinstance(hd["state_dict"], dict):
        hd = hd["state_dict"]
    if not isinstance(hd, dict):
        return None
    # pick any 2D '...weight' tensor as linear weight
    for k, v in hd.items():
        if k.endswith("weight") and isinstance(v, torch.Tensor) and v.ndim == 2:
            return int(v.shape[1])
    return None

def _map_in_features_to_backbone(in_features: int) -> Optional[str]:
    """Common mappings from probe dim to canonical backbones."""
    # ViT family
    if in_features == 384:  return "vit_small_patch16_224"
    if in_features == 768:  return "vit_base_patch16_224"
    if in_features == 1024: return "vit_large_patch16_224"
    # ResNet family (most common)
    if in_features == 2048: return "resnet50"
    if in_features == 512:  return "resnet34"
    # Unknown → None
    return None

def _resolve_wds_params(run_dir: Path, args):
    """Choose test set params: registry → CLI overrides → env/defaults."""
    test_dir = DEFAULT_WDS_TEST_DIR
    pattern = DEFAULT_WDS_PATTERN
    image_key = DEFAULT_IMAGE_KEY
    meta_key  = DEFAULT_META_KEY

    snap = read_yaml(run_dir / "configuration" / "experiment_snapshot.yaml") or {}
    dskey = None
    try:
        dskey = snap["data"]["webdataset"]["dataset_key"]
    except Exception:
        pass

    if dskey and dskey in DATASET_REGISTRY:
        reg = DATASET_REGISTRY[dskey]
        test_dir = reg.get("test_dir", test_dir)
        pattern  = reg.get("pattern", pattern)
        image_key = reg.get("image_key", image_key)
        meta_key  = reg.get("meta_key", meta_key)

    # CLI overrides (if provided)
    if args.wds_test_dir: test_dir = args.wds_test_dir
    if args.wds_pattern:  pattern  = args.wds_pattern
    if args.image_key:    image_key = args.image_key
    if args.meta_key:     meta_key  = args.meta_key
    return test_dir, pattern, image_key, meta_key

def build_eval_cfg(run_dir: Path, test_root: str, pattern: str, image_key: str, meta_key: str, labels: list) -> Path:
    """Compose a self-contained eval YAML for eval.py, saved under CFG_OUTPUT_DIR."""
    fm = read_json(run_dir / "metrics" / "final_metrics.json")
    snap_path = run_dir / "configuration" / "experiment_snapshot.yaml"

    ssl_backbone_rel = fm.get("ssl_backbone_path", "checkpoints/*__ssl_best.pt")
    ssl_head_rel     = fm.get("ssl_linear_ckpt_path", "checkpoints/*__ssl_linear_best.pt")

    ssl_backbone = _first_ckpt_match(run_dir, ssl_backbone_rel)
    ssl_head     = _first_ckpt_match(run_dir, ssl_head_rel)

    model_name     = f"{guess_model_name(run_dir)}_ssl_linear_best"
    # 1) Try to decide from the head dim (source of truth for the probe)
    head_in = _read_head_in_features(ssl_head)
    backbone_name = _map_in_features_to_backbone(head_in) if head_in is not None else None
    # 2) If still unknown, fall back to ckpt/snapshot heuristic
    if backbone_name is None:
        backbone_name = detect_backbone_name(snap_path, ssl_backbone)

    out_root       = (run_dir / "eval").resolve()

    labels_from_snapshot = None
    snap = read_yaml(snap_path)
    try:
        c2i = snap["data"]["webdataset"]["class_to_id"]
        labels_from_snapshot = [k for k, _ in sorted(c2i.items(), key=lambda kv: kv[1])]
    except Exception:
        pass
    labels = labels_from_snapshot or labels

    cfg = {
        "experiment": {"name": f"eval_{run_dir.name}", "seed": 1337, "outputs_root": str(out_root)},
        "data": {
            "backend": "webdataset", "img_size": 224, "imagenet_norm": False,
            "num_workers": 8, "batch_size": 256,
            "webdataset": {
                "test_dir": test_root, "pattern": pattern, "image_key": image_key, "meta_key": meta_key
            }
        },
        "labels": {"class_order": labels},
        "model": {
            "name": model_name, "arch_hint": "ssl_linear",
            "backbone_name": backbone_name,
            "ssl_backbone_ckpt": str(ssl_backbone.resolve()),
            "ssl_head_ckpt":     str(ssl_head.resolve()),
            "strict_load": False,
            "allow_arch_autoswap": False
        },
        "evaluation": {
            "save_logits": True, "save_embeddings": True, "save_preds_csv": True,
            "umap": {"enabled": True, "source": "features", "n_neighbors": 15, "min_dist": 0.1, "random_state": 1337}
        },
        "runtime": {"device": "cuda", "precision": "fp32"}
    }

    CFG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg_path = CFG_OUTPUT_DIR / f"{run_dir.parent.name}__{run_dir.name}.yaml"
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return cfg_path

def _discover_runs(root: Path):
    """Return [root] if root is itself a run (has metrics/final_metrics.json), else scan children exp_*."""
    if (root / "metrics" / "final_metrics.json").is_file():
        return [root]
    return sorted([p for p in root.glob("exp_*") if (p / "metrics" / "final_metrics.json").is_file()])

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mlruns-root", required=True, help="Path to experiment folder or single run folder")
    ap.add_argument("--submit", action="store_true", help="Submit SLURM jobs after generating YAMLs")
    ap.add_argument("--only-one-shard", action="store_true", help="Smoke eval on a single shard (env flag)")
    ap.add_argument("--sbatch-path", default=None, help="Absolute path to eval_models.sbatch (optional)")
    # Optional overrides (normally you can omit these)
    ap.add_argument("--wds-test-dir", default=None)
    ap.add_argument("--wds-pattern", default=None)
    ap.add_argument("--image-key", default=None)
    ap.add_argument("--meta-key", default=None)
    ap.add_argument("--labels", nargs="+", default=["ccRCC","pRCC","CHROMO","ONCO","NOT_TUMOR"])
    args = ap.parse_args()

    root = Path(args.mlruns_root)
    runs = _discover_runs(root)
    if not runs:
        raise SystemExit(f"No runs found under: {root}")

    out_cfgs = []
    for r in runs:
        test_dir, pattern, image_key, meta_key = _resolve_wds_params(r, args)
        cfgp = build_eval_cfg(
            r, test_root=test_dir, pattern=pattern,
            image_key=image_key, meta_key=meta_key, labels=args.labels
        )
        print(f"[OK] Config: {cfgp}")
        print(f"[INFO]  -> test_dir={test_dir}  pattern={pattern}  image_key={image_key}  meta_key={meta_key}")
        out_cfgs.append(cfgp)

    if args.submit:
        sb = Path(args.sbatch_path).resolve() if args.sbatch_path else SBATCH_DEFAULT
        if not sb.is_file():
            raise SystemExit(
                f"[ERROR] SBATCH file not found: {sb}\n"
                f"Pass --sbatch-path /home/mla_group_01/rcc-ssrl/src/evaluation/eval_models.sbatch if needed."
            )
        for cfg in out_cfgs:
            env = os.environ.copy()
            env["CFG_PATH"] = str(cfg)
            if args.only_one_shard:
                env["ONLY_ONE_SHARD"] = "1"
                env["SHARD_EXAMPLE"] = "shard-000000.tar"
            print(f"[SUBMIT] sbatch {sb}  CFG_PATH={cfg}")
            subprocess.run(["sbatch", str(sb)], check=True, env=env)

if __name__ == "__main__":
    main()
>>

tools/batch_patient_aggregation.py codice <<
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch patient-level aggregation for RCC subtype classification.

- Scans --mlruns-root (experiment with exp_* or a single run).
- For each run, finds the latest eval dir containing predictions.csv.
- Aggregates at patient level, ALWAYS excluding NOT_TUMOR and using ALL patches.
- Output is written under: <run>/eval/<model>/<timestamp>/per_patient/

Aggregation:
- 'prob_sum' (default): sum per-class softmax across patches; zero-out NOT_TUMOR before argmax.
- 'vote'            : majority vote over patch predictions ignoring NOT_TUMOR.

Per run outputs (inside per_patient/):
- patient_predictions.csv
- metrics_patient.json
- cm_patient_<model>.png
- info_patient.json (counts: total patients, tumor-evaluable, skipped non_tumor_only)
Also updates a global runs_summary_patient.csv at experiment root.
"""

from __future__ import annotations
import os, sys, json, argparse, glob, csv, time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, Counter

import numpy as np
import torch

from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, average_precision_score
)
from sklearn.preprocessing import label_binarize

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ------------------------- small utils -------------------------
def _read_json(p: Path) -> dict:
    try:
        return json.load(open(p))
    except Exception:
        return {}

def _softmax_logits(x: torch.Tensor) -> torch.Tensor:
    """Numerically-stable softmax over last dim."""
    x = x - x.max(dim=1, keepdim=True).values
    return torch.exp(x) / torch.exp(x).sum(dim=1, keepdim=True)

def _mode_excluding(items, exclude_value=None):
    """Most frequent item excluding a specific value; returns None if empty after exclusion."""
    vals = [v for v in items if v != exclude_value]
    if not vals:
        return None
    return Counter(vals).most_common(1)[0][0]

def _plot_confmat(cm: np.ndarray, labels: List[str], out_png: Path) -> None:
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Patient-level Confusion Matrix (tumor-only)")
    plt.colorbar()
    ticks = np.arange(len(labels))
    plt.xticks(ticks, labels, rotation=45, ha="right")
    plt.yticks(ticks, labels)
    thresh = cm.max() / 2.0 if cm.size else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = int(cm[i, j])
            plt.text(j, i, str(val), ha="center", va="center",
                     color="white" if val > thresh else "black")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()


# ------------------------- core aggregation -------------------------
def aggregate_patients(
    rows: List[Dict],                 # parsed predictions.csv
    logits_np: Optional[np.ndarray],  # (N, C) or None
    class_names: List[str],
    *,
    method: str = "prob_sum",         # "prob_sum" (default) or "vote"
) -> Tuple[List[Dict], Dict]:
    """
    Always exclude NOT_TUMOR and use ALL patches.

    Patient GT rule:
      - If any tumor labels exist among patch GTs: patient GT = mode over tumor-only labels.
      - If NO tumor labels (all NOT_TUMOR): mark as non_tumor_only → y_true_patient = -1 (excluded from metrics).

    Patient prediction:
      - prob_sum: sum softmax across patches; set NOT_TUMOR score=0; argmax over tumor classes.
      - vote   : majority vote on patch predictions, ignoring NOT_TUMOR; fallback to overall mode if empty.
    """
    n_classes = len(class_names)
    if "NOT_TUMOR" not in class_names:
        raise ValueError("Class 'NOT_TUMOR' is required in class_names.")
    excl_id = class_names.index("NOT_TUMOR")

    # Group patch indices by patient
    by_pat: Dict[str, List[int]] = defaultdict(list)
    for idx, r in enumerate(rows):
        pid = r.get("patient_id")
        if pid is not None and pid != "":
            by_pat[pid].append(idx)

    # Precompute probabilities for prob_sum
    probs = None
    if logits_np is not None and method == "prob_sum":
        t = torch.from_numpy(logits_np)
        probs = _softmax_logits(t).numpy()

    patient_rows: List[Dict] = []
    y_true_pat: List[int] = []
    y_pred_pat: List[int] = []

    # tumor-only mapping for metrics/plots
    keep_idx = [i for i, n in enumerate(class_names) if n != "NOT_TUMOR"]
    keep_labels = [class_names[i] for i in keep_idx]
    idx_remap = {c: i for i, c in enumerate(keep_idx)}

    n_total_pat, n_tumor_eval, n_non_tumor_only = 0, 0, 0

    for pid, idxs in by_pat.items():
        n_total_pat += 1

        # --------- ground-truth (tumor-only mode) ----------
        y_true_items = [int(rows[i]["y_true"]) for i in idxs]
        gt_pat = _mode_excluding(y_true_items, exclude_value=excl_id)
        if gt_pat is None:
            gt_status = "non_tumor_only"
            n_non_tumor_only += 1
        else:
            gt_status = "tumor"
            n_tumor_eval += 1

        # --------- prediction ----------
        if method == "prob_sum" and probs is not None:
            score = np.zeros((n_classes,), dtype=np.float64)
            for i in idxs:  # ALL patches
                vec = probs[i].copy()
                vec[excl_id] = 0.0  # zero-out NOT_TUMOR
                score += vec
            score[excl_id] = -1.0  # make sure NOT_TUMOR cannot be chosen
            pred_pat = int(np.argmax(score))
            tumor_mass = score[score >= 0].sum() if score[score >= 0].size else 1.0
            confidence = float(score[pred_pat] / max(tumor_mass, 1e-9))
            support = {class_names[c]: float(score[c]) for c in range(n_classes) if c != excl_id}
        else:
            votes = [int(rows[i]["y_pred"]) for i in idxs if int(rows[i]["y_pred"]) != excl_id]
            if votes:
                pred_pat = Counter(votes).most_common(1)[0][0]
            else:
                # fall back to mode over all predictions (including NOT_TUMOR) or first
                pred_pat = _mode_excluding([int(rows[i]["y_pred"]) for i in idxs], exclude_value=None)
                if pred_pat is None:
                    pred_pat = int(rows[idxs[0]]["y_pred"])
            confidence = 1.0
            support = {}

        patient_rows.append({
            "patient_id": pid,
            "gt_status": gt_status,  # "tumor" | "non_tumor_only"
            "y_true_patient": (int(gt_pat) if gt_pat is not None else -1),
            "y_pred_patient": int(pred_pat),
            "n_patches": int(len(idxs)),
            "n_used_patches": int(len(idxs)),  # all patches
            "confidence": confidence,
            "support_sum_by_class": support,
        })

        if gt_pat is not None:
            y_true_pat.append(int(gt_pat))
            y_pred_pat.append(int(pred_pat))

    # ----- Metrics @ patient-level (tumor-only) -----
    metrics: Dict = {
        "n_patients_total": int(n_total_pat),
        "n_patients_tumor_eval": int(n_tumor_eval),
        "n_patients_non_tumor_only_skipped": int(n_non_tumor_only),
    }

    if y_true_pat:
        y_true_arr = np.asarray(y_true_pat, dtype=int)
        y_pred_arr = np.asarray(y_pred_pat, dtype=int)

        metrics.update({
            "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true_arr, y_pred_arr)),
            "macro_f1": float(f1_score(y_true_arr, y_pred_arr, average="macro")),
        })

        # Tumor-only confusion matrix + classification report
        yt = np.array([idx_remap[y] for y in y_true_arr if y in idx_remap], dtype=int)
        yp = np.array([idx_remap[y] for y in y_pred_arr if y in idx_remap], dtype=int)
        if yt.size and yp.size:
            cm = confusion_matrix(yt, yp, labels=list(range(len(keep_idx))))
            metrics["_cm"] = cm.tolist()
            metrics["_labels"] = keep_labels
            report = classification_report(yt, yp, target_names=keep_labels, output_dict=True, zero_division=0)
            metrics["_report"] = report

        # AUC/AUPRC and Top-2 (only for prob_sum with logits)
        if method == "prob_sum" and logits_np is not None:
            tumor_scores = []
            tumor_targets = []
            for r in patient_rows:
                gt = r["y_true_patient"]
                if gt < 0 or gt == class_names.index("NOT_TUMOR"):
                    continue
                vec = np.zeros((len(keep_idx),), dtype=np.float64)
                for k, v in r["support_sum_by_class"].items():
                    if k != "NOT_TUMOR":
                        j = keep_labels.index(k)
                        vec[j] = float(v)
                s = vec / max(vec.sum(), 1e-9)
                tumor_scores.append(s)
                if gt in keep_idx:
                    tumor_targets.append(keep_idx.index(gt))
            if tumor_scores and tumor_targets and (len(tumor_scores) == len(tumor_targets)):
                S = np.vstack(tumor_scores)
                yb = label_binarize(np.asarray(tumor_targets), classes=list(range(len(keep_idx))))
                try:
                    metrics["macro_auc_ovr"] = float(roc_auc_score(yb, S, average="macro", multi_class="ovr"))
                    metrics["macro_auprc"] = float(average_precision_score(yb, S, average="macro"))
                except Exception:
                    pass
                # top-2 accuracy
                top2_correct = sum(tgt in np.argsort(svec)[-2:] for svec, tgt in zip(S, tumor_targets))
                if len(tumor_targets) > 0:
                    metrics["top2_accuracy"] = float(top2_correct / len(tumor_targets))

    return patient_rows, metrics


# ------------------------- discovery & I/O -------------------------
def _is_run_dir(p: Path) -> bool:
    return p.is_dir() and (p / "metrics" / "final_metrics.json").is_file()

def _discover_runs(root: Path) -> List[Path]:
    if _is_run_dir(root):
        return [root]
    cands = [d for d in root.glob("exp_*") if _is_run_dir(d)]
    if not cands:
        cands = [d for d in root.rglob("exp_*") if _is_run_dir(d)]
    return sorted(cands)

def _find_latest_eval_dir(run_dir: Path) -> Optional[Path]:
    eval_root = run_dir / "eval"
    if not eval_root.is_dir():
        return None
    hits = sorted(eval_root.glob("*/*/predictions.csv"))
    if not hits:
        hits = sorted(eval_root.glob("*/predictions.csv"))
    if not hits:
        return None
    hits.sort(key=lambda p: p.parent.stat().st_mtime, reverse=True)
    return hits[0].parent  # .../eval/<model>/<timestamp>

def _load_eval_artifacts(eval_dir: Path) -> Tuple[List[Dict], Optional[np.ndarray], List[str], str]:
    pred_csv = eval_dir / "predictions.csv"
    logits_npy = eval_dir / "logits_test.npy"
    metrics_json = None
    for cand in eval_dir.glob("metrics_*.json"):
        metrics_json = cand
        break

    # parse predictions.csv (no pandas)
    rows: List[Dict] = []
    with open(pred_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                rows.append({
                    "patient_id": r.get("patient_id"),
                    "y_true": int(r.get("y_true", -1)),
                    "y_pred": int(r.get("y_pred", -1)),
                })
            except Exception:
                continue

    logits_np = None
    if logits_npy.is_file():
        try:
            logits_np = np.load(logits_npy)
        except Exception:
            logits_np = None

    class_names = ["ccRCC","pRCC","CHROMO","ONCO","NOT_TUMOR"]
    model_name = eval_dir.parent.name if eval_dir.parent else "ssl_linear_best"
    if metrics_json and metrics_json.is_file():
        meta = _read_json(metrics_json)
        cn = meta.get("class_names")
        if isinstance(cn, list) and all(isinstance(x, str) for x in cn):
            class_names = cn
        m = meta.get("model", {})
        if isinstance(m, dict):
            maybe = m.get("name")
            if isinstance(maybe, str) and maybe:
                model_name = maybe

    return rows, logits_np, class_names, model_name


# ------------------------- summary I/O -------------------------
def _append_run_summary(summary_csv: Path, run_dir: Path, model_name: str, metrics: Dict) -> None:
    header = [
        "timestamp","run_dir","model_name",
        "n_patients_total","n_patients_tumor_eval","n_patients_non_tumor_only_skipped",
        "accuracy","balanced_accuracy","macro_f1","macro_auc_ovr","macro_auprc","top2_accuracy"
    ]
    row = {
        "timestamp": int(time.time()),
        "run_dir": str(run_dir),
        "model_name": model_name,
        "n_patients_total": metrics.get("n_patients_total",""),
        "n_patients_tumor_eval": metrics.get("n_patients_tumor_eval",""),
        "n_patients_non_tumor_only_skipped": metrics.get("n_patients_non_tumor_only_skipped",""),
        "accuracy": metrics.get("accuracy",""),
        "balanced_accuracy": metrics.get("balanced_accuracy",""),
        "macro_f1": metrics.get("macro_f1",""),
        "macro_auc_ovr": metrics.get("macro_auc_ovr",""),
        "macro_auprc": metrics.get("macro_auprc",""),
        "top2_accuracy": metrics.get("top2_accuracy",""),
    }
    exists = summary_csv.is_file()
    with open(summary_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if not exists:
            w.writeheader()
        w.writerow(row)


# ------------------------- main -------------------------
def main():
    ap = argparse.ArgumentParser(description="Aggregate patch-level eval to patient-level across multiple runs.")
    ap.add_argument("--mlruns-root", required=True, help="Experiment folder (with exp_*) or a single run folder")
    ap.add_argument("--method", choices=["prob_sum","vote"], default="prob_sum",
                    help="Aggregation method. NOT_TUMOR always excluded; ALL patches used.")
    args = ap.parse_args()

    root = Path(args.mlruns_root).resolve()
    runs = _discover_runs(root)
    if not runs:
        raise SystemExit(f"No runs found under: {root}")

    # global summary at experiment root (or parent if single run)
    summary_csv = (root / "runs_summary_patient.csv") if not _is_run_dir(root) else (root.parent / "runs_summary_patient.csv")

    for run in runs:
        eval_dir = _find_latest_eval_dir(run)
        if eval_dir is None:
            print(f"[WARN] No eval with predictions.csv for run: {run}")
            continue

        # ---- load artifacts from the eval dir ----
        rows, logits_np, class_names, model_name = _load_eval_artifacts(eval_dir)

        # ---- aggregate ----
        patients, metrics = aggregate_patients(rows, logits_np, class_names, method=args.method)

        # ---- write into per_patient/ subfolder ----
        per_dir = eval_dir / "per_patient"
        per_dir.mkdir(parents=True, exist_ok=True)

        # 1) patient_predictions.csv (add gt_status)
        pp_csv = per_dir / "patient_predictions.csv"
        with open(pp_csv, "w", newline="") as f:
            fn = ["patient_id","gt_status","y_true_patient","y_pred_patient","n_patches","n_used_patches","confidence","support_sum_by_class"]
            w = csv.DictWriter(f, fieldnames=fn)
            w.writeheader()
            for r in patients:
                w.writerow(r)

        # 2) metrics_patient.json
        mp_json = per_dir / "metrics_patient.json"
        with open(mp_json, "w") as f:
            json.dump({
                "class_names": class_names,
                "method": args.method,
                "exclude": "NOT_TUMOR (fixed)",
                "metrics": metrics
            }, f, indent=2)

        # 3) confusion matrix plot (tumor-only)
        if metrics.get("_cm") and metrics.get("_labels"):
            cm = np.array(metrics["_cm"])
            labels = metrics["_labels"]
            _plot_confmat(cm, labels, per_dir / f"cm_patient_{model_name}.png")

        # 4) info file with counts
        info_json = per_dir / "info_patient.json"
        with open(info_json, "w") as f:
            json.dump({
                "run_dir": str(run),
                "eval_dir": str(eval_dir),
                "per_patient_dir": str(per_dir),
                "counts": {
                    "n_patients_total": metrics.get("n_patients_total", 0),
                    "n_patients_tumor_eval": metrics.get("n_patients_tumor_eval", 0),
                    "n_patients_non_tumor_only_skipped": metrics.get("n_patients_non_tumor_only_skipped", 0),
                }
            }, f, indent=2)

        # 5) append run summary (at experiment root)
        try:
            _append_run_summary(summary_csv, run, model_name, metrics)
        except Exception as e:
            print(f"[WARN] Could not append summary: {e}")

        print(f"[OK] {run.name}: patient aggregation → {per_dir}")

    print(f"[DONE] Updated summary: {summary_csv}")


if __name__ == "__main__":
    main()
>>

