python /home/mla_group_01/rcc-ssrl/src/training/scripts/generate_ssl_ablation_configs.py --model i_jepa
 bash /home/mla_group_01/rcc-ssrl/src/training/scripts/launch_ssl_ablations.sh i_jepa


#reporting
EXP_ROOT="/beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251209_234747_moco_v3"

python /home/mla_group_01/rcc-ssrl/src/training/reporting/posthoc_diagnostics.py \
  --exp-root "$EXP_ROOT" \
  --model moco_v3
