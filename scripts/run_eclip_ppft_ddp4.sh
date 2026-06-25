#!/usr/bin/env bash
set -euo pipefail

cd /inspire/ssd/project/sais-bio/public/xiangwenkai/Protenix_v1

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
PROTENIX_ROOT_DIR=/inspire/ssd/project/sais-bio/public/xiangwenkai/Protenix_v1 \
PYTHONPATH=/inspire/ssd/project/sais-bio/public/xiangwenkai/Protenix_v1 \
LAYERNORM_TYPE=torch \
torchrun --standalone --nproc_per_node=4 runner/train_eclip_ppft.py \
  --model_name protenix_base_20250630_v1.0.0 \
  --run_name eclip_ppft_protenix \
  --base_dir ./output \
  --project protenix \
  --use_wandb True \
  --seed 42 \
  --dtype bf16 \
  --load_checkpoint_path /inspire/ssd/project/sais-bio/public/xiangwenkai/Protenix_v1/checkpoint/protenix_base_20250630_v1.0.0.pt \
  --load_strict True \
  --load_params_only True \
  --max_steps 40000 \
  --warmup_steps 500 \
  --lr 0.0001 \
  --lr_scheduler cosine_annealing \
  --grad_clip_norm 1 \
  --eval_interval 100 \
  --log_interval 10 \
  --checkpoint_interval -1 \
  --eclip_ppft.data_dir /inspire/ssd/project/sais-bio/public/xiangwenkai/GITHUB/parnet/data_process/high_quality_positive \
  --eclip_ppft.num_workers 16 \
  --eclip_ppft.max_protein_length 600 \
  --eclip_ppft.n_rollout_samples 1 \
  --eclip_ppft.n_rollout_steps 7 \
  --eclip_ppft.record_grad_steps 3,4,5 \
  --eclip_ppft.signal_profile_weight 1.0 \
  --eclip_ppft.signal_multinomial_min_height 3.0 \
  --eclip_ppft.signal_binary_threshold 2.0 \
  --eclip_ppft.signal_clip_value 100.0 \
  --eclip_ppft.signal_multinomial_max_total 100.0 \
  --eclip_ppft.save_every_steps 500 \
  --eclip_ppft.eval_every_steps 500
