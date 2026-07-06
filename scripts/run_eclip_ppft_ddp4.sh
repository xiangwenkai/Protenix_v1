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
  --project protenix_sft \
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
  --eclip_ppft.confidence_rollout_steps 7 \
  --eclip_ppft.save_every_steps 500

# Current v2 distogram-contact training command.
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
PROTENIX_ROOT_DIR=/inspire/ssd/project/sais-bio/public/xiangwenkai/Protenix_v1 \
PYTHONPATH=/inspire/ssd/project/sais-bio/public/xiangwenkai/Protenix_v1 \
LAYERNORM_TYPE=torch \
torchrun --standalone --nproc_per_node=4 runner/train_eclip_ppft.py \
  --model_name protenix_base_20250630_v1.0.0 \
  --run_name eclip_ppft_protenix_v2 \
  --base_dir ./output \
  --project protenix_sft \
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
  --eval_interval 500 \
  --log_interval 10 \
  --checkpoint_interval -1 \
  --eclip_ppft.data_dir /inspire/ssd/project/sais-bio/public/xiangwenkai/GITHUB/parnet/data_process/high_quality_positive \
  --eclip_ppft.num_workers 16 \
  --eclip_ppft.max_protein_length 600 \
  --eclip_ppft.eval_max_steps 32 \
  --eclip_ppft.distogram_contact_threshold 8.0 \
  --eclip_ppft.signal_profile_weight 1.0 \
  --eclip_ppft.signal_multinomial_min_height 3.0 \
  --eclip_ppft.signal_binary_threshold 2.0 \
  --eclip_ppft.signal_clip_value 100.0 \
  --eclip_ppft.signal_multinomial_max_total 100.0 \
  --eclip_ppft.confidence_quality_weight 0.4 \
  --eclip_ppft.confidence_quality_target 0.8 \
  --eclip_ppft.confidence_rollout_steps 20 \
  --eclip_ppft.train_last_pairformer_blocks -1 \
  --eclip_ppft.save_every_steps 500

# PDB protein-RNA structure training with additional RNA binding signal loss.
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
PROTENIX_ROOT_DIR=/inspire/ssd/project/sais-bio/public/xiangwenkai/Protenix_v1 \
PYTHONPATH=/inspire/ssd/project/sais-bio/public/xiangwenkai/Protenix_v1 \
LAYERNORM_TYPE=torch \
torchrun --standalone --nproc_per_node=4 runner/train_rna_signal_sft.py \
  --model_name protenix_base_default_v1.0.0 \
  --run_name pdb_rna_signal_sft \
  --eval_first True \
  --iters_to_accumulate 1 \
  --seed 42 \
  --base_dir ./output \
  --dtype bf16 \
  --project protenix \
  --load_strict False \
  --use_wandb True \
  --diffusion_batch_size 32 \
  --eval_interval 1000 \
  --log_interval 50 \
  --checkpoint_interval 1000 \
  --ema_decay 0.999 \
  --train_crop_size 640 \
  --max_steps 100000 \
  --warmup_steps 1000 \
  --lr 0.0001 \
  --sample_diffusion.N_step 20 \
  --loss.weight.alpha_pae 1.0 \
  --loss.weight.alpha_diffusion 4.0 \
  --loss.weight.alpha_distogram 0.03 \
  --loss.weight.alpha_bond 1.0 \
  --loss.weight.smooth_lddt 1.0 \
  --triangle_attention cuequivariance \
  --triangle_multiplicative cuequivariance \
  --load_checkpoint_path /inspire/ssd/project/sais-bio/public/xiangwenkai/Protenix_v1/checkpoint/protenix_base_default_v1.0.0.pt \
  --load_ema_checkpoint_path /inspire/ssd/project/sais-bio/public/xiangwenkai/Protenix_v1/checkpoint/protenix_base_default_v1.0.0.pt \
  --data.train_sets train_rna_before202606 \
  --data.test_sets test_rna_before202606 \
  --data.train_rna_before202606.base_info.bioassembly_dict_dir /inspire/ssd/project/sais-bio/public/xiangwenkai/GITHUB/Protenix_v1/data/train_signal \
  --data.test_rna_before202606.base_info.bioassembly_dict_dir /inspire/ssd/project/sais-bio/public/xiangwenkai/GITHUB/Protenix_v1/data/train_signal

# Mixed PDB structure + eCLIP signal training.
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
PROTENIX_ROOT_DIR=/inspire/ssd/project/sais-bio/public/xiangwenkai/Protenix_v1 \
PYTHONPATH=/inspire/ssd/project/sais-bio/public/xiangwenkai/Protenix_v1 \
LAYERNORM_TYPE=torch \
torchrun --standalone --nproc_per_node=4 runner/train_rna_eclip_mixed_sft.py \
  --model_name protenix_base_default_v1.0.0 \
  --run_name pdb_eclip_mixed_rna_signal_sft \
  --eval_first False \
  --iters_to_accumulate 1 \
  --seed 42 \
  --base_dir ./output \
  --dtype bf16 \
  --project protenix_sft \
  --load_strict False \
  --load_params_only True \
  --use_wandb True \
  --diffusion_batch_size 32 \
  --data.num_dl_workers 2 \
  --eval_interval 1000 \
  --log_interval 50 \
  --checkpoint_interval 1000 \
  --train_crop_size 640 \
  --max_steps 100000 \
  --warmup_steps 1000 \
  --lr 0.0001 \
  --sample_diffusion.N_step 20 \
  --loss.weight.alpha_pae 1.0 \
  --loss.weight.alpha_diffusion 4.0 \
  --loss.weight.alpha_distogram 0.03 \
  --loss.weight.alpha_bond 1.0 \
  --loss.weight.smooth_lddt 1.0 \
  --triangle_attention cuequivariance \
  --triangle_multiplicative cuequivariance \
  --load_checkpoint_path /inspire/ssd/project/sais-bio/public/xiangwenkai/Protenix_v1/checkpoint/protenix_base_default_v1.0.0.pt \
  --data.train_sets train_rna_before202606 \
  --data.test_sets test_rna_before202606 \
  --data.train_rna_before202606.base_info.bioassembly_dict_dir /inspire/ssd/project/sais-bio/public/xiangwenkai/GITHUB/Protenix_v1/data/train_signal \
  --data.test_rna_before202606.base_info.bioassembly_dict_dir /inspire/ssd/project/sais-bio/public/xiangwenkai/GITHUB/Protenix_v1/data/train_signal \
  --eclip_ppft.data_dir /inspire/ssd/project/sais-bio/public/xiangwenkai/GITHUB/parnet/data_process/high_quality_positive \
  --eclip_ppft.num_workers 0 \
  --eclip_ppft.max_protein_length 600 \
  --eclip_ppft.eval_max_steps 32 \
  --eclip_ppft.signal_profile_weight 1.0 \
  --eclip_ppft.signal_multinomial_min_height 3.0 \
  --eclip_ppft.signal_binary_threshold 2.0 \
  --eclip_ppft.signal_clip_value 100.0 \
  --eclip_ppft.signal_multinomial_max_total 100.0 \
  --eclip_ppft.confidence_quality_weight 0.2 \
  --eclip_ppft.confidence_quality_target 0.8 \
  --eclip_ppft.confidence_rollout_steps 20 \
  --rna_signal_sft.signal_loss_weight 0.05 \
  --mixed_sft.pdb_sample_prob 0.2 \
  --mixed_sft.rollout_metric_contact_cutoff 5.0
