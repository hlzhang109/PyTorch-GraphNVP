#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=1 python train_mle.py --batch_size 256 \
    --data_file qm9_relgcn_kekulized_ggnp.npz \
    --epochs 10 \
    --device cuda \
    --num_node_masks 9 \
    --num_channel_masks 9 \
    --num_node_coupling 18 \
    --num_channel_coupling 14 \
    --additive_transformations True \
    --apply_batch_norm True \
    --node_mask_size 15 \
    --mode train \
    --num_gen 100 \
    --show_loss_step 5 \
    --gen_path ./results/qm9_mols \
    --model_save_dir ./saved_qm9_models \
    --img_dir=./results/qm9_img
