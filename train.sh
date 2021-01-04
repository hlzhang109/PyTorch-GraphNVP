#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=1 python train_mle.py \
    --batch_size 64 \
    -f qm9_relgcn_kekulized_ggnp.npz \
    --epochs 200 \
    --device cuda:0 \
    --num_node_masks 9 \
    --num_channel_masks 9 \
    --num_node_coupling 36 \
    --num_channel_coupling 27 \
    --apply_batch_norm True \
    --node_mask_size 15 \
    --save_dir=results/qm9 \
    --learn_dist yes \
    --mcmc_iters 30 \
    --mode train \
    --gen_out_path ./mols/test_100mol.txt \
    --additive_transformations True \
    --model_save_dir ./saved_qm9_models \
    --show_loss_step 1 \
    --model_name ebm \
    --n_critic 3 \
    --alpha 0.01 \
    --temp 1.0 \
    --min_atoms 3 \
    --num_gen 200 \
    --min_gen_epoch 1\
    --post_method soft_gumbel \
    --use_switch False \
    --pretrained epoch95-mle-G.ckpt \
    #epoch150-mle-G.ckpt \
    #--resume True \
    #--resume_epoch 8 \
    #--resume_batch 472\
