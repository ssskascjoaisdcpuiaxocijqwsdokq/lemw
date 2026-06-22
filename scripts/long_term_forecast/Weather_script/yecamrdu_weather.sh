#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

run_experiment () {
  pred_len=$1
  python -u run2.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./data/weather/ \
    --data_path weather.csv \
    --model_id Weather_96_${pred_len}_yecamrdu \
    --model yecamrdu \
    --data custom \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len ${pred_len} \
    --enc_in 21 \
    --dec_in 21 \
    --c_out 21 \
    --e_layers 3 \
    --d_layers 1 \
    --factor 3 \
    --d_model 768 \
    --d_ff 1536 \
    --n_heads 8 \
    --lifting_levels 3 \
    --lifting_kernel_size 7 \
    --regu_details 0.001 \
    --regu_approx 0.001 \
    --learning_rate 0.0003 \
    --batch_size 32 \
    --dropout 0.2 \
    --patience 5 \
    --train_epochs 80 \
    --eca_kernel_size 5 \
    --mrfp_ratio 5 \
    --duet_d_model 384 \
    --moving_avg 19 \
    --sr_ratio 1 \
    --lradj type2 \
    --use_norm 1 \
    --use_gpu True \
    --gpu 0
}

run_experiment 96
run_experiment 192
run_experiment 336
run_experiment 720
