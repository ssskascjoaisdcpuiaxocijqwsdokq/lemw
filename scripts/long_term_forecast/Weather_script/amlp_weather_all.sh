#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

model_name=amlp

for pred_len in 96 192 336 720
do
  python -u run2.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./data/weather/ \
    --data_path weather.csv \
    --model_id weather_96_${pred_len}_${model_name} \
    --model ${model_name} \
    --data custom \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len ${pred_len} \
    --e_layers 3 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 21 \
    --dec_in 21 \
    --c_out 21 \
    --des Exp \
    --d_model 512 \
    --d_ff 512 \
    --itr 1 \
    --lifting_levels 3 \
    --lifting_kernel_size 7 \
    --learning_rate 0.0005 \
    --batch_size 16 \
    --router_temperature 1.0
done
