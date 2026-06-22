#!/usr/bin/env bash

# Run ZTONGFEN on ETTh1 and ETTm1 with pred_len in {96, 192, 336, 720}
# Adjust hyperparameters below if needed.

set -e

DATASETS=("ETTh1" "ETTm1")
PRED_LENS=(96 192 336 720)

for DATASET in "${DATASETS[@]}"; do
  for PRED in "${PRED_LENS[@]}"; do
    MODEL_ID="${DATASET,,}_ztongfen_pl${PRED}"
    python run2.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --model_id "${MODEL_ID}" \
      --model ZTONGFEN \
      --data "${DATASET}" \
      --root_path ./data/ETT/ \
      --data_path "${DATASET}.csv" \
      --features M \
      --target OT \
      --freq h \
      --seq_len 96 \
      --label_len 48 \
      --pred_len "${PRED}" \
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --d_model 512 \
      --n_heads 8 \
      --e_layers 2 \
      --d_layers 1 \
      --d_ff 512 \
      --dropout 0.1 \
      --factor 1 \
      --moving_avg 25 \
      --lifting_levels 1 \
      --lifting_kernel_size 7 \
      --regu_details 0.01 \
      --regu_approx 0.01 \
      --batch_size 32 \
      --learning_rate 0.0005 \
      --train_epochs 20 \
      --patience 2 \
      --itr 1
  done
done
