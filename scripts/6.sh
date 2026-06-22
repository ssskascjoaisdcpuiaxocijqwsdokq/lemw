export CUDA_VISIBLE_DEVICES=0

model_name=yangchanel

# YangChanel ETTh1 96->96
python -u run2.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data/ETT/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_96 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 720 \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --itr 1 \
  --lifting_levels 4 \
  --lifting_kernel_size 7 \
  --n_clusters 4 \
  --learning_rate 0.0003 \
  --batch_size 16 \
  --mrfp_ratio 2.0

# YangChanel ETTh1 96->192
