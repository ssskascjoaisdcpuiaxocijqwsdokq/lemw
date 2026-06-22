export CUDA_VISIBLE_DEVICES=0

model_name=AdaWaveNet

# pred_len = 96
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data/electricity/ \
  --data_path electricity.csv \
  --model_id Electricity_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --itr 1 \
  --lifting_levels 4 \
  --lifting_kernel_size 7 \
  --n_cluster 4 \
  --learning_rate 0.0005 \
  --batch_size 32

# pred_len = 192
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data/electricity/ \
  --data_path electricity.csv \
  --model_id Electricity_96_192 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --itr 1 \
  --lifting_levels 4 \
  --lifting_kernel_size 7 \
  --n_cluster 4 \
  --learning_rate 0.0005 \
  --batch_size 16

# pred_len = 336
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data/electricity/ \
  --data_path electricity.csv \
  --model_id Electricity_96_336 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 336 \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --itr 1 \
  --lifting_levels 4 \
  --lifting_kernel_size 7 \
  --n_cluster 4 \
  --learning_rate 0.0005 \
  --batch_size 16

# pred_len = 720
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data/electricity/ \
  --data_path electricity.csv \
  --model_id Electricity_96_720 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 720 \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --itr 1 \
  --lifting_levels 4 \
  --lifting_kernel_size 7 \
  --n_cluster 4 \
  --learning_rate 0.0005 \
  --batch_size 16
