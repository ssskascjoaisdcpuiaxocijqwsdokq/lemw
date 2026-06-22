export CUDA_VISIBLE_DEVICES=0

model_name=AdaWaveNet

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data/weather/ \
  --data_path weather.csv \
  --model_id weather_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --d_model 512\
  --d_ff 512\
  --itr 1 \
  --lifting_levels 3\
  --lifting_kernel_size 7\
  --n_clusters 4\
  --learning_rate 0.0005\
  --batch_size 16


python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data/weather/ \
  --data_path weather.csv \
  --model_id weather_192_192 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --d_model 512\
  --d_ff 512\
  --itr 1 \
  --lifting_levels 3\
  --lifting_kernel_size 7\
  --n_clusters 4\
  --learning_rate 0.0005\
  --batch_size 16\
  --train_epochs 20\
  --patience 7\
  --dropout 0.1\
  --regu_details 0.01\
  --regu_approx 0.01\
  --num_workers 0


python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data/weather/ \
  --data_path weather.csv \
  --model_id weather_336_336 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 336 \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --d_model 512\
  --d_ff 512\
  --itr 1 \
  --lifting_levels 3\
  --lifting_kernel_size 7\
  --n_clusters 4\
  --learning_rate 0.0005\
  --batch_size 16\
  --train_epochs 20\
  --patience 7\
  --dropout 0.1\
  --regu_details 0.01\
  --regu_approx 0.01\
  --num_workers 0


python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data/weather/ \
  --data_path weather.csv \
  --model_id weather_720_720 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 720 \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --d_model 512\
  --d_ff 512\
  --itr 1 \
  --lifting_levels 3\
  --lifting_kernel_size 7\
  --n_clusters 4\
  --learning_rate 0.0005\
  --batch_size 16\
  --train_epochs 20\
  --patience 7\
  --dropout 0.1\
  --regu_details 0.01\
  --regu_approx 0.01\
  --num_workers 0