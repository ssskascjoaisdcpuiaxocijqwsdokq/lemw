export CUDA_VISIBLE_DEVICES=0

# Use the ZDUWU model (YECAMRDU variant without DUET linear extractor)
model_name=zduwu

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id exchange_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --itr 1 \
  --kernel_size 7 \
  --geomattn_dropout 0.1 \
  --alpha 0.7 \
  --lifting_levels 4 \
  --lifting_kernel_size 7 \
  --learning_rate 0.0005 \
  --batch_size 32 \
  --adjust_lr True


python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id exchange_192_192 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --itr 1 \
  --kernel_size 7 \
  --geomattn_dropout 0.1 \
  --alpha 0.7 \
  --lifting_levels 5 \
  --lifting_kernel_size 7 \
  --learning_rate 0.0005 \
  --batch_size 16 \
  --adjust_lr True


python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id exchange_336_336 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 336 \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --itr 1 \
  --kernel_size 7 \
  --geomattn_dropout 0.1 \
  --alpha 0.7 \
  --lifting_levels 4 \
  --lifting_kernel_size 7 \
  --learning_rate 0.0005 \
  --batch_size 16 \
  --adjust_lr True


python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id exchange_720_720 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 720 \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --itr 1 \
  --kernel_size 7 \
  --geomattn_dropout 0.1 \
  --alpha 0.7 \
  --lifting_levels 1 \
  --lifting_kernel_size 7 \
  --learning_rate 0.0005 \
  --batch_size 32 \
  --adjust_lr True
