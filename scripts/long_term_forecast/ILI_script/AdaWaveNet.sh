export CUDA_VISIBLE_DEVICES=0

model_name=AdaWaveNet

# 实验1: ILI 36->24
echo "开始实验1: ILI 36->24 预测..."
python -u run.py \
   --task_name long_term_forecast \
   --is_training 1 \
   --root_path ./data/illness/ \
   --data_path national_illness.csv \
   --model_id ili_36_24 \
   --model $model_name \
   --data custom \
   --features M \
   --seq_len 36 \
   --label_len 18 \
   --pred_len 24 \
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
   --lifting_levels 2 \
   --lifting_kernel_size 7 \
   --n_clusters 1 \
   --learning_rate 0.0005 \
   --batch_size 32 \
   --adjust_lr True

# 实验2: ILI 36->36
echo "开始实验2: ILI 36->36 预测..."
python -u run.py \
   --task_name long_term_forecast \
   --is_training 1 \
   --root_path ./data/illness/ \
   --data_path national_illness.csv \
   --model_id ili_36_36 \
   --model $model_name \
   --data custom \
   --features M \
   --seq_len 36 \
   --label_len 18 \
   --pred_len 36 \
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
   --lifting_levels 2 \
   --lifting_kernel_size 7 \
   --n_clusters 1 \
   --learning_rate 0.0005 \
   --batch_size 32 \
   --adjust_lr True

# 实验3: ILI 36->48
echo "开始实验3: ILI 36->48 预测..."
python -u run.py \
   --task_name long_term_forecast \
   --is_training 1 \
   --root_path ./data/illness/ \
   --data_path national_illness.csv \
   --model_id ili_36_48 \
   --model $model_name \
   --data custom \
   --features M \
   --seq_len 36 \
   --label_len 18 \
   --pred_len 48 \
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
   --lifting_levels 2 \
   --lifting_kernel_size 7 \
   --n_clusters 1 \
   --learning_rate 0.0005 \
   --batch_size 32 \
   --adjust_lr True

# 实验4: ILI 36->60
echo "开始实验4: ILI 36->60 预测..."
python -u run.py \
   --task_name long_term_forecast \
   --is_training 1 \
   --root_path ./data/illness/ \
   --data_path national_illness.csv \
   --model_id ili_36_60 \
   --model $model_name \
   --data custom \
   --features M \
   --seq_len 36 \
   --label_len 18 \
   --pred_len 60 \
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
   --lifting_levels 2 \
   --lifting_kernel_size 7 \
   --n_clusters 1 \
   --learning_rate 0.0005 \
   --batch_size 32 \
   --adjust_lr True

echo "所有疾病数据集实验完成!"
