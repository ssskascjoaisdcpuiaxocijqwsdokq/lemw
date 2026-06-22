export CUDA_VISIBLE_DEVICES=0

# YECAMR610 ETTh1 长期预测实验脚本
# 融合ECA注意力和MRFP多感受野处理

# 预测96步 - 基础测试
python -u run2.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data/ETT/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_96_yecamr610 \
  --model yecamr610 \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 256 \
  --d_ff 256 \
  --lifting_levels 3 \
  --lifting_kernel_size 5 \
  --n_clusters 3 \
  --learning_rate 0.001 \
  --batch_size 32 \
  --dropout 0.1 \
  --patience 5 \
  --train_epochs 20 \
  --eca_kernel_size 3 \
  --mrfp_ratio 1.5 \
  --moving_avg 25

# 预测192步 - 中期预测
python -u run2.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data/ETT/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_192_192_yecamr610 \
  --model yecamr610 \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 384 \
  --d_ff 384 \
  --lifting_levels 3 \
  --lifting_kernel_size 5 \
  --n_clusters 3 \
  --learning_rate 0.0008 \
  --batch_size 24 \
  --dropout 0.12 \
  --patience 5 \
  --train_epochs 30 \
  --eca_kernel_size 3 \
  --mrfp_ratio 2.0 \
  --moving_avg 25

# 预测336步 - 长期预测
python -u run2.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data/ETT/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_336_336_yecamr610 \
  --model yecamr610 \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 336 \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 384 \
  --d_ff 384 \
  --lifting_levels 3 \
  --lifting_kernel_size 7 \
  --n_clusters 4 \
  --learning_rate 0.0006 \
  --batch_size 16 \
  --dropout 0.15 \
  --patience 7 \
  --train_epochs 50 \
  --eca_kernel_size 5 \
  --mrfp_ratio 2.5 \
  --moving_avg 25

# 预测720步 - 超长期预测
python -u run2.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data/ETT/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_720_720_yecamr610 \
  --model yecamr610 \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 720 \
  --e_layers 4 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 512 \
  --d_ff 512 \
  --lifting_levels 4 \
  --lifting_kernel_size 7 \
  --n_clusters 4 \
  --learning_rate 0.0005 \
  --batch_size 12 \
  --dropout 0.18 \
  --patience 10 \
  --train_epochs 100 \
  --eca_kernel_size 5 \
  --mrfp_ratio 3.0 \
  --moving_avg 25






