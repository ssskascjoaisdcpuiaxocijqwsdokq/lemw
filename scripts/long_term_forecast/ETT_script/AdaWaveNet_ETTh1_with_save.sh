#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

model_name=AdaWaveNet

# 创建结果保存目录
timestamp=$(date +"%Y%m%d_%H%M%S")
result_dir="experiment_results/ETTh1_${model_name}_${timestamp}"
mkdir -p $result_dir

# 创建日志文件
log_file="${result_dir}/experiment_log.txt"

echo "========================================" | tee -a $log_file
echo "AdaWaveNet ETTh1 实验开始" | tee -a $log_file
echo "开始时间: $(date)" | tee -a $log_file
echo "模型: $model_name" | tee -a $log_file
echo "结果保存目录: $result_dir" | tee -a $log_file
echo "========================================" | tee -a $log_file

# 实验1: ETTh1 96->96
echo "" | tee -a $log_file
echo "开始实验1: ETTh1 96->96 预测..." | tee -a $log_file
echo "开始时间: $(date)" | tee -a $log_file

python -u run.py \
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
  --pred_len 96 \
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
  --learning_rate 0.0005 \
  --batch_size 16 \
  2>&1 | tee -a $log_file

echo "实验1完成时间: $(date)" | tee -a $log_file

# 实验2: ETTh1 96->192
echo "" | tee -a $log_file
echo "开始实验2: ETTh1 96->192 预测..." | tee -a $log_file
echo "开始时间: $(date)" | tee -a $log_file

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data/ETT/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_192 \
  --model $model_name \
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
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --itr 1 \
  --lifting_levels 4 \
  --lifting_kernel_size 7 \
  --n_clusters 4 \
  --learning_rate 0.0005 \
  --batch_size 16 \
  2>&1 | tee -a $log_file

echo "实验2完成时间: $(date)" | tee -a $log_file

# 实验3: ETTh1 96->336
echo "" | tee -a $log_file
echo "开始实验3: ETTh1 96->336 预测..." | tee -a $log_file
echo "开始时间: $(date)" | tee -a $log_file

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data/ETT/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_336 \
  --model $model_name \
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
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --itr 1 \
  --lifting_levels 4 \
  --lifting_kernel_size 7 \
  --n_clusters 4 \
  --learning_rate 0.0005 \
  --batch_size 16 \
  2>&1 | tee -a $log_file

echo "实验3完成时间: $(date)" | tee -a $log_file

# 实验4: ETTh1 96->720
echo "" | tee -a $log_file
echo "开始实验4: ETTh1 96->720 预测..." | tee -a $log_file
echo "开始时间: $(date)" | tee -a $log_file

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data/ETT/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_720 \
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
  --learning_rate 0.0005 \
  --batch_size 16 \
  2>&1 | tee -a $log_file

echo "实验4完成时间: $(date)" | tee -a $log_file

echo "" | tee -a $log_file
echo "========================================" | tee -a $log_file
echo "所有ETTh1实验完成!" | tee -a $log_file
echo "完成时间: $(date)" | tee -a $log_file
echo "结果保存在: $result_dir" | tee -a $log_file
echo "========================================" | tee -a $log_file

# 复制重要结果文件到结果目录
echo "正在复制结果文件..." | tee -a $log_file
cp -r results/long_term_forecast_ETTh1_96_96_${model_name}_ETTh1_* $result_dir/ 2>/dev/null || true
cp -r results/long_term_forecast_ETTh1_96_192_${model_name}_ETTh1_* $result_dir/ 2>/dev/null || true
cp -r results/long_term_forecast_ETTh1_96_336_${model_name}_ETTh1_* $result_dir/ 2>/dev/null || true
cp -r results/long_term_forecast_ETTh1_96_720_${model_name}_ETTh1_* $result_dir/ 2>/dev/null || true

echo "实验完成! 所有结果已保存到: $result_dir" | tee -a $log_file
