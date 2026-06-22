#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

model_name=yecamr610

# 创建结果保存目录
timestamp=$(date +"%Y%m%d_%H%M%S")
result_dir="experiment_results/ETT_All_${model_name}_${timestamp}"
mkdir -p $result_dir

# 创建日志文件
log_file="${result_dir}/experiment_log.txt"

echo "========================================" | tee -a $log_file
echo "YECA1 ETT全数据集实验开始" | tee -a $log_file
echo "开始时间: $(date)" | tee -a $log_file
echo "模型: $model_name" | tee -a $log_file
echo "结果保存目录: $result_dir" | tee -a $log_file
echo "数据集: ETTh1, ETTh2, ETTm1, ETTm2" | tee -a $log_file
echo "预测长度: 96, 192, 336, 720" | tee -a $log_file
echo "========================================" | tee -a $log_file

# ETTh1 数据集实验
echo "" | tee -a $log_file
echo "开始ETTh1数据集实验..." | tee -a $log_file
echo "========================================" | tee -a $log_file

# ETTh1 96->96
echo "" | tee -a $log_file
echo "开始实验: ETTh1 96->96 预测..." | tee -a $log_file
echo "开始时间: $(date)" | tee -a $log_file

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
  --n_cluster 4 \
  --learning_rate 0.0005 \
  --batch_size 16 \
  --eca_kernel_size 3 \
  --dropout 0.1 2>&1 | tee -a $log_file

echo "ETTh1 96->96 完成时间: $(date)" | tee -a $log_file

# ETTh1 96->192
echo "" | tee -a $log_file
echo "开始实验: ETTh1 96->192 预测..." | tee -a $log_file
echo "开始时间: $(date)" | tee -a $log_file

python -u run2.py \
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
  --n_cluster 4 \
  --learning_rate 0.0005 \
  --batch_size 16 \
  --eca_kernel_size 3 \
  --dropout 0.1 2>&1 | tee -a $log_file

echo "ETTh1 96->192 完成时间: $(date)" | tee -a $log_file

# ETTh1 96->336
echo "" | tee -a $log_file
echo "开始实验: ETTh1 96->336 预测..." | tee -a $log_file
echo "开始时间: $(date)" | tee -a $log_file

python -u run2.py \
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
  --n_cluster 4 \
  --learning_rate 0.0005 \
  --batch_size 16 \
  --eca_kernel_size 3 \
  --dropout 0.1 2>&1 | tee -a $log_file

echo "ETTh1 96->336 完成时间: $(date)" | tee -a $log_file

# ETTh1 96->720
echo "" | tee -a $log_file
echo "开始实验: ETTh1 96->720 预测..." | tee -a $log_file
echo "开始时间: $(date)" | tee -a $log_file

python -u run2.py \
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
  --n_cluster 4 \
  --learning_rate 0.0005 \
  --batch_size 16 \
  --eca_kernel_size 3 \
  --dropout 0.1 2>&1 | tee -a $log_file

echo "ETTh1 96->720 完成时间: $(date)" | tee -a $log_file
echo "ETTh1数据集实验完成!" | tee -a $log_file

# ETTh2 数据集实验
echo "" | tee -a $log_file
echo "开始ETTh2数据集实验..." | tee -a $log_file
echo "========================================" | tee -a $log_file

# ETTh2 96->96
echo "" | tee -a $log_file
echo "开始实验: ETTh2 96->96 预测..." | tee -a $log_file
echo "开始时间: $(date)" | tee -a $log_file

python -u run2.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data/ETT/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_96_96 \
  --model $model_name \
  --data ETTh2 \
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
  --n_cluster 4 \
  --learning_rate 0.0005 \
  --batch_size 16 \
  --eca_kernel_size 3 \
  --dropout 0.1 2>&1 | tee -a $log_file

echo "ETTh2 96->96 完成时间: $(date)" | tee -a $log_file

# ETTh2 96->192
echo "" | tee -a $log_file
echo "开始实验: ETTh2 96->192 预测..." | tee -a $log_file
echo "开始时间: $(date)" | tee -a $log_file

python -u run2.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data/ETT/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_96_192 \
  --model $model_name \
  --data ETTh2 \
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
  --n_cluster 4 \
  --learning_rate 0.0005 \
  --batch_size 16 \
  --eca_kernel_size 3 \
  --dropout 0.1 2>&1 | tee -a $log_file

echo "ETTh2 96->192 完成时间: $(date)" | tee -a $log_file

# ETTh2 96->336
echo "" | tee -a $log_file
echo "开始实验: ETTh2 96->336 预测..." | tee -a $log_file
echo "开始时间: $(date)" | tee -a $log_file

python -u run2.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data/ETT/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_96_336 \
  --model $model_name \
  --data ETTh2 \
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
  --n_cluster 4 \
  --learning_rate 0.0005 \
  --batch_size 16 \
  --eca_kernel_size 3 \
  --dropout 0.1 2>&1 | tee -a $log_file

echo "ETTh2 96->336 完成时间: $(date)" | tee -a $log_file

# ETTh2 96->720
echo "" | tee -a $log_file
echo "开始实验: ETTh2 96->720 预测..." | tee -a $log_file
echo "开始时间: $(date)" | tee -a $log_file

python -u run2.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data/ETT/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_96_720 \
  --model $model_name \
  --data ETTh2 \
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
  --n_cluster 4 \
  --learning_rate 0.0005 \
  --batch_size 16 \
  --eca_kernel_size 3 \
  --dropout 0.1 2>&1 | tee -a $log_file

echo "ETTh2 96->720 完成时间: $(date)" | tee -a $log_file
echo "ETTh2数据集实验完成!" | tee -a $log_file

# ETTm1 数据集实验
echo "" | tee -a $log_file
echo "开始ETTm1数据集实验..." | tee -a $log_file
echo "========================================" | tee -a $log_file

# ETTm1 96->96
echo "" | tee -a $log_file
echo "开始实验: ETTm1 96->96 预测..." | tee -a $log_file
echo "开始时间: $(date)" | tee -a $log_file

python -u run2.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data/ETT/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_96_96 \
  --model $model_name \
  --data ETTm1 \
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
  --n_cluster 2 \
  --learning_rate 0.0005 \
  --batch_size 16 2>&1 | tee -a $log_file

echo "ETTm1 96->96 完成时间: $(date)" | tee -a $log_file

# ETTm1 96->192
echo "" | tee -a $log_file
echo "开始实验: ETTm1 96->192 预测..." | tee -a $log_file
echo "开始时间: $(date)" | tee -a $log_file

python -u run2.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data/ETT/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_96_192 \
  --model $model_name \
  --data ETTm1 \
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
  --n_cluster 2 \
  --learning_rate 0.0005 \
  --batch_size 16 2>&1 | tee -a $log_file

echo "ETTm1 96->192 完成时间: $(date)" | tee -a $log_file

# ETTm1 96->336
echo "" | tee -a $log_file
echo "开始实验: ETTm1 96->336 预测..." | tee -a $log_file
echo "开始时间: $(date)" | tee -a $log_file

python -u run2.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data/ETT/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_96_336 \
  --model $model_name \
  --data ETTm1 \
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
  --n_cluster 2 \
  --learning_rate 0.0005 \
  --batch_size 16 2>&1 | tee -a $log_file

echo "ETTm1 96->336 完成时间: $(date)" | tee -a $log_file

# ETTm1 96->720
echo "" | tee -a $log_file
echo "开始实验: ETTm1 96->720 预测..." | tee -a $log_file
echo "开始时间: $(date)" | tee -a $log_file

python -u run2.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data/ETT/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_96_720 \
  --model $model_name \
  --data ETTm1 \
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
  --n_cluster 2 \
  --learning_rate 0.0005 \
  --batch_size 16 2>&1 | tee -a $log_file

echo "ETTm1 96->720 完成时间: $(date)" | tee -a $log_file
echo "ETTm1数据集实验完成!" | tee -a $log_file

# ETTm2 数据集实验
echo "" | tee -a $log_file
echo "开始ETTm2数据集实验..." | tee -a $log_file
echo "========================================" | tee -a $log_file

# ETTm2 96->96
echo "" | tee -a $log_file
echo "开始实验: ETTm2 96->96 预测..." | tee -a $log_file
echo "开始时间: $(date)" | tee -a $log_file

python -u run2.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data/ETT/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_96_96 \
  --model $model_name \
  --data ETTm2 \
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
  --n_cluster 2 \
  --learning_rate 0.0005 \
  --batch_size 16 2>&1 | tee -a $log_file

echo "ETTm2 96->96 完成时间: $(date)" | tee -a $log_file

# ETTm2 96->192
echo "" | tee -a $log_file
echo "开始实验: ETTm2 96->192 预测..." | tee -a $log_file
echo "开始时间: $(date)" | tee -a $log_file

python -u run2.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data/ETT/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_96_192 \
  --model $model_name \
  --data ETTm2 \
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
  --n_cluster 2 \
  --learning_rate 0.0005 \
  --batch_size 16 2>&1 | tee -a $log_file

echo "ETTm2 96->192 完成时间: $(date)" | tee -a $log_file

# ETTm2 96->336
echo "" | tee -a $log_file
echo "开始实验: ETTm2 96->336 预测..." | tee -a $log_file
echo "开始时间: $(date)" | tee -a $log_file

python -u run2.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data/ETT/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_96_336 \
  --model $model_name \
  --data ETTm2 \
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
  --n_cluster 2 \
  --learning_rate 0.0005 \
  --batch_size 16 2>&1 | tee -a $log_file

echo "ETTm2 96->336 完成时间: $(date)" | tee -a $log_file

# ETTm2 96->720
echo "" | tee -a $log_file
echo "开始实验: ETTm2 96->720 预测..." | tee -a $log_file
echo "开始时间: $(date)" | tee -a $log_file

python -u run2.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data/ETT/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_96_720 \
  --model $model_name \
  --data ETTm2 \
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
  --n_cluster 2 \
  --learning_rate 0.0005 \
  --batch_size 16 2>&1 | tee -a $log_file

echo "ETTm2 96->720 完成时间: $(date)" | tee -a $log_file
echo "ETTm2数据集实验完成!" | tee -a $log_file

# 实验总结
echo "" | tee -a $log_file
echo "========================================" | tee -a $log_file
echo "所有ETT数据集实验完成!" | tee -a $log_file
echo "完成时间: $(date)" | tee -a $log_file
echo "结果保存在: $result_dir" | tee -a $log_file
echo "========================================" | tee -a $log_file

# 复制重要结果文件到结果目录
echo "正在复制结果文件..." | tee -a $log_file

# ETTh1 结果
cp -r results/long_term_forecast_ETTh1_96_96_${model_name}_ETTh1_* $result_dir/ 2>/dev/null || true
cp -r results/long_term_forecast_ETTh1_96_192_${model_name}_ETTh1_* $result_dir/ 2>/dev/null || true
cp -r results/long_term_forecast_ETTh1_96_336_${model_name}_ETTh1_* $result_dir/ 2>/dev/null || true
cp -r results/long_term_forecast_ETTh1_96_720_${model_name}_ETTh1_* $result_dir/ 2>/dev/null || true

# ETTh2 结果
cp -r results/long_term_forecast_ETTh2_96_96_${model_name}_ETTh2_* $result_dir/ 2>/dev/null || true
cp -r results/long_term_forecast_ETTh2_96_192_${model_name}_ETTh2_* $result_dir/ 2>/dev/null || true
cp -r results/long_term_forecast_ETTh2_96_336_${model_name}_ETTh2_* $result_dir/ 2>/dev/null || true
cp -r results/long_term_forecast_ETTh2_96_720_${model_name}_ETTh2_* $result_dir/ 2>/dev/null || true

# ETTm1 结果
cp -r results/long_term_forecast_ETTm1_96_96_${model_name}_ETTm1_* $result_dir/ 2>/dev/null || true
cp -r results/long_term_forecast_ETTm1_96_192_${model_name}_ETTm1_* $result_dir/ 2>/dev/null || true
cp -r results/long_term_forecast_ETTm1_96_336_${model_name}_ETTm1_* $result_dir/ 2>/dev/null || true
cp -r results/long_term_forecast_ETTm1_96_720_${model_name}_ETTm1_* $result_dir/ 2>/dev/null || true

# ETTm2 结果
cp -r results/long_term_forecast_ETTm2_96_96_${model_name}_ETTm2_* $result_dir/ 2>/dev/null || true
cp -r results/long_term_forecast_ETTm2_96_192_${model_name}_ETTm2_* $result_dir/ 2>/dev/null || true
cp -r results/long_term_forecast_ETTm2_96_336_${model_name}_ETTm2_* $result_dir/ 2>/dev/null || true
cp -r results/long_term_forecast_ETTm2_96_720_${model_name}_ETTm2_* $result_dir/ 2>/dev/null || true

echo "实验完成! 所有结果已保存到: $result_dir" | tee -a $log_file

# 创建实验总结
summary_file="${result_dir}/experiment_summary.txt"
echo "YECA1 ETT全数据集实验总结" > $summary_file
echo "=========================" >> $summary_file
echo "实验时间: $timestamp" >> $summary_file
echo "模型: $model_name" >> $summary_file
echo "数据集: ETTh1, ETTh2, ETTm1, ETTm2" >> $summary_file
echo "预测长度: 96, 192, 336, 720" >> $summary_file
echo "总实验数: 16个" >> $summary_file
echo "" >> $summary_file
echo "实验配置:" >> $summary_file
echo "- seq_len: 96" >> $summary_file
echo "- d_model: 512" >> $summary_file
echo "- batch_size: 16" >> $summary_file
echo "- learning_rate: 0.0005" >> $summary_file
echo "- lifting_levels: 4" >> $summary_file
echo "- n_clusters: 4" >> $summary_file
echo "- eca_kernel_size: 3" >> $summary_file
echo "" >> $summary_file

echo "实验总结已保存到: $summary_file" | tee -a $log_file
