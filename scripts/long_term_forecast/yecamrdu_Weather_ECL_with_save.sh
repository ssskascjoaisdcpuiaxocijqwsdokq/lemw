#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export PYTHONIOENCODING=utf-8
export LC_ALL=C.UTF-8
export LANG=C.UTF-8

model_name=yecamr610

# 创建结果保存目录
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_DIR="./results/yecamrdu_weather_ecl_experiments_${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"

# 创建日志文件
log_file="${RESULTS_DIR}/experiment_log.txt"

echo "========================================" | tee -a $log_file
echo "YECAMRDU Weather & ECL 实验开始" | tee -a $log_file
echo "开始时间: $(date)" | tee -a $log_file
echo "模型: $model_name" | tee -a $log_file
echo "结果保存目录: $RESULTS_DIR" | tee -a $log_file
echo "数据集: Weather, ECL" | tee -a $log_file
echo "预测长度: 96, 192, 336, 720" | tee -a $log_file
echo "========================================" | tee -a $log_file

# ==================== Weather 数据集实验 ====================
echo "" | tee -a $log_file
echo "开始Weather数据集实验..." | tee -a $log_file
echo "========================================" | tee -a $log_file

# Weather 96->96
echo "" | tee -a $log_file
echo "开始实验: Weather 96->96 预测..." | tee -a $log_file
echo "开始时间: $(date)" | tee -a $log_file

python -u run2.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data/weather/ \
  --data_path weather.csv \
  --model_id weather_96_96_yecamrdu \
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
  --d_model 512 \
  --d_ff 512 \
  --itr 1 \
  --lifting_levels 3 \
  --lifting_kernel_size 7 \
  --learning_rate 0.0005 \
  --batch_size 12 \
  --dropout 0.2 \
  --n_clusters 4 \
  --patience 2 \
  --train_epochs 100 \
  --eca_kernel_size 5 \
  --mrfp_ratio 7 \
  --duet_d_model 512 2>&1 | tee -a $log_file

echo "Weather 96->96 完成时间: $(date)" | tee -a $log_file

# Weather 96->192
echo "" | tee -a $log_file
echo "开始实验: Weather 96->192 预测..." | tee -a $log_file
echo "开始时间: $(date)" | tee -a $log_file

python -u run2.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data/weather/ \
  --data_path weather.csv \
  --model_id weather_96_192_yecamrdu \
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
  --d_model 512 \
  --d_ff 512 \
  --itr 1 \
  --lifting_levels 3 \
  --lifting_kernel_size 7 \
  --learning_rate 0.0005 \
  --batch_size 12 \
  --dropout 0.2 \
  --n_clusters 4 \
  --patience 2 \
  --train_epochs 100 \
  --eca_kernel_size 5 \
  --mrfp_ratio 7 \
  --duet_d_model 512 2>&1 | tee -a $log_file

echo "Weather 96->192 完成时间: $(date)" | tee -a $log_file

# Weather 96->336
echo "" | tee -a $log_file
echo "开始实验: Weather 96->336 预测..." | tee -a $log_file
echo "开始时间: $(date)" | tee -a $log_file

python -u run2.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data/weather/ \
  --data_path weather.csv \
  --model_id weather_96_336_yecamrdu \
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
  --d_model 512 \  
  --d_ff 512 \
  --itr 1 \
  --lifting_levels 3 \
  --lifting_kernel_size 7 \
  --n_clusters 4 \
  --learning_rate 0.0005 \
  --batch_size 12 \
  --dropout 0.2 \
  --patience 2 \
  --train_epochs 100 \
  --eca_kernel_size 5 \
  --mrfp_ratio 7 \
  --duet_d_model 512 2>&1 | tee -a $log_file

echo "Weather 96->336 完成时间: $(date)" | tee -a $log_file

# Weather 96->720
echo "" | tee -a $log_file
echo "开始实验: Weather 96->720 预测..." | tee -a $log_file
echo "开始时间: $(date)" | tee -a $log_file

python -u run2.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data/weather/ \
  --data_path weather.csv \
  --model_id weather_96_720_yecamrdu \
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
  --d_model 512 \
  --d_ff 512 \
  --itr 1 \
  --lifting_levels 3 \
  --lifting_kernel_size 7 \
  --n_clusters 4 \
  --learning_rate 0.0005 \
  --batch_size 12 \
  --dropout 0.2 \
  --patience 2 \
  --train_epochs 100 \
  --eca_kernel_size 5 \
  --mrfp_ratio 7 \
  --duet_d_model 512 2>&1 | tee -a $log_file

echo "Weather 96->720 完成时间: $(date)" | tee -a $log_file
echo "Weather数据集实验完成!" | tee -a $log_file

# ==================== ECL 数据集实验 ====================
echo "" | tee -a $log_file
echo "开始ECL数据集实验..." | tee -a $log_file
echo "========================================" | tee -a $log_file

# ECL 96->96
echo "" | tee -a $log_file
echo "开始实验: ECL 96->96 预测..." | tee -a $log_file
echo "开始时间: $(date)" | tee -a $log_file

python -u run2.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_96_96_yecamrdu \
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
  --lifting_levels 3 \
  --lifting_kernel_size 7 \
  --n_clusters 4 \
  --learning_rate 0.0005 \
  --batch_size 12 \
  --dropout 0.2 \
  --patience 2 \
  --train_epochs 100 \
  --eca_kernel_size 5 \
  --mrfp_ratio 7 \
  --duet_d_model 512 2>&1 | tee -a $log_file

echo "ECL 96->96 完成时间: $(date)" | tee -a $log_file

# ECL 96->192
echo "" | tee -a $log_file
echo "开始实验: ECL 96->192 预测..." | tee -a $log_file
echo "开始时间: $(date)" | tee -a $log_file

python -u run2.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_96_192_yecamrdu \
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
  --lifting_levels 3 \
  --lifting_kernel_size 7 \
  --n_clusters 4 \
  --learning_rate 0.0005 \
  --batch_size 12 \
  --dropout 0.2 \
  --patience 2 \
  --train_epochs 100 \
  --eca_kernel_size 5 \
  --mrfp_ratio 7 \
  --duet_d_model 512 2>&1 | tee -a $log_file

echo "ECL 96->192 完成时间: $(date)" | tee -a $log_file

# ECL 96->336
echo "" | tee -a $log_file
echo "开始实验: ECL 96->336 预测..." | tee -a $log_file
echo "开始时间: $(date)" | tee -a $log_file

python -u run2.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_96_336_yecamrdu \
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
  --lifting_levels 3 \
  --lifting_kernel_size 7 \
  --n_clusters 4 \
  --learning_rate 0.0005 \
  --batch_size 12 \
  --dropout 0.2 \
  --patience 2 \
  --train_epochs 100 \
  --eca_kernel_size 5 \
  --mrfp_ratio 7 \
  --duet_d_model 512 2>&1 | tee -a $log_file

echo "ECL 96->336 完成时间: $(date)" | tee -a $log_file

# ECL 96->720
echo "" | tee -a $log_file
echo "开始实验: ECL 96->720 预测..." | tee -a $log_file
echo "开始时间: $(date)" | tee -a $log_file

python -u run2.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_96_720_yecamrdu \
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
  --lifting_levels 3 \
  --lifting_kernel_size 7 \
  --n_clusters 4 \
  --learning_rate 0.0005 \
  --batch_size 12 \
  --dropout 0.2 \
  --patience 2 \
  --train_epochs 100 \
  --eca_kernel_size 5 \
  --mrfp_ratio 7 \
  --duet_d_model 512 2>&1 | tee -a $log_file

echo "ECL 96->720 完成时间: $(date)" | tee -a $log_file
echo "ECL数据集实验完成!" | tee -a $log_file

# 实验总结
echo "" | tee -a $log_file
echo "========================================" | tee -a $log_file
echo "所有YECAMRDU Weather & ECL实验完成!" | tee -a $log_file
echo "完成时间: $(date)" | tee -a $log_file
echo "结果保存在: $RESULTS_DIR" | tee -a $log_file
echo "========================================" | tee -a $log_file

# 复制重要结果文件到结果目录
echo "正在复制结果文件..." | tee -a $log_file

# Weather 结果
cp -r results/long_term_forecast_weather_96_96_${model_name}_custom_* $RESULTS_DIR/ 2>/dev/null || true
cp -r results/long_term_forecast_weather_96_192_${model_name}_custom_* $RESULTS_DIR/ 2>/dev/null || true
cp -r results/long_term_forecast_weather_96_336_${model_name}_custom_* $RESULTS_DIR/ 2>/dev/null || true
cp -r results/long_term_forecast_weather_96_720_${model_name}_custom_* $RESULTS_DIR/ 2>/dev/null || true

# ECL 结果
cp -r results/long_term_forecast_ECL_96_96_${model_name}_custom_* $RESULTS_DIR/ 2>/dev/null || true
cp -r results/long_term_forecast_ECL_96_192_${model_name}_custom_* $RESULTS_DIR/ 2>/dev/null || true
cp -r results/long_term_forecast_ECL_96_336_${model_name}_custom_* $RESULTS_DIR/ 2>/dev/null || true
cp -r results/long_term_forecast_ECL_96_720_${model_name}_custom_* $RESULTS_DIR/ 2>/dev/null || true

echo "实验完成! 所有结果已保存到: $RESULTS_DIR" | tee -a $log_file

# 创建实验总结
summary_file="${RESULTS_DIR}/experiment_summary.txt"
echo "YECAMRDU Weather & ECL 实验总结" > $summary_file
echo "=========================" >> $summary_file
echo "实验时间: $TIMESTAMP" >> $summary_file
echo "模型: $model_name" >> $summary_file
echo "数据集: Weather, ECL" >> $summary_file
echo "预测长度: 96, 192, 336, 720" >> $summary_file
echo "总实验数: 8个" >> $summary_file
echo "" >> $summary_file
echo "实验配置:" >> $summary_file
echo "- seq_len: 96" >> $summary_file
echo "- d_model: 1024" >> $summary_file
echo "- d_ff: 1024" >> $summary_file
echo "- batch_size: 12" >> $summary_file
echo "- learning_rate: 0.0005" >> $summary_file
echo "- lifting_levels: 3" >> $summary_file
echo "- dropout: 0.2" >> $summary_file
echo "- train_epochs: 100" >> $summary_file
echo "- patience: 2" >> $summary_file
echo "- eca_kernel_size: 5" >> $summary_file
echo "- mrfp_ratio: 7" >> $summary_file
echo "- duet_d_model: 512" >> $summary_file
echo "" >> $summary_file
echo "Weather数据集参数:" >> $summary_file
echo "- enc_in: 21" >> $summary_file
echo "- dec_in: 21" >> $summary_file
echo "- c_out: 21" >> $summary_file
echo "" >> $summary_file
echo "ECL数据集参数:" >> $summary_file
echo "- enc_in: 321" >> $summary_file
echo "- dec_in: 321" >> $summary_file
echo "- c_out: 321" >> $summary_file

echo "实验总结已保存到: $summary_file" | tee -a $log_file

