#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

model_name=AdaWaveNet

# 创建结果保存目录
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_DIR="./results/traffic_experiments_${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"

echo "开始Traffic数据集实验，结果将保存到: $RESULTS_DIR"

# 实验1: Traffic 96->96
echo "=== 运行实验 1: Traffic 96->96 ==="
EXP_DIR="$RESULTS_DIR/traffic_96_96"
mkdir -p "$EXP_DIR"

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data/traffic/ \
  --data_path traffic.csv \
  --model_id traffic_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 4 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --batch_size 32 \
  --learning_rate 0.001 \
  --itr 1 \
  --lifting_levels 1 \
  --lifting_kernel_size 7 \
  --n_cluster 9 \
  --train_epochs 20 > "$EXP_DIR/training_log.txt" 2>&1

# 保存结果
if [ -d "./results/long_term_forecast_traffic_96_96_AdaWaveNet_custom_ftM_sl96_ll48_pl96_dm512_nh8_el4_dl1_df512_fc3_ebtimeF_dtTrue_Exp_0/" ]; then
    cp -r "./results/long_term_forecast_traffic_96_96_AdaWaveNet_custom_ftM_sl96_ll48_pl96_dm512_nh8_el4_dl1_df512_fc3_ebtimeF_dtTrue_Exp_0/"* "$EXP_DIR/"
fi
echo "实验1完成，结果保存到: $EXP_DIR"

# 实验2: Traffic 96->192  
echo "=== 运行实验 2: Traffic 96->192 ==="
EXP_DIR="$RESULTS_DIR/traffic_96_192"
mkdir -p "$EXP_DIR"

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data/traffic/ \
  --data_path traffic.csv \
  --model_id traffic_96_192 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --e_layers 4 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --batch_size 32 \
  --learning_rate 0.001 \
  --itr 1 \
  --lifting_levels 1 \
  --lifting_kernel_size 7 \
  --n_cluster 9 \
  --train_epochs 20 > "$EXP_DIR/training_log.txt" 2>&1

# 保存结果
if [ -d "./results/long_term_forecast_traffic_96_192_AdaWaveNet_custom_ftM_sl96_ll48_pl192_dm512_nh8_el4_dl1_df512_fc3_ebtimeF_dtTrue_Exp_0/" ]; then
    cp -r "./results/long_term_forecast_traffic_96_192_AdaWaveNet_custom_ftM_sl96_ll48_pl192_dm512_nh8_el4_dl1_df512_fc3_ebtimeF_dtTrue_Exp_0/"* "$EXP_DIR/"
fi
echo "实验2完成，结果保存到: $EXP_DIR"

# 实验3: Traffic 96->336
echo "=== 运行实验 3: Traffic 96->336 ==="
EXP_DIR="$RESULTS_DIR/traffic_96_336"
mkdir -p "$EXP_DIR"

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data/traffic/ \
  --data_path traffic.csv \
  --model_id traffic_96_336 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 336 \
  --e_layers 4 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --batch_size 32 \
  --learning_rate 0.001 \
  --itr 1 \
  --lifting_levels 1 \
  --lifting_kernel_size 7 \
  --n_cluster 9 \
  --train_epochs 20 > "$EXP_DIR/training_log.txt" 2>&1

# 保存结果
if [ -d "./results/long_term_forecast_traffic_96_336_AdaWaveNet_custom_ftM_sl96_ll48_pl336_dm512_nh8_el4_dl1_df512_fc3_ebtimeF_dtTrue_Exp_0/" ]; then
    cp -r "./results/long_term_forecast_traffic_96_336_AdaWaveNet_custom_ftM_sl96_ll48_pl336_dm512_nh8_el4_dl1_df512_fc3_ebtimeF_dtTrue_Exp_0/"* "$EXP_DIR/"
fi
echo "实验3完成，结果保存到: $EXP_DIR"

# 实验4: Traffic 96->720
echo "=== 运行实验 4: Traffic 96->720 ==="
EXP_DIR="$RESULTS_DIR/traffic_96_720"
mkdir -p "$EXP_DIR"

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data/traffic/ \
  --data_path traffic.csv \
  --model_id traffic_96_720 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 720 \
  --e_layers 4 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --batch_size 32 \
  --learning_rate 0.001 \
  --itr 1 \
  --lifting_levels 1 \
  --lifting_kernel_size 7 \
  --n_cluster 9 \
  --train_epochs 20 > "$EXP_DIR/training_log.txt" 2>&1

# 保存结果
if [ -d "./results/long_term_forecast_traffic_96_720_AdaWaveNet_custom_ftM_sl96_ll48_pl720_dm512_nh8_el4_dl1_df512_fc3_ebtimeF_dtTrue_Exp_0/" ]; then
    cp -r "./results/long_term_forecast_traffic_96_720_AdaWaveNet_custom_ftM_sl96_ll48_pl720_dm512_nh8_el4_dl1_df512_fc3_ebtimeF_dtTrue_Exp_0/"* "$EXP_DIR/"
fi
echo "实验4完成，结果保存到: $EXP_DIR"

# 生成实验结果汇总文件
echo "正在生成实验结果汇总..."
SUMMARY_FILE="$RESULTS_DIR/experiment_summary.txt"

echo "=== 所有Traffic实验完成 ==="
echo "结果保存目录: $RESULTS_DIR"
echo "实验汇总文件: $SUMMARY_FILE"
echo "实验摘要:"
echo "  - traffic_96_96:  seq_len=96, pred_len=96"
echo "  - traffic_96_192: seq_len=96, pred_len=192"
echo "  - traffic_96_336: seq_len=96, pred_len=336"
echo "  - traffic_96_720: seq_len=96, pred_len=720"
echo ""
echo "所有实验参数保持与原始AdaWaveNet.sh一致，仅将seq_len统一改为96"
