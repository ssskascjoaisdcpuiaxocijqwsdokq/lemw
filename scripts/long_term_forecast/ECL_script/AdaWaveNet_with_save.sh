#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

model_name=AdaWaveNet

# 创建结果保存目录
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_DIR="./results/ecl_experiments_${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"

echo "开始ECL数据集实验，结果将保存到: $RESULTS_DIR"

# 实验1: ECL 96->96
echo "=== 运行实验 1: ECL 96->96 ==="
EXP_DIR="$RESULTS_DIR/ecl_96_96"
mkdir -p "$EXP_DIR"

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --d_model 256 \
  --d_ff 256 \
  --batch_size 16 \
  --learning_rate 0.0005 \
  --itr 1 \
  --lifting_levels 3 \
  --lifting_kernel_size 7 \
  --n_clusters 4 > "$EXP_DIR/training_log.txt" 2>&1

# 保存结果
if [ -d "./results/long_term_forecast_ECL_96_96_AdaWaveNet_custom_ftM_sl96_ll48_pl96_dm256_nh8_el2_dl1_df256_fc3_ebtimeF_dtTrue_Exp_0/" ]; then
    cp -r "./results/long_term_forecast_ECL_96_96_AdaWaveNet_custom_ftM_sl96_ll48_pl96_dm256_nh8_el2_dl1_df256_fc3_ebtimeF_dtTrue_Exp_0/"* "$EXP_DIR/"
fi
echo "实验1完成，结果保存到: $EXP_DIR"

# 实验2: ECL 96->192  
echo "=== 运行实验 2: ECL 96->192 ==="
EXP_DIR="$RESULTS_DIR/ecl_96_192"
mkdir -p "$EXP_DIR"

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_96_192 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --d_model 256 \
  --d_ff 256 \
  --batch_size 16 \
  --learning_rate 0.0005 \
  --itr 1 \
  --lifting_levels 3 \
  --lifting_kernel_size 7 \
  --n_clusters 4 > "$EXP_DIR/training_log.txt" 2>&1

# 保存结果
if [ -d "./results/long_term_forecast_ECL_96_192_AdaWaveNet_custom_ftM_sl96_ll48_pl192_dm256_nh8_el2_dl1_df256_fc3_ebtimeF_dtTrue_Exp_0/" ]; then
    cp -r "./results/long_term_forecast_ECL_96_192_AdaWaveNet_custom_ftM_sl96_ll48_pl192_dm256_nh8_el2_dl1_df256_fc3_ebtimeF_dtTrue_Exp_0/"* "$EXP_DIR/"
fi
echo "实验2完成，结果保存到: $EXP_DIR"

# 实验3: ECL 96->336
echo "=== 运行实验 3: ECL 96->336 ==="
EXP_DIR="$RESULTS_DIR/ecl_96_336"
mkdir -p "$EXP_DIR"

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_96_336 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 336 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --d_model 256 \
  --d_ff 256 \
  --batch_size 16 \
  --learning_rate 0.0005 \
  --itr 1 \
  --lifting_levels 3 \
  --lifting_kernel_size 7 \
  --n_clusters 4 > "$EXP_DIR/training_log.txt" 2>&1

# 保存结果
if [ -d "./results/long_term_forecast_ECL_96_336_AdaWaveNet_custom_ftM_sl96_ll48_pl336_dm256_nh8_el2_dl1_df256_fc3_ebtimeF_dtTrue_Exp_0/" ]; then
    cp -r "./results/long_term_forecast_ECL_96_336_AdaWaveNet_custom_ftM_sl96_ll48_pl336_dm256_nh8_el2_dl1_df256_fc3_ebtimeF_dtTrue_Exp_0/"* "$EXP_DIR/"
fi
echo "实验3完成，结果保存到: $EXP_DIR"

# 实验4: ECL 96->720
echo "=== 运行实验 4: ECL 96->720 ==="
EXP_DIR="$RESULTS_DIR/ecl_96_720"
mkdir -p "$EXP_DIR"

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_96_720 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 720 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --d_model 256 \
  --d_ff 256 \
  --batch_size 16 \
  --learning_rate 0.0005 \
  --itr 1 \
  --lifting_levels 3 \
  --lifting_kernel_size 7 \
  --n_clusters 4 > "$EXP_DIR/training_log.txt" 2>&1

# 保存结果
if [ -d "./results/long_term_forecast_ECL_96_720_AdaWaveNet_custom_ftM_sl96_ll48_pl720_dm256_nh8_el2_dl1_df256_fc3_ebtimeF_dtTrue_Exp_0/" ]; then
    cp -r "./results/long_term_forecast_ECL_96_720_AdaWaveNet_custom_ftM_sl96_ll48_pl720_dm256_nh8_el2_dl1_df256_fc3_ebtimeF_dtTrue_Exp_0/"* "$EXP_DIR/"
fi
echo "实验4完成，结果保存到: $EXP_DIR"

# 生成实验结果汇总文件
echo "正在生成实验结果汇总..."
python scripts/long_term_forecast/ECL_script/extract_metrics.py "$RESULTS_DIR"
SUMMARY_FILE="$RESULTS_DIR/experiment_summary.txt"

echo "=== 所有ECL实验完成 ==="
echo "结果保存目录: $RESULTS_DIR"
echo "实验汇总文件: $SUMMARY_FILE"
echo "实验摘要:"
echo "  - ecl_96_96:  seq_len=96, pred_len=96"
echo "  - ecl_96_192: seq_len=96, pred_len=192"
echo "  - ecl_96_336: seq_len=96, pred_len=336"
echo "  - ecl_96_720: seq_len=96, pred_len=720"
echo ""
echo "注意: ECL数据集seq_len统一设为96，与原脚本中的720不同"