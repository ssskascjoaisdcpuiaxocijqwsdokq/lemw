#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

model_name=yangchanel

# 创建结果保存目录
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_DIR="./results/yangchanel_weather_experiments_${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"

echo "开始YangChanel Weather数据集实验，结果将保存到: $RESULTS_DIR"

# 实验1: Weather 96->96
echo "=== 运行实验 1: YangChanel Weather 96->96 ==="
EXP_DIR="$RESULTS_DIR/weather_96_96"
mkdir -p "$EXP_DIR"

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
  --d_model 512 \
  --d_ff 512 \
  --itr 1 \
  --lifting_levels 3 \
  --lifting_kernel_size 7 \
  --n_clusters 4 \
  --learning_rate 0.0005 \
  --batch_size 16 \
  --cbam_reduction 16 \
  --cbam_kernel_size 7 \
  --mrfp_ratio 2.0 > "$EXP_DIR/training_log.txt" 2>&1

# 保存结果
if [ -d "./results/long_term_forecast_weather_96_96_yangchanel_custom_ftM_sl96_ll48_pl96_dm512_nh8_el3_dl1_df512_fc3_ebtimeF_dtTrue_Exp_0/" ]; then
    cp -r "./results/long_term_forecast_weather_96_96_yangchanel_custom_ftM_sl96_ll48_pl96_dm512_nh8_el3_dl1_df512_fc3_ebtimeF_dtTrue_Exp_0/"* "$EXP_DIR/"
fi
echo "实验1完成，结果保存到: $EXP_DIR"

# 实验2: Weather 96->192  
echo "=== 运行实验 2: YangChanel Weather 96->192 ==="
EXP_DIR="$RESULTS_DIR/weather_96_192"
mkdir -p "$EXP_DIR"

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data/weather/ \
  --data_path weather.csv \
  --model_id weather_96_192 \
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
  --n_clusters 4 \
  --learning_rate 0.0005 \
  --batch_size 16 \
  --cbam_reduction 16 \
  --cbam_kernel_size 7 \
  --mrfp_ratio 2.0 > "$EXP_DIR/training_log.txt" 2>&1

# 保存结果
if [ -d "./results/long_term_forecast_weather_96_192_yangchanel_custom_ftM_sl96_ll48_pl192_dm512_nh8_el3_dl1_df512_fc3_ebtimeF_dtTrue_Exp_0/" ]; then
    cp -r "./results/long_term_forecast_weather_96_192_yangchanel_custom_ftM_sl96_ll48_pl192_dm512_nh8_el3_dl1_df512_fc3_ebtimeF_dtTrue_Exp_0/"* "$EXP_DIR/"
fi
echo "实验2完成，结果保存到: $EXP_DIR"

# 实验3: Weather 96->336
echo "=== 运行实验 3: YangChanel Weather 96->336 ==="
EXP_DIR="$RESULTS_DIR/weather_96_336"
mkdir -p "$EXP_DIR"

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data/weather/ \
  --data_path weather.csv \
  --model_id weather_96_336 \
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
  --batch_size 16 \
  --cbam_reduction 16 \
  --cbam_kernel_size 7 \
  --mrfp_ratio 2.0 > "$EXP_DIR/training_log.txt" 2>&1

# 保存结果
if [ -d "./results/long_term_forecast_weather_96_336_yangchanel_custom_ftM_sl96_ll48_pl336_dm512_nh8_el3_dl1_df512_fc3_ebtimeF_dtTrue_Exp_0/" ]; then
    cp -r "./results/long_term_forecast_weather_96_336_yangchanel_custom_ftM_sl96_ll48_pl336_dm512_nh8_el3_dl1_df512_fc3_ebtimeF_dtTrue_Exp_0/"* "$EXP_DIR/"
fi
echo "实验3完成，结果保存到: $EXP_DIR"

# 实验4: Weather 96->720
echo "=== 运行实验 4: YangChanel Weather 96->720 ==="
EXP_DIR="$RESULTS_DIR/weather_96_720"
mkdir -p "$EXP_DIR"

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data/weather/ \
  --data_path weather.csv \
  --model_id weather_96_720 \
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
  --batch_size 16 \
  --cbam_reduction 16 \
  --cbam_kernel_size 7 \
  --mrfp_ratio 2.0 > "$EXP_DIR/training_log.txt" 2>&1

# 保存结果
if [ -d "./results/long_term_forecast_weather_96_720_yangchanel_custom_ftM_sl96_ll48_pl720_dm512_nh8_el3_dl1_df512_fc3_ebtimeF_dtTrue_Exp_0/" ]; then
    cp -r "./results/long_term_forecast_weather_96_720_yangchanel_custom_ftM_sl96_ll48_pl720_dm512_nh8_el3_dl1_df512_fc3_ebtimeF_dtTrue_Exp_0/"* "$EXP_DIR/"
fi
echo "实验4完成，结果保存到: $EXP_DIR"

# 生成实验结果汇总文件
echo "正在生成实验结果汇总..."
SUMMARY_FILE="$RESULTS_DIR/experiment_summary.txt"

# 创建汇总文件
cat > "$SUMMARY_FILE" << EOF
YangChanel Weather Dataset Experiment Results Summary
===================================================
Timestamp: $(date '+%Y-%m-%d %H:%M:%S')
Model: YangChanel (AdaWaveNet + CBAM + MRFP)
Input Length (seq_len): 96
Features: 21 (weather variables)

Model Configuration:
- CBAM Reduction: 16
- CBAM Kernel Size: 7
- MRFP Ratio: 2.0
- Lifting Levels: 3
- N Clusters: 4
- Learning Rate: 0.0005
- Batch Size: 16

Experiment Details:
- weather_96_96:  seq_len=96, pred_len=96
- weather_96_192: seq_len=96, pred_len=192
- weather_96_336: seq_len=96, pred_len=336
- weather_96_720: seq_len=96, pred_len=720

Results saved in: $RESULTS_DIR

YangChanel Model Features:
- CBAM通道空间融合注意力机制
- MRFP多感受野特征处理
- AdaWaveNet小波变换架构
- K-means聚类线性预测
EOF

echo "=== 所有YangChanel Weather实验完成 ==="
echo "结果保存目录: $RESULTS_DIR"
echo "实验汇总文件: $SUMMARY_FILE"
echo "实验摘要:"
echo "  - weather_96_96:  seq_len=96, pred_len=96"
echo "  - weather_96_192: seq_len=96, pred_len=192"
echo "  - weather_96_336: seq_len=96, pred_len=336"
echo "  - weather_96_720: seq_len=96, pred_len=720"
echo ""
echo "YangChanel模型特点:"
echo "  - CBAM通道空间融合注意力"
echo "  - MRFP多感受野处理"
echo "  - AdaWaveNet小波变换架构"
echo "  - 21个天气特征变量"
