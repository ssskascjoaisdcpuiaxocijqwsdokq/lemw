#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

model_name=yeca1

# 创建结果保存目录
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_DIR="./results/yangchanel_weather_ecl_experiments_${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"

echo "开始YangChanel Weather & ECL数据集综合实验，结果将保存到: $RESULTS_DIR"

# ========================================
# Weather数据集实验
# ========================================
echo ""
echo "########## Weather数据集实验开始 ##########"

# Weather 实验1: 96->96
echo "=== 运行实验: YangChanel Weather 96->96 ==="
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

# 保存Weather 96->96结果
if [ -d "./results/long_term_forecast_weather_96_96_yangchanel_custom_ftM_sl96_ll48_pl96_dm512_nh8_el3_dl1_df512_fc3_ebtimeF_dtTrue_Exp_0/" ]; then
    cp -r "./results/long_term_forecast_weather_96_96_yangchanel_custom_ftM_sl96_ll48_pl96_dm512_nh8_el3_dl1_df512_fc3_ebtimeF_dtTrue_Exp_0/"* "$EXP_DIR/"
fi
echo "Weather 96->96实验完成，结果保存到: $EXP_DIR"

# Weather 实验2: 96->192
echo "=== 运行实验: YangChanel Weather 96->192 ==="
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

# 保存Weather 96->192结果
if [ -d "./results/long_term_forecast_weather_96_192_yangchanel_custom_ftM_sl96_ll48_pl192_dm512_nh8_el3_dl1_df512_fc3_ebtimeF_dtTrue_Exp_0/" ]; then
    cp -r "./results/long_term_forecast_weather_96_192_yangchanel_custom_ftM_sl96_ll48_pl192_dm512_nh8_el3_dl1_df512_fc3_ebtimeF_dtTrue_Exp_0/"* "$EXP_DIR/"
fi
echo "Weather 96->192实验完成，结果保存到: $EXP_DIR"

# Weather 实验3: 96->336
echo "=== 运行实验: YangChanel Weather 96->336 ==="
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

# 保存Weather 96->336结果
if [ -d "./results/long_term_forecast_weather_96_336_yangchanel_custom_ftM_sl96_ll48_pl336_dm512_nh8_el3_dl1_df512_fc3_ebtimeF_dtTrue_Exp_0/" ]; then
    cp -r "./results/long_term_forecast_weather_96_336_yangchanel_custom_ftM_sl96_ll48_pl336_dm512_nh8_el3_dl1_df512_fc3_ebtimeF_dtTrue_Exp_0/"* "$EXP_DIR/"
fi
echo "Weather 96->336实验完成，结果保存到: $EXP_DIR"

# Weather 实验4: 96->720
echo "=== 运行实验: YangChanel Weather 96->720 ==="
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

# 保存Weather 96->720结果
if [ -d "./results/long_term_forecast_weather_96_720_yangchanel_custom_ftM_sl96_ll48_pl720_dm512_nh8_el3_dl1_df512_fc3_ebtimeF_dtTrue_Exp_0/" ]; then
    cp -r "./results/long_term_forecast_weather_96_720_yangchanel_custom_ftM_sl96_ll48_pl720_dm512_nh8_el3_dl1_df512_fc3_ebtimeF_dtTrue_Exp_0/"* "$EXP_DIR/"
fi
echo "Weather 96->720实验完成，结果保存到: $EXP_DIR"

echo "########## Weather数据集实验完成 ##########"

# ========================================
# ECL数据集实验
# ========================================
echo ""
echo "########## ECL数据集实验开始 ##########"

# ECL 实验1: 96->96
echo "=== 运行实验: YangChanel ECL 96->96 ==="
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
  --n_clusters 4 \
  --cbam_reduction 32 \
  --cbam_kernel_size 5 \
  --mrfp_ratio 1.5 > "$EXP_DIR/training_log.txt" 2>&1

# 保存ECL 96->96结果
if [ -d "./results/long_term_forecast_ECL_96_96_yangchanel_custom_ftM_sl96_ll48_pl96_dm256_nh8_el2_dl1_df256_fc3_ebtimeF_dtTrue_Exp_0/" ]; then
    cp -r "./results/long_term_forecast_ECL_96_96_yangchanel_custom_ftM_sl96_ll48_pl96_dm256_nh8_el2_dl1_df256_fc3_ebtimeF_dtTrue_Exp_0/"* "$EXP_DIR/"
fi
echo "ECL 96->96实验完成，结果保存到: $EXP_DIR"

# ECL 实验2: 96->192
echo "=== 运行实验: YangChanel ECL 96->192 ==="
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
  --n_clusters 4 \
  --cbam_reduction 32 \
  --cbam_kernel_size 5 \
  --mrfp_ratio 1.5 > "$EXP_DIR/training_log.txt" 2>&1

# 保存ECL 96->192结果
if [ -d "./results/long_term_forecast_ECL_96_192_yangchanel_custom_ftM_sl96_ll48_pl192_dm256_nh8_el2_dl1_df256_fc3_ebtimeF_dtTrue_Exp_0/" ]; then
    cp -r "./results/long_term_forecast_ECL_96_192_yangchanel_custom_ftM_sl96_ll48_pl192_dm256_nh8_el2_dl1_df256_fc3_ebtimeF_dtTrue_Exp_0/"* "$EXP_DIR/"
fi
echo "ECL 96->192实验完成，结果保存到: $EXP_DIR"

# ECL 实验3: 96->336
echo "=== 运行实验: YangChanel ECL 96->336 ==="
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
  --n_clusters 4 \
  --cbam_reduction 32 \
  --cbam_kernel_size 5 \
  --mrfp_ratio 1.5 > "$EXP_DIR/training_log.txt" 2>&1

# 保存ECL 96->336结果
if [ -d "./results/long_term_forecast_ECL_96_336_yangchanel_custom_ftM_sl96_ll48_pl336_dm256_nh8_el2_dl1_df256_fc3_ebtimeF_dtTrue_Exp_0/" ]; then
    cp -r "./results/long_term_forecast_ECL_96_336_yangchanel_custom_ftM_sl96_ll48_pl336_dm256_nh8_el2_dl1_df256_fc3_ebtimeF_dtTrue_Exp_0/"* "$EXP_DIR/"
fi
echo "ECL 96->336实验完成，结果保存到: $EXP_DIR"

# ECL 实验4: 96->720
echo "=== 运行实验: YangChanel ECL 96->720 ==="
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
  --n_clusters 4 \
  --cbam_reduction 32 \
  --cbam_kernel_size 5 \
  --mrfp_ratio 1.5 > "$EXP_DIR/training_log.txt" 2>&1

# 保存ECL 96->720结果
if [ -d "./results/long_term_forecast_ECL_96_720_yangchanel_custom_ftM_sl96_ll48_pl720_dm256_nh8_el2_dl1_df256_fc3_ebtimeF_dtTrue_Exp_0/" ]; then
    cp -r "./results/long_term_forecast_ECL_96_720_yangchanel_custom_ftM_sl96_ll48_pl720_dm256_nh8_el2_dl1_df256_fc3_ebtimeF_dtTrue_Exp_0/"* "$EXP_DIR/"
fi
echo "ECL 96->720实验完成，结果保存到: $EXP_DIR"

echo "########## ECL数据集实验完成 ##########"

# ========================================
# 生成综合实验结果汇总
# ========================================
echo ""
echo "正在生成综合实验结果汇总..."
SUMMARY_FILE="$RESULTS_DIR/comprehensive_experiment_summary.txt"

# 创建综合汇总文件
cat > "$SUMMARY_FILE" << EOF
YangChanel Comprehensive Experiment Results Summary
=================================================
Timestamp: $(date '+%Y-%m-%d %H:%M:%S')
Model: YangChanel (AdaWaveNet + CBAM + MRFP)

========================================
Weather Dataset Results
========================================
Features: 21 (weather variables)
Input Length (seq_len): 96

Weather Configuration:
- d_model: 512, d_ff: 512
- e_layers: 3, lifting_levels: 3
- CBAM Reduction: 16, Kernel Size: 7
- MRFP Ratio: 2.0
- Learning Rate: 0.0005, Batch Size: 16

Weather Experiments:
- weather_96_96:  seq_len=96, pred_len=96
- weather_96_192: seq_len=96, pred_len=192
- weather_96_336: seq_len=96, pred_len=336
- weather_96_720: seq_len=96, pred_len=720

========================================
ECL Dataset Results
========================================
Features: 321 (electricity consumption variables)
Input Length (seq_len): 96

ECL Configuration:
- d_model: 256, d_ff: 256
- e_layers: 2, lifting_levels: 3
- CBAM Reduction: 32, Kernel Size: 5
- MRFP Ratio: 1.5
- Learning Rate: 0.0005, Batch Size: 16

ECL Experiments:
- ecl_96_96:  seq_len=96, pred_len=96
- ecl_96_192: seq_len=96, pred_len=192
- ecl_96_336: seq_len=96, pred_len=336
- ecl_96_720: seq_len=96, pred_len=720

========================================
YangChanel Model Architecture
========================================
Core Components:
1. CBAM通道空间融合注意力机制
   - 通道注意力: 全局平均/最大池化 + MLP
   - 空间注意力: 通道维度池化 + 1D卷积
   - 融合机制: 联合MLP建立通道-空间相关性

2. MRFP多感受野特征处理
   - 多尺度深度可分离卷积 (3,5,7,9,11,13)
   - 自适应多尺度特征融合
   - 时序数据三层级处理

3. AdaWaveNet小波变换架构
   - 自适应小波分解与重构
   - 多层级小波编码解码
   - 保持原始小波变换核心不变

4. K-means聚类线性预测
   - 通道级聚类分组
   - 聚类特定的线性变换
   - 趋势和季节性分离处理

Results saved in: $RESULTS_DIR
Total Experiments: 8 (4 Weather + 4 ECL)
EOF

echo ""
echo "=========================================="
echo "=== 所有YangChanel综合实验完成 ==="
echo "=========================================="
echo "结果保存目录: $RESULTS_DIR"
echo "综合汇总文件: $SUMMARY_FILE"
echo ""
echo "Weather数据集实验:"
echo "  - weather_96_96:  21特征, seq_len=96, pred_len=96"
echo "  - weather_96_192: 21特征, seq_len=96, pred_len=192"
echo "  - weather_96_336: 21特征, seq_len=96, pred_len=336"
echo "  - weather_96_720: 21特征, seq_len=96, pred_len=720"
echo ""
echo "ECL数据集实验:"
echo "  - ecl_96_96:  321特征, seq_len=96, pred_len=96"
echo "  - ecl_96_192: 321特征, seq_len=96, pred_len=192"
echo "  - ecl_96_336: 321特征, seq_len=96, pred_len=336"
echo "  - ecl_96_720: 321特征, seq_len=96, pred_len=720"
echo ""
echo "YangChanel模型核心特性:"
echo "  ✓ CBAM通道空间融合注意力"
echo "  ✓ MRFP多感受野特征处理"
echo "  ✓ AdaWaveNet小波变换架构"
echo "  ✓ K-means聚类线性预测"
echo "  ✓ 针对不同数据集优化超参数"
echo ""
echo "实验完成时间: $(date '+%Y-%m-%d %H:%M:%S')"
