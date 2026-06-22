#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

model_name=AdaWaveNet

# 创建结果保存目录
timestamp=$(date +"%Y%m%d_%H%M%S")
result_dir="experiment_results/ETTm1_${model_name}_${timestamp}"
mkdir -p $result_dir

# 创建日志文件
log_file="${result_dir}/experiment_log.txt"

echo "========================================" | tee -a $log_file
echo "AdaWaveNet ETTm1 实验开始" | tee -a $log_file
echo "开始时间: $(date)" | tee -a $log_file
echo "模型: $model_name" | tee -a $log_file
echo "结果保存目录: $result_dir" | tee -a $log_file
echo "========================================" | tee -a $log_file

# 实验1: ETTm1 96->96
echo "" | tee -a $log_file
echo "开始实验1: ETTm1 96->96 预测..." | tee -a $log_file
echo "开始时间: $(date)" | tee -a $log_file

python -u run.py \
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
  --n_clusters 2 \
  --learning_rate 0.0005 \
  --batch_size 16 2>&1 | tee -a "${result_dir}/exp1_96_96_log.txt"

# 保存实验1结果
if [ $? -eq 0 ]; then
    echo "实验1完成: $(date)" | tee -a $log_file
    echo "✅ 实验1 (96->96) 成功完成" | tee -a $log_file
    
    # 复制结果文件
    cp -r ./results/long_term_forecast_ETTm1_96_96_* "${result_dir}/exp1_results/" 2>/dev/null || echo "⚠️ 实验1结果文件复制失败" | tee -a $log_file
    cp -r ./checkpoints/long_term_forecast_ETTm1_96_96_* "${result_dir}/exp1_checkpoints/" 2>/dev/null || echo "⚠️ 实验1模型文件复制失败" | tee -a $log_file
    
    # 提取关键指标
    tail -n 20 "${result_dir}/exp1_96_96_log.txt" | grep -E "(mse|mae)" >> "${result_dir}/exp1_metrics.txt" 2>/dev/null
else
    echo "❌ 实验1 (96->96) 失败" | tee -a $log_file
fi

echo "----------------------------------------" | tee -a $log_file

# 实验2: ETTm1 96->192
echo "" | tee -a $log_file
echo "开始实验2: ETTm1 96->192 预测..." | tee -a $log_file
echo "开始时间: $(date)" | tee -a $log_file

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data/ETT/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_192_192 \
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
  --n_clusters 2 \
  --learning_rate 0.0005 \
  --batch_size 16 2>&1 | tee -a "${result_dir}/exp2_96_192_log.txt"

# 保存实验2结果
if [ $? -eq 0 ]; then
    echo "实验2完成: $(date)" | tee -a $log_file
    echo "✅ 实验2 (96->192) 成功完成" | tee -a $log_file
    
    # 复制结果文件
    cp -r ./results/long_term_forecast_ETTm1_192_192_* "${result_dir}/exp2_results/" 2>/dev/null || echo "⚠️ 实验2结果文件复制失败" | tee -a $log_file
    cp -r ./checkpoints/long_term_forecast_ETTm1_192_192_* "${result_dir}/exp2_checkpoints/" 2>/dev/null || echo "⚠️ 实验2模型文件复制失败" | tee -a $log_file
    
    # 提取关键指标
    tail -n 20 "${result_dir}/exp2_96_192_log.txt" | grep -E "(mse|mae)" >> "${result_dir}/exp2_metrics.txt" 2>/dev/null
else
    echo "❌ 实验2 (96->192) 失败" | tee -a $log_file
fi

echo "----------------------------------------" | tee -a $log_file

# 实验3: ETTm1 96->336
echo "" | tee -a $log_file
echo "开始实验3: ETTm1 96->336 预测..." | tee -a $log_file
echo "开始时间: $(date)" | tee -a $log_file

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data/ETT/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_336_336 \
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
  --n_clusters 2 \
  --learning_rate 0.0005 \
  --batch_size 16 2>&1 | tee -a "${result_dir}/exp3_96_336_log.txt"

# 保存实验3结果
if [ $? -eq 0 ]; then
    echo "实验3完成: $(date)" | tee -a $log_file
    echo "✅ 实验3 (96->336) 成功完成" | tee -a $log_file
    
    # 复制结果文件
    cp -r ./results/long_term_forecast_ETTm1_336_336_* "${result_dir}/exp3_results/" 2>/dev/null || echo "⚠️ 实验3结果文件复制失败" | tee -a $log_file
    cp -r ./checkpoints/long_term_forecast_ETTm1_336_336_* "${result_dir}/exp3_checkpoints/" 2>/dev/null || echo "⚠️ 实验3模型文件复制失败" | tee -a $log_file
    
    # 提取关键指标
    tail -n 20 "${result_dir}/exp3_96_336_log.txt" | grep -E "(mse|mae)" >> "${result_dir}/exp3_metrics.txt" 2>/dev/null
else
    echo "❌ 实验3 (96->336) 失败" | tee -a $log_file
fi

echo "----------------------------------------" | tee -a $log_file

# 实验4: ETTm1 96->720
echo "" | tee -a $log_file
echo "开始实验4: ETTm1 96->720 预测..." | tee -a $log_file
echo "开始时间: $(date)" | tee -a $log_file

python -u run.py \
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
  --n_clusters 2 \
  --learning_rate 0.0005 \
  --batch_size 16 2>&1 | tee -a "${result_dir}/exp4_96_720_log.txt"

# 保存实验4结果
if [ $? -eq 0 ]; then
    echo "实验4完成: $(date)" | tee -a $log_file
    echo "✅ 实验4 (96->720) 成功完成" | tee -a $log_file
    
    # 复制结果文件
    cp -r ./results/long_term_forecast_ETTm1_96_720_* "${result_dir}/exp4_results/" 2>/dev/null || echo "⚠️ 实验4结果文件复制失败" | tee -a $log_file
    cp -r ./checkpoints/long_term_forecast_ETTm1_96_720_* "${result_dir}/exp4_checkpoints/" 2>/dev/null || echo "⚠️ 实验4模型文件复制失败" | tee -a $log_file
    
    # 提取关键指标
    tail -n 20 "${result_dir}/exp4_96_720_log.txt" | grep -E "(mse|mae)" >> "${result_dir}/exp4_metrics.txt" 2>/dev/null
else
    echo "❌ 实验4 (96->720) 失败" | tee -a $log_file
fi

echo "----------------------------------------" | tee -a $log_file

# 生成实验总结
echo "" | tee -a $log_file
echo "========================================" | tee -a $log_file
echo "所有实验完成!" | tee -a $log_file
echo "结束时间: $(date)" | tee -a $log_file
echo "结果保存在: $result_dir" | tee -a $log_file
echo "========================================" | tee -a $log_file

# 创建实验总结文件
summary_file="${result_dir}/experiment_summary.txt"
echo "AdaWaveNet ETTm1 实验总结" > $summary_file
echo "=========================" >> $summary_file
echo "实验时间: $timestamp" >> $summary_file
echo "模型: $model_name" >> $summary_file
echo "" >> $summary_file

# 汇总所有指标
echo "实验结果汇总:" >> $summary_file
echo "-------------" >> $summary_file

for i in {1..4}; do
    if [ -f "${result_dir}/exp${i}_metrics.txt" ]; then
        echo "实验${i}:" >> $summary_file
        cat "${result_dir}/exp${i}_metrics.txt" >> $summary_file
        echo "" >> $summary_file
    fi
done

# 复制总结到主目录
cp $summary_file "./ETTm1_experiment_summary_${timestamp}.txt"

echo ""
echo "🎉 所有实验已完成并保存!"
echo "📁 详细结果保存在: $result_dir"
echo "📊 实验总结: ./ETTm1_experiment_summary_${timestamp}.txt"
echo ""





