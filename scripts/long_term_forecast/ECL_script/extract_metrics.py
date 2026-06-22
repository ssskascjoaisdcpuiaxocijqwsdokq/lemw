#!/usr/bin/env python3
"""
提取ECL实验结果中的MAE和MSE指标，生成汇总文件
"""
import os
import sys
import numpy as np
from datetime import datetime

def extract_metrics_from_npy(exp_dir):
    """从.npy文件中提取指标"""
    try:
        # 查找metrics.npy文件
        metrics_file = None
        for file in os.listdir(exp_dir):
            if file == 'metrics.npy':
                metrics_file = os.path.join(exp_dir, file)
                break
        
        if metrics_file and os.path.exists(metrics_file):
            metrics = np.load(metrics_file)
            # 通常metrics数组格式为 [mae, mse, rmse, mape, mspe]
            if len(metrics) >= 2:
                mae = float(metrics[0])
                mse = float(metrics[1])
                rmse = float(metrics[2]) if len(metrics) > 2 else (mse ** 0.5)
                mape = float(metrics[3]) if len(metrics) > 3 else None
                mspe = float(metrics[4]) if len(metrics) > 4 else None
                return mae, mse, rmse, mape, mspe
    except Exception as e:
        print(f"Error reading metrics from {exp_dir}: {e}")
    
    return None, None, None, None, None

def extract_metrics_from_log(exp_dir):
    """从训练日志中提取指标"""
    log_file = os.path.join(exp_dir, 'training_log.txt')
    if not os.path.exists(log_file):
        return None, None, None, None, None
    
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 查找包含test结果的行
        mae, mse = None, None
        for line in reversed(lines):  # 从后往前找最新的结果
            if 'test' in line.lower() and ('mae' in line.lower() or 'mse' in line.lower()):
                # 尝试提取数字
                parts = line.split()
                for i, part in enumerate(parts):
                    if 'mae' in part.lower() and i + 1 < len(parts):
                        try:
                            mae = float(parts[i + 1].replace(',', ''))
                        except:
                            pass
                    elif 'mse' in part.lower() and i + 1 < len(parts):
                        try:
                            mse = float(parts[i + 1].replace(',', ''))
                        except:
                            pass
                if mae is not None and mse is not None:
                    break
        
        if mae is not None and mse is not None:
            rmse = mse ** 0.5
            return mae, mse, rmse, None, None
            
    except Exception as e:
        print(f"Error reading log from {exp_dir}: {e}")
    
    return None, None, None, None, None

def generate_summary(results_dir):
    """生成实验结果汇总"""
    summary_file = os.path.join(results_dir, 'experiment_summary.txt')
    
    # 实验配置
    experiments = [
        ('96', 'ecl_96_96'),
        ('192', 'ecl_96_192'),
        ('336', 'ecl_96_336'),
        ('720', 'ecl_96_720')
    ]
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("ECL (Electricity Consuming Load) Dataset Experiment Results Summary\n")
        f.write("================================================================\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("Model: AdaWaveNet\n")
        f.write("Input Length (seq_len): 96\n")
        f.write("Features: 321 (electricity consumption variables)\n")
        f.write("\n")
        f.write("Pred_Length    MAE        MSE        RMSE       MAPE       MSPE\n")
        f.write("--------------------------------------------------------------\n")
        
        for pred_len, exp_name in experiments:
            exp_dir = os.path.join(results_dir, exp_name)
            
            if os.path.exists(exp_dir):
                # 首先尝试从.npy文件提取
                mae, mse, rmse, mape, mspe = extract_metrics_from_npy(exp_dir)
                
                # 如果.npy文件没有找到，尝试从日志提取
                if mae is None:
                    mae, mse, rmse, mape, mspe = extract_metrics_from_log(exp_dir)
                
                # 格式化输出
                mae_str = f"{mae:.6f}" if mae is not None else "N/A"
                mse_str = f"{mse:.6f}" if mse is not None else "N/A"
                rmse_str = f"{rmse:.6f}" if rmse is not None else "N/A"
                mape_str = f"{mape:.6f}" if mape is not None else "N/A"
                mspe_str = f"{mspe:.6f}" if mspe is not None else "N/A"
                
                f.write(f"{pred_len:<12} {mae_str:<10} {mse_str:<10} {rmse_str:<10} {mape_str:<10} {mspe_str}\n")
            else:
                f.write(f"{pred_len:<12} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A'}\n")
        
        f.write("\n")
        f.write("Experiment Details:\n")
        f.write("- ecl_96_96:  seq_len=96, pred_len=96\n")
        f.write("- ecl_96_192: seq_len=96, pred_len=192\n")
        f.write("- ecl_96_336: seq_len=96, pred_len=336\n")
        f.write("- ecl_96_720: seq_len=96, pred_len=720\n")
        f.write("\n")
        f.write("Model Configuration:\n")
        f.write("- d_model: 256\n")
        f.write("- d_ff: 256\n")
        f.write("- e_layers: 2\n")
        f.write("- d_layers: 1\n")
        f.write("- lifting_levels: 3\n")
        f.write("- n_clusters: 4\n")
        f.write("- batch_size: 16\n")
        f.write("- learning_rate: 0.0005\n")
        f.write(f"\nResults saved in: {results_dir}\n")
    
    print(f"ECL实验汇总文件已生成: {summary_file}")
    return summary_file

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python extract_metrics.py <results_directory>")
        sys.exit(1)
    
    results_dir = sys.argv[1]
    if not os.path.exists(results_dir):
        print(f"Error: Directory {results_dir} does not exist")
        sys.exit(1)
    
    generate_summary(results_dir)




















