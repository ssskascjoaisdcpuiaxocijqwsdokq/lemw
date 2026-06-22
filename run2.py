import argparse
import os
import torch
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from exp.exp_imputation import Exp_Imputation
from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast
from exp.exp_anomaly_detection import Exp_Anomaly_Detection
from exp.exp_classification import Exp_Classification
# from exp.exp_super_resolution import Exp_Super_Resolution  # 模块不存在，注释掉
from utils.print_args import print_args
import random
import numpy as np
import json
from datetime import datetime
import shutil

def save_experiment_results(args, setting, results_metrics=None):
    """
    保存实验结果，包括模型名称、参数和最终结果
    """
    # 创建保存目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"./experiment_results/{args.model}_{args.data}_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. 保存实验参数
    args_dict = vars(args)
    args_file = os.path.join(save_dir, "experiment_args.json")
    with open(args_file, 'w', encoding='utf-8') as f:
        json.dump(args_dict, f, indent=2, ensure_ascii=False, default=str)
    
    # 2. 保存实验配置摘要
    summary_file = os.path.join(save_dir, "experiment_summary.txt")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("Experiment Results Summary\n")
        f.write("=" * 50 + "\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Dataset: {args.data}\n")
        f.write(f"Task: {args.task_name}\n")
        f.write(f"Setting: {setting}\n")
        f.write("\n")
        
        # 关键参数
        f.write("Key Parameters:\n")
        f.write("-" * 30 + "\n")
        f.write(f"seq_len: {args.seq_len}\n")
        f.write(f"pred_len: {args.pred_len}\n")
        f.write(f"d_model: {args.d_model}\n")
        f.write(f"n_heads: {args.n_heads}\n")
        f.write(f"e_layers: {args.e_layers}\n")
        f.write(f"d_layers: {args.d_layers}\n")
        f.write(f"batch_size: {args.batch_size}\n")
        f.write(f"learning_rate: {args.learning_rate}\n")
        
        # YECAMR610 and YECAMRDU specific parameters
        if hasattr(args, 'mrfp_ratio'):
            f.write(f"mrfp_ratio: {args.mrfp_ratio}\n")
        if hasattr(args, 'eca_kernel_size'):
            f.write(f"eca_kernel_size: {args.eca_kernel_size}\n")
        if hasattr(args, 'duet_d_model'):
            f.write(f"duet_d_model: {args.duet_d_model}\n")
        f.write(f"train_epochs: {args.train_epochs}\n")
        
        # 模型特定参数
        if hasattr(args, 'n_clusters'):
            f.write(f"n_clusters: {args.n_clusters}\n")
        if hasattr(args, 'lifting_levels'):
            f.write(f"lifting_levels: {args.lifting_levels}\n")
        if hasattr(args, 'lifting_kernel_size'):
            f.write(f"lifting_kernel_size: {args.lifting_kernel_size}\n")
        
        f.write("\n")
        
        # 如果有结果指标，保存它们
        if results_metrics:
            f.write("Results:\n")
            f.write("-" * 30 + "\n")
            for key, value in results_metrics.items():
                f.write(f"{key}: {value}\n")
    
    # 3. 复制原始结果文件
    original_result_dir = f"./results/{setting}"
    if os.path.exists(original_result_dir):
        target_result_dir = os.path.join(save_dir, "original_results")
        try:
            shutil.copytree(original_result_dir, target_result_dir)
        except Exception as e:
            print(f"Warning: Could not copy original results: {e}")
    
    # 4. 保存命令行重现脚本
    cmd_file = os.path.join(save_dir, "reproduce_command.txt")
    with open(cmd_file, 'w', encoding='utf-8') as f:
        f.write("# Command to reproduce this experiment\n")
        f.write("python run.py \\\n")
        for key, value in args_dict.items():
            if key not in ['use_gpu', 'gpu', 'use_multi_gpu', 'devices', 'device_ids']:
                if isinstance(value, bool):
                    if value:
                        f.write(f"  --{key} \\\n")
                else:
                    f.write(f"  --{key} {value} \\\n")
        f.write("\n")
    
    print(f"实验结果已保存到: {save_dir}")
    return save_dir

def extract_metrics_from_results(setting):
    """
    从结果文件中提取指标
    """
    result_dir = f"./results/{setting}"
    metrics = {}
    
    try:
        # 尝试读取metrics.npy
        metrics_file = os.path.join(result_dir, "metrics.npy")
        if os.path.exists(metrics_file):
            metrics_array = np.load(metrics_file)
            if len(metrics_array) >= 2:
                metrics['MAE'] = float(metrics_array[0])
                metrics['MSE'] = float(metrics_array[1])
                if len(metrics_array) > 2:
                    metrics['RMSE'] = float(metrics_array[2])
                if len(metrics_array) > 3:
                    metrics['MAPE'] = float(metrics_array[3])
                if len(metrics_array) > 4:
                    metrics['MSPE'] = float(metrics_array[4])
        
        # 尝试读取result文件
        result_files = [f for f in os.listdir(result_dir) if f.startswith('result_') and f.endswith('.txt')]
        if result_files:
            result_file = os.path.join(result_dir, result_files[0])
            with open(result_file, 'r') as f:
                lines = f.readlines()
                if lines:
                    # 通常最后一行包含最终结果
                    last_line = lines[-1].strip()
                    if last_line and not last_line.startswith('#'):
                        parts = last_line.split()
                        if len(parts) >= 2:
                            try:
                                metrics['Final_MAE'] = float(parts[0])
                                metrics['Final_MSE'] = float(parts[1])
                                if len(parts) > 2:
                                    metrics['Final_RMSE'] = float(parts[2])
                                if len(parts) > 3:
                                    metrics['Final_MAPE'] = float(parts[3])
                                if len(parts) > 4:
                                    metrics['Final_MSPE'] = float(parts[4])
                            except ValueError:
                                pass
    except Exception as e:
        print(f"Warning: Could not extract metrics: {e}")
    
    return metrics

if __name__ == '__main__':
    fix_seed = 2024
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='TimesNet')

    # basic config
    parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='yeca',
                        help='model name, options: [Autoformer, Transformer, TimesNet]')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--adjust_lr', type=bool, default=False, help='use learning rate decay')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

    # inputation task
    parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')
    parser.add_argument('--mask_type', type=str, default="random", help='mask_type: [random, extended]')

    # super resolution task
    parser.add_argument('--sr_ratio', type=int, default=1, help='super resolution ratio')
    
    # anomaly detection task
    parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (percent)')

    # model define
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--lifting_kernel_size', type=int, default=7, help='conv kernel size of lifting scheme')
    parser.add_argument('--lifting_levels', type=int, default=1, help='levels of lifting scheme')
    parser.add_argument('--regu_details', type=float, default=0.01, help='regu_details of lifting scheme')
    parser.add_argument('--regu_approx', type=float, default=0.01, help='regu_approx of lifting scheme')
    parser.add_argument('--n_clusters', type=int, default=4, help='number of clusters for AdaWaveNet')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--channel_independence', type=int, default=1,
                        help='1: channel dependence 0: channel independence for FreTS model')
    
    # YECAMR610 and YECAMRDU specific params
    parser.add_argument('--mrfp_ratio', type=float, default=2.0, help='hidden layer expansion ratio for MRFP in YECAMR610/YECAMRDU')
    parser.add_argument('--eca_kernel_size', type=int, default=3, help='kernel size for ECA attention')
    parser.add_argument('--duet_d_model', type=int, default=512, help='DUET linear extractor model dimension for YECAMRDU')
    parser.add_argument('--trend_num_experts', type=int, default=4, help='number of experts for AAD trend MoE')
    parser.add_argument('--trend_num_heads', type=int, default=4, help='number of projection heads for AAD trend extractor')
    parser.add_argument('--trend_expert_expansion', type=float, default=2.0, help='hidden expansion ratio inside each AAD trend expert')
    parser.add_argument('--trend_mlp_expansion', type=float, default=2.0, help='hidden expansion ratio of AAD post-MoE MLP')
    parser.add_argument('--router_temperature', type=float, default=1.0, help='temperature for prototype soft router in AROUTDUET')
    
    # SimpleTM specific params
    parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize for SimpleTM')
    parser.add_argument('--geomattn_dropout', type=float, default=0.1, help='geometric attention dropout for SimpleTM')
    parser.add_argument('--alpha', type=float, default=0.5, help='alpha parameter for geometric attention in SimpleTM')
    parser.add_argument('--kernel_size', type=int, default=3, help='kernel size for geometric attention in SimpleTM')
    parser.add_argument('--requires_grad', type=bool, default=True, help='whether to require grad for SimpleTM')
    parser.add_argument('--wv', type=int, default=1, help='wv parameter for SimpleTM')
    parser.add_argument('--m', type=int, default=1, help='m parameter for SimpleTM')
    
    # optimization
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=50, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
    parser.add_argument('--debug', action='store_true', help='enable verbose debug logs in model forward')
    parser.add_argument('--log_file', type=str, default='./zadaw_debug.log', help='debug log file path')
    parser.add_argument('--log_limit', type=int, default=50, help='max debug log entries per epoch')

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

    # ECA attention params
    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print_args(args)

    if args.task_name == 'long_term_forecast':
        Exp = Exp_Long_Term_Forecast
    elif args.task_name == 'short_term_forecast':
        Exp = Exp_Short_Term_Forecast
    elif args.task_name == 'imputation':
        Exp = Exp_Imputation
    elif args.task_name == 'anomaly_detection':
        Exp = Exp_Anomaly_Detection
    elif args.task_name == 'classification':
        Exp = Exp_Classification
    # elif args.task_name == 'super_resolution':
    #     Exp = Exp_Super_Resolution  # 模块不存在，注释掉
    else:
        Exp = Exp_Long_Term_Forecast

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            exp = Exp(args)  # set experiments
            setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
                args.task_name,
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.factor,
                args.embed,
                args.distil,
                args.des, ii)

            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)
            
            # 提取并保存实验结果
            print('>>>>>>>saving results : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            metrics = extract_metrics_from_results(setting)
            save_experiment_results(args, setting, metrics)
            
            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
            args.task_name,
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.factor,
            args.embed,
            args.distil,
            args.des, ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        
        # 提取并保存实验结果
        print('>>>>>>>saving results : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        metrics = extract_metrics_from_results(setting)
        save_experiment_results(args, setting, metrics)
        
        torch.cuda.empty_cache()
        exp.test(setting, test=1)
        
        # 提取并保存实验结果
        print('>>>>>>>saving results : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        metrics = extract_metrics_from_results(setting)
        save_experiment_results(args, setting, metrics)
        
        torch.cuda.empty_cache()
