@echo off
setlocal enabledelayedexpansion

echo ========================================
echo AdaWaveNet ETTm1 实验 (带结果保存)
echo ========================================

:: 设置变量
set model_name=AdaWaveNet
set timestamp=%date:~0,4%%date:~5,2%%date:~8,2%_%time:~0,2%%time:~3,2%%time:~6,2%
set timestamp=!timestamp: =0!
set result_dir=experiment_results\ETTm1_!model_name!_!timestamp!

:: 创建结果目录
mkdir !result_dir! 2>nul

:: 创建日志文件
set log_file=!result_dir!\experiment_log.txt

echo 实验开始时间: %date% %time% > !log_file!
echo 模型: !model_name! >> !log_file!
echo 结果保存目录: !result_dir! >> !log_file!
echo ======================================== >> !log_file!

echo 📝 日志文件: !log_file!
echo 📁 结果目录: !result_dir!
echo.

:: 实验1: 96->96
echo 🚀 开始实验1: ETTm1 96-^>96 预测...
echo. >> !log_file!
echo 实验1开始: %date% %time% >> !log_file!

python run.py --task_name long_term_forecast --is_training 1 --root_path ./data/ETT/ --data_path ETTm1.csv --model_id ETTm1_96_96 --model !model_name! --data ETTm1 --features M --seq_len 96 --label_len 48 --pred_len 96 --e_layers 3 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --des Exp --d_model 512 --d_ff 512 --itr 1 --lifting_levels 4 --lifting_kernel_size 7 --n_clusters 2 --learning_rate 0.0005 --batch_size 16 > !result_dir!\exp1_96_96_log.txt 2>&1

if !errorlevel! equ 0 (
    echo ✅ 实验1 ^(96-^>96^) 成功完成
    echo 实验1完成: %date% %time% >> !log_file!
    
    :: 复制结果文件
    xcopy /E /I /Q results\long_term_forecast_ETTm1_96_96_* !result_dir!\exp1_results\ 2>nul
    xcopy /E /I /Q checkpoints\long_term_forecast_ETTm1_96_96_* !result_dir!\exp1_checkpoints\ 2>nul
    
    :: 提取指标
    findstr /C:"mse" /C:"mae" !result_dir!\exp1_96_96_log.txt > !result_dir!\exp1_metrics.txt 2>nul
) else (
    echo ❌ 实验1 ^(96-^>96^) 失败
    echo 实验1失败: %date% %time% >> !log_file!
)

echo ----------------------------------------

:: 实验2: 96->192
echo 🚀 开始实验2: ETTm1 96-^>192 预测...
echo. >> !log_file!
echo 实验2开始: %date% %time% >> !log_file!

python run.py --task_name long_term_forecast --is_training 1 --root_path ./data/ETT/ --data_path ETTm1.csv --model_id ETTm1_192_192 --model !model_name! --data ETTm1 --features M --seq_len 96 --label_len 48 --pred_len 192 --e_layers 3 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --des Exp --d_model 512 --d_ff 512 --itr 1 --lifting_levels 4 --lifting_kernel_size 7 --n_clusters 2 --learning_rate 0.0005 --batch_size 16 > !result_dir!\exp2_96_192_log.txt 2>&1

if !errorlevel! equ 0 (
    echo ✅ 实验2 ^(96-^>192^) 成功完成
    echo 实验2完成: %date% %time% >> !log_file!
    
    xcopy /E /I /Q results\long_term_forecast_ETTm1_192_192_* !result_dir!\exp2_results\ 2>nul
    xcopy /E /I /Q checkpoints\long_term_forecast_ETTm1_192_192_* !result_dir!\exp2_checkpoints\ 2>nul
    findstr /C:"mse" /C:"mae" !result_dir!\exp2_96_192_log.txt > !result_dir!\exp2_metrics.txt 2>nul
) else (
    echo ❌ 实验2 ^(96-^>192^) 失败
    echo 实验2失败: %date% %time% >> !log_file!
)

echo ----------------------------------------

:: 实验3: 96->336
echo 🚀 开始实验3: ETTm1 96-^>336 预测...
echo. >> !log_file!
echo 实验3开始: %date% %time% >> !log_file!

python run.py --task_name long_term_forecast --is_training 1 --root_path ./data/ETT/ --data_path ETTm1.csv --model_id ETTm1_336_336 --model !model_name! --data ETTm1 --features M --seq_len 96 --label_len 48 --pred_len 336 --e_layers 3 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --des Exp --d_model 512 --d_ff 512 --itr 1 --lifting_levels 4 --lifting_kernel_size 7 --n_clusters 2 --learning_rate 0.0005 --batch_size 16 > !result_dir!\exp3_96_336_log.txt 2>&1

if !errorlevel! equ 0 (
    echo ✅ 实验3 ^(96-^>336^) 成功完成
    echo 实验3完成: %date% %time% >> !log_file!
    
    xcopy /E /I /Q results\long_term_forecast_ETTm1_336_336_* !result_dir!\exp3_results\ 2>nul
    xcopy /E /I /Q checkpoints\long_term_forecast_ETTm1_336_336_* !result_dir!\exp3_checkpoints\ 2>nul
    findstr /C:"mse" /C:"mae" !result_dir!\exp3_96_336_log.txt > !result_dir!\exp3_metrics.txt 2>nul
) else (
    echo ❌ 实验3 ^(96-^>336^) 失败
    echo 实验3失败: %date% %time% >> !log_file!
)

echo ----------------------------------------

:: 实验4: 96->720
echo 🚀 开始实验4: ETTm1 96-^>720 预测...
echo. >> !log_file!
echo 实验4开始: %date% %time% >> !log_file!

python run.py --task_name long_term_forecast --is_training 1 --root_path ./data/ETT/ --data_path ETTm1.csv --model_id ETTm1_96_720 --model !model_name! --data ETTm1 --features M --seq_len 96 --label_len 48 --pred_len 720 --e_layers 3 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --des Exp --d_model 512 --d_ff 512 --itr 1 --lifting_levels 4 --lifting_kernel_size 7 --n_clusters 2 --learning_rate 0.0005 --batch_size 16 > !result_dir!\exp4_96_720_log.txt 2>&1

if !errorlevel! equ 0 (
    echo ✅ 实验4 ^(96-^>720^) 成功完成
    echo 实验4完成: %date% %time% >> !log_file!
    
    xcopy /E /I /Q results\long_term_forecast_ETTm1_96_720_* !result_dir!\exp4_results\ 2>nul
    xcopy /E /I /Q checkpoints\long_term_forecast_ETTm1_96_720_* !result_dir!\exp4_checkpoints\ 2>nul
    findstr /C:"mse" /C:"mae" !result_dir!\exp4_96_720_log.txt > !result_dir!\exp4_metrics.txt 2>nul
) else (
    echo ❌ 实验4 ^(96-^>720^) 失败
    echo 实验4失败: %date% %time% >> !log_file!
)

:: 生成总结
echo.
echo ========================================
echo 🎉 所有实验完成!
echo 📁 结果保存在: !result_dir!
echo 📊 查看日志: !log_file!
echo ========================================

:: 创建实验总结
set summary_file=!result_dir!\experiment_summary.txt
echo AdaWaveNet ETTm1 实验总结 > !summary_file!
echo ========================= >> !summary_file!
echo 实验时间: !timestamp! >> !summary_file!
echo 模型: !model_name! >> !summary_file!
echo. >> !summary_file!
echo 实验结果汇总: >> !summary_file!
echo ------------- >> !summary_file!

:: 汇总指标
for %%i in (1 2 3 4) do (
    if exist "!result_dir!\exp%%i_metrics.txt" (
        echo 实验%%i: >> !summary_file!
        type "!result_dir!\exp%%i_metrics.txt" >> !summary_file!
        echo. >> !summary_file!
    )
)

:: 复制总结到主目录
copy !summary_file! "ETTm1_experiment_summary_!timestamp!.txt" >nul

echo 📄 实验总结已保存: ETTm1_experiment_summary_!timestamp!.txt
echo.
pause





