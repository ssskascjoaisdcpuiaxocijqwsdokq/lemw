@echo off
echo 开始运行AdaWaveNet Weather实验（带结果保存）...
echo.

cd /d "D:\alunwen\AdaWaveNet-main"

bash scripts/long_term_forecast/Weather_script/AdaWaveNet_with_save.sh

echo.
echo 实验完成！请查看results目录中的结果。
pause
