export CUDA_VISIBLE_DEVICES=0

# Common hyperparameters for electricity + yemrdu
COMMON="--task_name long_term_forecast --is_training 1 --model yemrdu --data electricity --features M --freq h \
--seq_len 96 --label_len 48 --enc_in 321 --dec_in 321 --c_out 321 --d_model 256 --n_heads 8 \
--e_layers 2 --d_layers 1 --d_ff 256 --factor 3 --moving_avg 25 --lifting_levels 2 --lifting_kernel_size 7 \
--duet_d_model 256 --regu_details 0.005 --regu_approx 0.005 --dropout 0.1 --embed timeF --activation gelu \
--learning_rate 0.0005 --train_epochs 80 --patience 3 --lradj type1 --des electricity_yemrdu"
SUMMARY_FILE="./experiment_results/yemrdu_electricity_summary.txt"
FIX_SEED=2025

run_case () {
  PRED=$1
  MID=$2
  BS=$3
  LLVLS=$4

  python -u run3.py $COMMON \
    --root_path ./data/electricity/ \
    --data_path electricity.csv \
    --pred_len $PRED \
    --model_id $MID \
    --batch_size $BS \
    --lifting_levels $LLVLS

  # 根据 run3.py 的 setting 规则生成结果目录名并记录 metrics
  SETTING="${MID}_electricity_sl96_pl${PRED}_dm512_df512_el3_nh8_ll${LLVLS}_lk7_dd0.005_da0.005_lr0.0005_bs${BS}_seed${FIX_SEED}"
  METRIC_PATH="./results/${SETTING}/metrics.npy"
  if [ -f "$METRIC_PATH" ]; then
    python - <<PY
import numpy as np
mae, mse, *_ = np.load("$METRIC_PATH")
print(f"{SETTING} | pred_len={{{PRED}}} | MAE={{mae:.4f}} | MSE={{mse:.4f}}")
with open("$SUMMARY_FILE","a",encoding="utf-8") as f:
    f.write(f"{SETTING} | pred_len={{{PRED}}} | MAE={{mae:.4f}} | MSE={{mse:.4f}}\n")
PY
  else
    echo "metrics.npy not found for ${SETTING}, skip summary."
  fi
}

# 96 / 192 / 336 / 720
run_case 96  electricity_96_96_yemrdu   32 2
run_case 192 electricity_96_192_yemrdu  24 2
run_case 336 electricity_96_336_yemrdu  16 2
run_case 720 electricity_96_720_yemrdu  16 2
