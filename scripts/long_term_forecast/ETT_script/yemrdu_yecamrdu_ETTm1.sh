export CUDA_VISIBLE_DEVICES=0

# Summary file
SUMMARY_FILE="./experiment_results/ettm1_yemrdu_yecamrdu_summary.txt"

# Common hyperparameters
COMMON_YEMRDU="--task_name long_term_forecast --is_training 1 --model yemrdu --data ETTm1 --features M --freq t \
--seq_len 96 --label_len 48 --enc_in 7 --dec_in 7 --c_out 7 --d_model 512 --n_heads 8 --e_layers 3 --d_layers 1 --d_ff 512 \
--factor 3 --moving_avg 25 --lifting_levels 4 --lifting_kernel_size 7 --duet_d_model 512 --dropout 0.1 --embed timeF \
--activation gelu --learning_rate 5e-4 --train_epochs 50 --patience 3 --lradj type1 --des ETTm1_yemrdu --regu_details 0.0 --regu_approx 0.0"

COMMON_YECAMRDU="--task_name long_term_forecast --is_training 1 --model yecamrdu --data ETTm1 --features M --freq t \
--seq_len 96 --label_len 48 --enc_in 7 --dec_in 7 --c_out 7 --d_model 512 --n_heads 8 --e_layers 3 --d_layers 1 --d_ff 2048 \
--factor 3 --moving_avg 25 --lifting_levels 2 --lifting_kernel_size 7 --duet_d_model 768 --mrfp_ratio 2.5 --eca_kernel_size 5 \
--regu_details 0.005 --regu_approx 0.005 --dropout 0.1 --embed timeF --activation gelu --learning_rate 5e-4 \
--train_epochs 80 --patience 5 --lradj type1 --des ETTm1_yecamrdu"

run_yemrdu () {
  PRED=$1
  MID=$2
  BS=$3
  LLVLS=$4
  python -u run3.py $COMMON_YEMRDU \
    --root_path ./data/ETT/ \
    --data_path ETTm1.csv \
    --pred_len $PRED \
    --model_id $MID \
    --batch_size $BS \
    --lifting_levels $LLVLS

  SETTING="${MID}_ETTm1_sl96_pl${PRED}_dm512_df512_el3_nh8_ll${LLVLS}_lk7_dd0.0_da0.0_lr0.0005_bs${BS}_seed2025"
  METRIC_PATH=$(ls -1dt ./results/${SETTING}*/metrics.npy 2>/dev/null | head -1)
  if [ -f "$METRIC_PATH" ]; then
    python - <<PY
import numpy as np
mae, mse, *_ = np.load("$METRIC_PATH")
print(f"{SETTING} | MAE={mae:.4f} | MSE={mse:.4f}")
with open("$SUMMARY_FILE","a",encoding="utf-8") as f:
    f.write(f"{SETTING} | MAE={mae:.4f} | MSE={mse:.4f}\n")
PY
  else
    echo "metrics.npy not found for ${SETTING}, skip summary."
  fi
}

run_yecamrdu () {
  PRED=$1
  MID=$2
  BS=$3
  python -u run.py $COMMON_YECAMRDU \
    --root_path ./data/ETT/ \
    --data_path ETTm1.csv \
    --pred_len $PRED \
    --model_id $MID \
    --batch_size $BS

  # yecamrdu setting name较长，用glob查找对应结果目录
  METRIC_PATH=$(ls -1dt ./results/long_term_forecast_${MID}_ETTm1_ftM_sl96_ll48_pl${PRED}_dm512_nh8_el3_dl1_df2048_fc3_ebtimeF_dtTrue_*/*metrics.npy 2>/dev/null | head -1)
  if [ -z "$METRIC_PATH" ]; then
    METRIC_PATH=$(ls -1dt ./results/long_term_forecast_${MID}_ETTm1_ftM_sl96_ll48_pl${PRED}_dm512_nh8_el3_dl1_df2048_fc3_ebtimeF_dtTrue_*metrics.npy 2>/dev/null | head -1)
  fi
  if [ -f "$METRIC_PATH" ]; then
    python - <<PY
import numpy as np
mae, mse, *_ = np.load("$METRIC_PATH")
print(f"{MID} | pred_len={{{PRED}}} | MAE={{mae:.4f}} | MSE={{mse:.4f}}")
with open("$SUMMARY_FILE","a",encoding="utf-8") as f:
    f.write(f"{MID} | pred_len={{{PRED}}} | MAE={{mae:.4f}} | MSE={{mse:.4f}}\n")
PY
  else
    echo "metrics.npy not found for ${MID} pred_len=${PRED}, skip summary."
  fi
}

export CUDA_VISIBLE_DEVICES=0
mkdir -p ./experiment_results
echo "ETTm1 runs (yemrdu + yecamrdu)" > "$SUMMARY_FILE"

# yemrdu 96/192/336/720
run_yemrdu 96  ETTm1_96_96_yemrdu   32 4
run_yemrdu 192 ETTm1_96_192_yemrdu  16 5
run_yemrdu 336 ETTm1_96_336_yemrdu  16 4
run_yemrdu 720 ETTm1_96_720_yemrdu  32 1

# yecamrdu 96/192/336/720
run_yecamrdu 96  ETTm1_96_96_yecamrdu   32
run_yecamrdu 192 ETTm1_96_192_yecamrdu  24
run_yecamrdu 336 ETTm1_96_336_yecamrdu  16
run_yecamrdu 720 ETTm1_96_720_yecamrdu  16
