export CUDA_VISIBLE_DEVICES=0

# Common hyperparameters for Traffic + yemrdu
COMMON="--task_name long_term_forecast --is_training 1 --model yemrdu --features M --freq h \
--seq_len 96 --label_len 48 --enc_in 862 --dec_in 862 --c_out 862 --d_model 512 --n_heads 8 \
--e_layers 3 --d_layers 1 --d_ff 512 --factor 3 --moving_avg 25 --lifting_levels 4 --lifting_kernel_size 7 \
--duet_d_model 512 --regu_details 0.0 --regu_approx 0.0 --dropout 0.1 --embed timeF --activation gelu \
--learning_rate 5e-4 --train_epochs 50 --patience 3 --lradj type1 --des traffic_yemrdu"

run_case () {
  PRED=$1
  MID=$2
  BS=$3
  LLVLS=$4
  python -u run3.py $COMMON \
    --root_path ./data/traffic/ \
    --data_path traffic.csv \
    --data Traffic \
    --pred_len $PRED \
    --model_id $MID \
    --batch_size $BS \
    --lifting_levels $LLVLS
}

# 96 / 192 / 336 / 720
run_case 96  traffic_96_96_yemrdu   32 4
run_case 192 traffic_96_192_yemrdu  16 5
run_case 336 traffic_96_336_yemrdu  16 4
run_case 720 traffic_96_720_yemrdu  32 1
