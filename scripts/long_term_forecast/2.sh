export CUDA_VISIBLE_DEVICES=0

python -u run2.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data/ETT/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_720_720_yecamr610 \
  --model yecamr610 \
  --data ETTh2 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 720\
  --e_layers 4 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 512\
  --d_ff 512 \
  --lifting_levels 4 \
  --lifting_kernel_size 7 \
  --n_clusters 4 \
  --learning_rate 0.0005 \
  --batch_size 16 \
  --dropout 0.2\
  --eca_kernel_size 27\
  --mrfp_ratio 3.0\
  --patience 2 \
 

