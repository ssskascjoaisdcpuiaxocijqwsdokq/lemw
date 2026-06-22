import argparse
import random
import numpy as np
import torch
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast


def build_parser():
    parser = argparse.ArgumentParser(description='yemrdu runner')

    # basic config
    parser.add_argument('--task_name', type=str, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, imputation, anomaly_detection, super_resolution]')
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='yemrdu',
                        help='model name, options: [yemrdu]')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='ETTh1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data csv file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s,t,h,d,b,w,m]')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--adjust_lr', type=bool, default=False, help='use learning rate decay')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4 (if applicable)')
    parser.add_argument('--inverse', action='store_true', default=False, help='inverse output data')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

    # model define (yemrdu)
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=3, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=512, help='dimension of ffn')
    parser.add_argument('--factor', type=int, default=3, help='attn factor')
    parser.add_argument('--moving_avg', type=int, default=25, help='moving avg window')
    parser.add_argument('--lifting_levels', type=int, default=4, help='lifting levels')
    parser.add_argument('--lifting_kernel_size', type=int, default=7, help='lifting kernel size')
    parser.add_argument('--regu_details', type=float, default=0.0, help='detail regu')
    parser.add_argument('--regu_approx', type=float, default=0.0, help='approx regu')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention')
    parser.add_argument('--duet_d_model', type=int, default=512, help='DUET trend head hidden dim')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

    # optimization
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=50, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=5e-4, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    parser.add_argument('--exp_name', type=str, required=False, default='yemrdu', help='experiment name')
    parser.add_argument('--fix_seed', type=int, default=2025, help='seed')
    return parser


if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    fix_seed = args.fix_seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args)

    Exp = Exp_Long_Term_Forecast

    if args.is_training:
        for ii in range(args.itr):
            setting = '{}_{}_sl{}_pl{}_dm{}_df{}_el{}_nh{}_ll{}_lk{}_dd{}_da{}_lr{}_bs{}_seed{}'.format(
                args.model_id,
                args.data,
                args.seq_len,
                args.pred_len,
                args.d_model,
                args.d_ff,
                args.e_layers,
                args.n_heads,
                args.lifting_levels,
                args.lifting_kernel_size,
                args.regu_details,
                args.regu_approx,
                args.learning_rate,
                args.batch_size,
                args.fix_seed)

            exp = Exp(args)  # set experiments
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)

            if args.do_predict:
                print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.predict(setting, True)

            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_sl{}_pl{}_dm{}_df{}_el{}_nh{}_ll{}_lk{}_dd{}_da{}_lr{}_bs{}_seed{}'.format(
            args.data,
            args.seq_len,
            args.pred_len,
            args.d_model,
            args.d_ff,
            args.e_layers,
            args.n_heads,
            args.lifting_levels,
            args.lifting_kernel_size,
            args.regu_details,
            args.regu_approx,
            args.learning_rate,
            args.batch_size,
            args.fix_seed)

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()
