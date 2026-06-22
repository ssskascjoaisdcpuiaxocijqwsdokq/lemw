import os
import torch
from models import Autoformer, Transformer, TimesNet, Nonstationary_Transformer, DLinear, FEDformer, \
    Informer, LightTS, Reformer, ETSformer, Pyraformer, PatchTST, MICN, Crossformer, FiLM, Koopa, TiDE, FreTS, AdaWaveNet, iTransformer, YANGNet, YANGNetZhu, YANGNetDU, YANGNetDU_SENet, YANGSE, YANGSEOptimized, YANGSEN, YANGC, YANGSETD, YANGSETD1IVATE, YangChanl, YangCBA, YangDU2, YECA, YECA1, YECA2, YECAMF, YECAMR610, YECAMR611, YECAMRDU, AAD, AROUTDUET, AMLP, ACONCAT, YEMRDU, YANX, YFT, YFF, YangChanel, TimeKAN, yef35, yftt, yfttv2, yfttopt, yfttmsf, zaaafinal, YIOP, ZAAA, ZAAA_Optimized, ZAAA_Final, ZAAA_Ultimate, ZAAA_Stable, ZAAB, ZAABB, ZAABB_Simplified, ZAABB_Minimal, ZAABB_Ultra, ZAAA_Performance_Optimized, ZAAA_Optimized_V2, Za3, Zxx, Zabb, ZabbOptimized, ZabbUltra, ZabbRefined, FFAAdaWaveNet, Zabca, Zaas, ZABCA_Optimized, ZACCS, ZXCC, ZXCVB, ZXCVB_Optimized, ZMEM, ZXXCV, ZXXCV_V2, ZANNN, ZDCA, zxcxcx, zxxxp, SimpleTM, zecdu, zyes, yancc, ZZZZZZ, zadaw, ZDUWU, ZAWN, ZDZ, ZBOWU, ZTLUAN, ZYECAMRDU, ZYECAMEDUXX, ZXIAN, ZZZXP, ZYDU, ZCVX, ZTONGFEN


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'TimesNet': TimesNet,
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Nonstationary_Transformer': Nonstationary_Transformer,
            'DLinear': DLinear,
            'FEDformer': FEDformer,
            'Informer': Informer,
            'LightTS': LightTS,
            'Reformer': Reformer,
            'ETSformer': ETSformer,
            'PatchTST': PatchTST,
            'Pyraformer': Pyraformer,
            'MICN': MICN,
            'Crossformer': Crossformer,
            'FiLM': FiLM,
            'Koopa': Koopa,
            'TiDE': TiDE,
            'FreTS': FreTS,
            'AdaWaveNet':AdaWaveNet,
            'iTransformer': iTransformer,
            'YANGNet': YANGNet,
            'YANGNetZhu': YANGNetZhu,
            'YANGNetDU': YANGNetDU,
            'yangnetdu_senet': YANGNetDU_SENet,
            'YANGSE': YANGSE,
            'YANGSEOptimized': YANGSEOptimized,
            'YANGSEN': YANGSEN,
            'YANGC': YANGC,
            'YANGSETD': YANGSETD,
            'YANGSETD1IVATE': YANGSETD1IVATE,
            'yangchanl': YangChanl,
            'yangcba': YangCBA,
            'YangDU2': YangDU2,
            'yeca': YECA,
            'yeca1': YECA1,
            'yeca2': YECA2,
            'yecamf': YECAMF,
            'yecamr610': YECAMR610,
            'YECAMR611': YECAMR611,
            'yecamrdu': YECAMRDU
            , 'aad': AAD
            , 'aroutduet': AROUTDUET
            , 'amlp': AMLP
            , 'aconcat': ACONCAT
            , 'yemrdu': YEMRDU
            , 'yanx': YANX
            , 'yft': YFT
            , 'yff': YFF
            , 'yangchanel': YangChanel
            , 'timekan': TimeKAN
            , 'yef35': yef35
            , 'yftt': yftt
            , 'yfttv2': yfttv2
            , 'yfttopt': yfttopt
            , 'yfttmsf': yfttmsf
            , 'zaaafinal': zaaafinal
            # , 'yrr': YRR  # 临时注释掉
            , 'yiop': YIOP
            , 'zaaa': ZAAA
            , 'zaaa_opt': ZAAA_Optimized
            , 'zaaa_final': ZAAA_Final
            , 'zaaa_ultimate': ZAAA_Ultimate
            , 'zaaa_stable': ZAAA_Stable
            , 'zaab': ZAAB
            , 'zaabb': ZAABB
            , 'zaabb_simplified': ZAABB_Simplified
            , 'zaabb_minimal': ZAABB_Minimal
            , 'zaabb_ultra': ZAABB_Ultra
            , 'zaaa_perf_opt': ZAAA_Performance_Optimized
            , 'zaaa_opt_v2': ZAAA_Optimized_V2
            , 'za3': Za3
            , 'zxx': Zxx
            , 'zabb': Zabb
            , 'zabb_opt': ZabbOptimized
            , 'zabb_ultra': ZabbUltra
            , 'zabb_refined': ZabbRefined
            , 'ffa_adawavenet': FFAAdaWaveNet
            , 'zabca': Zabca
            , 'zabca_opt': ZABCA_Optimized
            , 'zaccs': ZACCS
            , 'zxcc': ZXCC
            , 'zxcvb': ZXCVB
            , 'zxcvb_opt': ZXCVB_Optimized
            , 'zmem': ZMEM
            # , 'zxxxxx': ZXXXXX  # 文件为空，暂时注释
            , 'zaas': Zaas
            , 'zxxcv': ZXXCV
            , 'zxxcv_v2': ZXXCV_V2
            , 'zannn': ZANNN
            , 'zdca': ZDCA
            , 'zxcxcx': zxcxcx
            , 'zxxxp': zxxxp
            , 'zecdu': zecdu
            , 'zyes': zyes
            , 'yancc': yancc
            , 'ZZZZZZ': ZZZZZZ
            , 'zzzzzz': ZZZZZZ
            , 'SimpleTM': SimpleTM
            , 'zadaw': zadaw
            , 'zduwu': ZDUWU
            , 'zawn': ZAWN
            , 'zdz': ZDZ
            , 'zbowu': ZBOWU
            , 'ztluan': ZTLUAN
            , 'zyecamrdu': ZYECAMRDU
            , 'zyecameduxx': ZYECAMEDUXX
            , 'zxian': ZXIAN
            , 'zzzxp': ZZZXP
            , 'zydu': ZYDU
            , 'zcvx': ZCVX
            , 'ztongfen': ZTONGFEN
            , 'ZTONGFEN': ZTONGFEN
        }
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _set_model_iter(self, iteration: int):
        pass

    def _set_model_epoch(self, epoch: int):
        """当模型支持 set_epoch 时，通知模型当前 epoch，便于控制日志输出"""
        target = self.model.module if hasattr(self.model, "module") else self.model
        if hasattr(target, "set_epoch"):
            target.set_epoch(epoch)

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
