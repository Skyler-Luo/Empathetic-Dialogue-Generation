"""
Notebook 专用对抗训练脚本 - 避免 CUDA 相关问题，参考 run_train.py 的风格
"""
import os
import sys
import numpy as np
from copy import deepcopy

# 在导入任何 torch 相关模块之前，先设置环境变量，强制使用 CPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

import torch
# 强制禁用 CUDA 相关函数以避免 Notebook 中的 GPU 问题
torch.cuda.is_available = lambda: False
torch.cuda.device_count = lambda: 0
torch.cuda.set_device = lambda *args, **kwargs: None
# 防止模块/张量的 .cuda() 方法触发 CUDA 初始化（在无 CUDA / CPU-only 环境会抛 AssertionError）
try:
    import torch.nn as _nn
    # 将 Module.cuda 重写为 no-op，返回 self
    _nn.Module.cuda = lambda self, device=None: self
except Exception:
    pass
try:
    # 将 Tensor.cuda 重写为 identity，避免在 Module._apply 中调用时初始化 CUDA
    torch.Tensor.cuda = lambda self, device=None: self
except Exception:
    pass

# 固定随机种子
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)


def adver_train_notebook(
    device_id="0",
    model: str = "EmpDG",  # 支持 EmpDG / EmpDG_woG / EmpDG_woD 等，但 Notebook 主要用 EmpDG
    batch_size: int = 16,
    save_path: str = "results/tb_results/EmpDG/",
    epochs: int = 1,
    lr: float = 1e-4,
    emb_dim: int = 300,
    hidden_dim: int = 300,
    hop: int = 1,
    heads: int = 2,
    d_steps: int = 1,
    g_steps: int = 5,
    adver_itr_num: int = 500,
    resume_g: bool = False,
    resume_d: bool = False,
    pretrain_emb: bool = True,
    rnn_hidden_dim: int = 300,
    label_smoothing: bool = True,
    noam: bool = True,
    pointer_gen: bool = True,
    test: bool = False,
    enable_test: bool = False,
    verbose_test: bool = False,
):
    """
    Notebook 专用的对抗训练入口：在 CPU 下运行，临时替换 `utils.config`，并复用 `adver_train.py` 中的函数。

    返回值：如果执行测试，会返回与 `run_train.train_notebook` 相似的结果字典；否则返回训练完成的标记。
    """

    # 构造一个轻量级的配置对象，供其他模块导入时使用
    class Config:
        def __init__(self):
            # 基本索引
            self.UNK_idx = 0
            self.PAD_idx = 1
            self.EOS_idx = 2
            self.SOS_idx = 3
            self.USR_idx = 4
            self.SYS_idx = 5
            self.CLS_idx = 6
            self.LAB_idx = 7

            # 设备相关
            self.USE_CUDA = False
            self.device_id = device_id
            self.device = torch.device("cpu")

            # 模型相关
            self.model = model
            self.emb_dim = emb_dim
            self.hidden_dim = hidden_dim
            self.hop = hop
            self.heads = heads
            self.emb_file = f"vectors/glove.6B.{emb_dim}d.txt"
            # Transformer / Encoder 层相关（与原 train.py 保持一致）
            self.depth = 40
            self.filter = 50
            self.universal = False
            self.weight_sharing = False
            # RNN / 其他结构维度
            self.rnn_hidden_dim = rnn_hidden_dim
            self.emotion_state_emb = False
            # 添加缺失的project属性
            self.project = False

            # 训练相关
            self.batch_size = batch_size
            self.epochs = epochs
            self.lr = lr
            self.save_path = save_path
            self.save_path_dataset = save_path
            self.resume_path = "result/"

            # 对抗训练相关
            self.d_steps = d_steps
            self.g_steps = g_steps
            self.adver_itr_num = adver_itr_num
            self.resume_g = resume_g
            self.resume_d = resume_d
            self.adver_train = True
            # 添加梯度惩罚系数
            self.gp_lambda = 0.1
            # 添加情感判别器标志
            self.emotion_disc = True

            # flags
            self.test = test
            self.specify_model = False
            self.pretrain_emb = pretrain_emb
            self.label_smoothing = label_smoothing
            self.noam = noam
            self.pointer_gen = pointer_gen
            self.enable_test = enable_test
            self.verbose_test = verbose_test
            self.device = torch.device("cpu")

            # 其他默认值（与项目中使用的字段保持兼容）
            self.max_grad_norm = 2.0
            self.beam_size = 5
            self.save_path = save_path

            # 添加 arg 属性以兼容 write_config 等函数
            self.arg = self

    # 创建配置实例并临时注入到 sys.modules
    config = Config()
    original_config = None
    if 'utils.config' in sys.modules:
        original_config = sys.modules['utils.config']
    sys.modules['utils.config'] = config

    try:
        # 延迟导入依赖 adver 训练流程的模块（它们会读取我们注入的 utils.config）
        import adver_train as adv_mod
        from utils.data_loader import prepare_data_seq
        # 确保 pickle 能找到 Lang 类（utils.data_reader.Lang）
        from utils.data_reader import Lang
        import __main__
        __main__.Lang = Lang

        # 加载数据（对抗训练需要 adver_train=True）
        print("加载数据 (adversarial)...")
        data_loader_tra, data_loader_val, data_loader_tst, vocab, program_number = prepare_data_seq(batch_size=config.batch_size, adver_train=True)

        # 模型流程（参考 adver_train.py 中的 main 逻辑）
        if config.model == "EmpDG":
            print('=====================STEP 1: Pre-train Empathetic Generator=====================')
            model_g = adv_mod.EmpDG_G(vocab, emotion_number=program_number)

            if config.test:
                # 测试模式：直接加载并评估
                model_g = model_g.eval()
                checkpoint = torch.load("result/EmpDG_best.tar", map_location='cpu')
                model_g.load_state_dict(checkpoint)
                loss_test, ppl_test, bce_test, acc_test = adv_mod.evaluate(model_g, data_loader_tst, ty="test", max_dec_step=50)
                print("Model: ", config.model, "End .")
                return {
                    'training_completed': False,
                    'test_performed': True,
                    'loss': loss_test,
                    'ppl': ppl_test,
                    'accuracy': acc_test,
                }
            else:
                # 预训练生成器 G
                model_g = adv_mod.pre_train_g(model_g, resume=config.resume_g)

                print('=====================STEP 2: Pre-train Discriminators==========================')
                model_d = adv_mod.EmpDG_D(vocab)
                model_d = adv_mod.pre_train_d(model_g, model_d, iters=1000, resume=config.resume_d)

                print('=====================STEP 3: Adversarial joint learning=======================')
                adv_mod.adver_joint_train_gd(model_g, model_d, itr_num=config.adver_itr_num)
                
                print('=====================对抗训练完成！=====================')
                return {'training_completed': True, 'test_performed': False}
        else:
            # 其他模式（如只用 EmoPrepend 作为生成器的 D 训练），尽量复用 adver_train 中的逻辑
            print(f"不在 Notebook 模式下支持的模型: {config.model}。Notebook 主要用于 EmpDG 对抗流程。")
            return {'training_completed': False, 'test_performed': False}

    finally:
        # 恢复原始 utils.config，避免污染全局模块状态
        if original_config is not None:
            sys.modules['utils.config'] = original_config
        elif 'utils.config' in sys.modules:
            del sys.modules['utils.config']


if __name__ == '__main__':
    # 简单的命令行快速测试（Notebook 外也可运行）
    result = adver_train_notebook(
        epochs=1,
        emb_dim=300,
        hidden_dim=300,
        hop=1,
        heads=2,
        pretrain_emb=True,
        label_smoothing=True,
        noam=True,
        pointer_gen=True,
        save_path="results/tb_results/EmpDG/",
        resume_g=True,
        resume_d=True,
        enable_test=False,      # 执行测试
        verbose_test=False,    # 静默测试
        model="EmpDG",
    )

    print(result)
