"""
Notebook 专用训练脚本 - 避免 CUDA 相关问题
"""
import os
import sys
import numpy as np
from copy import deepcopy

# 在导入任何 torch 相关模块之前，先设置环境变量
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

# 强制禁用 CUDA
import torch
torch.cuda.is_available = lambda: False
torch.cuda.device_count = lambda: 0
torch.cuda.set_device = lambda *args, **kwargs: None

# 设置随机种子
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)

def train_notebook(
    emb_dim=300,
    hidden_dim=300,
    hop=1,
    heads=2,
    device_id="0",
    save_path="results/tb_results/EmoPrepend/",
    batch_size=16,
    epochs=100,
    lr=1e-4,
    pretrain_emb=True,
    label_smoothing=True,
    noam=True,
    pointer_gen=True,
    test=False,
    enable_test=True,    # 新增参数：训练完成后是否执行测试
    verbose_test=False,  # 是否详细显示测试输出
    model: str = "EmoPrepend",  # 选择模型类型（EmoPrepend / Transformer）
):
    """
    在 Notebook 中训练（轻量化版本）模型，默认使用 EmoPrepend。

    可选 model：
    - "EmoPrepend"（默认）
    - "Transformer"

    对抗训练相关模型（如 EmpDG、EmpDG_woD、EmpDG_woG）请使用 train.py 或 adver_train*.py。
    """
    
    # 创建一个临时的配置对象
    class Config:
        def __init__(self):
            # 基础配置
            self.UNK_idx = 0
            self.PAD_idx = 1
            self.EOS_idx = 2
            self.SOS_idx = 3
            self.USR_idx = 4
            self.SYS_idx = 5
            self.CLS_idx = 6
            self.LAB_idx = 7
            
            # 设备配置
            self.USE_CUDA = False  # 强制使用 CPU
            self.device_id = device_id
            self.device = torch.device("cpu")
            
            # 模型配置
            self.model = model
            self.dataset = "empathetic"
            self.hidden_dim = hidden_dim
            self.emb_dim = emb_dim
            self.batch_size = batch_size
            self.epochs = epochs
            self.lr = lr
            self.max_grad_norm = 2.0
            self.beam_size = 5
            self.save_path = save_path
            self.save_path_dataset = save_path
            self.resume_path = "result/"
            self.dropout = 0.2
            self.pointer_gen = pointer_gen
            self.beam_search = False
            self.oracle = False
            self.basic_learner = False
            self.project = False
            self.emotion_bia = False
            self.global_update = False
            self.topk = 0
            self.teacher_ratio = 1.0
            self.l1 = 0.0
            self.softmax = False
            self.mean_query = False
            self.schedule = 0
            self.large_decoder = False
            self.multitask = False
            self.is_coverage = False
            self.use_oov_emb = False
            self.pretrain_emb = pretrain_emb
            self.test = test
            self.weight_sharing = False
            self.label_smoothing = label_smoothing
            self.noam = noam
            self.universal = False
            self.act = False
            self.act_loss_weight = 0.001
            self.emb_file = f"vectors/glove.6B.{emb_dim}d.txt"
            self.hop = hop
            self.heads = heads
            self.depth = 40
            self.filter = 50
            self.resume_g = False
            self.resume_d = False
            self.adver_train = False
            self.gp_lambda = 0.1
            self.rnn_hidden_dim = 400
            self.d_steps = 1
            self.g_steps = 1
            self.emotion_disc = False
            self.adver_itr_num = 5000
            self.specify_model = False
            self.emotion_state_emb = False
            
            # 其他配置
            self.adagrad_init_acc = 0.1
            self.rand_unif_init_mag = 0.02
            self.trunc_norm_init_std = 1e-4
            self.cov_loss_wt = 1.0
            self.lr_coverage = 0.15
            self.eps = 1e-12
            self.collect_stats = False
            
            # 添加 arg 属性以兼容 write_config 函数
            self.arg = self
    
    # 创建配置实例
    config = Config()
    
    # 临时替换 utils.config
    original_config = None
    if 'utils.config' in sys.modules:
        original_config = sys.modules['utils.config']
    sys.modules['utils.config'] = config
    
    try:
        # 现在可以安全导入其他模块
        from utils.data_reader import Lang
        
        # 将 Lang 类添加到 __main__ 模块，以便 pickle 可以找到它
        import __main__
        __main__.Lang = Lang
        
        from utils.data_loader import prepare_data_seq
        from utils.common import count_parameters, make_infinite, print_custum, distinctEval
        from Model.EmoPrepend import EmoP
        # 可选导入 Transformer
        try:
            from Model.transformer import Transformer as TransformerModel
        except Exception:
            TransformerModel = None
        from tensorboardX import SummaryWriter
        from torch.nn.init import xavier_uniform_
        from tqdm import tqdm
        import math
        
        def evaluate_quiet(model, data, ty='valid', max_dec_step=30, verbose=True):
            """自定义评估函数，可以控制输出详细程度"""
            outputs = None
            if ty == "test" and verbose:
                outputs = open("Predictions/{}.txt".format(config.model), "w", encoding="utf-8")
            
            model.eval()
            model.__id__logger = 0
            ref, hyp_g = [], []
            
            if ty == "test" and verbose:
                print("testing generation:")
            elif ty == "test":
                print("testing generation (quiet mode)...")
                
            l = []
            p = []
            bce = []
            acc = []

            pbar = tqdm(enumerate(data), total=len(data))
            for j, batch in pbar:
                loss, ppl, bce_prog, acc_prog = model.train_one_batch(batch, 0, train=False)
                l.append(loss)
                p.append(ppl)
                bce.append(bce_prog)
                acc.append(acc_prog)
                
                if ty == "test":
                    sent_g = model.decoder_greedy(batch, max_dec_step=max_dec_step)
                    for i, greedy_sent in enumerate(sent_g):
                        rf = " ".join(batch["target_txt"][i])
                        hyp_g.append(greedy_sent)
                        ref.append(rf)
                        
                        # 只在 verbose 模式下打印详细信息
                        if verbose:
                            print_custum(emotion=batch["emotion_txt"][i],
                                       dial=[" ".join(s) for s in batch['context_txt'][i]],
                                       emotion_context=str(batch['emotion_context_txt'][i]),
                                       ref=rf,
                                       hyp_g=greedy_sent,
                                       hyp_b=[])
                        
                        if outputs is not None:
                            outputs.write("emotion:{} \n".format(batch["emotion_txt"][i]))
                            outputs.write("Context:{} \n".format(
                                [" ".join(s) for s in batch['context_txt'][i]]))
                            outputs.write("Emotion_context:{} \n".format(batch["emotion_context_txt"][i]))
                            outputs.write("Feedback:{} \n".format(batch["feedback_txt"][i]))
                            outputs.write("Pred:{} \n".format(greedy_sent))
                            outputs.write("Ref:{} \n".format(rf))

                pbar.set_description("loss:{:.4f}; ppl:{:.1f}".format(np.mean(l), math.exp(np.mean(l))))

            loss = np.mean(l)
            ppl = np.mean(p)
            bce = np.mean(bce)
            acc = np.mean(acc)
            
            if outputs is not None:
                outputs.close()

            if ty == "test":
                # 计算多样性指标
                preds = [hyp_g[i].split() for i in range(len(hyp_g))]
                dist1, dist2, response_len_ave = distinctEval(preds)
                
                if verbose:
                    print("Dist-1 {:.4f}; Dist-2 {:.4f}; len {:.4f}".format(dist1, dist2, response_len_ave))
                
                return loss, ppl, bce, acc, dist1, dist2
            else:
                return loss, ppl, bce, acc
        
        # 抽出日志写入的辅助函数，减少重复
        def log_train_metrics(writer: "SummaryWriter", n_iter: int, loss: float, ppl: float, bce: float, acc: float) -> None:
            writer.add_scalars('loss', {'loss_train': loss}, n_iter)
            writer.add_scalars('ppl', {'ppl_train': ppl}, n_iter)
            writer.add_scalars('bce', {'bce_train': bce}, n_iter)
            writer.add_scalars('accuracy', {'acc_train': acc}, n_iter)
        
        def log_valid_metrics(writer: "SummaryWriter", n_iter: int, loss: float, ppl: float, bce: float, acc: float) -> None:
            writer.add_scalars('loss', {'loss_valid': loss}, n_iter)
            writer.add_scalars('ppl', {'ppl_valid': ppl}, n_iter)
            writer.add_scalars('bce', {'bce_valid': bce}, n_iter)
            writer.add_scalars('accuracy', {'acc_valid': acc}, n_iter)
        
        # 校验与提示
        supported_models = {"EmoPrepend", "Transformer"}
        if config.model not in supported_models:
            raise ValueError(
                f"当前 Notebook 轻量化训练仅支持 {supported_models}，收到: {config.model}. "
                "请使用 train.py（监督训练）或 adver_train*.py（对抗训练）来训练 EmpDG/EmpDG_woD/EmpDG_woG。"
            )
        
        print(f"开始训练 {config.model} 模型...")
        print(f"配置: emb_dim={emb_dim}, hidden_dim={hidden_dim}, epochs={epochs}")
        
        # 创建保存目录
        if not os.path.exists(config.save_path):
            os.makedirs(config.save_path)
        
        # 加载数据
        print("加载数据...")
        data_loader_tra, data_loader_val, data_loader_tst, vocab, program_number = prepare_data_seq(
            batch_size=config.batch_size
        )
        steps_per_epoch = len(data_loader_tra)
        
        # 创建模型
        print("创建模型...")
        if config.model == "EmoPrepend":
            model_instance = EmoP(vocab, decoder_number=program_number)
        elif config.model == "Transformer":
            if TransformerModel is None:
                raise ImportError("找不到 Transformer 模型实现（Model/transformer.py）。")
            model_instance = TransformerModel(vocab, decoder_number=program_number)
        else:
            # 理论上已在 supported_models 中过滤，这里兜底
            raise ValueError(f"不支持的模型类型: {config.model}")
        
        model = model_instance
        
        # 参数初始化
        for n, p in model.named_parameters():
            if p.dim() > 1 and (n != "embedding.lut.weight" and config.pretrain_emb):
                xavier_uniform_(p)
        
        print("MODEL USED", config.model)
        print("TRAINABLE PARAMETERS", count_parameters(model))
        
        if config.test:
            print("测试模式...")
            model.eval()
            checkpoint = torch.load('result/' + config.model + '_best.tar', map_location='cpu')
            weights_best = checkpoint['models']
            model.load_state_dict({name: weights_best[name] for name in weights_best})
            model.eval()
            loss_test, ppl_test, bce_test, acc_test, dist1, dist2 = evaluate_quiet(
                model, data_loader_tst, ty="test", max_dec_step=50, verbose=verbose_test
            )
            print(f"Test Loss: {loss_test:.4f}, PPL: {ppl_test:.4f}, Acc: {acc_test:.4f}; Dist1: {dist1:.4f}; Dist2: {dist2:.4f}")
            return
        
        # 训练模式
        print("开始训练...")
        model = model.train()
        
        # 训练变量
        best_ppl = 1000
        patient = 0
        
        # TensorBoard
        from tensorboardX import SummaryWriter as _SW
        writer = SummaryWriter(log_dir=config.save_path)
        
        # 保存最佳权重
        weights_best = deepcopy(model.state_dict())
        loss_val, ppl_val, bce_val, acc_val = 0.0, 0.0, 0.0, 0.0
        
        # 数据迭代器
        data_iter = make_infinite(data_loader_tra)
        
        # 训练循环
        check_iter = steps_per_epoch
        from tqdm import tqdm as _tqdm
        for n_iter in tqdm(range(config.epochs * steps_per_epoch)):
            # 训练一步
            loss, ppl, bce, acc = model.train_one_batch(next(data_iter), n_iter)
            
            # 记录指标
            log_train_metrics(writer, n_iter, loss, ppl, bce, acc)
            
            if config.noam:
                writer.add_scalars('lr', {'learning_rate': model.optimizer._rate}, n_iter)
            
            # 验证
            if (n_iter + 1) % check_iter == 0:
                model = model.eval()
                model.epoch = n_iter
                model.__id__logger = 0
                loss_val, ppl_val, bce_val, acc_val = evaluate_quiet(
                    model, data_loader_val, ty="valid", max_dec_step=50, verbose=True
                )
                
                log_valid_metrics(writer, n_iter, loss_val, ppl_val, bce_val, acc_val)
                
                model = model.train()
                
                # 预热
                warmup_iters = min(13000, 2 * steps_per_epoch)
                if n_iter < warmup_iters:
                    continue
                
                # 保存最佳模型
                if ppl_val <= best_ppl:
                    best_ppl = ppl_val
                    patient = 0
                    model_save_path = os.path.join(
                        config.save_path, 'model_{}_{:.4f}'.format(n_iter, best_ppl)
                    )
                    torch.save(model.state_dict(), model_save_path)
                    weights_best = deepcopy(model.state_dict())
                    print("best_ppl: {}; patient: {}".format(best_ppl, patient))
                else:
                    patient += 1
                
                # 早停
                if patient > 2:
                    break
        
        # 保存最终结果
        torch.save({
            "models": weights_best,
            'result': [loss_val, ppl_val, bce_val, acc_val],
        }, os.path.join('result', f'{config.model}_best.tar'))
        
        print("训练完成!")
        
        # 可选的测试阶段
        if enable_test:
            print("开始测试...")
            model.load_state_dict({name: weights_best[name] for name in weights_best})
            model.eval()
            model.epoch = 100
            loss_test, ppl_test, bce_test, acc_test, dist1, dist2 = evaluate_quiet(
                model, data_loader_tst, ty="test", max_dec_step=50, verbose=verbose_test
            )
            
            print(f"测试完成! Test Loss: {loss_test:.4f}, PPL: {ppl_test:.4f}, Acc: {acc_test:.4f}; Dist1: {dist1:.4f}; Dist2: {dist2:.4f}")
            
            # 写入总结
            file_summary = config.save_path + "summary.txt"
            with open(file_summary, 'w') as the_file:
                the_file.write("EVAL\tLoss\tPPL\tAccuracy\n")
                the_file.write("{}\t{:.4f}\t{:.4f}\t{:.4f}".format("test", loss_test, ppl_test, acc_test))
            
            return {
                'training_completed': True,
                'test_performed': True,
                'loss': loss_test,
                'ppl': ppl_test,
                'accuracy': acc_test,
                'best_val_ppl': best_ppl,
                'dist1': dist1,
                'dist2': dist2,
            }
        else:
            print("跳过测试阶段")
            return {
                'training_completed': True,
                'test_performed': False,
                'best_val_ppl': best_ppl
            }
        
    finally:
        # 恢复原始配置
        if original_config is not None:
            sys.modules['utils.config'] = original_config
        elif 'utils.config' in sys.modules:
            del sys.modules['utils.config'] 