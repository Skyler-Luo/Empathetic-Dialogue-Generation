"""
EmpDG 项目训练脚本（train.py）

功能概览：
- 加载配置与数据集，构建词表与数据迭代器
- 按配置选择不同的模型结构（Transformer / EmoPrepend / EmpDG 等）
- 可选使用 Xavier 初始化参数（除预训练词向量）
- 执行训练循环，记录 TensorBoard 指标，定期在验证集上评估并保存最优权重
- 在训练结束后进行测试评估；或在 test 模式下直接加载权重进行测试

使用要点：
- 通过 `utils.config` 中的配置项控制设备、批大小、模型类型、是否使用 noam 学习率调度、是否测试模式等
- 训练过程会将日志写入 `config.save_path`，并将最佳模型/指标保存至 `result/` 目录
"""
import os
from tensorboardX import SummaryWriter
from copy import deepcopy
from torch.nn.init import xavier_uniform_

from utils.data_loader import prepare_data_seq
from utils.common import *

torch.manual_seed(0)  # 设定随机种子
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)
from utils.data_reader import Lang
from Model.transformer import Transformer
from Model.EmoPrepend import EmoP
from Model.EmpDG_G import EmpDG_G


# 通过环境变量与 PyTorch API 指定 GPU
os.environ["CUDA_VISIBLE_DEVICES"] = config.device_id
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
if torch.cuda.is_available():
    torch.cuda.set_device(int(config.device_id))

if __name__ == '__main__':
    if not os.path.exists(config.save_path):
        os.makedirs(config.save_path)

    # 加载数据与词表
    # 返回：训练/验证/测试数据加载器、词表对象、program_number（用于情感/解码器数量等）
    data_loader_tra, data_loader_val, data_loader_tst, vocab, program_number = prepare_data_seq(batch_size=config.batch_size)
    steps_per_epoch = len(data_loader_tra)

    # 模型选择：根据配置构建不同的模型实例
    if config.model == "Transformer":
        model = Transformer(vocab, decoder_number=program_number)

    if (config.model == "EmoPrepend") or (config.model == "EmpDG_woG"):
        model = EmoP(vocab, decoder_number=program_number)

    if (config.model == "EmpDG_woD") or (config.model == "EmpDG"):  # EmpDG_woD 训练/测试；EmpDG 仅测试 G
        model = EmpDG_G(vocab, emotion_number=program_number)

    # 参数初始化：对除预训练词向量以外的多维参数使用 Xavier 均匀初始化
    for n, p in model.named_parameters():
        if p.dim() > 1 and (n != "embedding.lut.weight" and config.pretrain_emb):
            xavier_uniform_(p)

    print("MODEL USED", config.model)
    print("TRAINABLE PARAMETERS", count_parameters(model))

    # 训练/测试分支
    if config.test is False:
        # 每隔 `check_iter` 步在验证集上评估一次
        check_iter = steps_per_epoch
        try:
            # 将模型放置到 GPU（如可用）并设置为训练模式
            if config.USE_CUDA:
                model.cuda()
            model = model.train()

            # 记录最佳指标与早停相关变量
            best_ppl = 1000
            patient = 0

            # TensorBoard 日志记录器
            writer = SummaryWriter(log_dir=config.save_path)

            # 备份最佳权重（深拷贝）
            weights_best = deepcopy(model.state_dict())
            # 初始化验证指标以防未触发验证步骤也能正常保存结果
            loss_val, ppl_val, bce_val, acc_val = 0.0, 0.0, 0.0, 0.0

            # 构建一个“无限”数据迭代器，简化训练循环书写
            data_iter = make_infinite(data_loader_tra)

            # 主训练循环（上限较大，实际将由早停控制）
            for n_iter in tqdm(range(config.epochs * steps_per_epoch)):
                # 前向+反向+优化一步；返回训练损失、困惑度、BCE、多分类/二分类准确率等
                loss, ppl, bce, acc = model.train_one_batch(next(data_iter),n_iter)

                # 记录训练指标
                writer.add_scalars('loss', {'loss_train': loss}, n_iter)
                writer.add_scalars('ppl', {'ppl_train': ppl}, n_iter)
                writer.add_scalars('bce', {'bce_train': bce}, n_iter)
                writer.add_scalars('accuracy', {'acc_train': acc}, n_iter)

                # 可选：记录 Noam 学习率调度器的当前学习率
                if config.noam:
                    writer.add_scalars('lr', {'learning_rate': model.optimizer._rate}, n_iter)

                # 定期在验证集上评估
                if (n_iter + 1) % check_iter == 0:
                    model = model.eval()
                    model.epoch = n_iter
                    model.__id__logger = 0
                    loss_val, ppl_val, bce_val, acc_val = evaluate(model, data_loader_val, ty="valid", max_dec_step=50)

                    # 记录验证指标
                    writer.add_scalars('loss', {'loss_valid': loss_val}, n_iter)
                    writer.add_scalars('ppl', {'ppl_valid': ppl_val}, n_iter)
                    writer.add_scalars('bce', {'bce_valid': bce_val}, n_iter)
                    # 注意：此处键为 'acc_train' 但记录了验证准确率，命名可能不一致
                    writer.add_scalars('accuracy', {'acc_train': acc_val}, n_iter)

                    model = model.train()

                    # 动态预热步数，避免小数据集下永不验证
                    warmup_iters = min(13000, 2 * steps_per_epoch)
                    if n_iter < warmup_iters:
                        continue

                    # 保存更优模型
                    if ppl_val <= best_ppl:
                        best_ppl = ppl_val
                        patient = 0
                        # 保存当前最优模型权重快照
                        model_save_path = os.path.join(config.save_path,
                                                       'model_{}_{:.4f}'.format(n_iter, best_ppl))
                        torch.save(model.state_dict(), model_save_path)
                        weights_best = deepcopy(model.state_dict())
                        print("best_ppl: {}; patient: {}".format(best_ppl, patient))
                    else:
                        patient += 1
                    # 简单早停：验证集 PPL 连续劣化超过阈值则停止
                    if patient > 2: break

        except KeyboardInterrupt:
            # 支持 Ctrl+C 手动中断训练
            print('-' * 89)
            print('Exiting from training early')

        # 训练完成后，持久化最佳权重与最终验证指标，供后续直接测试/对比
        torch.save({"models": weights_best,
                    'result': [loss_val, ppl_val, bce_val, acc_val], },
                   os.path.join('result/' + config.model + '_best.tar'))

        # 使用最佳权重在测试集上做最终评估
        model.load_state_dict({name: weights_best[name] for name in weights_best})
        model.eval()
        model.epoch = 100
        loss_test, ppl_test, bce_test, acc_test = evaluate(model, data_loader_tst, ty="test", max_dec_step=50)
    else:  # test
        print("TESTING !!!")
        # 测试模式下仅进行前向评估；可从指定路径或默认 best.tar 加载
        model.cuda()
        model = model.eval()
        if config.specify_model:
            checkpoint = torch.load(config.resume_path, map_location=lambda storage, location: storage)
            model.load_state_dict(checkpoint)
        else:
            checkpoint = torch.load('result/' + config.model + '_best.tar', map_location=lambda storage, location: storage)
            if config.model == "EmpDG" or config.model == "EmpDG_woG":
                weights_best = checkpoint['models_g']
            else:
                weights_best = checkpoint['models']
            model.load_state_dict({name: weights_best[name] for name in weights_best})
        model.eval()
        # 记录测试集上的各类指标（并计算多样性指标 dist1/dist2）
        loss_test, ppl_test, bce_test, acc_test, dist1, dist2 = evaluate(model, data_loader_tst, ty="test", max_dec_step=50)

    print("Model: ", config.model, "End .")

    # 写入测试汇总到文件
    if config.specify_model:
        file_summary = "_summary.txt"
    else:
        file_summary = config.save_path + "summary.txt"
    with open(file_summary, 'w') as the_file:
        the_file.write("EVAL\tLoss\tPPL\tAccuracy\n")
        the_file.write(
            "{}\t{:.4f}\t{:.4f}\t{:.4f}".format("test", loss_test, ppl_test, acc_test))
