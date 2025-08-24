import os
import logging 
import argparse
import torch
import sys

UNK_idx = 0  # 未知词标记索引，用于表示未在词表中出现的词
PAD_idx = 1  # 填充符标记索引，用于对齐不同长度的序列
EOS_idx = 2  # 序列结束标记索引，表示一个序列的结束
SOS_idx = 3  # 序列开始标记索引，表示一个序列的起始
USR_idx = 4  # 用户发言状态标记索引，用于区分对话中的说话者
SYS_idx = 5  # 系统（或模型）发言状态标记索引
CLS_idx = 6  # 分类标记索引，通常用于句子分类的开头标记
LAB_idx = 7  # 情感标签索引，用于表示对话中的情感类别

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 计算设备，优先使用GPU

if (os.cpu_count() > 8):
    USE_CUDA = True  # 如果CPU核心数大于8，则默认启用CUDA加速
else:
    USE_CUDA = False  # 否则禁用CUDA

parser = argparse.ArgumentParser(allow_abbrev=False)  # 禁用参数缩写，避免 --f 误匹配 --filter
parser.add_argument("--dataset", type=str, default="empathetic")  # 数据集名称，默认为 empathetic
parser.add_argument("--hidden_dim", type=int, default=300)  # 隐藏层维度大小，默认300
parser.add_argument("--emb_dim", type=int, default=300)  # 词嵌入维度，默认300
parser.add_argument("--batch_size", type=int, default=16)  # 每个训练批次的样本数，默认16
parser.add_argument("--epochs", type=int, default=100)  # 训练轮数，默认100
parser.add_argument("--lr", type=float, default=0.0001)  # 学习率，默认0.0001
parser.add_argument("--max_grad_norm", type=float, default=2.0)  # 梯度裁剪的最大范数，默认2.0
parser.add_argument("--beam_size", type=int, default=5)  # Beam Search 的大小，默认5
parser.add_argument("--save_path", type=str, default="results/tb_results/test/")  # 模型和日志的保存路径，默认 results/tb_results/test/
parser.add_argument("--save_path_dataset", type=str, default="results/tb_results/")  # 数据集保存路径，默认 results/tb_results/
parser.add_argument("--resume_path", type=str, default="result/")  # 恢复训练时模型路径，默认 result/
parser.add_argument("--cuda", action="store_true")  # 是否启用 CUDA 训练，默认 False
parser.add_argument("--device_id", dest="device_id", type=str, default="0")  # 指定使用哪个 GPU，默认0
parser.add_argument("--dropout", dest="dropout", type=float, default=0.2)  # Dropout 概率，默认0.2
parser.add_argument("--pointer_gen", action="store_true", default=True)  # 是否使用指针生成机制，默认 True
parser.add_argument("--beam_search", action="store_true")  # 是否在解码阶段使用 Beam Search，默认 False
parser.add_argument("--oracle", action="store_true")  # 是否使用 Oracle 模式，默认 False
parser.add_argument("--basic_learner", action="store_true")  # 是否使用基础学习器，默认 False
parser.add_argument("--project", action="store_true")  # 是否启用投影层，默认 False
parser.add_argument("--emotion_bia", action="store_true")  # 是否使用情感偏差机制，默认 False
parser.add_argument("--global_update", action="store_true")  # 是否启用全局更新，默认 False
parser.add_argument("--topk", type=int, default=0)  # Top-K 采样的 K 值，默认0（不使用）
parser.add_argument("--teacher_ratio", type=float, default=1.0)  # 教师强制比例，默认1.0
parser.add_argument("--l1", type=float, default=.0)  # L1 正则化权重，默认0.0
parser.add_argument("--softmax", action="store_true")  # 是否在注意力中使用 Softmax，默认 False
parser.add_argument("--mean_query", action="store_true")  # 是否对查询取平均，默认 False
parser.add_argument("--schedule", type=float, default=0)  # 学习率调度参数，默认0
parser.add_argument("--large_decoder", action="store_true")  # 是否使用大型解码器，默认 False
parser.add_argument("--multitask", action="store_true")  # 是否启用多任务学习，默认 False
parser.add_argument("--is_coverage", action="store_true")  # 是否启用覆盖机制，默认 False
parser.add_argument("--use_oov_emb", action="store_true")  # 是否使用 OOV 词嵌入，默认 False
parser.add_argument("--pretrain_emb", action="store_true", default=True)  # 是否使用预训练词嵌入，默认 True
parser.add_argument("--test", action="store_true")  # 是否为测试模式，默认 False
parser.add_argument("--model", type=str, default="EmpDG")  # 模型类型，默认 EmpDG
parser.add_argument("--weight_sharing", action="store_true")  # 是否共享权重，默认 False
parser.add_argument("--label_smoothing", action="store_true", default=True)  # 是否使用标签平滑，默认 True
parser.add_argument("--noam", action="store_true", default=True)  # 是否使用 Noam 学习率策略，默认 True
parser.add_argument("--universal", action="store_true")  # 是否使用 Universal 编码器，默认 False
parser.add_argument("--act", action="store_true")  # 是否使用动作损失，默认 False
parser.add_argument("--act_loss_weight", type=float, default=0.001)  # 动作损失权重，默认0.001
parser.add_argument("--emb_file", type=str)  # 词嵌入文件路径，默认为根据 emb_dim 自动设置
parser.add_argument("--hop", type=int, default=1)  # Transformer 的 hop 数量，默认1
parser.add_argument("--heads", type=int, default=2)  # 注意力头数，默认2
parser.add_argument("--depth", type=int, default=40)  # Transformer 深度，默认40
parser.add_argument("--filter", type=int, default=50)  # 卷积滤波器大小，默认50
parser.add_argument("--resume_g", action="store_true")  # 是否恢复生成器模型，默认 False
parser.add_argument("--resume_d", action="store_true")  # 是否恢复判别器模型，默认 False
parser.add_argument("--adver_train", action="store_true")  # 是否使用对抗训练，默认 False
parser.add_argument("--gp_lambda", type=int, default=0.1)  # 对抗训练的梯度惩罚系数，默认0.1
parser.add_argument("--rnn_hidden_dim", type=int, default=300)  # RNN 隐层维度，默认300
parser.add_argument("--d_steps", type=int, default=1)  # 判别器训练步数，默认1
parser.add_argument("--g_steps", type=int, default=5)  # 生成器训练步数，默认5
parser.add_argument("--emotion_disc", action="store_true")  # 是否使用情感判别器，默认 False
parser.add_argument("--adver_itr_num", type=int, default=5000)  # 对抗训练迭代次数，默认5000
parser.add_argument("--specify_model", action="store_true")  # 是否指定模型名称，默认 False
parser.add_argument("--emotion_state_emb", action="store_true")  # 是否使用情感状态嵌入，默认 False


def print_opts(opts):
    """Prints the values of all command-line arguments.
    """
    print('=' * 80)
    print('Opts'.center(80))
    print('-' * 80)
    for key in opts.__dict__:
        if opts.__dict__[key]:
            print('{:>30}: {:<30}'.format(key, opts.__dict__[key]).center(80))
    print('=' * 80)

arg = None
# 兼容 Jupyter/交互环境与命令行
in_notebook = ('ipykernel' in sys.modules) or ('IPython' in sys.modules)
if in_notebook:
    # 忽略 Notebook 启动参数（如 --f kernel.json）
    arg, _ = parser.parse_known_args([])
else:
    # 命令行严格解析参数
    arg = parser.parse_args()
print_opts(arg)
model = arg.model  # 模型类型，从命令行参数中读取
dataset = arg.dataset  # 数据集名称，从命令行参数中读取
large_decoder = arg.large_decoder  # 是否使用大型解码器
emotion_bia = arg.emotion_bia  # 是否使用情感偏差机制
global_update = arg.global_update  # 是否启用全局更新
topk = arg.topk  # Top-K 采样的 K 值
dropout = arg.dropout  # Dropout 概率
l1 = arg.l1  # L1 正则化权重
oracle = arg.oracle  # 是否使用 Oracle 模式
beam_search = arg.beam_search  # 是否使用 Beam Search
basic_learner = arg.basic_learner  # 是否使用基础学习器
teacher_ratio = arg.teacher_ratio  # 教师强制比例
multitask = arg.multitask  # 是否启用多任务学习
softmax = arg.softmax  # 是否在注意力中使用 Softmax
mean_query = arg.mean_query  # 是否对查询取平均
schedule = arg.schedule  # 学习率调度参数
hidden_dim = arg.hidden_dim  # 隐藏层维度
emb_dim = arg.emb_dim  # 词嵌入维度
batch_size = arg.batch_size  # 批次大小
lr = arg.lr  # 学习率
beam_size = arg.beam_size  # Beam Search 大小
project = arg.project  # 是否启用投影层
adagrad_init_acc = 0.1  # Adagrad 初始累积梯度
rand_unif_init_mag = 0.02  # 权重随机均匀初始化范围
trunc_norm_init_std = 1e-4  # 截断正态分布初始化标准差
max_grad_norm = arg.max_grad_norm  # 最大梯度裁剪范数
USE_CUDA = arg.cuda or torch.cuda.is_available()  # 若命令行启用或本机有GPU则启用 CUDA
device_id = arg.device_id  # 指定使用的 GPU ID
pointer_gen = arg.pointer_gen  # 是否使用指针生成机制
is_coverage = arg.is_coverage  # 是否启用覆盖机制
use_oov_emb = arg.use_oov_emb  # 是否使用 OOV 词嵌入
cov_loss_wt = 1.0  # 覆盖损失权重
lr_coverage = 0.15  # 覆盖机制学习率
eps = 1e-12  # 数值稳定性常量
epochs = arg.epochs  # 最大训练轮数
emb_file = arg.emb_file or "vectors/glove.6B.{}d.txt".format(str(emb_dim))  # 词嵌入文件路径
pretrain_emb = arg.pretrain_emb  # 是否使用预训练词嵌入
save_path = arg.save_path  # 模型保存路径
save_path_dataset = arg.save_path_dataset  # 数据集保存路径
test = arg.test  # 测试模式标志
if not test:
    save_path_dataset = save_path  # 非测试模式时，数据集保存路径等同于模型保存路径

hop = arg.hop  # Transformer 中的 hop 数量
heads = arg.heads  # 注意力头数
depth = arg.depth  # Transformer 深度
filter = arg.filter  # 卷积滤波器大小

label_smoothing = arg.label_smoothing  # 是否使用标签平滑
weight_sharing = arg.weight_sharing  # 是否共享权重
noam = arg.noam  # 是否使用 Noam 学习率策略
universal = arg.universal  # 是否使用 Universal 编码器
act = arg.act  # 是否使用动作损失
act_loss_weight = arg.act_loss_weight  # 动作损失权重

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m-%d %H:%M')#,filename='results/tb_results/logs/{}.log'.format(str(name)))
collect_stats = False

resume_path = arg.resume_path  # 恢复模型路径
resume_g = arg.resume_g  # 是否恢复生成器模型
gp_lambda = arg.gp_lambda  # 对抗训练梯度惩罚系数
rnn_hidden_dim = arg.rnn_hidden_dim  # RNN 隐层维度
d_steps = arg.d_steps  # 判别器训练步数
g_steps = arg.g_steps  # 生成器训练步数
adver_train = arg.adver_train  # 是否使用对抗训练
emotion_disc = arg.emotion_disc  # 是否使用情感判别器
resume_d = arg.resume_d  # 是否恢复判别器模型
adver_itr_num = arg.adver_itr_num  # 对抗训练迭代次数
specify_model = arg.specify_model  # 是否指定模型名称
emotion_state_emb = arg.emotion_state_emb  # 是否使用情感状态嵌入
