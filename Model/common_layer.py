import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.init as I
import numpy as np
import math
import os
from utils import config
import pdb

import pprint
from tqdm import tqdm
pp = pprint.PrettyPrinter(indent=1)
import numpy as np

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)


class EncoderLayer(nn.Module):
    """
    Transformer 编码器（Encoder）中的单层结构。
    参考论文：https://arxiv.org/pdf/1706.03762.pdf
    """
    def __init__(self, hidden_size, total_key_depth, total_value_depth, filter_size, num_heads,
                 bias_mask=None, layer_dropout=0.0, attention_dropout=0.0, relu_dropout=0.0):
        """
        参数:
            hidden_size (int): 隐藏层维度大小
            total_key_depth (int): 键（Key）张量最后一维的大小，需能被 num_heads 整除
            total_value_depth (int): 值（Value）张量最后一维的大小，需能被 num_heads 整除
            output_depth (int): 输出张量最后一维的大小
            filter_size (int): 前馈网络（FFN）中间隐藏层的大小
            num_heads (int): 多头注意力头数
            bias_mask (Tensor or None): 用于屏蔽未来位置的掩码，防止信息泄露
            layer_dropout (float): 本层残差连接后的 Dropout 概率
            attention_dropout (float): 注意力权重上的 Dropout 概率（训练时非零）
            relu_dropout (float): 前馈网络中 ReLU 后的 Dropout 概率（训练时非零）
        """
        super(EncoderLayer, self).__init__()
        self.multi_head_attention = MultiHeadAttention(hidden_size, total_key_depth, total_value_depth,
                                                       hidden_size, num_heads, bias_mask, attention_dropout)
        self.positionwise_feed_forward = PositionwiseFeedForward(hidden_size, filter_size, hidden_size,
                                                                 layer_config='cc', padding='both',
                                                                 dropout=relu_dropout)
        self.dropout = nn.Dropout(layer_dropout)
        self.layer_norm_mha = LayerNorm(hidden_size)
        self.layer_norm_ffn = LayerNorm(hidden_size)

    def forward(self, inputs, mask=None):
        x = inputs
        # 层归一化：对输入进行标准化处理，稳定模型训练
        x_norm = self.layer_norm_mha(x)
        # 多头注意力：执行自注意力计算，融合全局信息
        y, _ = self.multi_head_attention(x_norm, x_norm, x_norm, mask)
        # 残差连接并进行 Dropout
        x = self.dropout(x + y)
        # 层归一化：对残差输出进行标准化
        x_norm = self.layer_norm_ffn(x)
        # 前馈网络：逐位置进行两层线性变换及激活
        y = self.positionwise_feed_forward(x_norm)
        # 残差连接并进行 Dropout
        y = self.dropout(x + y)
        return y


class GraphLayer(nn.Module):
    """
    Transformer 解码器（Decoder）中的图网络层（GraphLayer）。
    参考论文：https://arxiv.org/pdf/1706.03762.pdf
    """

    def __init__(self, hidden_size, total_key_depth, total_value_depth, filter_size, num_heads,
                 bias_mask, layer_dropout=0.0, attention_dropout=0.0, relu_dropout=0.0):
        """
        参数:
            hidden_size (int): 隐藏层维度
            total_key_depth (int): Key 张量最后一维维度，需被 num_heads 整除
            total_value_depth (int): Value 张量最后一维维度，需被 num_heads 整除
            filter_size (int): 前馈网络中间层维度
            num_heads (int): 注意力头数量
            bias_mask (Tensor): 屏蔽未来位置的掩码
            layer_dropout (float): 本层残差后的 Dropout 概率
            attention_dropout (float): 注意力权重上的 Dropout 概率
            relu_dropout (float): 前馈网络中 ReLU 后的 Dropout 概率
        """

        super(GraphLayer, self).__init__()
        self.multi_head_attention_enc_dec = MultiHeadAttention(hidden_size, total_key_depth, total_value_depth,
                                                               hidden_size, num_heads, None, attention_dropout)
        self.positionwise_feed_forward = PositionwiseFeedForward(hidden_size, filter_size, hidden_size,
                                                                 layer_config='cc', padding='left',
                                                                 dropout=relu_dropout)
        self.dropout = nn.Dropout(layer_dropout)
        self.layer_norm_mha_dec = LayerNorm(hidden_size)
        self.layer_norm_mha_enc = LayerNorm(hidden_size)
        self.layer_norm_ffn = LayerNorm(hidden_size)

    def forward(self, inputs):
        """
        注：inputs 为元组，包含解码器输入、编码器输出、注意力权重、源掩码
        """
        x, encoder_outputs, attention_weight, mask_src = inputs

        # 在进行编码器-解码器注意力前先做层归一化，稳定训练
        x_norm = self.layer_norm_mha_enc(x)

        # 多头编码器-解码器注意力，融合编码器信息
        y, attention_weight = self.multi_head_attention_enc_dec(x_norm, encoder_outputs, encoder_outputs, mask_src)

        # 编码器-解码器注意力后的残差连接并 Dropout
        x = self.dropout(x + y)

        # 层归一化
        x_norm = self.layer_norm_ffn(x)

        # 前馈网络：逐位置线性 + 激活 + 线性
        y = self.positionwise_feed_forward(x_norm)

        # 前馈网络后的残差连接并 Dropout
        y = self.dropout(x + y)

        return y, encoder_outputs, attention_weight


class DecoderLayer(nn.Module):
    """
    Transformer 解码器（Decoder）中的单层结构。
    参考论文：https://arxiv.org/pdf/1706.03762.pdf
    """
    def __init__(self, hidden_size, total_key_depth, total_value_depth, filter_size, num_heads,
                 bias_mask, layer_dropout=0.0, attention_dropout=0.0, relu_dropout=0.0):
        """
        参数:
            hidden_size (int): 隐藏层维度
            total_key_depth (int): Key 张量最后一维维度，需被 num_heads 整除
            total_value_depth (int): Value 张量最后一维维度，需被 num_heads 整除
            filter_size (int): 前馈网络中间层维度
            num_heads (int): 注意力头数量
            bias_mask (Tensor): 屏蔽未来位置的掩码
            layer_dropout (float): 本层残差后的 Dropout 概率
            attention_dropout (float): 注意力权重上的 Dropout 概率
            relu_dropout (float): 前馈网络中 ReLU 后的 Dropout 概率
        """

        super(DecoderLayer, self).__init__()
        self.multi_head_attention_dec = MultiHeadAttention(hidden_size, total_key_depth, total_value_depth,
                                                           hidden_size, num_heads, bias_mask, attention_dropout)
        self.multi_head_attention_enc_dec = MultiHeadAttention(hidden_size, total_key_depth, total_value_depth,
                                                               hidden_size, num_heads, None, attention_dropout)
        self.positionwise_feed_forward = PositionwiseFeedForward(hidden_size, filter_size, hidden_size,
                                                                 layer_config='cc', padding='left',
                                                                 dropout=relu_dropout)
        self.dropout = nn.Dropout(layer_dropout)
        self.layer_norm_mha_dec = LayerNorm(hidden_size)
        self.layer_norm_mha_enc = LayerNorm(hidden_size)
        self.layer_norm_ffn = LayerNorm(hidden_size)

    def forward(self, inputs):
        """
        注：inputs 为元组，包含解码器输入、编码器输出、注意力权重、源掩码
        """
        x, encoder_outputs, attention_weight, mask = inputs
        mask_src, dec_mask = mask

        # 在进行解码器自注意力前先做层归一化，稳定训练
        x_norm = self.layer_norm_mha_dec(x)

        # 掩码多头自注意力：对自身序列进行注意力计算，防止未来信息泄露
        y, _ = self.multi_head_attention_dec(x_norm, x_norm, x_norm, dec_mask)

        # 自注意力后的残差连接并 Dropout
        x = self.dropout(x + y)

        # 在进行编码器-解码器注意力前先做层归一化，稳定训练
        x_norm = self.layer_norm_mha_enc(x)

        # 多头编码器-解码器注意力，融合编码器输出
        y, attention_weight = self.multi_head_attention_enc_dec(x_norm, encoder_outputs, encoder_outputs, mask_src)

        # 编码器-解码器注意力后的残差连接并 Dropout
        x = self.dropout(x + y)

        # 层归一化
        x_norm = self.layer_norm_ffn(x)

        # 前馈网络：逐位置两个线性变换及激活
        y = self.positionwise_feed_forward(x_norm)

        # 前馈网络后的残差连接并 Dropout
        y = self.dropout(x + y)

        return y, encoder_outputs, attention_weight, mask


class MultiHeadAttention(nn.Module):
    """
    多头注意力机制，参考论文：https://arxiv.org/pdf/1706.03762.pdf
    """
    def __init__(self, input_depth, total_key_depth, total_value_depth, output_depth,
                 num_heads, bias_mask=None, dropout=0.0):
        """
        参数:
            input_depth (int): 输入张量最后一维维度
            total_key_depth (int): 键张量最后一维维度，需被 num_heads 整除
            total_value_depth (int): 值张量最后一维维度，需被 num_heads 整除
            output_depth (int): 输出张量最后一维维度
            num_heads (int): 注意力头数量
            bias_mask (Tensor or None): 用于屏蔽未来位置的掩码
            dropout (float): 注意力权重上的 Dropout 概率（训练时非零）
        """
        super(MultiHeadAttention, self).__init__()

        if total_key_depth % num_heads != 0:
            print("Key depth (%d) must be divisible by the number of "
                  "attention heads (%d)." % (total_key_depth, num_heads))
            total_key_depth = total_key_depth - (total_key_depth % num_heads)
        if total_value_depth % num_heads != 0:
            print("Value depth (%d) must be divisible by the number of "
                  "attention heads (%d)." % (total_value_depth, num_heads))
            total_value_depth = total_value_depth - (total_value_depth % num_heads)

        self.num_heads = num_heads
        self.query_scale = (total_key_depth // num_heads) ** -0.5  ## sqrt
        self.bias_mask = bias_mask

        self.query_linear = nn.Linear(input_depth, total_key_depth, bias=False)
        self.key_linear = nn.Linear(input_depth, total_key_depth, bias=False)
        self.value_linear = nn.Linear(input_depth, total_value_depth, bias=False)
        self.output_linear = nn.Linear(total_value_depth, output_depth, bias=False)

        self.emotion_output_linear = nn.Linear(2 * output_depth, output_depth, bias=False)

        self.W_vad = nn.Parameter(torch.FloatTensor(1))

        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x):
        """
        拆分多头：在特征维度上增加 num_heads 维度
        输入:
            x: 形状为 [batch_size, seq_length, depth] 的张量
        返回:
            形状为 [batch_size, num_heads, seq_length, depth/num_heads] 的张量
        """
        if len(x.shape) != 3:
            raise ValueError("x must have rank 3")
        shape = x.shape
        return x.view(shape[0], shape[1], self.num_heads, shape[2] // self.num_heads).permute(0, 2, 1, 3)

    def _merge_heads(self, x):
        """
        合并多头：将 num_heads 维度合并回最后一维
        输入:
            x: 形状为 [batch_size, num_heads, seq_length, depth/num_heads] 的张量
        返回:
            形状为 [batch_size, seq_length, depth] 的张量
        """
        if len(x.shape) != 4:
            raise ValueError("x must have rank 4")
        shape = x.shape
        return x.permute(0, 2, 1, 3).contiguous().view(shape[0], shape[2], shape[3] * self.num_heads)

    def forward(self, queries, keys, values, mask):

        # 对查询、键、值分别进行线性映射
        queries = self.query_linear(queries)
        keys = self.key_linear(keys)
        values = self.value_linear(values)

        # 拆分多头
        queries = self._split_heads(queries)  # (bsz, heads, len, key_depth-20)
        keys = self._split_heads(keys)
        values = self._split_heads(values)

        # 缩放查询向量
        queries *= self.query_scale

        # 计算注意力得分
        logits = torch.matmul(queries, keys.permute(0, 1, 3, 2))  # (bsz, head, tgt_len, src_len)

        if mask is not None:
            mask = mask.unsqueeze(1)  # 扩展掩码维度至 [B, 1, 1, T_values]
            logits = logits.masked_fill(mask, -1e18)

        # 计算整体注意力权重 (平均各头)
        attetion_weights = logits.sum(dim=1) / self.num_heads  # (bsz, tgt_len, src_len)

        # 将得分转换为概率分布
        weights = nn.functional.softmax(logits, dim=-1)  # (bsz, 2, tgt_len, src_len)

        # 对注意力概率进行 Dropout
        weights = self.dropout(weights)

        # 与值向量相乘，获取上下文表示
        contexts = torch.matmul(weights, values)

        # 合并多头，恢复上下文向量形状
        contexts = self._merge_heads(contexts)
        # contexts = torch.tanh(contexts)

        # 线性变换输出
        outputs = self.output_linear(contexts)  # 50 -> 300

        # 返回输出及注意力权重（已归一化）
        return outputs, torch.softmax(attetion_weights, dim=-1)


class Conv(nn.Module):
    """
    卷积类，对形状为 [batch_size, sequence_length, hidden_size] 的输入进行 Padding 和卷积
    """
    def __init__(self, input_size, output_size, kernel_size, pad_type):
        """
        参数:
            input_size (int): 输入特征维度
            output_size (int): 输出特征维度
            kernel_size (int): 卷积核大小
            pad_type (str): 'left' -> 在左侧填充（屏蔽未来数据）；'both' -> 在两侧填充
        """
        super(Conv, self).__init__()
        padding = (kernel_size - 1, 0) if pad_type == 'left' else (kernel_size // 2, (kernel_size - 1) // 2)
        self.pad = nn.ConstantPad1d(padding, 0)
        self.conv = nn.Conv1d(input_size, output_size, kernel_size=kernel_size, padding=0)

    def forward(self, inputs):
        inputs = self.pad(inputs.permute(0, 2, 1))
        outputs = self.conv(inputs).permute(0, 2, 1)
        return outputs


class PositionwiseFeedForward(nn.Module):
    """
    对序列中的每个位置先后执行线性变换 + ReLU + 线性变换
    """
    def __init__(self, input_depth, filter_size, output_depth, layer_config='ll', padding='left', dropout=0.0):
        """
        参数:
            input_depth (int): 输入张量最后一维维度
            filter_size (int): 前馈网络中间层维度
            output_depth (int): 输出张量最后一维维度
            layer_config (str): 'll' -> 全线性结构；'cc' -> 卷积结构
            padding (str): 'left' -> 在左侧填充；'both' -> 在两侧填充
            dropout (float): 前馈网络中 Dropout 概率（训练时非零）
        """
        super(PositionwiseFeedForward, self).__init__()
        layers = []
        sizes = ([(input_depth, filter_size)] +
                 [(filter_size, filter_size)] * (len(layer_config) - 2) +
                 [(filter_size, output_depth)])

        for lc, s in zip(list(layer_config), sizes):
            if lc == 'l':
                layers.append(nn.Linear(*s))
            elif lc == 'c':
                layers.append(Conv(*s, kernel_size=3, pad_type=padding))
            else:
                raise ValueError("Unknown layer type {}".format(lc))

        self.layers = nn.ModuleList(layers)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        x = inputs
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers):
                x = self.relu(x)
                x = self.dropout(x)

        return x


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


def _gen_bias_mask(max_length):
    """
    生成偏置掩码值（-Inf），用于屏蔽注意力中的未来时间步
    """
    np_mask = np.triu(np.full([max_length, max_length], -np.inf), 1)
    torch_mask = torch.from_numpy(np_mask).type(torch.FloatTensor)

    return torch_mask.unsqueeze(0).unsqueeze(1)


def _gen_timing_signal(length, channels, min_timescale=1.0, max_timescale=1.0e4):
    """
    生成形状为 [1, length, channels] 的正余弦时序信号，用于位置编码
    引用自 Tensor2Tensor 实现：
    https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py
    """
    position = np.arange(length)
    num_timescales = channels // 2
    log_timescale_increment = (math.log(float(max_timescale) / float(min_timescale)) / (float(num_timescales) - 1))
    inv_timescales = min_timescale * np.exp(np.arange(num_timescales).astype(float) * -log_timescale_increment)
    scaled_time = np.expand_dims(position, 1) * np.expand_dims(inv_timescales, 0)

    signal = np.concatenate([np.sin(scaled_time), np.cos(scaled_time)], axis=1)
    signal = np.pad(signal, [[0, 0], [0, channels % 2]], 'constant', constant_values=[0.0, 0.0])
    signal = signal.reshape([1, length, channels])

    return torch.from_numpy(signal).type(torch.FloatTensor)


def _get_attn_subsequent_mask(size):
    """
    获取后续位置的注意力掩码，用于阻止访问后续信息
    参数:
        size (int): 序列长度
    返回:
        subsequent_mask (LongTensor)，形状为 [1, size, size]
    """
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    subsequent_mask = torch.from_numpy(subsequent_mask)
    if (config.USE_CUDA):
        return subsequent_mask.cuda()
    else:
        return subsequent_mask

def gen_embeddings(vocab):
    """
    生成初始词向量矩阵。
    若未提供预训练文件或词不在该文件中，则随机初始化向量。
    """
    embeddings = np.random.randn(vocab.n_words, config.emb_dim) * 0.01
    print('Embeddings: %d x %d' % (vocab.n_words, config.emb_dim))
    if config.emb_file is not None:
        print('Loading embedding file: %s' % config.emb_file)
        pre_trained = 0
        with open(config.emb_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                sp = line.split()
                if(len(sp) == config.emb_dim + 1):
                    if sp[0] in vocab.word2index:
                        pre_trained += 1
                        embeddings[vocab.word2index[sp[0]]] = [float(x) for x in sp[1:]]
                else:
                    print(sp[0])
        print('Pre-trained: %d (%.2f%%)' % (pre_trained, pre_trained * 100.0 / vocab.n_words))
    return embeddings

class Embeddings(nn.Module):
    def __init__(self,vocab, d_model, padding_idx=None):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model, padding_idx=padding_idx)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

def share_embedding(vocab, pretrain=True):
    embedding = Embeddings(vocab.n_words, config.emb_dim, padding_idx=config.PAD_idx)
    if(pretrain):
        pre_embedding = gen_embeddings(vocab)
        embedding.lut.weight.data.copy_(torch.FloatTensor(pre_embedding))
        embedding.lut.weight.data.requires_grad = True
    return embedding


class LabelSmoothing(nn.Module):
    """实现标签平滑（Label Smoothing）"""

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.size()[0] > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist)


class NoamOpt:
    """学习率优化封装，根据步数动态调整学习率。"""
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def state_dict(self):
        return self.optimizer.state_dict()

    def step(self):
        """更新模型参数并根据公式调整学习率"""
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        """根据 Noam 学习率公式计算当前学习率"""
        if step is None:
            step = self._step
        return self.factor * \
               (self.model_size ** (-0.5) *
                min(step ** (-0.5), step * self.warmup ** (-1.5)))
