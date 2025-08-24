import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math
from Model.common_layer import EncoderLayer, DecoderLayer, MultiHeadAttention, Conv, PositionwiseFeedForward, LayerNorm , _gen_bias_mask ,_gen_timing_signal, share_embedding, LabelSmoothing, NoamOpt, _get_attn_subsequent_mask
from utils import config
import random
import os
import pprint
from tqdm import tqdm

pp = pprint.PrettyPrinter(indent=1)
import os
import time
from copy import deepcopy
from sklearn.metrics import accuracy_score
import pdb
from utils.common import get_emotion_words

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)


class Semantic_Encoder(nn.Module):
    """
    Transformer 编码器（语义通道）。
    输入形状:
    - inputs: [batch_size, seq_len, emb_dim]
    - mask:   [batch_size, 1, seq_len]，其中 True 表示需要被屏蔽（PAD）的位置

    输出形状:
    - [batch_size, seq_len, hidden_dim]

    参考: Vaswani et al., "Attention Is All You Need"
    """

    def __init__(self, embedding_size, hidden_size, num_layers, num_heads, total_key_depth, total_value_depth,
                 filter_size, max_length=1000, input_dropout=0.0, layer_dropout=0.0,
                 attention_dropout=0.0, relu_dropout=0.0, use_mask=False, universal=False, concept=False):
        """
        参数:
        - embedding_size: 词向量维度
        - hidden_size: 隐藏层维度
        - num_layers: 编码器层数（例如 2）
        - num_heads: 注意力头数（例如 2）
        - total_key_depth: Key 向量维度（需能被 num_heads 整除，例如 40）
        - total_value_depth: Value 向量维度（需能被 num_heads 整除，例如 40）
        - filter_size: 前馈网络中间层维度（例如 50）
        - max_length: 最大序列长度（用于时间/位置信号）
        - input_dropout: 嵌入后的输入端 dropout
        - layer_dropout: 每层的 dropout 概率
        - attention_dropout: 注意力后的 dropout（仅训练时非零）
        - relu_dropout: 前馈网络激活后的 dropout（仅训练时非零）
        - use_mask: 是否启用未来位置掩码
        """

        super(Semantic_Encoder, self).__init__()
        self.universal = universal
        self.num_layers = num_layers
        self.timing_signal = _gen_timing_signal(max_length, hidden_size)

        if (self.universal):
            ## 用于层位置的时序信号
            self.position_signal = _gen_timing_signal(num_layers, hidden_size)

        params = (hidden_size,
                  total_key_depth or hidden_size,
                  total_value_depth or hidden_size,
                  filter_size,
                  num_heads,
                  _gen_bias_mask(max_length) if use_mask else None,
                  layer_dropout,
                  attention_dropout,
                  relu_dropout)

        self.embedding_proj = nn.Linear(embedding_size, hidden_size, bias=False)
        if (self.universal):
            self.enc = EncoderLayer(*params)
        else:
            self.enc = nn.ModuleList([EncoderLayer(*params) for _ in range(num_layers)])

        self.layer_norm = LayerNorm(hidden_size)
        self.input_dropout = nn.Dropout(input_dropout)

    def forward(self, inputs, mask):
        """
        参数:
        - inputs: FloatTensor，形状 [B, L, E]，词向量或其与对话状态嵌入相加的结果
        - mask:   BoolTensor，形状 [B, 1, L]，PAD 掩码

        返回:
        - y: FloatTensor，形状 [B, L, H]，经过若干层 Transformer 与 LayerNorm 的输出
        """
        # 添加输入端 dropout
        x = self.input_dropout(inputs)

        # 映射到隐藏维度
        x = self.embedding_proj(x)

        if (self.universal):
            if (config.act):
                x, (self.remainders, self.n_updates) = self.act_fn(x, inputs, self.enc, self.timing_signal,
                                                                   self.position_signal, self.num_layers)
                y = self.layer_norm(x)
            else:
                for l in range(self.num_layers):
                    x += self.timing_signal[:, :inputs.shape[1], :].type_as(inputs.data)
                    x += self.position_signal[:, l, :].unsqueeze(1).repeat(1, inputs.shape[1], 1).type_as(inputs.data)
                    x = self.enc(x, mask=mask)
                y = self.layer_norm(x)
        else:
            # 添加时间/位置编码
            x += self.timing_signal[:, :inputs.shape[1], :].type_as(inputs.data)

            for i in range(self.num_layers):
                x = self.enc[i](x, mask)

            y = self.layer_norm(x)
        return y


class Emotion_Encoder(nn.Module):
    """
    Transformer 编码器（情感通道）。

    与语义编码器结构相同，用于对情感相关上下文进行表征。
    输入/输出形状同 `Semantic_Encoder`。
    """

    def __init__(self, embedding_size, hidden_size, num_layers, num_heads, total_key_depth, total_value_depth,
                 filter_size, max_length=1000, input_dropout=0.0, layer_dropout=0.0,
                 attention_dropout=0.0, relu_dropout=0.0, use_mask=False, universal=False, concept=False):
        """
        参数:
        - embedding_size: 词向量维度
        - hidden_size: 隐藏层维度
        - num_layers: 编码器层数（例如 2）
        - num_heads: 注意力头数（例如 2）
        - total_key_depth: Key 向量维度（需能被 num_heads 整除，例如 40）
        - total_value_depth: Value 向量维度（需能被 num_heads 整除，例如 40）
        - filter_size: 前馈网络中间层维度（例如 50）
        - max_length: 最大序列长度（用于时间/位置信号）
        - input_dropout: 嵌入后的输入端 dropout
        - layer_dropout: 每层的 dropout 概率
        - attention_dropout: 注意力后的 dropout（仅训练时非零）
        - relu_dropout: 前馈网络激活后的 dropout（仅训练时非零）
        - use_mask: 是否启用未来位置掩码
        """

        super(Emotion_Encoder, self).__init__()
        self.universal = universal
        self.num_layers = num_layers
        self.timing_signal = _gen_timing_signal(max_length, hidden_size)

        if (self.universal):
            ## 用于层位置的时序信号
            self.position_signal = _gen_timing_signal(num_layers, hidden_size)

        params = (hidden_size,
                  total_key_depth or hidden_size,
                  total_value_depth or hidden_size,
                  filter_size,
                  num_heads,
                  _gen_bias_mask(max_length) if use_mask else None,
                  layer_dropout,
                  attention_dropout,
                  relu_dropout)

        self.embedding_proj = nn.Linear(embedding_size, hidden_size, bias=False)
        if (self.universal):
            self.enc = EncoderLayer(*params)
        else:
            self.enc = nn.ModuleList([EncoderLayer(*params) for _ in range(num_layers)])

        self.layer_norm = LayerNorm(hidden_size)
        self.input_dropout = nn.Dropout(input_dropout)

    def forward(self, inputs, mask):
        """
        参数:
        - inputs: FloatTensor，形状 [B, L, E]
        - mask:   BoolTensor，形状 [B, 1, L]

        返回:
        - y: FloatTensor，形状 [B, L, H]
        """
        # 添加输入端 dropout
        x = self.input_dropout(inputs)

        # 映射到隐藏维度
        x = self.embedding_proj(x)

        if (self.universal):
            if (config.act):
                x, (self.remainders, self.n_updates) = self.act_fn(x, inputs, self.enc, self.timing_signal,
                                                                   self.position_signal, self.num_layers)
                y = self.layer_norm(x)
            else:
                for l in range(self.num_layers):
                    x += self.timing_signal[:, :inputs.shape[1], :].type_as(inputs.data)
                    x += self.position_signal[:, l, :].unsqueeze(1).repeat(1, inputs.shape[1], 1).type_as(inputs.data)
                    x = self.enc(x, mask=mask)
                y = self.layer_norm(x)
        else:
            # 添加时间/位置编码
            x += self.timing_signal[:, :inputs.shape[1], :].type_as(inputs.data)

            for i in range(self.num_layers):
                x = self.enc[i](x, mask)

            y = self.layer_norm(x)
        return y


class Decoder(nn.Module):
    """
    Transformer 解码器。

    - 支持通用（universal）与堆叠式两种变体
    - 内部集成未来位置的子序列掩码，确保自回归生成
    """

    def __init__(self, embedding_size, hidden_size, num_layers, num_heads, total_key_depth, total_value_depth,
                 filter_size, max_length=1000, input_dropout=0.0, layer_dropout=0.0,
                 attention_dropout=0.0, relu_dropout=0.0, universal=False):
        """
        参数:
            embedding_size: 词向量维度
            hidden_size: 隐藏层维度
            num_layers: 解码器层数
            num_heads: 多头注意力头数
            total_key_depth: Key 向量维度（需能被 num_heads 整除）
            total_value_depth: Value 向量维度（需能被 num_heads 整除）
            output_depth: 最终输出维度
            filter_size: 前馈网络中间层维度
            max_length: 最大序列长度（用于时间/位置信号）
            input_dropout: 嵌入后的输入端 dropout
            layer_dropout: 每层的 dropout 概率
            attention_dropout: 注意力后的 dropout（仅训练时非零）
            relu_dropout: 前馈网络激活后的 dropout（仅训练时非零）
        """

        super(Decoder, self).__init__()
        self.universal = universal
        self.num_layers = num_layers
        self.timing_signal = _gen_timing_signal(max_length, hidden_size)

        if (self.universal):
            ## for t
            self.position_signal = _gen_timing_signal(num_layers, hidden_size)

        self.mask = _get_attn_subsequent_mask(max_length)

        params = (hidden_size,
                  total_key_depth or hidden_size,
                  total_value_depth or hidden_size,
                  filter_size,
                  num_heads,
                  _gen_bias_mask(max_length),  # mandatory
                  layer_dropout,
                  attention_dropout,
                  relu_dropout)

        if (self.universal):
            self.dec = DecoderLayer(*params)
        else:
            self.dec = nn.Sequential(*[DecoderLayer(*params) for l in range(num_layers)])

        self.embedding_proj = nn.Linear(embedding_size, hidden_size, bias=False)
        self.layer_norm = LayerNorm(hidden_size)
        self.input_dropout = nn.Dropout(input_dropout)

    def forward(self, inputs, encoder_output, mask=None):
        """
        参数:
        - inputs: FloatTensor，形状 [B, T, E]，已嵌入的目标端序列（含起始 token 的情感向量）
        - encoder_output: FloatTensor，形状 [B, S, H]，由双通道编码器拼接得到的上下文表示
        - mask: (mask_src, mask_trg)
          - mask_src: BoolTensor，形状 [B, 1, S]
          - mask_trg: BoolTensor，形状 [B, 1, T]，将与子序列掩码合并

        返回:
        - y: FloatTensor，形状 [B, T, H]，解码器隐藏状态
        - attn_dist: FloatTensor，形状 [B, T, S]，注意力分布（供指针网络使用）
        """
        mask_src, mask_trg = mask
        dec_mask = torch.gt(mask_trg.bool() + self.mask[:, :mask_trg.size(-1), :mask_trg.size(-1)].bool(), 0).to(config.device)
        # 添加输入端 dropout
        x = self.input_dropout(inputs)
        x = self.embedding_proj(x)

        if (self.universal):
            if (config.act):
                x, attn_dist, (self.remainders, self.n_updates) = self.act_fn(x, inputs, self.dec, self.timing_signal,
                                                                              self.position_signal, self.num_layers,
                                                                              encoder_output, decoding=True)
                y = self.layer_norm(x)

            else:
                x += self.timing_signal[:, :inputs.shape[1], :].type_as(inputs.data)
                for l in range(self.num_layers):
                    x += self.position_signal[:, l, :].unsqueeze(1).repeat(1, inputs.shape[1], 1).type_as(inputs.data)
                    x, _, attn_dist, _ = self.dec(
                        (x, encoder_output, [], (mask_src, dec_mask)))
                y = self.layer_norm(x)
        else:
            # 添加时间/位置编码
            x += self.timing_signal[:, :inputs.shape[1], :].type_as(inputs.data)

            # 运行解码器
            y, _, attn_dist, _ = self.dec((x, encoder_output, [], (mask_src, dec_mask)))

            # 最终层归一化
            y = self.layer_norm(y)

        return y, attn_dist


class Generator(nn.Module):
    """
    输出层与（可选）指针生成器。

    - 当 config.pointer_gen=True 时：融合词表分布与注意力分布，支持复制 OOV 词
    - 当为 False 时：普通线性层 + log_softmax
    """
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)
        self.emo_proj = nn.Linear(2 * d_model, vocab)
        self.p_gen_linear = nn.Linear(config.hidden_dim, 1)

    def forward(self, x, attn_dist=None, enc_batch_extend_vocab=None,
                max_oov_length=None, temp=1, beam_search=False, attn_dist_db=None):
        """
        参数:
        - x: FloatTensor，形状 [B, T, H]，解码器输出
        - attn_dist: FloatTensor，形状 [B, T, S]，注意力分布
        - enc_batch_extend_vocab: LongTensor，[B, S]，扩展词表索引（含 OOV 的临时 id）
        - max_oov_length: int，本 batch 的 OOV 表长度
        - temp: 温度系数，用于 softmax 平滑
        - beam_search: 预留参数（未使用）
        - attn_dist_db: 兼容参数（未使用）

        返回:
        - logit: FloatTensor，log 概率分布，形状 [B, T, vocab_size(+oov)]
        """
        if config.pointer_gen:
            p_gen = self.p_gen_linear(x)
            alpha = torch.sigmoid(p_gen)

        logit = self.proj(x)  # x: (批量大小, 目标序列长度, 嵌入维度)

        if config.pointer_gen:
            vocab_dist = F.softmax(logit / temp, dim=2)
            vocab_dist_ = alpha * vocab_dist

            attn_dist = F.softmax(attn_dist / temp, dim=-1)
            attn_dist_ = (1 - alpha) * attn_dist
            enc_batch_extend_vocab_ = torch.cat([enc_batch_extend_vocab.unsqueeze(1)] * x.size(1),1)  ## 扩展到整个目标序列

            extra_zeros = Variable(torch.zeros((logit.size(0), max_oov_length))).to(config.device)
            if extra_zeros is not None:
                extra_zeros = torch.cat([extra_zeros.unsqueeze(1)] * x.size(1), 1)
                vocab_dist_ = torch.cat([vocab_dist_, extra_zeros], 2)

            logit = torch.log(vocab_dist_.scatter_add(2, enc_batch_extend_vocab_, attn_dist_) + 1e-18)

            return logit
        else:
            return F.log_softmax(logit, dim=-1)


class EmpDG_G(nn.Module):
    """
    EmpDG 生成器：包含
    - 语义理解（Semantic_Encoder）
    - 情感感知与识别（Emotion_Encoder + 线性分类器）
    - 以情感向量初始化的解码器（Decoder）
    - 指针生成器（Generator）

    训练目标：
    - 响应生成的 NLL / Label Smoothing Loss
    - 情感识别的交叉熵
    - 对抗训练时引入的判别器损失（可选）
    """
    def __init__(self, vocab, emotion_number, model_file_path=None, is_eval=False, load_optim=False):
        """
        参数:
        - vocab: 词表对象，需包含 `n_words`、`index2word` 等属性
        - emotion_number: 情感类别数（如 32）
        - model_file_path: 可选，权重加载路径
        - is_eval: 是否评估模式（保留参数）
        - load_optim: 是否同时加载优化器状态
        """
        '''
        :param decoder_number: the number of emotion labels, i.e., 32
        '''
        super(EmpDG_G, self).__init__()
        self.vocab = vocab
        self.vocab_size = vocab.n_words

        self.embedding = share_embedding(self.vocab, config.pretrain_emb)
        self.semantic_und = Semantic_Encoder(config.emb_dim, config.hidden_dim, num_layers=config.hop,
                                             num_heads=config.heads,
                                             total_key_depth=config.depth, total_value_depth=config.depth,
                                             filter_size=config.filter, universal=config.universal)
        self.emotion_pec = Emotion_Encoder(config.emb_dim, config.hidden_dim, num_layers=config.hop,
                                           num_heads=config.heads,
                                           total_key_depth=config.depth, total_value_depth=config.depth,
                                           filter_size=config.filter, universal=config.universal)
        self.map_emo = {0: 'surprised', 1: 'excited', 2: 'annoyed', 3: 'proud',
                        4: 'angry', 5: 'sad', 6: 'grateful', 7: 'lonely', 8: 'impressed',
                        9: 'afraid', 10: 'disgusted', 11: 'confident', 12: 'terrified',
                        13: 'hopeful', 14: 'anxious', 15: 'disappointed', 16: 'joyful',
                        17: 'prepared', 18: 'guilty', 19: 'furious', 20: 'nostalgic',
                        21: 'jealous', 22: 'anticipating', 23: 'embarrassed', 24: 'content',
                        25: 'devastated', 26: 'sentimental', 27: 'caring', 28: 'trusting',
                        29: 'ashamed', 30: 'apprehensive', 31: 'faithful'}

        ## 情感信号蒸馏
        self.identify = nn.Linear(config.emb_dim, emotion_number, bias=False)
        self.identify_new = nn.Linear(2*config.emb_dim, emotion_number, bias=False)
        self.activation = nn.Softmax(dim=1)

        ## 解码器
        self.emotion_embedding = nn.Linear(emotion_number, config.emb_dim)
        self.decoder = Decoder(config.emb_dim, hidden_size=config.hidden_dim, num_layers=config.hop,
                               num_heads=config.heads,
                               total_key_depth=config.depth, total_value_depth=config.depth,
                               filter_size=config.filter)

        self.decoder_key = nn.Linear(config.hidden_dim, emotion_number, bias=False)
        self.generator = Generator(config.hidden_dim, self.vocab_size)

        if config.weight_sharing:
            # Share the weight matrix between target word embedding & the final logit dense layer
            self.generator.proj.weight = self.embedding.lut.weight

        self.criterion = nn.NLLLoss(ignore_index=config.PAD_idx)
        if config.label_smoothing:
            self.criterion = LabelSmoothing(size=self.vocab_size, padding_idx=config.PAD_idx, smoothing=0.1)
            self.criterion_ppl = nn.NLLLoss(ignore_index=config.PAD_idx)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=config.lr)
        if config.noam:
            self.optimizer = NoamOpt(config.hidden_dim, 1, 8000,
                                     torch.optim.Adam(self.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

        if model_file_path is not None:
            # 说明：该加载/保存段沿用上游代码键名。注意当前类未显式定义 `self.encoder`，
            # 若需严格对应，请确认权重字典中的键与本类属性一致。
            print("loading weights")
            state = torch.load(model_file_path, map_location=lambda storage, location: storage)
            self.encoder.load_state_dict(state['encoder_state_dict'])
            self.decoder.load_state_dict(state['decoder_state_dict'])
            self.generator.load_state_dict(state['generator_dict'])
            self.embedding.load_state_dict(state['embedding_dict'])
            self.decoder_key.load_state_dict(state['decoder_key_state_dict'])
            if load_optim:
                self.optimizer.load_state_dict(state['optimizer'])
            self.eval()

        self.model_dir = config.save_path
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.best_path = ""

    def save_model(self, running_avg_ppl, iter, f1_g, f1_b, ent_g, ent_b):
        """
        保存模型权重快照与指标。

        参数:
        - running_avg_ppl: 最近的平均困惑度
        - iter: 迭代步编号
        - f1_g, f1_b, ent_g, ent_b: 训练/验证时的若干指标，用于命名

        注意：键名与加载逻辑需对应；当前类未显式定义 `self.encoder`，
        若保持此键名，请确保外部权重字典结构一致。
        """
        state = {
            'iter': iter,
            'encoder_state_dict': self.encoder.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
            'generator_dict': self.generator.state_dict(),
            'decoder_key_state_dict': self.decoder_key.state_dict(),
            'embedding_dict': self.embedding.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'current_loss': running_avg_ppl
        }
        model_save_path = os.path.join(self.model_dir,
                                       'model_{}_{:.4f}_{:.4f}_{:.4f}_{:.4f}_{:.4f}'.format(iter, running_avg_ppl, f1_g,
                                                                                            f1_b, ent_g, ent_b))
        self.best_path = model_save_path
        torch.save(state, model_save_path)

    def train_one_batch(self, batch, iter, train=True, loss_from_d=0.0):
        """
        处理单个 batch 的前向与（可选）反向。

        流程:
        1) 语义编码（加入对话状态嵌入 E_d）得到 C_u
        2) 情感编码得到 C_e，并进行情感识别（交叉熵 + 准确率）
        3) 拼接 C_u 与 C_e 作为源上下文，使用情感向量初始化解码端输入
        4) 解码得到隐藏表示与注意力分布
        5) 指针生成器融合词表分布与注意力分布，计算生成损失
        6) 总损失 = 生成损失 + 情感识别损失 + 对抗损失（可选）

        参数:
        - batch: 字典，包含 context/target/oovs 等张量
        - iter: 训练步，用于命名或日志
        - train: 是否执行反向传播与优化
        - loss_from_d: 判别器传入的损失（对抗训练）

        返回:
        - 若 label_smoothing=True: (loss_ppl, ppl, loss_emotion, emotion_acc)
        - 否则: (loss, ppl, 0, 0)
        """
        enc_batch = batch["context_batch"]
        enc_batch_ext = batch["context_ext_batch"]
        enc_emo_batch = batch['emotion_context_batch']
        enc_emo_batch_ext = batch["emotion_context_ext_batch"]

        oovs = batch["oovs"]
        max_oov_length = len(sorted(oovs, key=lambda i: len(i), reverse=True)[0])

        dec_batch = batch["target_batch"]

        if config.noam:
            self.optimizer.optimizer.zero_grad()
        else:
            self.optimizer.zero_grad()

        ## 语义理解编码
        mask_semantic = enc_batch.data.eq(config.PAD_idx).unsqueeze(1)  # (bsz, src_len)->(bsz, 1, src_len)
        sem_emb_mask = self.embedding(batch["mask_context"])  # dialogue state  E_d
        sem_emb = self.embedding(enc_batch) + sem_emb_mask  # E_w+E_d
        sem_encoder_outputs = self.semantic_und(sem_emb, mask_semantic)  # C_u  (bsz, sem_w_len, emb_dim)

        ## 多粒度情感感知（理解与识别）
        mask_emotion = enc_emo_batch.data.eq(config.PAD_idx).unsqueeze(1)
        # emo_emb_mask = self.embedding(batch["mask_emotion_context"])
        # emo_emb = self.embedding(enc_emo_batch) + emo_emb_mask
        emo_encoder_outputs = self.emotion_pec(self.embedding(enc_emo_batch), mask_emotion)  # C_e  (bsz, emo_w_len, emb_dim)

        # emotion_logit = self.identify(emo_encoder_outputs[:,0,:])  # (bsz, emotion_number)
        emotion_logit = self.identify_new(torch.cat((emo_encoder_outputs[:,0,:],sem_encoder_outputs[:,0,:]), dim=-1))  # (bsz, emotion_number)
        loss_emotion = nn.CrossEntropyLoss(reduction='sum')(emotion_logit, batch['emotion_label'])
        pred_emotion = np.argmax(emotion_logit.detach().cpu().numpy(), axis=1)
        emotion_acc = accuracy_score(batch["emotion_label"].cpu().numpy(), pred_emotion)


        ## 合并两路上下文
        src_emb = torch.cat((sem_encoder_outputs, emo_encoder_outputs), dim=1)  # (bsz, src_len, emb_dim)
        mask_src = torch.cat((mask_semantic, mask_emotion), dim=2)  # (bsz, 1, src_len)

        ## 共情回复生成
        sos_emb = self.emotion_embedding(emotion_logit).unsqueeze(1)
        dec_emb = self.embedding(dec_batch[:, :-1])
        dec_emb = torch.cat((sos_emb, dec_emb), dim=1)  # (bsz, 1+tgt_len, emb_dim)

        mask_trg = dec_batch.data.eq(config.PAD_idx).unsqueeze(1)
        # inputs, encoder_output, pred_emotion=None, emotion_contexts=None, mask=None
        pre_logit, attn_dist = self.decoder(dec_emb, src_emb, (mask_src, mask_trg))

        ## 计算输出分布
        enc_ext_batch = torch.cat((enc_batch_ext, enc_emo_batch_ext), dim=1)
        logit = self.generator(pre_logit, attn_dist, enc_ext_batch if config.pointer_gen else None,
                               max_oov_length, attn_dist_db=None)
        # logit = F.log_softmax(logit,dim=-1) #fix the name later
        # 损失：指针网络用负对数似然，否则交叉熵
        loss = self.criterion(logit.contiguous().view(-1, logit.size(-1)), dec_batch.contiguous().view(-1))

        loss += loss_emotion
        loss += loss_from_d

        if config.label_smoothing:
            loss_ppl = self.criterion_ppl(logit.contiguous().view(-1, logit.size(-1)),
                                          dec_batch.contiguous().view(-1)).item()

        if train:
            loss.backward()
            self.optimizer.step()

        if config.label_smoothing:
            return loss_ppl, math.exp(min(loss_ppl, 100)), loss_emotion.item(), emotion_acc
        else:
            return loss.item(), math.exp(min(loss.item(), 100)), 0, 0

    def compute_act_loss(self, module):
        """
        计算 ACT（Adaptive Computation Time）损失，用于 Universal Transformer 变体。

        参数:
        - module: 含有 `remainders` 与 `n_updates` 属性的模块

        返回:
        - 标量 float，按批次平均的惩罚项
        """
        R_t = module.remainders
        N_t = module.n_updates
        p_t = R_t + N_t
        avg_p_t = torch.sum(torch.sum(p_t, dim=1) / p_t.size(1)) / p_t.size(0)
        loss = config.act_loss_weight * avg_p_t.item()
        return loss

    def decoder_greedy(self, batch, max_dec_step=30):
        """
        贪心解码（多步，逐 token 选择最大概率）。

        典型用法：交互式推理或调试，batch size 一般为 1。

        参数:
        - batch: 与训练时相同的输入字典
        - max_dec_step: 最大解码步数（含 <EOS> 截止）

        返回:
        - sent: List[str]，每个样本对应的生成句子
        """
        enc_batch_ext, extra_zeros = None, None
        enc_batch = batch["context_batch"]
        enc_batch_ext = batch["context_ext_batch"]
        enc_emo_batch = batch['emotion_context_batch']
        enc_emo_batch_ext = batch["emotion_context_ext_batch"]

        oovs = batch["oovs"]
        max_oov_length = len(sorted(oovs, key=lambda i: len(i), reverse=True)[0])

        if config.noam:
            self.optimizer.optimizer.zero_grad()
        else:
            self.optimizer.zero_grad()

        ## 语义理解编码
        mask_semantic = enc_batch.data.eq(config.PAD_idx).unsqueeze(1)  # (bsz, src_len)->(bsz, 1, src_len)
        sem_emb_mask = self.embedding(batch["mask_context"])  # dialogue state  E_d
        sem_emb = self.embedding(enc_batch) + sem_emb_mask  # E_w+E_d
        sem_encoder_outputs = self.semantic_und(sem_emb, mask_semantic)  # C_u  (bsz, sem_w_len, emb_dim)

        # Multi-resolution Emotion Perception (understanding & predicting)
        mask_emotion = enc_emo_batch.data.eq(config.PAD_idx).unsqueeze(1)
        # emo_emb_mask = self.embedding(batch["mask_emotion_context"])
        # emo_emb = self.embedding(enc_emo_batch) + emo_emb_mask
        emo_encoder_outputs = self.emotion_pec(self.embedding(enc_emo_batch), mask_emotion)  # C_e  (bsz, emo_w_len, emb_dim)

        ## Identify
        # emotion_logit = self.identify(emo_encoder_outputs[:,0,:])  # (bsz, emotion_number)
        emotion_logit = self.identify_new(torch.cat((emo_encoder_outputs[:,0,:],sem_encoder_outputs[:,0,:]), dim=-1))  # (bsz, emotion_number)

        ## Combine Two Contexts
        src_emb = torch.cat((sem_encoder_outputs, emo_encoder_outputs), dim=1)  # (bsz, src_len, emb_dim)
        mask_src = torch.cat((mask_semantic, mask_emotion), dim=2)  # (bsz, 1, src_len)
        enc_ext_batch = torch.cat((enc_batch_ext, enc_emo_batch_ext), dim=1)

        ys = torch.ones(1, 1).fill_(config.SOS_idx).long()
        ys_emb = self.emotion_embedding(emotion_logit).unsqueeze(1)  # (bsz, 1, emb_dim)
        if config.USE_CUDA:
            ys = ys.cuda()
        mask_trg = ys.data.eq(config.PAD_idx).unsqueeze(1)
        decoded_words = []
        for i in range(max_dec_step + 1):
            if config.project:
                out, attn_dist = self.decoder(self.embedding_proj_in(ys_emb), self.embedding_proj_in(src_emb),
                                              (mask_src, mask_trg))
            else:
                out, attn_dist = self.decoder(ys_emb, src_emb, (mask_src, mask_trg))

            prob = self.generator(out, attn_dist, enc_ext_batch, max_oov_length, attn_dist_db=None)
            # logit = F.log_softmax(logit,dim=-1) #fix the name later
            # 过滤采样示例：对分布做 top-k/top-p 过滤后再采样
            # filtered_logit = top_k_top_p_filtering(logit[:, -1], top_k=0, top_p=0, filter_value=-float('Inf'))
            # next_word = torch.multinomial(F.softmax(filtered_logit, dim=-1), 1).squeeze()
            _, next_word = torch.max(prob[:, -1], dim=1)

            batch_words = []
            for i_batch, ni in enumerate(next_word.view(-1)):
                if ni.item() == config.EOS_idx:
                    batch_words.append('<EOS>')
                elif ni.item() in self.vocab.index2word:
                    batch_words.append(self.vocab.index2word[ni.item()])
                else:
                    batch_words.append(oovs[i_batch][ni.item() - self.vocab.n_words])
                    next_word[i_batch] = config.UNK_idx
            decoded_words.append(batch_words)
            next_word = next_word.data[0]

            if config.USE_CUDA:
                ys = torch.cat([ys, torch.ones(1, 1).long().fill_(next_word).cuda()], dim=1)
                ys = ys.cuda()
                ys_emb = torch.cat((ys_emb, self.embedding(torch.ones(1, 1).long().fill_(next_word).cuda())), dim=1)
            else:
                ys = torch.cat([ys, torch.ones(1, 1).long().fill_(next_word)], dim=1)
                ys_emb = torch.cat((ys_emb, self.embedding(torch.ones(1, 1).long().fill_(next_word))), dim=1)
            mask_trg = ys.data.eq(config.PAD_idx).unsqueeze(1)

        sent = []
        for _, row in enumerate(np.transpose(decoded_words)):
            st = ''
            for e in row:
                if e == '<EOS>':
                    break
                else:
                    st += e + ' '
            sent.append(st)
        return sent

    def predict(self, batch, max_dec_step=30):
        """
        批量贪心推理，接口与 `decoder_greedy` 类似，但支持 B>1。

        参数:
        - batch: 输入字典
        - max_dec_step: 最大解码步数

        返回:
        - sent: List[str]，每个样本一条生成文本
        """
        enc_batch_ext, extra_zeros = None, None
        enc_batch = batch["context_batch"]
        enc_batch_ext = batch["context_ext_batch"]
        enc_emo_batch = batch['emotion_context_batch']
        enc_emo_batch_ext = batch["emotion_context_ext_batch"]

        oovs = batch["oovs"]
        max_oov_length = len(sorted(oovs, key=lambda i: len(i), reverse=True)[0])

        ## 语义理解编码
        mask_semantic = enc_batch.data.eq(config.PAD_idx).unsqueeze(1)  # (bsz, src_len)->(bsz, 1, src_len)
        sem_emb_mask = self.embedding(batch["mask_context"])  # dialogue state  E_d
        sem_emb = self.embedding(enc_batch) + sem_emb_mask  # E_w+E_d
        sem_encoder_outputs = self.semantic_und(sem_emb, mask_semantic)  # C_u  (bsz, sem_w_len, emb_dim)

        # Multi-resolution Emotion Perception (understanding & predicting)
        mask_emotion = enc_emo_batch.data.eq(config.PAD_idx).unsqueeze(1)
        # emo_emb_mask = self.embedding(batch["mask_emotion_context"])
        # emo_emb = self.embedding(enc_emo_batch) + emo_emb_mask
        emo_encoder_outputs = self.emotion_pec(self.embedding(enc_emo_batch), mask_emotion)  # C_e  (bsz, emo_w_len, emb_dim)

        ## Identify
        # emotion_logit = self.identify(emo_encoder_outputs[:,0,:])  # (bsz, emotion_number)
        emotion_logit = self.identify_new(torch.cat((emo_encoder_outputs[:,0,:],sem_encoder_outputs[:,0,:]), dim=-1))  # (bsz, emotion_number)

        ## Combine Two Contexts
        src_emb = torch.cat((sem_encoder_outputs, emo_encoder_outputs), dim=1)  # (bsz, src_len, emb_dim)
        mask_src = torch.cat((mask_semantic, mask_emotion), dim=2)  # (bsz, 1, src_len)
        enc_ext_batch = torch.cat((enc_batch_ext, enc_emo_batch_ext), dim=1)

        ys = torch.ones(enc_batch.size(0), 1).fill_(config.SOS_idx).long()
        ys_emb = self.emotion_embedding(emotion_logit).unsqueeze(1)  # (bsz, 1, emb_dim)
        if config.USE_CUDA:
            ys = ys.cuda()
        mask_trg = ys.data.eq(config.PAD_idx).unsqueeze(1)
        decoded_words = []
        for i in range(max_dec_step + 1):
            if config.project:
                out, attn_dist = self.decoder(self.embedding_proj_in(ys_emb), self.embedding_proj_in(src_emb),
                                              (mask_src, mask_trg))
            else:
                out, attn_dist = self.decoder(ys_emb, src_emb, (mask_src, mask_trg))

            prob = self.generator(out, attn_dist, enc_ext_batch, max_oov_length, attn_dist_db=None)
            # logit = F.log_softmax(logit,dim=-1) #fix the name later
            # filtered_logit = top_k_top_p_filtering(logit[:, -1], top_k=0, top_p=0, filter_value=-float('Inf'))
            # Sample from the filtered distribution
            # next_word = torch.multinomial(F.softmax(filtered_logit, dim=-1), 1).squeeze()
            _, next_word = torch.max(prob[:, -1], dim=1)
            decoded_words.append(['<EOS>' if ni.item() == config.EOS_idx else self.vocab.index2word[ni.item()] for ni in
                                  next_word.view(-1)])
            next_word = next_word.data[0]

            if config.USE_CUDA:
                ys = torch.cat([ys, torch.ones(enc_batch.size(0), 1).long().fill_(next_word).cuda()], dim=1)
                ys = ys.cuda()
                ys_emb = torch.cat((ys_emb, self.embedding(torch.ones(enc_batch.size(0), 1).long().fill_(next_word).cuda())), dim=1)
            else:
                ys = torch.cat([ys, torch.ones(enc_batch.size(0), 1).long().fill_(next_word)], dim=1)
                ys_emb = torch.cat((ys_emb, self.embedding(torch.ones(enc_batch.size(0), 1).long().fill_(next_word))), dim=1)
            mask_trg = ys.data.eq(config.PAD_idx).unsqueeze(1)

        sent = []
        for _, row in enumerate(np.transpose(decoded_words)):
            st = ''
            for e in row:
                if e == '<EOS>':
                    break
                else:
                    st += e + ' '
            sent.append(st)
        return sent

    def g_for_d(self, batch, max_dec_step=30):
        """
        为判别器（D）生成样本：
        - 返回分词后的句子（不拼接）
        - 从句子中抽取情感词（启发式）
        - 同时返回两种上下文的首 token 表示，供 D 使用

        参数:
        - batch: 输入字典
        - max_dec_step: 最大解码步数

        返回:
        - sent: List[List[str]]，每个样本的分词列表
        - sent_emo: List[List[str]]，对应样本的情感词列表
        - sem_first: FloatTensor [B, H]，语义上下文首 token 表示
        - emo_first: FloatTensor [B, H]，情感上下文首 token 表示
        """
        enc_batch_ext, extra_zeros = None, None
        enc_batch = batch["context_batch"]
        enc_batch_ext = batch["context_ext_batch"]
        enc_emo_batch = batch['emotion_context_batch']
        enc_emo_batch_ext = batch["emotion_context_ext_batch"]

        oovs = batch["oovs"]
        max_oov_length = len(sorted(oovs, key=lambda i: len(i), reverse=True)[0])

        ## 语义理解编码
        mask_semantic = enc_batch.data.eq(config.PAD_idx).unsqueeze(1)  # (bsz, src_len)->(bsz, 1, src_len)
        sem_emb_mask = self.embedding(batch["mask_context"])  # dialogue state  E_d
        sem_emb = self.embedding(enc_batch) + sem_emb_mask  # E_w+E_d
        sem_encoder_outputs = self.semantic_und(sem_emb, mask_semantic)  # C_u  (bsz, sem_w_len, emb_dim)

        # Multi-resolution Emotion Perception (understanding & predicting)
        mask_emotion = enc_emo_batch.data.eq(config.PAD_idx).unsqueeze(1)
        # emo_emb_mask = self.embedding(batch["mask_emotion_context"])
        # emo_emb = self.embedding(enc_emo_batch) + emo_emb_mask
        emo_encoder_outputs = self.emotion_pec(self.embedding(enc_emo_batch), mask_emotion)  # C_e  (bsz, emo_w_len, emb_dim)

        ## Identify
        # emotion_logit = self.identify(emo_encoder_outputs[:,0,:])  # (bsz, emotion_number)
        emotion_logit = self.identify_new(torch.cat((emo_encoder_outputs[:,0,:],sem_encoder_outputs[:,0,:]), dim=-1))  # (bsz, emotion_number)

        ## Combine Two Contexts
        src_emb = torch.cat((sem_encoder_outputs, emo_encoder_outputs), dim=1)  # (bsz, src_len, emb_dim)
        mask_src = torch.cat((mask_semantic, mask_emotion), dim=2)  # (bsz, 1, src_len)
        enc_ext_batch = torch.cat((enc_batch_ext, enc_emo_batch_ext), dim=1)

        ys = torch.ones(enc_batch.size(0), 1).fill_(config.SOS_idx).long()
        ys_emb = self.emotion_embedding(emotion_logit).unsqueeze(1)  # (bsz, 1, emb_dim)
        if config.USE_CUDA:
            ys = ys.cuda()
        mask_trg = ys.data.eq(config.PAD_idx).unsqueeze(1)
        decoded_words = []
        for i in range(max_dec_step + 1):
            if config.project:
                out, attn_dist = self.decoder(self.embedding_proj_in(ys_emb), self.embedding_proj_in(src_emb),
                                              (mask_src, mask_trg))
            else:
                out, attn_dist = self.decoder(ys_emb, src_emb, (mask_src, mask_trg))

            prob = self.generator(out, attn_dist, enc_ext_batch, max_oov_length, attn_dist_db=None)
            # logit = F.log_softmax(logit,dim=-1) #fix the name later
            # filtered_logit = top_k_top_p_filtering(logit[:, -1], top_k=0, top_p=0, filter_value=-float('Inf'))
            # Sample from the filtered distribution
            # next_word = torch.multinomial(F.softmax(filtered_logit, dim=-1), 1).squeeze()
            _, next_word = torch.max(prob[:, -1], dim=1)
            decoded_words.append(['<EOS>' if ni.item() == config.EOS_idx else self.vocab.index2word[ni.item()] for ni in
                                  next_word.view(-1)])
            next_word = next_word.data[0]

            if config.USE_CUDA:
                ys = torch.cat([ys, torch.ones(enc_batch.size(0), 1).long().fill_(next_word).cuda()], dim=1)
                ys = ys.cuda()
                ys_emb = torch.cat((ys_emb, self.embedding(torch.ones(enc_batch.size(0), 1).long().fill_(next_word).cuda())), dim=1)
            else:
                ys = torch.cat([ys, torch.ones(enc_batch.size(0), 1).long().fill_(next_word)], dim=1)
                ys_emb = torch.cat((ys_emb, self.embedding(torch.ones(enc_batch.size(0), 1).long().fill_(next_word))), dim=1)
            mask_trg = ys.data.eq(config.PAD_idx).unsqueeze(1)

        sent = []
        sent_emo = []
        for _, row in enumerate(np.transpose(decoded_words)):
            st = []
            for e in row:
                if e == '<EOS>':
                    break
                else:
                    st.append(e)
            sent.append(st)
            sent_emo.append(get_emotion_words(st))
        return sent, sent_emo, sem_encoder_outputs[:, 0, :], emo_encoder_outputs[:, 0, :]
