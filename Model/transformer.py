"""
本文件实现了基于 Transformer 的对话生成模型（含指针生成机制）。

主要组成部分：
- Encoder: 将输入序列编码为上下文表示；支持通用（共享）层或多层堆叠；支持位置/时间编码与掩码。
- Decoder: 自回归式解码生成目标序列；内部包含自注意力、交叉注意力、前馈网络与遮罩。
- Generator: 将解码器隐状态映射为词表分布；可选指针网络，将注意力分布与词表分布融合以复制源词。
- Transformer: 封装完整训练/推断流程，负责前向计算、损失计算、模型保存与贪心/Top-k 解码。

关键特性：
- 支持词向量共享（weight sharing）。
- 可选 Noam 学习率调度与 Label Smoothing。
- 指针生成（pointer-generator）用于 OOV 复制。
- 通过掩码屏蔽 PAD 与未来位以保证自回归性质。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import math
from Model.common_layer import EncoderLayer, DecoderLayer, MultiHeadAttention, Conv, PositionwiseFeedForward, LayerNorm , _gen_bias_mask ,_gen_timing_signal, share_embedding, LabelSmoothing, NoamOpt, _get_attn_subsequent_mask
from utils import config
import random
# from numpy import random
import os
import pprint
from tqdm import tqdm
pp = pprint.PrettyPrinter(indent=1)
import os
import time
from copy import deepcopy
from sklearn.metrics import accuracy_score
import pdb


torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)


class Encoder(nn.Module):
    """
    Transformer 编码器模块。
    功能:
    - 将输入的词向量先做丢弃与线性映射到 hidden_size
    - 叠加时间/位置信号，经由若干编码层提取上下文表示
    - 最后做层归一化
    参数:
    - embedding_size: 词向量维度
    - hidden_size: 隐层/模型维度
    - num_layers: 编码层数
    - num_heads: 注意力头数
    - total_key_depth/total_value_depth: 多头注意力键/值聚合后的总维度
    - filter_size: 前馈网络内部中间层维度
    - max_length: 最大序列长度（用于生成时间信号）
    - input_dropout/layer_dropout/attention_dropout/relu_dropout: 各处的 dropout 概率
    - use_mask: 是否对未来位进行遮罩
    - universal: 是否使用“共享同一层多次迭代”的通用 Transformer 设置
    """
    def __init__(self, embedding_size, hidden_size, num_layers, num_heads, total_key_depth, total_value_depth,
                 filter_size, max_length=1000, input_dropout=0.0, layer_dropout=0.0, 
                 attention_dropout=0.0, relu_dropout=0.0, use_mask=False, universal=False, concept=False):
        # 参数说明见类 docstring。下方实现按照是否 universal 选择编码层结构。
        
        super(Encoder, self).__init__()
        self.universal = universal
        self.num_layers = num_layers
        # 生成时间（位置）信号，用于为序列位置加入位置信息
        self.timing_signal = _gen_timing_signal(max_length, hidden_size)
        
        if(self.universal):  
            # 当使用共享层时，还会为“层索引”生成一组时间信号
            self.position_signal = _gen_timing_signal(num_layers, hidden_size)

        params =(hidden_size, 
                 total_key_depth or hidden_size,
                 total_value_depth or hidden_size,
                 filter_size, 
                 num_heads, 
                 _gen_bias_mask(max_length) if use_mask else None,
                 layer_dropout, 
                 attention_dropout, 
                 relu_dropout)
        
        # 将输入 embedding 投影到 hidden_size
        self.embedding_proj = nn.Linear(embedding_size, hidden_size, bias=False)
        if(self.universal):
            self.enc = EncoderLayer(*params)
        else:
            # 多层独立的编码层堆叠
            self.enc = nn.ModuleList([EncoderLayer(*params) for _ in range(num_layers)])
        
        self.layer_norm = LayerNorm(hidden_size)
        self.input_dropout = nn.Dropout(input_dropout)


    def forward(self, inputs, mask):
        """
        前向计算流程：
        1) 对输入做 dropout 并线性投影
        2) 加入时间/位置编码
        3) 通过编码层（共享或堆叠）计算
        4) 层归一化得到输出
        参数:
        - inputs: [batch, src_len, embedding_size]
        - mask: [batch, 1, src_len]，为 True 的位置将被 attention 层遮罩
        返回:
        - y: [batch, src_len, hidden_size]
        """
        # Add input dropout
        x = self.input_dropout(inputs)
        
        # Project to hidden size
        x = self.embedding_proj(x)
        
        if(self.universal):
            if(config.act):
                # 若使用 ACT（Adaptive Computation Time），enc 层将按动态步数执行
                x, (self.remainders, self.n_updates) = self.act_fn(x, inputs, self.enc, self.timing_signal, self.position_signal, self.num_layers)
                y = self.layer_norm(x)
            else:
                # 逐层重复使用同一层权重，并叠加层索引位置信号
                for l in range(self.num_layers):
                    x += self.timing_signal[:, :inputs.shape[1], :].type_as(inputs.data)
                    x += self.position_signal[:, l, :].unsqueeze(1).repeat(1,inputs.shape[1],1).type_as(inputs.data)
                    x = self.enc(x, mask=mask)
                y = self.layer_norm(x)
        else:
            # Add timing signal
            x += self.timing_signal[:, :inputs.shape[1], :].type_as(inputs.data)
            
            # 多层独立编码器
            for i in range(self.num_layers):
                x = self.enc[i](x, mask)
        
            y = self.layer_norm(x)
        return y


class Decoder(nn.Module):
    """
    Transformer 解码器模块。
    功能:
    - 自回归式地根据已生成的目标序列与编码器输出，预测下一个词的分布
    - 内部包含自注意力（带未来位遮罩）、交叉注意力（对编码器输出）、前馈网络
    参数:
    - embedding_size/hidden_size/num_layers/num_heads 等含义与编码器一致
    - max_length: 最大目标序列长度，用于构造后续位遮罩矩阵
    - universal: 是否使用共享层
    """
    def __init__(self, embedding_size, hidden_size, num_layers, num_heads, total_key_depth, total_value_depth,
                 filter_size, max_length=1000, input_dropout=0.0, layer_dropout=0.0, 
                 attention_dropout=0.0, relu_dropout=0.0, universal=False):
        # 解码器构造，包含生成后续位遮罩所需的下三角矩阵。
        
        super(Decoder, self).__init__()
        self.universal = universal
        self.num_layers = num_layers
        self.timing_signal = _gen_timing_signal(max_length, hidden_size)
        
        if(self.universal):  
            # 用于共享层时的“层位置”信号
            self.position_signal = _gen_timing_signal(num_layers, hidden_size)

        # 未来位遮罩：确保解码只能看到已生成的历史
        self.mask = _get_attn_subsequent_mask(max_length)

        params =(hidden_size, 
                 total_key_depth or hidden_size,
                 total_value_depth or hidden_size,
                 filter_size, 
                 num_heads, 
                 _gen_bias_mask(max_length), # mandatory
                 layer_dropout, 
                 attention_dropout, 
                 relu_dropout)
        
        if(self.universal):
            self.dec = DecoderLayer(*params)
        else:
            # 多层解码器结构
            self.dec = nn.Sequential(*[DecoderLayer(*params) for l in range(num_layers)])
        
        self.embedding_proj = nn.Linear(embedding_size, hidden_size, bias=False)
        self.layer_norm = LayerNorm(hidden_size)
        self.input_dropout = nn.Dropout(input_dropout)


    def forward(self, inputs, encoder_output, mask):
        """
        前向计算：
        - 构造解码端遮罩 dec_mask = padding_mask OR subsequent_mask
        - 对输入做丢弃和线性映射，叠加时间（位置）信号
        - 通过解码层（共享/堆叠），返回解码隐状态与注意力分布
        参数:
        - inputs: [batch, tgt_len, embedding_size]
        - encoder_output: [batch, src_len, hidden_size]
        - mask: (mask_src, mask_trg)，分别对应源端与目标端的 padding 遮罩
        返回:
        - y: [batch, tgt_len, hidden_size]
        - attn_dist: [batch, tgt_len, src_len] 与源的注意力权重（供指针生成使用）
        """
        mask_src, mask_trg = mask
        # subsequent mask + padding mask，确保自回归与 PAD 位置不被关注
        dec_mask = torch.gt(mask_trg.bool() + self.mask[:, :mask_trg.size(-1), :mask_trg.size(-1)].bool(), 0)
        # Add input dropout
        x = self.input_dropout(inputs)
        x = self.embedding_proj(x)
            
        if(self.universal):
            if(config.act):
                x, attn_dist, (self.remainders,self.n_updates) = self.act_fn(x, inputs, self.dec, self.timing_signal, self.position_signal, self.num_layers, encoder_output, decoding=True)
                y = self.layer_norm(x)

            else:
                x += self.timing_signal[:, :inputs.shape[1], :].type_as(inputs.data)
                for l in range(self.num_layers):
                    x += self.position_signal[:, l, :].unsqueeze(1).repeat(1,inputs.shape[1],1).type_as(inputs.data)
                    x, _, attn_dist, _ = self.dec((x, encoder_output, [], (mask_src,dec_mask)))
                y = self.layer_norm(x)
        else:
            # Add timing signal
            x += self.timing_signal[:, :inputs.shape[1], :].type_as(inputs.data)
            
            # Run decoder
            y, _, attn_dist, _ = self.dec((x, encoder_output, [], (mask_src,dec_mask)))

            # Final layer normalization
            y = self.layer_norm(y)
        return y, attn_dist


class Generator(nn.Module):
    """将解码器隐状态映射为词表对数概率的生成器。
    - 若启用指针生成（pointer_gen=True），则按公式融合词表分布与注意力分布，实现从源序列“复制”。
    - 否则采用标准的线性映射 + log softmax。
    参数:
    - d_model: 隐状态维度
    - vocab: 词表大小
    """
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)
        self.p_gen_linear = nn.Linear(config.hidden_dim, 1)

    def forward(self, x, attn_dist=None, enc_batch_extend_vocab=None, max_oov_length=None, temp=1, beam_search=False, attn_dist_db=None):
        """
        将隐状态 x 转换为对数概率：
        - pointer_gen=True 时：
          1) 计算复制门 p_gen=σ(Linear(x))
          2) vocab_dist = softmax(Wx/temp)
          3) attn_dist = softmax(attn_dist/temp)
          4) 融合: logit = log( p_gen * vocab_dist + (1-p_gen) * scatter(attn_dist, enc_ext_idx) )
        - pointer_gen=False 时：直接返回 log_softmax(Wx)
        参数:
        - x: [batch, tgt_len, hidden]
        - attn_dist: [batch, tgt_len, src_len]，来自解码器与 encoder 的注意力
        - enc_batch_extend_vocab: [batch, src_len]，将源 token 映射到扩展词表的索引
        - max_oov_length: OOV 列表最大长度（用于扩展词表大小）
        - temp: 温度系数
        返回:
        - logit: [batch, tgt_len, vocab(+oov)] 的对数概率
        """

        if config.pointer_gen:
            p_gen = self.p_gen_linear(x)
            alpha = torch.sigmoid(p_gen)

        logit = self.proj(x)

        if config.pointer_gen:
            vocab_dist = F.softmax(logit/temp, dim=2)
            vocab_dist_ = alpha * vocab_dist

            attn_dist = F.softmax(attn_dist/temp, dim=-1)
            attn_dist_ = (1 - alpha) * attn_dist            
            # 将注意力分布根据源位置索引 scatter 到扩展词表维度
            enc_batch_extend_vocab_ = torch.cat([enc_batch_extend_vocab.unsqueeze(1)]*x.size(1),1)

            logit = torch.log(vocab_dist_.scatter_add(2, enc_batch_extend_vocab_, attn_dist_))

            return logit
        else:
            return F.log_softmax(logit, dim=-1)


class Transformer(nn.Module):
    """
    对话生成用的 Transformer 模型封装。
    组成：共享的 embedding、编码器、解码器与生成器；包含训练/保存与解码（贪心、Top-k）方法。
    依赖的超参数来自 `utils.config`。
    """
    def __init__(self, vocab, decoder_number,  model_file_path=None, is_eval=False, load_optim=False):
        super(Transformer, self).__init__()
        self.vocab = vocab
        self.vocab_size = vocab.n_words

        # 共享词嵌入（可选择加载预训练词向量）
        self.embedding = share_embedding(self.vocab,config.pretrain_emb)
        # 编码器：对上下文序列进行表征
        self.encoder = Encoder(config.emb_dim, config.hidden_dim, num_layers=config.hop, num_heads=config.heads, 
                                total_key_depth=config.depth, total_value_depth=config.depth,
                                filter_size=config.filter,universal=config.universal)


        # 单一解码器（若需多解码器，可扩展此处）
        self.decoder = Decoder(config.emb_dim, hidden_size = config.hidden_dim,  num_layers=config.hop, num_heads=config.heads, 
                                    total_key_depth=config.depth,total_value_depth=config.depth,
                                    filter_size=config.filter)
        
        # 根据解码器选择的权重（若使用多解码器场景）
        self.decoder_key = nn.Linear(config.hidden_dim ,decoder_number, bias=False)
        self.generator = Generator(config.hidden_dim, self.vocab_size)

        if config.weight_sharing:
            # 词嵌入与输出层权重共享
            self.generator.proj.weight = self.embedding.lut.weight

        # 训练目标：默认 NLLLoss；若启用 label smoothing，替换为平滑损失
        self.criterion = nn.NLLLoss(ignore_index=config.PAD_idx)
        if config.label_smoothing:
            self.criterion = LabelSmoothing(size=self.vocab_size, padding_idx=config.PAD_idx, smoothing=0.1)
            self.criterion_ppl = nn.NLLLoss(ignore_index=config.PAD_idx)
        
        # 优化器：可选 Noam 学习率调度
        self.optimizer = torch.optim.Adam(self.parameters(), lr=config.lr)
        if config.noam:
            self.optimizer = NoamOpt(config.hidden_dim, 1, 8000, torch.optim.Adam(self.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

        # 可选加载已有模型
        if model_file_path is not None:
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

    def save_model(self, running_avg_ppl, iter, f1_g,f1_b,ent_g,ent_b):
        """
        将当前模型权重与训练状态保存到磁盘。
        文件名包含迭代步与若干指标，便于对比与回溯。
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
        model_save_path = os.path.join(self.model_dir, 'model_{}_{:.4f}_{:.4f}_{:.4f}_{:.4f}_{:.4f}'.format(iter,running_avg_ppl,f1_g,f1_b,ent_g,ent_b) )
        self.best_path = model_save_path
        torch.save(state, model_save_path)


    def train_one_batch(self, batch, iter, train=True):
        """
        处理一个 batch 的训练/评估前向与损失计算。
        流程：
        - 构造源端与目标端的 padding 掩码
        - 编码 source；目标序列右移一位并送入解码器
        - 若启用指针生成，融合注意力分布以处理 OOV 复制
        - 计算 NLL 或 Label Smoothing 损失
        返回：loss, ppl, 0, 0（后两项保留位）
        """
        enc_batch = batch["context_batch"]
        enc_batch_extend_vocab = batch["context_ext_batch"]
        oovs = batch["oovs"]
        max_oov_length = len(sorted(oovs, key=lambda i: len(i), reverse=True)[0])

        dec_batch = batch["target_batch"]
        dec_ext_batch = batch["target_ext_batch"]

        if config.noam:
            self.optimizer.optimizer.zero_grad()
        else:
            self.optimizer.zero_grad()

        # 源端 embedding 与掩码
        mask_src = enc_batch.data.eq(config.PAD_idx).unsqueeze(1)  # [bsz, 1, src_len]
        emb_mask = self.embedding(batch["mask_context"])  # 位置/类型等附加 embedding
        src_emb = self.embedding(enc_batch)+emb_mask
        encoder_outputs = self.encoder(src_emb, mask_src)  # [bsz, src_len, hidden]

        # 目标端右移：在句首拼接 SOS，便于自回归解码输入
        sos_token = torch.LongTensor([config.SOS_idx] * enc_batch.size(0)).unsqueeze(1)  # [bsz, 1]
        if config.USE_CUDA: sos_token = sos_token.cuda()
        dec_batch_shift = torch.cat((sos_token, dec_batch[:, :-1]), 1)  # [bsz, tgt_len]

        mask_trg = dec_batch_shift.data.eq(config.PAD_idx).unsqueeze(1)
        pre_logit, attn_dist = self.decoder(self.embedding(dec_batch_shift), encoder_outputs, (mask_src,mask_trg))

        enc_ext_batch = enc_batch_extend_vocab

        logit = self.generator(pre_logit, attn_dist, enc_ext_batch if config.pointer_gen else None, max_oov_length, attn_dist_db=None)
        # loss: 指针时为 NLL（对数概率），否则为交叉熵等价形式
        loss = self.criterion(logit.contiguous().view(-1, logit.size(-1)), dec_batch.contiguous().view(-1))

        if config.label_smoothing:
            loss_ppl = self.criterion_ppl(logit.contiguous().view(-1, logit.size(-1)), dec_batch.contiguous().view(-1)).item()

        if train:
            loss.backward()
            self.optimizer.step()

        if config.label_smoothing:
            return loss_ppl, math.exp(min(loss_ppl, 100)), 0, 0
        else:
            return loss.item(), math.exp(min(loss.item(), 100)), 0, 0

    def compute_act_loss(self,module):    
        """
        若使用 ACT，自适应步数的正则化损失：约束平均计算步数，避免过多迭代。
        返回标量 loss（Python float）。
        """
        R_t = module.remainders
        N_t = module.n_updates
        p_t = R_t + N_t
        avg_p_t = torch.sum(torch.sum(p_t,dim=1)/p_t.size(1))/p_t.size(0)
        loss = config.act_loss_weight * avg_p_t.item()
        return loss

    def decoder_greedy(self, batch, max_dec_step=30):
        """
        贪心解码：每一步选取最大概率的词，生成至达到最大步数或遇到 EOS。
        仅支持 batch_size=1 的交互式推断。
        返回：生成的句子列表（长度为 batch）。
        """
        enc_batch_extend_vocab, extra_zeros = None, None
        enc_batch = batch["context_batch"]
        enc_batch_extend_vocab = batch["context_ext_batch"]

        oovs = batch["oovs"]
        max_oov_length = len(sorted(oovs, key=lambda i: len(i), reverse=True)[0])

        # Encode - context
        mask_src = enc_batch.data.eq(config.PAD_idx).unsqueeze(1)  # [bsz, 1, src_len]
        emb_mask = self.embedding(batch["mask_context"])
        src_emb = self.embedding(enc_batch) + emb_mask
        encoder_outputs = self.encoder(src_emb, mask_src)  # [bsz, src_len, hidden]

        enc_ext_batch = enc_batch_extend_vocab

        ys = torch.ones(1, 1).fill_(config.SOS_idx).long()  # 推断时 bsz = 1
        if config.USE_CUDA:
            ys = ys.cuda()
        mask_trg = ys.data.eq(config.PAD_idx).unsqueeze(1)
        decoded_words = []
        for i in range(max_dec_step+1):
            if config.project:
                out, attn_dist = self.decoder(self.embedding_proj_in(self.embedding(ys)),self.embedding_proj_in(encoder_outputs), (mask_src,mask_trg))
            else:
                out, attn_dist = self.decoder(self.embedding(ys), encoder_outputs, (mask_src,mask_trg))
            
            prob = self.generator(out, attn_dist, enc_ext_batch, max_oov_length, attn_dist_db=None)
            _, next_word = torch.max(prob[:, -1], dim = 1)
            decoded_words.append(['<EOS>' if ni.item() == config.EOS_idx else self.vocab.index2word[ni.item()] for ni in next_word.view(-1)])
            next_word = next_word.data[0]

            if config.USE_CUDA:
                ys = torch.cat([ys, torch.ones(1, 1).long().fill_(next_word).cuda()], dim=1)
                ys = ys.cuda()
            else:
                ys = torch.cat([ys, torch.ones(1, 1).long().fill_(next_word)], dim=1)
            mask_trg = ys.data.eq(config.PAD_idx).unsqueeze(1)

        sent = []
        for _, row in enumerate(np.transpose(decoded_words)):
            st = ''
            for e in row:
                if e == '<EOS>': break
                else: st+= e + ' '
            sent.append(st)
        return sent
        
    def decoder_topk(self, batch, max_dec_step=30):
        """
        Top-k 采样解码：
        - 每一步在 top-k 概率的词中按概率采样，提升多样性
        - 本实现默认 k=3，可按需调整
        返回：生成的句子列表（长度为 batch）。
        """
        enc_batch_extend_vocab, extra_zeros = None, None
        enc_batch = batch["context_batch"]

        mask_src = enc_batch.data.eq(config.PAD_idx).unsqueeze(1)
        emb_mask = self.embedding(batch["mask_input"])
        encoder_outputs = self.encoder(self.embedding(enc_batch)+emb_mask, mask_src)

        ys = torch.ones(1, 1).fill_(config.SOS_idx).long()
        if config.USE_CUDA:
            ys = ys.cuda()
        mask_trg = ys.data.eq(config.PAD_idx).unsqueeze(1)
        decoded_words = []
        for i in range(max_dec_step+1):
            if config.project:
                out, attn_dist = self.decoder(self.embedding_proj_in(self.embedding(ys)),self.embedding_proj_in(encoder_outputs), (mask_src,mask_trg))
            else:
                out, attn_dist = self.decoder(self.embedding(ys),encoder_outputs, (mask_src,mask_trg))
            
            logit = self.generator(out,attn_dist,enc_batch_extend_vocab, extra_zeros, attn_dist_db=None)
            # 取 top-k（k=3）后按概率采样
            filtered_logit = top_k_top_p_filtering(logit[:, -1], top_k=3, top_p=0, filter_value=-float('Inf'))
            next_word = torch.multinomial(F.softmax(filtered_logit, dim=-1), 1).squeeze()
            decoded_words.append(['<EOS>' if ni.item() == config.EOS_idx else self.vocab.index2word[ni.item()] for ni in next_word.view(-1)])
            next_word = next_word.data[0]

            if config.USE_CUDA:
                ys = torch.cat([ys, torch.ones(1, 1).long().fill_(next_word).cuda()], dim=1)
                ys = ys.cuda()
            else:
                ys = torch.cat([ys, torch.ones(1, 1).long().fill_(next_word)], dim=1)
            mask_trg = ys.data.eq(config.PAD_idx).unsqueeze(1)

        sent = []
        for _, row in enumerate(np.transpose(decoded_words)):
            st = ''
            for e in row:
                if e == '<EOS>': break
                else: st+= e + ' '
            sent.append(st)
        return sent
