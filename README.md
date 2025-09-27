# 🤖 Empathetic-Dialogue-Generation (RAICOM2025)

🏆 本项目为**睿抗 2025 智海人工智能算法应用赛国赛**参赛项目，荣获**二等奖**。🎉

## 项目背景

随着人工智能技术的发展，构建能够进行流畅、有逻辑、多轮对话的智能系统成为了热门方向。Transformer 架构，凭借其强大的自注意力机制，在序列建模任务上取得了突破性进展。本项目旨在训练一个**基于 Transformer 的对话生成模型**。该模型能够理解用户意图，并生成连贯、相关的回复，模拟真实的对话场景。学生将学习 **Transformer 的核心组件**（如自注意力、多头注意力、位置编码），以及如何将其应用于**对话生成任务**，理解其在解决长距离依赖和并行计算方面的优势。

## 🌟 项目亮点

- 🎭 **多分辨率情感建模**：融合对话级与词语级的情感信息，提升共情能力。
- ⚔️ **交互式对抗训练**：双判别器（语义 + 情感）协同，提高生成质量与多样性。
- 🔥 **Transformer 架构**：强大的注意力建模，适配长距离依赖与并行训练。
- 💬 **情感引导生成**：情感类别向量指导解码，回复更贴合情境与情绪。

## 🛠️ 环境依赖

* Python 3.8+
* PyTorch（需匹配本地 CUDA 版本）
* 其他依赖见 `requirements.txt`

```bash
# 1) 创建与激活虚拟环境（推荐 Conda）
conda create -n empdg python=3.8 -y
conda activate empdg

# 2) 安装依赖
pip install -r requirements.txt

# 3) 检查 GPU 可用性
python utils/check_torch_gpu.py
```

### 📥 预训练向量（必需）

从这里 [GloVe 向量](http://nlp.stanford.edu/data/glove.6B.zip) 下载 GloVe 向量并将其放入 /vectors/ 中。

## 📂 数据集说明

本项目使用 **EmpatheticDialogues** 数据集，特点：

* 🗣️ **对话上下文**：模拟真实场景中的多轮对话
* 🎭 **情感标签**：覆盖多种情绪（开心、伤心、惊讶等）
* 📖 **情境描述**：每段对话均有对应的背景说明
* 💡 **目标回复**：由人工撰写的情感化回复

> 预处理后的数据已存放在 `datasets/empathetic-dialogue/`

## 🏗️ 模型架构

本项目包含一个基准模型 `EmoPrepend` 和一个核心对抗模型 `EmpDG`。

### 🔹 基础模型 (EmoPrepend)

`EmoPrepend` 是一个基于 Transformer 的增强模型，其核心思想是“情感引导”：
*   **情感前置**：利用外部情感词典 `NRCDict.json`，在输入端为上下文中的情感词注入可学习的“情感提示嵌入”，引导编码器关注情感信号。
*   **情感分类头**：在编码器之上增加一个分类头，用于预测对话的整体情感类别，并通过多任务学习增强模型的情感感知能力。
*   **指针生成网络**：允许模型在生成回复时，从原始对话上下文中直接复制词语，有效处理未登录词（OOV）和专有名词。

### 🔹 核心对抗模型 (EmpDG)

`EmpDG` (Empathetic Dialogue Generation) 是一个采用**交互式对抗训练**的共情对话生成模型，建立在 WGAN-GP 框架之上，由一个生成器（Generator）和两个判别器（Discriminator）组成。

*   **生成器 (EmpDG_G)**
    *   **多分辨率情感建模**：采用双编码器结构，一个用于理解**语义上下文**，另一个专门用于感知**多分辨率情感信息**（对话级情感标签 + 词语级情感词）。
    *   **情感条件解码**：将编码器预测的情感类别向量作为解码器的初始输入，引导解码器生成与情感一致的回复。

*   **交互式判别器 (EmpDG_D)**
    *   **双判别器架构**：包含一个**语义判别器**和一个**情感判别器**。
        *   语义判别器：评估回复在内容上是否连贯、合理。
        *   情感判别器：评估回复在情感上是否与上下文匹配。
    *   **利用用户反馈**：判别器不仅区分真实/生成回复，还创新性地引入对话的**下一轮用户回复**作为隐式反馈信号，从而更精准地指导生成器优化。


## 🚀 模型训练

项目提供了对多个模型的训练支持，主要分为基础模型训练和对抗模型训练。所有训练参数的默认值可以在 `utils/config.py` 中查看和修改。

### 🔹 基础模型训练

使用 `train.py` 脚本训练 `Transformer`、`EmoPrepend` 等基础模型。

- **训练 EmoPrepend 模型** (推荐的基础模型):
  ```bash
  python train.py --cuda --label_smoothing --noam --emb_dim 300 --hidden_dim 300 --hop 1 --heads 2 --pretrain_emb --model EmoPrepend --device_id 0 --save_path results/tb_results/EmoPrepend/ --pointer_gen
  ```
- **训练 Transformer 基础模型**:
  ```bash
  python train.py --cuda --label_smoothing --noam --emb_dim 300 --hidden_dim 300 --hop 1 --heads 2 --pretrain_emb --model Transformer --device_id 0 --save_path results/tb_results/Transformer/ --pointer_gen
  ```
- **训练 EmpDG_woD 模型** (仅生成器部分):
  ```bash
  python train.py --cuda --label_smoothing --noam --emb_dim 300 --hidden_dim 300 --hop 1 --heads 2 --pretrain_emb --model EmpDG_woD --device_id 0 --save_path results/tb_results/EmpDG_woD/ --pointer_gen
  ```

### 🔹 对抗训练 (EmpDG)

使用 `adver_train.py` 脚本对 `EmpDG` 模型进行三阶段的对抗训练：

1.  **预训练生成器**：首先训练一个基础的共情生成器。
2.  **预训练判别器**：固定生成器，训练语义和情感判别器。
3.  **联合对抗训练**：生成器和判别器交替训练，进行博弈优化。

- **运行完整的 EmpDG 对抗训练**:
  ```bash
  python3 adver_train.py --cuda --label_smoothing --noam --emb_dim 300 --rnn_hidden_dim 300 --hidden_dim 300 --hop 1 --heads 2 --emotion_disc --pretrain_emb --model EmpDG --device_id 0 --save_path results/tb_results/EmpDG/ --d_steps 1 --g_steps 5 --pointer_gen
  ```
- **从断点恢复对抗训练**:
  ```bash
  python3 adver_train.py --cuda --label_smoothing --noam --emb_dim 300 --rnn_hidden_dim 300 --hidden_dim 300 --hop 1 --heads 2 --emotion_disc --pretrain_emb --model EmpDG --device_id 0 --save_path results/tb_results/EmpDG/ --d_steps 1 --g_steps 5 --pointer_gen --resume_g --resume_d
  ```

训练日志与可视化结果将保存到 `results/tb_results/`，模型权重存放在 `result/`。

训练好的对抗模型权重可从[Google Drive](https://drive.google.com/drive/folders/1EIIZ9SFJCE1JavUal39J_NN2WxP5JK6H?usp=sharing)中获取，您可以下载并将其移动到 /result/ 下。

## 📊 模型评估

我们采用以下指标来综合评估模型性能：

*   **Perplexity (PPL)**：困惑度，衡量模型生成回复的流畅性以及对真实数据分布的拟合程度。PPL 越低通常表示流畅性越好。
*   **Distinct-1 / Distinct-2**：衡量生成回复的多样性，分别计算不重复的一元文法（unigrams）和二元文法（bigrams）的比例。该指标越高，表示模型生成的回复越不容易出现重复和通用的套话。
*   **Emotion Accuracy**：情感准确率，衡量生成器内部的情感分类器从对话上下文中识别正确情感类别的能力。这是模型能否产生共情回复的关键前提。

> 注：本项目不使用 BLEU，因其与对话质量的相关性较弱。所有生成示例与预测结果存放在 `Predictions/` 目录下。

## 🧪 模型测试与交互

### 🔹 模型测试

您可以使用 `train.py` 脚本加载训练好的模型，并在测试集上进行评估。请确保对应的模型权重文件（如 `result/EmpDG_best.tar`）存在。

```bash
# 在测试集上评估 EmpDG 模型
python train.py --test --model EmpDG --cuda --label_smoothing --noam --emb_dim 300 --hidden_dim 300 --hop 1 --heads 2 --pretrain_emb --pointer_gen
```

### 🔹 交互式对话

通过 `interact.py` 脚本，您可以与训练好的模型进行实时多轮对话。

```bash
# 与 EmpDG 模型进行对话
python interact.py --model EmpDG --cuda --label_smoothing --noam --emb_dim 300 --hidden_dim 300 --hop 1 --heads 2 --pretrain_emb --pointer_gen
```

> **注意**：在测试或交互时，请确保所使用的模型超参数（如 `emb_dim`, `hidden_dim` 等）与训练时保持一致。

### 🔹 生成示例

以下是一些在交互环境中与 `EmpDG` 模型对话的示例：

**示例 1：**

| 角色  | 回复                                                                |
| :---- | :------------------------------------------------------------------ |
| User  | i am so excited because i am finally going to visit my old friend ! |
| EmpDG | that is awesome ! what kind of visit ?                              |
| User  | we will enjoy chatting and playing .                                |
| EmpDG | that is a lot of nostalgia .                                        |

**示例 2：**

| 角色  | 回复                                    |
| :---- | :-------------------------------------- |
| User  | i argued with my brother .              |
| EmpDG | oh no ! what happened ?                 |
| User  | i don't want to recall .                |
| EmpDG | i am sorry , that is not a good thing . |

## 📁 项目结构

```
├── Model/              # 模型实现 (EmpDG_G, EmpDG_D, Transformer)
├── datasets/           # 数据集与预处理文件
├── utils/              # 工具函数 (数据加载、配置等)
├── vectors/            # 预训练 GloVe 词向量
├── results/            # TensorBoard 日志与可视化
├── result/             # 训练好的模型权重
├── Predictions/        # 模型生成结果与评估
├── *.ipynb             # Jupyter 教程笔记本
├── train.py            # 基础模型训练脚本
├── adver_train.py      # 对抗训练脚本
└── interact.py         # 交互式对话脚本
```

## 📒 教程笔记本

0. `00、实训项目教案.ipynb`
1. `01、项目介绍与数据集分析处理.ipynb`
2. `02、预备知识点学习.ipynb`
3. `03、基础对话模型搭建.ipynb`
4. `04、进阶知识与对抗模型实现.ipynb`
5. `05、模型训练与评估.ipynb`
6. `06、调用模型进行对话.ipynb`

## 📚 参考资料

* [Transformers](https://github.com/huggingface/transformers)
* [EmpatheticDialogues](https://github.com/facebookresearch/EmpatheticDialogues)
* [MoEL: Mixture of Empathetic Listeners](https://github.com/HLTCHKUST/MoEL)
* [EmpDG: Multi-resolution Interactive Empathetic Dialogue Generation](https://github.com/qtli/EmpDG)
* [learn-nlp-with-transformer](https://github.com/datawhalechina/learn-nlp-with-transformers)

---

✨ **本项目所有内容仅供学习所需，如您有任何任何问题，请及时联系我！欢迎一起交流学习！** 🌈
