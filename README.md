<p align="center">
  <h1 align="center">🚀 Transformer Demo</h1>
  <p align="center">
    <strong>从零实现 Transformer 模型 — PyTorch + uv</strong>
  </p>
  <p align="center">
    <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python">
    <img src="https://img.shields.io/badge/PyTorch-2.0+-red.svg" alt="PyTorch">
    <img src="https://img.shields.io/badge/uv-package%20manager-purple.svg" alt="uv">
    <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
    <img src="https://img.shields.io/github/stars/mg1094/transformer-demo.svg" alt="Stars">
  </p>
</p>

---

## 🎯 一句话

**不调库，从零写每一行 — Multi-Head Attention、Positional Encoding、FFN、残差连接、LayerNorm，全手写。**

## 🧠 为什么做这个项目

市面上的 Transformer 教程要么太抽象（只讲公式），要么太黑盒（直接调 `nn.TransformerEncoder`）。

这个项目把每一层拆开给你看：
- Q/K/V 矩阵怎么算的 → `attention.py`
- 位置编码怎么加的 → `attention.py`
- 残差连接 + LayerNorm 在哪 → `transformer.py`
- 训练循环怎么写 → `trainer.py`
- 注意力权重长什么样 → `attention_visualization.py`

## 🏗️ 架构

### 模型架构

```mermaid
graph TD
    A["输入文本<br/>[batch, seq_len]"] --> B["Token Embedding<br/>[batch, seq, d_model]"]
    B --> C["× √d_model 缩放"]
    C --> D["+ Positional Encoding"]
    D --> E["Dropout"]
    E --> F["Padding Mask"]

    F --> G["Transformer Block × N"]
    
    subgraph G["Transformer Block（重复 N 层）"]
        G1["Multi-Head Attention"] --> G2["Dropout"]
        G2 --> G3["+ 残差"]
        G3 --> G4["LayerNorm"]
        G4 --> G5["Feed Forward<br/>Linear → ReLU → Dropout → Linear"]
        G5 --> G6["Dropout"]
        G6 --> G7["+ 残差"]
        G7 --> G8["LayerNorm"]
    end

    G --> H["Masked Mean Pooling"]
    H --> I["Classifier<br/>Linear → ReLU → Dropout → Linear"]
    I --> J["输出 logits<br/>[batch, num_classes]"]

    style A fill:#e1f5fe
    style J fill:#e8f5e9
    style G1 fill:#fff3e0
    style G5 fill:#fce4ec
```

### 训练流程

```mermaid
flowchart LR
    A["样本数据生成"] --> B["Tokenizer 编码"]
    B --> C["DataLoader 分批"]
    C --> D["前向传播"]
    D --> E["CrossEntropy Loss"]
    E --> F["反向传播 + 梯度裁剪"]
    F --> G["Optimizer Step"]
    G --> H{"验证集评估"}
    H -->|"改善"| I["保存最佳模型"]
    H -->|"未改善"| J["早停计数 +1"]
    J --> K{"达到 patience?"}
    K -->|"是"| L["停止训练"]
    K -->|"否"| D
    I --> D
```

### 项目结构

```mermaid
graph LR
    Root["transformer-demo/"] --> Src["src/"]
    Root --> Examples["examples/"]
    Root --> Notebooks["notebooks/"]
    Root --> Tests["tests/"]

    Src --> Models["models/"]
    Src --> Data["data/"]
    Src --> Training["training/"]
    Src --> Utils["utils/"]

    Models --> M1["attention.py<br/>MultiHeadAttention + PositionalEncoding"]
    Models --> M2["transformer.py<br/>TransformerBlock + SimpleTransformer + Tokenizer"]

    Data --> D1["dataset.py<br/>样本数据生成 + DataLoader"]

    Training --> T1["trainer.py<br/>训练循环 + 验证 + 保存"]
    Training --> T2["utils.py<br/>EarlyStopping + 工具函数"]

    Utils --> U1["visualization.py<br/>注意力热力图 + 混淆矩阵 + 嵌入可视化"]

    Examples --> E1["simple_classification.py<br/>文本分类训练"]
    Examples --> E2["attention_visualization.py<br/>注意力权重可视化"]
```

## 📁 目录结构

```
transformer-demo/
├── src/
│   ├── models/
│   │   ├── attention.py        # MultiHeadAttention + PositionalEncoding
│   │   └── transformer.py      # TransformerBlock + SimpleTransformer + Tokenizer
│   ├── data/
│   │   └── dataset.py          # 样本数据生成 + DataLoader
│   ├── training/
│   │   ├── trainer.py          # 训练循环 + 验证 + checkpoint
│   │   └── utils.py            # EarlyStopping + set_seed + 工具函数
│   └── utils/
│       └── visualization.py    # 注意力热力图 + 混淆矩阵 + 嵌入可视化
├── examples/
│   ├── simple_classification.py     # 文本分类训练示例
│   └── attention_visualization.py   # 注意力权重可视化
├── notebooks/
│   └── transformer_tutorial.ipynb   # 交互式教程
├── tests/
│   └── test_models.py               # 单元测试
├── demo.py                          # 快速演示入口
├── pyproject.toml                   # 项目配置（uv）
└── README.md
```

## 🚀 快速开始

### 1. 安装依赖

```bash
# 安装 uv（推荐）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 同步依赖
uv sync

# 安装 PyTorch（按你的系统选）
source .venv/bin/activate
pip install torch torchvision torchaudio
```

### 2. 运行

```bash
# 快速演示
python demo.py

# 训练文本分类
python examples/simple_classification.py

# 可视化注意力
python examples/attention_visualization.py

# 交互式教程
jupyter lab notebooks/transformer_tutorial.ipynb
```

## 📊 模型参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `d_model` | 128 | 模型维度 |
| `num_heads` | 8 | 注意力头数 |
| `num_layers` | 6 | Transformer 层数 |
| `d_ff` | 512 | 前馈网络隐藏层维度 |
| `max_len` | 512 | 最大序列长度 |
| `num_classes` | 2 | 分类类别数 |
| `dropout` | 0.1 | Dropout 概率 |

## 📚 学习路径

```mermaid
graph LR
    A["1. demo.py<br/>了解项目结构"] --> B["2. attention.py<br/>理解 Q/K/V + Multi-Head"]
    B --> C["3. transformer.py<br/>完整模型 + 残差 + LayerNorm"]
    C --> D["4. simple_classification.py<br/>实际训练"]
    D --> E["5. attention_visualization.py<br/>注意力热力图"]
    E --> F["6. transformer_tutorial.ipynb<br/>深入学习"]

    style A fill:#e1f5fe
    style F fill:#e8f5e9
```

## ✨ 特性

- 🔧 **从零实现** — 不调 `nn.TransformerEncoder`，每一层手写
- 📊 **完整训练流程** — 前向 + 反向 + 梯度裁剪 + 早停 + checkpoint
- 📈 **可视化** — 注意力热力图、多头对比、混淆矩阵、嵌入降维
- 🎯 **文本分类示例** — 情感分析（正面/负面）
- 📚 **详细注释** — 每行关键代码都有中文注释
- 🧪 **单元测试** — 模型维度、注意力输出形状验证

## 📄 License

MIT © [mg1094](https://github.com/mg1094)
