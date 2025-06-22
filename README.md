# Transformer Demo 项目

这是一个用于学习Transformer架构的小型演示项目，使用PyTorch和uv包管理器。

## 项目特性

- 🔧 从零实现简单的Transformer模型
- 📊 包含完整的训练和推理流程
- 📈 可视化训练过程和结果
- 🎯 简单的文本分类任务示例
- 📚 详细的代码注释和文档

## 安装和设置

### 1. 安装uv（如果还没有安装）
```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# 或者使用pip
pip install uv
```

### 2. 安装项目依赖
```bash
uv sync
```

### 3. 激活虚拟环境并安装PyTorch
```bash
source .venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
# 或者访问 https://pytorch.org/get-started/locally/ 获取适合您系统的安装命令
```

### 4. 运行演示
```bash
python demo.py
```

**注意**: 由于PyTorch的平台兼容性问题，可能需要手动安装PyTorch。请根据您的系统选择合适的安装命令。

## 项目结构

```
transformer-demo/
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── transformer.py      # Transformer模型实现
│   │   └── attention.py        # 注意力机制实现
│   ├── data/
│   │   ├── __init__.py
│   │   └── dataset.py          # 数据加载和处理
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py          # 训练器
│   │   └── utils.py            # 训练工具函数
│   └── utils/
│       ├── __init__.py
│       └── visualization.py    # 可视化工具
├── examples/
│   ├── simple_classification.py  # 文本分类示例
│   └── attention_visualization.py # 注意力可视化
├── notebooks/
│   └── transformer_tutorial.ipynb # 教程notebook
├── tests/
│   └── test_models.py          # 单元测试
├── pyproject.toml
└── README.md
```

## 快速开始

### 1. 运行简单分类示例
```bash
python examples/simple_classification.py
```

### 2. 可视化注意力机制
```bash
python examples/attention_visualization.py
```

### 3. 打开教程notebook
```bash
jupyter lab notebooks/transformer_tutorial.ipynb
```

## 学习目标

通过这个项目，您将学到：

1. **Transformer架构基础**
   - 多头注意力机制
   - 位置编码
   - 前馈网络
   - 残差连接和层归一化

2. **实际实现经验**
   - PyTorch模型构建
   - 训练循环设计
   - 损失函数和优化器使用

3. **可视化和调试**
   - 注意力权重可视化
   - 训练过程监控
   - 模型性能分析

## 贡献

欢迎提交问题和改进建议！

## 许可证

MIT License 


📚 学习路径
从demo.py开始了解项目结构
阅读src/models/attention.py理解注意力机制
查看src/models/transformer.py学习完整模型
运行examples/simple_classification.py进行实际训练
使用examples/attention_visualization.py可视化注意力
打开notebooks/transformer_tutorial.ipynb深入学习