"""
Transformer Demo 项目演示

这个文件展示了如何使用项目中的组件。
"""

import numpy as np
import matplotlib.pyplot as plt
import platform
import matplotlib.font_manager as fm

# 设置中文字体
def setup_chinese_font():
    """设置中文字体以正确显示中文字符"""
    system = platform.system()
    
    if system == "Darwin":  # macOS
        chinese_fonts = [
            'PingFang SC',
            'Arial Unicode MS', 
            'STHeiti',
            'SimHei',
            'Heiti SC'
        ]
    elif system == "Windows":
        chinese_fonts = [
            'SimHei',
            'Microsoft YaHei',
            'SimSun',
            'KaiTi'
        ]
    else:  # Linux
        chinese_fonts = [
            'WenQuanYi Micro Hei',
            'WenQuanYi Zen Hei',
            'SimHei',
            'DejaVu Sans'
        ]
    
    # 获取系统可用字体
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    # 找到第一个可用的中文字体
    for font in chinese_fonts:
        if font in available_fonts:
            plt.rcParams['font.sans-serif'] = [font]
            print(f"✅ 使用中文字体: {font}")
            break
    else:
        print("⚠️  警告: 未找到合适的中文字体，中文可能显示异常")
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    
    plt.rcParams['axes.unicode_minus'] = False

# 初始化中文字体
setup_chinese_font()

# 模拟一些基本功能，无需PyTorch
print("🚀 Transformer Demo 项目演示")
print("=" * 50)

print("\n📊 项目结构:")
print("""
transformer-demo/
├── src/
│   ├── models/          # Transformer模型实现
│   ├── data/           # 数据处理
│   ├── training/       # 训练相关
│   └── utils/          # 工具函数
├── examples/           # 示例代码
├── tests/             # 单元测试
├── notebooks/         # Jupyter教程
└── README.md          # 项目文档
""")

print("\n🔧 主要组件:")
print("1. MultiHeadAttention - 多头注意力机制")
print("2. PositionalEncoding - 位置编码")
print("3. SimpleTransformer - 完整的Transformer模型")
print("4. SimpleTokenizer - 简单分词器")
print("5. Trainer - 训练器")

print("\n📈 功能特性:")
print("- 从零实现Transformer架构")
print("- 完整的训练和验证流程")
print("- 注意力权重可视化")
print("- 模型性能评估")
print("- 详细的代码注释")

print("\n🎯 学习目标:")
print("- 理解注意力机制原理")
print("- 掌握Transformer模型结构")
print("- 学会使用PyTorch构建深度学习模型")
print("- 了解模型训练和评估流程")

print("\n💡 开始使用:")
print("1. 安装依赖: uv sync")
print("2. 安装PyTorch: pip install torch")
print("3. 运行示例: python examples/simple_classification.py")
print("4. 查看教程: jupyter lab notebooks/transformer_tutorial.ipynb")

print("\n✅ 项目创建完成！")
print("📚 查看 README.md 获取详细使用说明")

# 简单的可视化演示
if __name__ == "__main__":
    # 模拟注意力权重可视化
    np.random.seed(42)
    attention_matrix = np.random.rand(8, 8)
    attention_matrix = attention_matrix / attention_matrix.sum(axis=1, keepdims=True)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(attention_matrix, cmap='Blues')
    plt.colorbar(label='注意力权重')
    plt.title('模拟注意力权重矩阵')
    plt.xlabel('Key位置')
    plt.ylabel('Query位置')
    
    # 添加数值标注
    for i in range(8):
        for j in range(8):
            plt.text(j, i, f'{attention_matrix[i, j]:.2f}', 
                    ha='center', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('demo_attention.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n🎨 注意力矩阵可视化已保存为 demo_attention.png") 