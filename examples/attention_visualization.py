"""
注意力权重可视化示例

演示如何可视化Transformer的注意力权重。
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import matplotlib.pyplot as plt

from src.models import SimpleTransformer, SimpleTokenizer
from src.data import create_sample_data
from src.training import set_seed, get_device
from src.utils import plot_attention_weights, plot_multiple_attention_heads, setup_chinese_font

# 设置中文字体
setup_chinese_font()


def main():
    """主函数"""
    print("👁️ Transformer注意力可视化示例")
    print("=" * 50)
    
    # 设置随机种子
    set_seed(42)
    
    # 设置设备
    device = get_device()
    
    # 创建示例数据用于构建词汇表
    print("📊 创建示例数据...")
    texts, labels = create_sample_data(num_samples=1000)
    
    # 创建分词器并构建词汇表
    print("🔤 构建词汇表...")
    tokenizer = SimpleTokenizer(vocab_size=1000)
    tokenizer.build_vocab(texts)
    
    # 创建模型
    print("🏗️ 创建Transformer模型...")
    model = SimpleTransformer(
        vocab_size=len(tokenizer.word_to_id),
        d_model=128,
        num_heads=8,
        num_layers=4,
        d_ff=256,
        max_len=64,
        num_classes=2,
        dropout=0.0  # 关闭dropout以获得一致的注意力权重
    ).to(device)
    
    # 如果有预训练模型，加载它
    checkpoint_path = "checkpoints/best_model.pth"
    if os.path.exists(checkpoint_path):
        print(f"📥 加载预训练模型: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("⚠️ 未找到预训练模型，使用随机初始化的权重")
        print("💡 提示: 先运行 examples/simple_classification.py 训练模型")
    
    # 准备测试文本
    test_texts = [
        "This movie is amazing and wonderful",
        "The film was terrible and boring",
        "Great story with fantastic acting",
        "Poor plot and bad characters"
    ]
    
    print("\n🎯 分析注意力权重...")
    
    model.eval()
    for i, text in enumerate(test_texts):
        print(f"\n📝 文本 {i+1}: '{text}'")
        
        # 编码文本
        token_ids = tokenizer.encode(text, max_len=32)  # 使用较短的序列便于可视化
        input_ids = torch.tensor([token_ids], dtype=torch.long).to(device)
        
        # 获取tokens（用于显示）
        tokens = tokenizer.decode(token_ids).split()
        # 限制token数量以便可视化
        if len(tokens) > 12:
            tokens = tokens[:12]
            token_ids = token_ids[:12]
            input_ids = input_ids[:, :12]
        
        # 获取注意力权重
        with torch.no_grad():
            outputs, attention_weights = model(input_ids, return_attention=True)
            
            # 预测结果
            probs = torch.softmax(outputs, dim=1)
            pred = torch.argmax(outputs, dim=1).item()
            confidence = probs[0, pred].item()
            
            label_name = "正面" if pred == 1 else "负面"
            print(f"预测: {label_name} (置信度: {confidence:.3f})")
        
        # 可视化第一层的第一个注意力头
        print(f"可视化第1层第1个注意力头...")
        plot_attention_weights(
            attention_weights,
            tokens,
            layer_idx=0,
            head_idx=0,
            save_path=f"attention_layer0_head0_text{i+1}.png"
        )
        
        # 可视化第一层的多个注意力头
        print(f"可视化第1层的多个注意力头...")
        plot_multiple_attention_heads(
            attention_weights,
            tokens,
            layer_idx=0,
            num_heads=4,
            save_path=f"attention_multiple_heads_text{i+1}.png"
        )
        
        # 分析注意力模式
        analyze_attention_patterns(attention_weights, tokens, text)
    
    # 创建注意力权重对比
    print("\n🔄 创建注意力权重对比...")
    create_attention_comparison(model, tokenizer, device)
    
    print("\n✅ 注意力可视化完成！")
    print("📁 可视化结果已保存到当前目录")


def analyze_attention_patterns(attention_weights, tokens, text):
    """
    分析注意力模式
    
    Args:
        attention_weights: 注意力权重
        tokens: token列表
        text: 原始文本
    """
    print(f"  📊 注意力模式分析:")
    
    # 计算每层的平均注意力分布
    for layer_idx, layer_attn in enumerate(attention_weights):
        # 平均所有头的注意力权重
        avg_attn = layer_attn[0].mean(dim=0)  # [seq_len, seq_len]
        
        # 找到每个token最关注的其他token
        max_attn_indices = torch.argmax(avg_attn, dim=1)
        
        print(f"    层 {layer_idx + 1}:")
        for i, (token, max_idx) in enumerate(zip(tokens, max_attn_indices)):
            if i < len(tokens) and max_idx < len(tokens):
                attention_score = avg_attn[i, max_idx].item()
                most_attended_token = tokens[max_idx]
                print(f"      '{token}' -> '{most_attended_token}' (权重: {attention_score:.3f})")


def create_attention_comparison(model, tokenizer, device):
    """
    创建不同情感文本的注意力对比
    
    Args:
        model: 模型
        tokenizer: 分词器
        device: 设备
    """
    positive_text = "This movie is absolutely amazing and wonderful"
    negative_text = "This movie is completely terrible and awful"
    
    texts = [positive_text, negative_text]
    labels = ["正面文本", "负面文本"]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    model.eval()
    for i, (text, label) in enumerate(zip(texts, labels)):
        # 编码文本
        token_ids = tokenizer.encode(text, max_len=20)
        input_ids = torch.tensor([token_ids], dtype=torch.long).to(device)
        tokens = tokenizer.decode(token_ids).split()[:15]  # 限制长度
        
        # 获取注意力权重
        with torch.no_grad():
            _, attention_weights = model(input_ids, return_attention=True)
        
        # 绘制第一层和最后一层的注意力
        for j, layer_idx in enumerate([0, -1]):
            layer_name = "第1层" if layer_idx == 0 else "最后一层"
            avg_attn = attention_weights[layer_idx][0].mean(dim=0).cpu().numpy()
            
            # 调整矩阵大小以匹配tokens
            attn_size = min(len(tokens), avg_attn.shape[0])
            avg_attn = avg_attn[:attn_size, :attn_size]
            display_tokens = tokens[:attn_size]
            
            im = axes[i, j].imshow(avg_attn, cmap='Blues')
            axes[i, j].set_title(f'{label} - {layer_name}')
            axes[i, j].set_xticks(range(len(display_tokens)))
            axes[i, j].set_yticks(range(len(display_tokens)))
            axes[i, j].set_xticklabels(display_tokens, rotation=45)
            axes[i, j].set_yticklabels(display_tokens)
            
            # 添加颜色条
            plt.colorbar(im, ax=axes[i, j])
    
    plt.tight_layout()
    plt.savefig("attention_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print("💡 注意力对比图已保存: attention_comparison.png")


if __name__ == "__main__":
    main() 