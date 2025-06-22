"""
可视化工具

包含注意力权重可视化、混淆矩阵等可视化功能。
"""

import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.font_manager as fm


# 设置支持中文的字体
import platform
import matplotlib.font_manager as fm

def setup_chinese_font():
    """设置中文字体"""
    system = platform.system()
    
    if system == "Darwin":  # macOS
        # macOS常用中文字体
        chinese_fonts = [
            'PingFang SC',
            'Arial Unicode MS', 
            'STHeiti',
            'SimHei',
            'Heiti SC'
        ]
    elif system == "Windows":
        # Windows常用中文字体
        chinese_fonts = [
            'SimHei',
            'Microsoft YaHei',
            'SimSun',
            'KaiTi'
        ]
    else:  # Linux
        # Linux常用中文字体
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
            break
    else:
        # 如果没有找到中文字体，使用默认字体
        print("警告: 未找到合适的中文字体，中文可能显示异常")
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    
    # 设置负号正常显示
    plt.rcParams['axes.unicode_minus'] = False

# 初始化中文字体设置
setup_chinese_font()


def plot_attention_weights(attention_weights, tokens, layer_idx=0, head_idx=0, save_path=None):
    """
    可视化注意力权重
    
    Args:
        attention_weights: 注意力权重 [num_layers, batch_size, num_heads, seq_len, seq_len]
        tokens: token列表
        layer_idx: 要可视化的层索引
        head_idx: 要可视化的注意力头索引
        save_path: 保存路径
    """
    # 提取指定层和头的注意力权重
    attn = attention_weights[layer_idx][0, head_idx].cpu().numpy()  # [seq_len, seq_len]
    
    # 创建图像
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 绘制热力图
    sns.heatmap(
        attn,
        xticklabels=tokens,
        yticklabels=tokens,
        cmap='Blues',
        annot=True,
        fmt='.2f',
        ax=ax
    )
    
    ax.set_title(f'注意力权重可视化 (层 {layer_idx + 1}, 头 {head_idx + 1})')
    ax.set_xlabel('Key Position')
    ax.set_ylabel('Query Position')
    
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"注意力可视化已保存到: {save_path}")
    
    plt.show()


def plot_multiple_attention_heads(attention_weights, tokens, layer_idx=0, num_heads=4, save_path=None):
    """
    可视化多个注意力头
    
    Args:
        attention_weights: 注意力权重
        tokens: token列表
        layer_idx: 层索引
        num_heads: 要显示的注意力头数量
        save_path: 保存路径
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()
    
    for head_idx in range(min(num_heads, 4)):
        attn = attention_weights[layer_idx][0, head_idx].cpu().numpy()
        
        sns.heatmap(
            attn,
            xticklabels=tokens,
            yticklabels=tokens,
            cmap='Blues',
            ax=axes[head_idx],
            cbar=True
        )
        
        axes[head_idx].set_title(f'头 {head_idx + 1}')
        axes[head_idx].set_xlabel('Key Position')
        axes[head_idx].set_ylabel('Query Position')
    
    fig.suptitle(f'层 {layer_idx + 1} 的多头注意力可视化')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"多头注意力可视化已保存到: {save_path}")
    
    plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names=None, save_path=None):
    """
    绘制混淆矩阵
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        class_names: 类别名称
        save_path: 保存路径
    """
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    
    # 创建图像
    plt.figure(figsize=(8, 6))
    
    # 绘制热力图
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names or range(len(cm)),
        yticklabels=class_names or range(len(cm))
    )
    
    plt.title('混淆矩阵')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"混淆矩阵已保存到: {save_path}")
    
    plt.show()


def plot_layer_attention_summary(attention_weights, tokens, save_path=None):
    """
    可视化所有层的注意力权重摘要
    
    Args:
        attention_weights: 注意力权重
        tokens: token列表
        save_path: 保存路径
    """
    num_layers = len(attention_weights)
    num_heads = attention_weights[0].shape[1]
    
    fig, axes = plt.subplots(num_layers, 2, figsize=(12, 3 * num_layers))
    
    for layer_idx in range(num_layers):
        # 计算该层所有头的平均注意力
        layer_attn = attention_weights[layer_idx][0].mean(dim=0).cpu().numpy()  # 平均所有头
        
        # 绘制平均注意力
        sns.heatmap(
            layer_attn,
            xticklabels=tokens,
            yticklabels=tokens,
            cmap='Blues',
            ax=axes[layer_idx, 0],
            cbar=True
        )
        axes[layer_idx, 0].set_title(f'层 {layer_idx + 1} - 平均注意力')
        
        # 绘制注意力强度分布
        attn_scores = layer_attn.flatten()
        axes[layer_idx, 1].hist(attn_scores, bins=50, alpha=0.7)
        axes[layer_idx, 1].set_title(f'层 {layer_idx + 1} - 注意力分布')
        axes[layer_idx, 1].set_xlabel('注意力权重')
        axes[layer_idx, 1].set_ylabel('频次')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"层注意力摘要已保存到: {save_path}")
    
    plt.show()


def visualize_embeddings(embeddings, labels, method='pca', save_path=None):
    """
    可视化词嵌入
    
    Args:
        embeddings: 嵌入向量 [num_samples, embedding_dim]
        labels: 标签
        method: 降维方法 ('pca' 或 'tsne')
        save_path: 保存路径
    """
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    
    # 降维
    if method == 'pca':
        reducer = PCA(n_components=2)
        embeddings_2d = reducer.fit_transform(embeddings)
    else:  # tsne
        reducer = TSNE(n_components=2, random_state=42)
        embeddings_2d = reducer.fit_transform(embeddings)
    
    # 绘制散点图
    plt.figure(figsize=(10, 8))
    
    unique_labels = np.unique(labels)
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
    
    for label, color in zip(unique_labels, colors):
        mask = labels == label
        plt.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=[color],
            label=f'类别 {label}',
            alpha=0.7
        )
    
    plt.title(f'词嵌入可视化 ({method.upper()})')
    plt.xlabel('维度 1')
    plt.ylabel('维度 2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"嵌入可视化已保存到: {save_path}")
    
    plt.show() 