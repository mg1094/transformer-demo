"""
简单文本分类示例

演示如何使用Transformer模型进行文本分类。
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import classification_report

from src.models import SimpleTransformer, SimpleTokenizer
from src.data import TextClassificationDataset, create_sample_data
from src.training import Trainer, set_seed, get_device
from src.utils import plot_confusion_matrix, setup_chinese_font

# 设置中文字体
setup_chinese_font()


def main():
    """主函数"""
    print("🚀 Transformer文本分类示例")
    print("=" * 50)
    
    # 设置随机种子
    set_seed(42)
    
    # 设置设备
    device = get_device()
    
    # 创建示例数据
    print("📊 创建示例数据...")
    texts, labels = create_sample_data(num_samples=2000)
    print(f"生成了 {len(texts)} 个样本")
    
    # 显示一些示例
    print("\n📝 数据示例:")
    for i in range(5):
        label_name = "正面" if labels[i] == 1 else "负面"
        print(f"  {i+1}. [{label_name}] {texts[i]}")
    
    # 创建分词器并构建词汇表
    print("\n🔤 构建词汇表...")
    tokenizer = SimpleTokenizer(vocab_size=1000)
    tokenizer.build_vocab(texts)
    print(f"词汇表大小: {len(tokenizer.word_to_id)}")
    
    # 创建数据集
    dataset = TextClassificationDataset(texts, labels, tokenizer, max_len=64)
    
    # 划分训练集和验证集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    
    # 创建模型
    print("\n🏗️ 创建Transformer模型...")
    model = SimpleTransformer(
        vocab_size=len(tokenizer.word_to_id),
        d_model=128,
        num_heads=8,
        num_layers=4,
        d_ff=256,
        max_len=64,
        num_classes=2,
        dropout=0.1
    ).to(device)
    
    # 计算参数数量
    from src.training.utils import count_parameters
    total_params, trainable_params = count_parameters(model)
    print(f"总参数数量: {total_params:,}")
    print(f"可训练参数数量: {trainable_params:,}")
    
    # 定义优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        save_dir="checkpoints"
    )
    
    # 开始训练
    print("\n🎯 开始训练...")
    trainer.train(num_epochs=10, save_best=True)
    
    # 绘制训练历史
    print("\n📈 绘制训练历史...")
    trainer.plot_training_history(save_path="training_history.png")
    
    # 加载最佳模型进行评估
    print("\n🧪 评估模型...")
    trainer.load_model("best_model.pth")
    
    # 在验证集上进行预测
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 打印分类报告
    class_names = ['负面', '正面']
    print("\n📊 分类报告:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    # 绘制混淆矩阵
    print("\n🔍 绘制混淆矩阵...")
    plot_confusion_matrix(all_labels, all_preds, class_names, save_path="confusion_matrix.png")
    
    # 测试一些新样本
    print("\n🎪 测试新样本:")
    test_texts = [
        "This movie is amazing and wonderful",
        "The film was terrible and boring",
        "Great acting and fantastic story",
        "Awful plot and bad characters"
    ]
    
    model.eval()
    with torch.no_grad():
        for text in test_texts:
            # 编码文本
            token_ids = tokenizer.encode(text, max_len=64)
            input_ids = torch.tensor([token_ids], dtype=torch.long).to(device)
            
            # 预测
            outputs = model(input_ids)
            probs = torch.softmax(outputs, dim=1)
            pred = torch.argmax(outputs, dim=1).item()
            confidence = probs[0, pred].item()
            
            label_name = "正面" if pred == 1 else "负面"
            print(f"  文本: '{text}'")
            print(f"  预测: {label_name} (置信度: {confidence:.3f})")
            print()
    
    print("✅ 训练完成！")
    print(f"📁 模型已保存到: checkpoints/best_model.pth")
    print(f"📊 训练历史图: training_history.png")
    print(f"🔍 混淆矩阵: confusion_matrix.png")


if __name__ == "__main__":
    main() 