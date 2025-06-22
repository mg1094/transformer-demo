"""
训练工具函数

包含训练过程中用到的辅助函数。
"""

import torch
import numpy as np
import random


def set_seed(seed=42):
    """
    设置随机种子以确保结果可复现
    
    Args:
        seed: 随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 保证每次结果一样
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_accuracy(outputs, targets):
    """
    计算分类准确率
    
    Args:
        outputs: 模型输出 [batch_size, num_classes]
        targets: 真实标签 [batch_size]
        
    Returns:
        accuracy: 准确率
    """
    with torch.no_grad():
        predictions = torch.argmax(outputs, dim=1)
        correct = (predictions == targets).float()
        accuracy = correct.mean().item()
    return accuracy


def count_parameters(model):
    """
    计算模型参数数量
    
    Args:
        model: PyTorch模型
        
    Returns:
        total_params: 总参数数量
        trainable_params: 可训练参数数量
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return total_params, trainable_params


def get_device():
    """
    获取可用的设备
    
    Returns:
        device: torch.device
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"使用GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device('cpu')
        print("使用CPU")
    
    return device


def save_predictions(model, dataloader, tokenizer, device, save_path):
    """
    保存模型预测结果
    
    Args:
        model: 训练好的模型
        dataloader: 数据加载器
        tokenizer: 分词器
        device: 设备
        save_path: 保存路径
    """
    model.eval()
    predictions = []
    texts = []
    true_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            batch_texts = batch['text']
            
            outputs = model(input_ids)
            preds = torch.argmax(outputs, dim=1)
            
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
            texts.extend(batch_texts)
    
    # 保存到文件
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("文本\t真实标签\t预测标签\n")
        for text, true_label, pred in zip(texts, true_labels, predictions):
            f.write(f"{text}\t{true_label}\t{pred}\n")
    
    print(f"预测结果已保存到: {save_path}")


class EarlyStopping:
    """早停机制"""
    
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True):
        """
        初始化早停
        
        Args:
            patience: 容忍没有改善的epoch数
            min_delta: 最小改善阈值
            restore_best_weights: 是否恢复最佳权重
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        """
        检查是否应该早停
        
        Args:
            val_loss: 验证损失
            model: 模型
            
        Returns:
            should_stop: 是否应该停止训练
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False
    
    def save_checkpoint(self, model):
        """保存最佳权重"""
        self.best_weights = model.state_dict().copy() 