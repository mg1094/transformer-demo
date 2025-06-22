"""
训练器类

包含模型训练和验证的完整逻辑。
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from .utils import calculate_accuracy
import os


class Trainer:
    """模型训练器"""
    
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        device,
        save_dir="checkpoints"
    ):
        """
        初始化训练器
        
        Args:
            model: 要训练的模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            optimizer: 优化器
            criterion: 损失函数
            device: 设备
            save_dir: 模型保存目录
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.save_dir = save_dir
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        # 训练历史
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        
    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        # 使用tqdm显示进度条
        pbar = tqdm(self.train_loader, desc="Training")
        
        for batch in pbar:
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # 清零梯度
            self.optimizer.zero_grad()
            
            # 前向传播
            outputs = self.model(input_ids)
            loss = self.criterion(outputs, labels)
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            # 计算指标
            total_loss += loss.item()
            accuracy = calculate_accuracy(outputs, labels)
            total_correct += accuracy * labels.size(0)
            total_samples += labels.size(0)
            
            # 更新进度条
            current_acc = total_correct / total_samples
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{current_acc:.4f}'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        avg_accuracy = total_correct / total_samples
        
        return avg_loss, avg_accuracy
    
    def validate(self):
        """验证模型"""
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # 前向传播
                outputs = self.model(input_ids)
                loss = self.criterion(outputs, labels)
                
                # 计算指标
                total_loss += loss.item()
                accuracy = calculate_accuracy(outputs, labels)
                total_correct += accuracy * labels.size(0)
                total_samples += labels.size(0)
        
        avg_loss = total_loss / len(self.val_loader)
        avg_accuracy = total_correct / total_samples
        
        return avg_loss, avg_accuracy
    
    def train(self, num_epochs, save_best=True):
        """
        训练模型
        
        Args:
            num_epochs: 训练轮数
            save_best: 是否保存最佳模型
        """
        best_val_accuracy = 0
        
        print(f"开始训练，共{num_epochs}个epoch")
        print("-" * 50)
        
        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")
            
            # 训练
            train_loss, train_acc = self.train_epoch()
            
            # 验证
            val_loss, val_acc = self.validate()
            
            # 记录历史
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            # 打印结果
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # 保存最佳模型
            if save_best and val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                self.save_model(f"best_model.pth")
                print(f"保存最佳模型 (验证准确率: {val_acc:.4f})")
            
            print("-" * 50)
        
        print(f"训练完成！最佳验证准确率: {best_val_accuracy:.4f}")
    
    def save_model(self, filename):
        """
        保存模型
        
        Args:
            filename: 文件名
        """
        filepath = os.path.join(self.save_dir, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'train_accuracies': self.train_accuracies,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
        }, filepath)
    
    def load_model(self, filename):
        """
        加载模型
        
        Args:
            filename: 文件名
        """
        filepath = os.path.join(self.save_dir, filename)
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.train_accuracies = checkpoint.get('train_accuracies', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.val_accuracies = checkpoint.get('val_accuracies', [])
    
    def plot_training_history(self, save_path=None):
        """
        绘制训练历史
        
        Args:
            save_path: 保存路径
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # 损失曲线
        ax1.plot(self.train_losses, label='训练损失', color='blue')
        ax1.plot(self.val_losses, label='验证损失', color='red')
        ax1.set_title('训练和验证损失')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('损失')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 准确率曲线
        ax2.plot(self.train_accuracies, label='训练准确率', color='blue')
        ax2.plot(self.val_accuracies, label='验证准确率', color='red')
        ax2.set_title('训练和验证准确率')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('准确率')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"训练历史图已保存到: {save_path}")
        
        plt.show() 