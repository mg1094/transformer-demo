"""
注意力机制实现

包含多头注意力机制的完整实现。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    """多头注意力机制实现"""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        """
        初始化多头注意力层
        
        Args:
            d_model: 模型维度
            num_heads: 注意力头数
            dropout: dropout概率
        """
        super(MultiHeadAttention, self).__init__()
        
        assert d_model % num_heads == 0, "d_model必须能被num_heads整除"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 每个头的维度
        
        # 线性变换层
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        缩放点积注意力
        
        Args:
            Q: 查询矩阵 [batch_size, num_heads, seq_len, d_k]
            K: 键矩阵 [batch_size, num_heads, seq_len, d_k]
            V: 值矩阵 [batch_size, num_heads, seq_len, d_k]
            mask: 注意力掩码
            
        Returns:
            attention_output: 注意力输出
            attention_weights: 注意力权重
        """
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 应用掩码（如果提供）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 应用softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 计算注意力输出
        attention_output = torch.matmul(attention_weights, V)
        
        return attention_output, attention_weights
    
    def forward(self, query, key, value, mask=None):
        """
        前向传播
        
        Args:
            query: 查询 [batch_size, seq_len, d_model]
            key: 键 [batch_size, seq_len, d_model]
            value: 值 [batch_size, seq_len, d_model]
            mask: 注意力掩码
            
        Returns:
            output: 多头注意力输出
            attention_weights: 注意力权重
        """
        batch_size = query.size(0)
        
        # 1. 线性变换并重塑为多头格式
        Q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 2. 应用缩放点积注意力
        attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # 3. 连接多头
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        # 4. 最终线性变换
        output = self.w_o(attention_output)
        
        return output, attention_weights


class PositionalEncoding(nn.Module):
    """位置编码实现"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        """
        初始化位置编码
        
        Args:
            d_model: 模型维度
            max_len: 最大序列长度
        """
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        添加位置编码到输入
        
        Args:
            x: 输入张量 [seq_len, batch_size, d_model]
            
        Returns:
            带位置编码的输出
        """
        return x + self.pe[:x.size(0), :] 