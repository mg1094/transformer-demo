"""
简单的Transformer模型实现

包含编码器和解码器的完整Transformer模型。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import MultiHeadAttention, PositionalEncoding


class TransformerBlock(nn.Module):
    """Transformer编码器块"""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        """
        初始化Transformer块
        
        Args:
            d_model: 模型维度
            num_heads: 注意力头数
            d_ff: 前馈网络隐藏层维度
            dropout: dropout概率
        """
        super(TransformerBlock, self).__init__()
        
        # 多头注意力层
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        
        # 前馈网络
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, seq_len, d_model]
            mask: 注意力掩码
            
        Returns:
            output: 输出张量
            attention_weights: 注意力权重
        """
        # 自注意力 + 残差连接 + 层归一化
        attn_output, attention_weights = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 前馈网络 + 残差连接 + 层归一化
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x, attention_weights


class SimpleTransformer(nn.Module):
    """简单的Transformer模型，用于文本分类"""
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 512,
        max_len: int = 512,
        num_classes: int = 2,
        dropout: float = 0.1
    ):
        """
        初始化Transformer模型
        
        Args:
            vocab_size: 词汇表大小
            d_model: 模型维度
            num_heads: 注意力头数
            num_layers: Transformer层数
            d_ff: 前馈网络隐藏层维度
            max_len: 最大序列长度
            num_classes: 分类类别数
            dropout: dropout概率
        """
        super(SimpleTransformer, self).__init__()
        
        self.d_model = d_model
        
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        # Transformer层
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # 初始化权重
        self._init_weights()
        
    def _init_weights(self):
        """初始化模型权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.1)
    
    def create_padding_mask(self, x, pad_token_id=0):
        """
        创建填充掩码
        
        Args:
            x: 输入序列 [batch_size, seq_len]
            pad_token_id: 填充token的ID
            
        Returns:
            mask: 填充掩码
        """
        return (x != pad_token_id).unsqueeze(1).unsqueeze(2)
    
    def forward(self, x, return_attention=False):
        """
        前向传播
        
        Args:
            x: 输入序列 [batch_size, seq_len]
            return_attention: 是否返回注意力权重
            
        Returns:
            output: 分类输出
            attention_weights: 注意力权重（如果return_attention=True）
        """
        # 创建填充掩码
        mask = self.create_padding_mask(x)
        
        # 词嵌入 + 位置编码
        x = self.embedding(x) * (self.d_model ** 0.5)  # 缩放嵌入
        x = x.transpose(0, 1)  # [seq_len, batch_size, d_model]
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)  # [batch_size, seq_len, d_model]
        x = self.dropout(x)
        
        # 通过Transformer层
        attention_weights_list = []
        for layer in self.transformer_layers:
            x, attention_weights = layer(x, mask)
            if return_attention:
                attention_weights_list.append(attention_weights)
        
        # 全局平均池化
        # 使用掩码确保只对非填充token进行池化
        mask_expanded = mask.squeeze(1).squeeze(1).unsqueeze(-1).float()  # [batch_size, seq_len, 1]
        x_masked = x * mask_expanded
        x_pooled = x_masked.sum(dim=1) / mask_expanded.sum(dim=1)  # [batch_size, d_model]
        
        # 分类
        output = self.classifier(x_pooled)
        
        if return_attention:
            return output, attention_weights_list
        return output
    
    def get_attention_weights(self, x):
        """
        获取注意力权重（用于可视化）
        
        Args:
            x: 输入序列
            
        Returns:
            attention_weights: 所有层的注意力权重
        """
        _, attention_weights = self.forward(x, return_attention=True)
        return attention_weights


class SimpleTokenizer:
    """简单的分词器，用于演示"""
    
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.word_to_id = {"<PAD>": 0, "<UNK>": 1}
        self.id_to_word = {0: "<PAD>", 1: "<UNK>"}
        self.pad_token_id = 0
        self.unk_token_id = 1
        
    def build_vocab(self, texts):
        """
        从文本构建词汇表
        
        Args:
            texts: 文本列表
        """
        word_freq = {}
        for text in texts:
            words = text.lower().split()
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # 按频率排序，保留最高频的词
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        
        for i, (word, _) in enumerate(sorted_words[:self.vocab_size - 2]):
            word_id = i + 2  # 0和1已被占用
            self.word_to_id[word] = word_id
            self.id_to_word[word_id] = word
    
    def encode(self, text, max_len=None):
        """
        编码文本为ID序列
        
        Args:
            text: 输入文本
            max_len: 最大长度
            
        Returns:
            token_ids: token ID列表
        """
        words = text.lower().split()
        token_ids = [self.word_to_id.get(word, self.unk_token_id) for word in words]
        
        if max_len:
            if len(token_ids) > max_len:
                token_ids = token_ids[:max_len]
            else:
                token_ids.extend([self.pad_token_id] * (max_len - len(token_ids)))
        
        return token_ids
    
    def decode(self, token_ids):
        """
        解码ID序列为文本
        
        Args:
            token_ids: token ID列表
            
        Returns:
            text: 解码后的文本
        """
        words = [self.id_to_word.get(token_id, "<UNK>") for token_id in token_ids]
        # 移除填充token
        words = [word for word in words if word != "<PAD>"]
        return " ".join(words) 