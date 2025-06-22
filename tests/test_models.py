"""
模型单元测试

测试Transformer模型的各个组件。
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pytest
from src.models import SimpleTransformer, MultiHeadAttention, SimpleTokenizer
from src.models.attention import PositionalEncoding


class TestMultiHeadAttention:
    """测试多头注意力机制"""
    
    def test_attention_forward(self):
        """测试注意力前向传播"""
        d_model = 128
        num_heads = 8
        seq_len = 10
        batch_size = 2
        
        attention = MultiHeadAttention(d_model, num_heads)
        
        # 创建输入
        x = torch.randn(batch_size, seq_len, d_model)
        
        # 前向传播
        output, attention_weights = attention(x, x, x)
        
        # 检查输出形状
        assert output.shape == (batch_size, seq_len, d_model)
        assert attention_weights.shape == (batch_size, num_heads, seq_len, seq_len)
        
        # 检查注意力权重是否归一化
        attn_sum = attention_weights.sum(dim=-1)
        assert torch.allclose(attn_sum, torch.ones_like(attn_sum), atol=1e-6)
    
    def test_attention_with_mask(self):
        """测试带掩码的注意力"""
        d_model = 64
        num_heads = 4
        seq_len = 5
        batch_size = 1
        
        attention = MultiHeadAttention(d_model, num_heads)
        
        # 创建输入和掩码
        x = torch.randn(batch_size, seq_len, d_model)
        mask = torch.ones(batch_size, 1, seq_len, seq_len)
        mask[:, :, 2:, 2:] = 0  # 掩盖后面的位置
        
        # 前向传播
        output, attention_weights = attention(x, x, x, mask)
        
        # 检查掩码是否生效
        assert attention_weights[:, :, 2, 3].item() < 1e-6  # 被掩盖的位置应该接近0


class TestPositionalEncoding:
    """测试位置编码"""
    
    def test_positional_encoding_shape(self):
        """测试位置编码形状"""
        d_model = 128
        max_len = 100
        seq_len = 20
        batch_size = 2
        
        pos_encoding = PositionalEncoding(d_model, max_len)
        
        # 创建输入
        x = torch.randn(seq_len, batch_size, d_model)
        
        # 应用位置编码
        output = pos_encoding(x)
        
        # 检查输出形状
        assert output.shape == (seq_len, batch_size, d_model)
    
    def test_positional_encoding_values(self):
        """测试位置编码值"""
        d_model = 4
        max_len = 10
        
        pos_encoding = PositionalEncoding(d_model, max_len)
        
        # 检查位置编码的周期性
        pe = pos_encoding.pe.squeeze(1)  # [max_len, d_model]
        
        # 检查奇偶位置的差异
        assert not torch.allclose(pe[:, 0], pe[:, 1])  # sin和cos应该不同


class TestSimpleTransformer:
    """测试Transformer模型"""
    
    def test_transformer_forward(self):
        """测试Transformer前向传播"""
        vocab_size = 1000
        d_model = 128
        num_heads = 8
        num_layers = 2
        max_len = 32
        num_classes = 2
        batch_size = 2
        seq_len = 16
        
        model = SimpleTransformer(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            max_len=max_len,
            num_classes=num_classes
        )
        
        # 创建输入
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        # 前向传播
        output = model(input_ids)
        
        # 检查输出形状
        assert output.shape == (batch_size, num_classes)
    
    def test_transformer_with_attention(self):
        """测试返回注意力权重"""
        vocab_size = 100
        d_model = 64
        num_heads = 4
        num_layers = 2
        
        model = SimpleTransformer(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers
        )
        
        # 创建输入
        input_ids = torch.randint(0, vocab_size, (1, 10))
        
        # 前向传播并返回注意力权重
        output, attention_weights = model(input_ids, return_attention=True)
        
        # 检查注意力权重
        assert len(attention_weights) == num_layers
        assert attention_weights[0].shape[1] == num_heads  # 检查头数
    
    def test_transformer_padding_mask(self):
        """测试填充掩码"""
        vocab_size = 100
        model = SimpleTransformer(vocab_size=vocab_size)
        
        # 创建包含填充的输入
        input_ids = torch.tensor([[1, 2, 3, 0, 0], [4, 5, 0, 0, 0]])  # 0是填充token
        
        # 创建掩码
        mask = model.create_padding_mask(input_ids, pad_token_id=0)
        
        # 检查掩码形状和值
        assert mask.shape == (2, 1, 1, 5)
        assert mask[0, 0, 0, 0].item() == 1  # 非填充位置
        assert mask[0, 0, 0, 3].item() == 0  # 填充位置


class TestSimpleTokenizer:
    """测试简单分词器"""
    
    def test_tokenizer_build_vocab(self):
        """测试词汇表构建"""
        texts = ["hello world", "world is great", "hello great world"]
        tokenizer = SimpleTokenizer(vocab_size=10)
        
        tokenizer.build_vocab(texts)
        
        # 检查特殊token
        assert tokenizer.word_to_id["<PAD>"] == 0
        assert tokenizer.word_to_id["<UNK>"] == 1
        
        # 检查常见词是否在词汇表中
        assert "world" in tokenizer.word_to_id
        assert "hello" in tokenizer.word_to_id
    
    def test_tokenizer_encode_decode(self):
        """测试编码和解码"""
        texts = ["hello world test", "world is good"]
        tokenizer = SimpleTokenizer(vocab_size=20)
        tokenizer.build_vocab(texts)
        
        # 测试编码
        text = "hello world"
        token_ids = tokenizer.encode(text, max_len=5)
        
        # 检查长度
        assert len(token_ids) == 5
        
        # 测试解码
        decoded_text = tokenizer.decode(token_ids)
        
        # 解码后应该包含原始词汇
        assert "hello" in decoded_text
        assert "world" in decoded_text
    
    def test_tokenizer_padding(self):
        """测试填充功能"""
        texts = ["short text"]
        tokenizer = SimpleTokenizer(vocab_size=20)
        tokenizer.build_vocab(texts)
        
        text = "short"
        token_ids = tokenizer.encode(text, max_len=10)
        
        # 检查填充
        assert len(token_ids) == 10
        assert token_ids[-1] == tokenizer.pad_token_id  # 最后应该是填充token
    
    def test_tokenizer_truncation(self):
        """测试截断功能"""
        texts = ["very long text with many words here"]
        tokenizer = SimpleTokenizer(vocab_size=50)
        tokenizer.build_vocab(texts)
        
        text = "very long text with many words"
        token_ids = tokenizer.encode(text, max_len=3)
        
        # 检查截断
        assert len(token_ids) == 3


def run_tests():
    """运行所有测试"""
    print("🧪 运行模型单元测试...")
    
    # 测试多头注意力
    print("  测试多头注意力机制...")
    test_attn = TestMultiHeadAttention()
    test_attn.test_attention_forward()
    test_attn.test_attention_with_mask()
    print("    ✅ 多头注意力测试通过")
    
    # 测试位置编码
    print("  测试位置编码...")
    test_pe = TestPositionalEncoding()
    test_pe.test_positional_encoding_shape()
    test_pe.test_positional_encoding_values()
    print("    ✅ 位置编码测试通过")
    
    # 测试Transformer
    print("  测试Transformer模型...")
    test_transformer = TestSimpleTransformer()
    test_transformer.test_transformer_forward()
    test_transformer.test_transformer_with_attention()
    test_transformer.test_transformer_padding_mask()
    print("    ✅ Transformer模型测试通过")
    
    # 测试分词器
    print("  测试分词器...")
    test_tokenizer = TestSimpleTokenizer()
    test_tokenizer.test_tokenizer_build_vocab()
    test_tokenizer.test_tokenizer_encode_decode()
    test_tokenizer.test_tokenizer_padding()
    test_tokenizer.test_tokenizer_truncation()
    print("    ✅ 分词器测试通过")
    
    print("🎉 所有测试通过！")


if __name__ == "__main__":
    run_tests() 