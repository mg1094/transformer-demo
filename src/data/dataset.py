"""
数据集类和数据处理工具

包含文本分类数据集和示例数据生成。
"""

import torch
from torch.utils.data import Dataset
import random


class TextClassificationDataset(Dataset):
    """文本分类数据集"""
    
    def __init__(self, texts, labels, tokenizer, max_len=128):
        """
        初始化数据集
        
        Args:
            texts: 文本列表
            labels: 标签列表
            tokenizer: 分词器
            max_len: 最大序列长度
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # 编码文本
        token_ids = self.tokenizer.encode(text, max_len=self.max_len)
        
        return {
            'input_ids': torch.tensor(token_ids, dtype=torch.long),
            'labels': torch.tensor(label, dtype=torch.long),
            'text': text
        }


def create_sample_data(num_samples=1000):
    """
    创建示例数据用于演示
    
    Args:
        num_samples: 样本数量
        
    Returns:
        texts: 文本列表
        labels: 标签列表
    """
    # 正面情感的词汇
    positive_words = [
        "great", "excellent", "amazing", "wonderful", "fantastic", "awesome",
        "brilliant", "perfect", "outstanding", "superb", "magnificent", "incredible",
        "love", "like", "enjoy", "happy", "pleased", "satisfied", "delighted",
        "good", "nice", "beautiful", "stunning", "impressive", "remarkable"
    ]
    
    # 负面情感的词汇
    negative_words = [
        "terrible", "awful", "horrible", "bad", "worst", "disgusting",
        "disappointing", "frustrating", "annoying", "boring", "dull", "stupid",
        "hate", "dislike", "angry", "sad", "upset", "disappointed", "unhappy",
        "poor", "ugly", "useless", "worthless", "pathetic", "ridiculous"
    ]
    
    # 中性词汇
    neutral_words = [
        "movie", "film", "book", "story", "character", "plot", "scene", "actor",
        "product", "service", "experience", "time", "place", "thing", "person",
        "day", "way", "work", "life", "world", "people", "today", "yesterday"
    ]
    
    texts = []
    labels = []
    
    for _ in range(num_samples):
        # 随机选择情感类别
        sentiment = random.choice([0, 1])  # 0: 负面, 1: 正面
        
        if sentiment == 1:  # 正面
            # 选择正面词汇作为主要内容
            main_words = random.sample(positive_words, random.randint(2, 4))
            filler_words = random.sample(neutral_words, random.randint(1, 3))
        else:  # 负面
            # 选择负面词汇作为主要内容
            main_words = random.sample(negative_words, random.randint(2, 4))
            filler_words = random.sample(neutral_words, random.randint(1, 3))
        
        # 组合词汇生成句子
        all_words = main_words + filler_words
        random.shuffle(all_words)
        
        # 添加一些连接词
        connectors = ["is", "was", "very", "really", "quite", "so", "the", "a", "an"]
        sentence_words = []
        
        for i, word in enumerate(all_words):
            sentence_words.append(word)
            if i < len(all_words) - 1 and random.random() < 0.3:
                sentence_words.append(random.choice(connectors))
        
        text = " ".join(sentence_words)
        
        texts.append(text)
        labels.append(sentiment)
    
    return texts, labels


def create_imdb_style_data(num_samples=1000):
    """
    创建类似IMDB风格的电影评论数据
    
    Args:
        num_samples: 样本数量
        
    Returns:
        texts: 文本列表
        labels: 标签列表
    """
    positive_templates = [
        "This movie is {} and {}. The acting was {} and the plot was {}.",
        "I {} this film! It was {} with {} characters.",
        "Amazing {}! The director did a {} job with this {} story.",
        "What a {} experience! The cinematography was {} and the music was {}.",
        "This is one of the {} movies I've ever seen. {} performance by the lead actor."
    ]
    
    negative_templates = [
        "This movie is {} and {}. The acting was {} and the plot was {}.",
        "I {} this film! It was {} with {} characters.",
        "Terrible {}! The director did a {} job with this {} story.",
        "What a {} experience! The cinematography was {} and the music was {}.",
        "This is one of the {} movies I've ever seen. {} performance by the lead actor."
    ]
    
    positive_adjectives = [
        "amazing", "brilliant", "excellent", "fantastic", "wonderful", "outstanding",
        "superb", "incredible", "beautiful", "perfect", "great", "awesome"
    ]
    
    negative_adjectives = [
        "terrible", "awful", "horrible", "disappointing", "boring", "bad",
        "worst", "disgusting", "annoying", "stupid", "pathetic", "useless"
    ]
    
    positive_verbs = ["love", "adore", "enjoy", "recommend"]
    negative_verbs = ["hate", "dislike", "regret watching"]
    
    nouns = ["movie", "film", "picture", "story", "drama", "comedy", "thriller"]
    
    texts = []
    labels = []
    
    for _ in range(num_samples):
        sentiment = random.choice([0, 1])
        
        if sentiment == 1:  # 正面
            template = random.choice(positive_templates)
            adjectives = random.sample(positive_adjectives, 4)
            verb = random.choice(positive_verbs)
            noun = random.choice(nouns)
            
            if "I {} this film" in template:
                text = template.format(verb, adjectives[0], adjectives[1])
            elif "Amazing {}!" in template:
                text = template.format(noun, adjectives[0], adjectives[1])
            elif "What a {} experience" in template:
                text = template.format(adjectives[0], adjectives[1], adjectives[2])
            else:
                text = template.format(adjectives[0], adjectives[1], adjectives[2], adjectives[3])
                
        else:  # 负面
            template = random.choice(negative_templates)
            adjectives = random.sample(negative_adjectives, 4)
            verb = random.choice(negative_verbs)
            noun = random.choice(nouns)
            
            if "I {} this film" in template:
                text = template.format(verb, adjectives[0], adjectives[1])
            elif "Terrible {}!" in template:
                text = template.format(noun, adjectives[0], adjectives[1])
            elif "What a {} experience" in template:
                text = template.format(adjectives[0], adjectives[1], adjectives[2])
            else:
                text = template.format(adjectives[0], adjectives[1], adjectives[2], adjectives[3])
        
        texts.append(text)
        labels.append(sentiment)
    
    return texts, labels 