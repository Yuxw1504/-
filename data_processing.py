#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 17:00:12 2024

@author: yxw
"""

import json
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import BertTokenizer
import torch
import joblib
import warnings
from emotion_dataset import EmotionDataset  # 导入EmotionDataset类

# 忽略FutureWarning警告
warnings.simplefilter(action='ignore', category=FutureWarning)

def load_data(file_path):
    texts, labels = [], []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split(';')
            if len(parts) == 2:
                texts.append(parts[0])
                labels.append(parts[1])
    return texts, labels

def preprocess_data(data_path, max_len=128, batch_size=16):
    texts, labels = load_data(data_path)

    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)

    # 分割数据集为训练集和验证集
    texts_train, texts_val, labels_train, labels_val = train_test_split(texts, labels, test_size=0.2, random_state=42)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', force_download=True)

    # 创建训练集和验证集的数据集对象
    train_dataset = EmotionDataset(texts_train, labels_train, tokenizer, max_len)
    val_dataset = EmotionDataset(texts_val, labels_val, tokenizer, max_len)

    # 创建训练集和验证集的数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 保存处理后的数据
    joblib.dump((texts_train, labels_train, texts_val, labels_val, tokenizer, max_len, label_encoder, batch_size), '/Users/yxw/Desktop/代码/emotion_nlp/data/preprocessed_data.pkl')

    return train_loader, val_loader, label_encoder

if __name__ == "__main__":
    train_loader, val_loader, label_encoder = preprocess_data('/Users/yxw/Desktop/代码/emotion_nlp/data/data.txt')
    print("Data preprocessing completed.")

