#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 19:30:15 2024

@author: yxw
"""

import json
from nlpaug.augmenter.word import SynonymAug

# 加载数据
def load_data(file_path):
    texts, labels = [], []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split(';')
            if len(parts) == 2:
                texts.append(parts[0])
                labels.append(parts[1])
    return texts, labels

train_texts, train_labels = load_data('train.txt')

# 数据增强
def augment_data(texts, labels, augmenter, num_augmentations=1):
    augmented_texts, augmented_labels = [], []
    for _ in range(num_augmentations):
        for text, label in zip(texts, labels):
            augmented_text = augmenter.augment(text)
            augmented_texts.append(augmented_text)
            augmented_labels.append(label)
    return augmented_texts, augmented_labels

augmenter = SynonymAug(aug_src='wordnet')
aug_train_texts, aug_train_labels = augment_data(train_texts, train_labels, augmenter)

# 保存增强后的数据
with open('augmented_data.json', 'w', encoding='utf-8') as f:
    json.dump({'texts': aug_train_texts, 'labels': aug_train_labels}, f, ensure_ascii=False)


