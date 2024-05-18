#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: yxw
"""

import numpy as np
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
import joblib
import warnings
from emotion_dataset import EmotionDataset  # 导入EmotionDataset类

# 忽略FutureWarning警告
warnings.simplefilter(action='ignore', category=FutureWarning)

class EmotionClassifier(nn.Module):
    def __init__(self, n_classes):
        super(EmotionClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased', force_download=True)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs[1]  # 从BERT模型输出中获取池化输出
        output = self.drop(pooled_output)
        return self.out(output)

def load_model_and_preprocessing():
    # 加载预处理后的对象
    texts_train, labels_train, texts_val, labels_val, tokenizer, max_len, label_encoder, batch_size = joblib.load('/Users/yxw/Desktop/代码/emotion_nlp/data/preprocessed_data.pkl')

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = EmotionClassifier(n_classes=len(label_encoder.classes_))
    model.load_state_dict(torch.load('model/best_model_state.bin', map_location=device))
    model = model.to(device)
    model.eval()

    return model, tokenizer, label_encoder, max_len, device

def predict_emotion(model, tokenizer, label_encoder, max_len, device, text):
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_len,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        _, preds = torch.max(outputs, dim=1)
        predicted_label = label_encoder.inverse_transform(preds.cpu().numpy())

    return predicted_label[0]

if __name__ == "__main__":
    model, tokenizer, label_encoder, max_len, device = load_model_and_preprocessing()
    print("Model and preprocessing objects loaded. Enter 'esc' to exit.")

    while True:
        text = input("Enter a sentence: ")
        if text.lower() == 'esc':
            break
        emotion = predict_emotion(model, tokenizer, label_encoder, max_len, device, text)
        print(f"Predicted emotion: {emotion}")

