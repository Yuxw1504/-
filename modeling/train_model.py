#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: yxw
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertModel
import joblib
import warnings
from torch.utils.data import DataLoader
from emotion_dataset import EmotionDataset  # 导入EmotionDataset类
import os
from tqdm import tqdm  # 导入tqdm库

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

def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
    model = model.train()
    losses = []
    correct_predictions = 0

    for d in tqdm(data_loader, desc="Training", leave=False):
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        labels = d["labels"].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, labels)
        
        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
    scheduler.step()
    
    return correct_predictions.float() / n_examples, np.mean(losses)

def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()
    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for d in tqdm(data_loader, desc="Evaluating", leave=False):
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            labels = d["labels"].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, labels)
            
            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())
            
    return correct_predictions.float() / n_examples, np.mean(losses)

def train_model():
    # 加载预处理后的数据
    texts_train, labels_train, texts_val, labels_val, tokenizer, max_len, label_encoder, batch_size = joblib.load('data/preprocessed_data.pkl')
    
    # 重新创建训练集和验证集的数据集对象和数据加载器
    train_dataset = EmotionDataset(texts_train, labels_train, tokenizer, max_len)
    val_dataset = EmotionDataset(texts_val, labels_val, tokenizer, max_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    device = torch.device("mps" if torch.backends.mps.is_built() else "cpu")  # 使用 M1 GPU（MPS）或 CPU
    model = EmotionClassifier(n_classes=len(label_encoder.classes_))
    model = model.to(device)
    
    EPOCHS = 10
    optimizer = optim.Adam(model.parameters(), lr=2e-5)
    total_steps = len(train_loader) * EPOCHS
    
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
    loss_fn = nn.CrossEntropyLoss().to(device)
    
    best_accuracy = 0
    os.makedirs('/Users/yxw/Desktop/代码/emotion_nlp/model_checkpoints', exist_ok=True)
    
    for epoch in range(EPOCHS):
        print(f'Epoch {epoch + 1}/{EPOCHS}')
        print('-' * 10)
        
        train_acc, train_loss = train_epoch(
            model,
            train_loader,
            loss_fn,
            optimizer,
            device,
            scheduler,
            len(train_loader.dataset)
        )
        
        print(f'Train loss {train_loss} accuracy {train_acc}')
        
        val_acc, val_loss = eval_model(
            model,
            val_loader,
            loss_fn,
            device,
            len(val_loader.dataset)
        )
        
        print(f'Val   loss {val_loss} accuracy {val_acc}')
        
        if val_acc > best_accuracy:
            torch.save(model.state_dict(), '/Users/yxw/Desktop/代码/emotion_nlp/model/best_model_state.bin')
            best_accuracy = val_acc
    
    # 保存最终模型
    torch.save(model.state_dict(), '/Users/yxw/Desktop/代码/emotion_nlp/model/final_model_state.bin')
    
    print(f'Best val accuracy: {best_accuracy}')
    
def test_model():
    # 加载预处理后的数据
    texts_train, labels_train, texts_val, labels_val, tokenizer, max_len, label_encoder, batch_size = joblib.load('data/preprocessed_data.pkl')
    
    # 重新创建验证集的数据集对象和数据加载器
    val_dataset = EmotionDataset(texts_val, labels_val, tokenizer, max_len)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    device = torch.device("mps" if torch.backends.mps.is_built() else "cpu")  # 使用 M1 GPU（MPS）或 CPU
    model = EmotionClassifier(n_classes=len(label_encoder.classes_))
    model.load_state_dict(torch.load('model/best_model_state.bin'))
    model = model.to(device)
    
    test_acc, _ = eval_model(
        model,
        val_loader,
        nn.CrossEntropyLoss().to(device),
        device,
        len(val_loader.dataset)
    )
    
    print(f'Test Accuracy: {test_acc}')

if __name__ == "__main__":
    train_model()
    test_model()
