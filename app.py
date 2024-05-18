#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 18:47:31 2024

@author: yxw
"""

import numpy as np
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
import joblib
import warnings
from flask import Flask, request, render_template
import requests

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
    texts_train, labels_train, texts_val, labels_val, tokenizer, max_len, label_encoder, batch_size = joblib.load('data/preprocessed_data.pkl')

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

def search_images(emotion, text):
    search_query = f"感情：{emotion} {text}"
    search_url = "https://image.baidu.com/search/acjson"
    params = {
        "tn": "resultjson_com",
        "logid": "8383748883020709728",
        "ipn": "rj",
        "ct": 201326592,
        "is": "",
        "fp": "result",
        "queryWord": search_query,
        "cl": 2,
        "lm": -1,
        "ie": "utf-8",
        "oe": "utf-8",
        "adpicid": "",
        "st": -1,
        "z": "",
        "ic": 0,
        "hd": "",
        "latest": "",
        "copyright": "",
        "word": search_query,
        "s": "",
        "se": "",
        "tab": "",
        "width": "",
        "height": "",
        "face": 0,
        "istype": 2,
        "qc": "",
        "nc": 1,
        "fr": "",
        "expermode": "",
        "force": "",
        "pn": 0,
        "rn": 3,
    }
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    response = requests.get(search_url, params=params, headers=headers)
    response.raise_for_status()
    search_results = response.json()["data"]
    image_urls = [result["thumbURL"] for result in search_results if "thumbURL" in result]
    
    return image_urls

app = Flask(__name__)
model, tokenizer, label_encoder, max_len, device = load_model_and_preprocessing()

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        text = request.form["text"]
        if text:
            emotion = predict_emotion(model, tokenizer, label_encoder, max_len, device, text)
            images = search_images(emotion, text)
            return render_template("index.html", text=text, emotion=emotion, images=images)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True, port=5001)  