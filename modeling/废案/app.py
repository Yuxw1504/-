#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 20:50:32 2024

@author: yxw
"""

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import numpy as np
import json
import webbrowser

MODEL_PATH = 'emotion_transformer_model.h5'
TOKENIZER_PATH = 'tokenizer.json'
EMOTIONS = ['happy', 'sad', 'angry', 'fear', 'surprise', 'neutral']  # 你的情绪类别

def load_tokenizer():
    with open(TOKENIZER_PATH, 'r', encoding='utf-8') as file:
        data = json.load(file)
        tokenizer = Tokenizer()
        tokenizer.word_index = data['word_index']
    return tokenizer

def load_emotion_model(model_path):
    return load_model(model_path)

def prepare_input(text, tokenizer, max_len=50):
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post')
    return padded_sequences

def predict_emotion(text, model, tokenizer):
    prepared_text = prepare_input(text, tokenizer)
    prediction = model.predict(prepared_text)
    return prediction

def main():
    print("加载模型和分词器...")
    model = load_emotion_model(MODEL_PATH)
    tokenizer = load_tokenizer()
    while True:
        text_input = input("请输入您的文本（输入'quit'退出）: ")
        if text_input.lower() == 'quit':
            break
        prediction = predict_emotion(text_input, model, tokenizer)
        emotion_index = np.argmax(prediction, axis=1)[0]
        emotion = EMOTIONS[emotion_index]
        print(f"预测结果：{emotion}")
        
        # 打开网页浏览器进行搜索
        '''search_query = f"{emotion} emoji"
        url = f"https://www.google.com/search?q={search_query}"
        webbrowser.open(url)'''

if __name__ == "__main__":
    main()







