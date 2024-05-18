#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: yxw
"""
import json
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, MultiHeadAttention, Dense, LayerNormalization, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
import numpy as np
import os
import matplotlib.pyplot as plt

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
val_texts, val_labels = load_data('val.txt')
test_texts, test_labels = load_data('test.txt')

# 将数据集转换为DataFrame
train_df = pd.DataFrame({'Text': train_texts, 'Emotion': train_labels})
val_df = pd.DataFrame({'Text': val_texts, 'Emotion': val_labels})
test_df = pd.DataFrame({'Text': test_texts, 'Emotion': test_labels})

# 合并数据集以进行整体统计分析
all_data_df = pd.concat([train_df, val_df, test_df])

# 获取文本统计信息
text_stats = all_data_df['Text'].describe()

# 获取情感分布
emotion_counts = all_data_df['Emotion'].value_counts()

# 可视化情感分布
plt.figure(figsize=(10, 5))
emotion_counts.plot(kind='bar')
plt.title('Emotion Distribution')
plt.xlabel('Emotion')
plt.ylabel('Counts')
plt.show()

# 输出文本统计信息和情感分布
print(text_stats)
print(emotion_counts)

# 文本分词和序列化
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_texts)
train_seqs = tokenizer.texts_to_sequences(train_texts)
val_seqs = tokenizer.texts_to_sequences(val_texts)
test_seqs = tokenizer.texts_to_sequences(test_texts)

# 序列填充
max_len = 50  # 根据需要调整
train_seqs = pad_sequences(train_seqs, maxlen=max_len, padding='post')
val_seqs = pad_sequences(val_seqs, maxlen=max_len, padding='post')
test_seqs = pad_sequences(test_seqs, maxlen=max_len, padding='post')

# 标签编码
label_tokenizer = Tokenizer()
label_tokenizer.fit_on_texts(train_labels)
train_labels = np.array(label_tokenizer.texts_to_sequences(train_labels))
val_labels = np.array(label_tokenizer.texts_to_sequences(val_labels))
test_labels = np.array(label_tokenizer.texts_to_sequences(test_labels))

#构建模型
def transformer_model(vocab_size, num_labels):
    inputs = Input(shape=(max_len,))
    x = Embedding(input_dim=vocab_size, output_dim=128)(inputs)

    # Transformer block with Dropout and possible L2 regularization
    attn_output = MultiHeadAttention(num_heads=2, key_dim=128)(x, x) #多头注意力
    attn_output = Dropout(0.2)(attn_output)  # Dropout率
    out1 = LayerNormalization(epsilon=1e-6)(x + attn_output)

    ffn_output = Dense(2048, activation='relu', kernel_regularizer=l2(0.05))(out1)  # L2正则化
    ffn_output = Dense(128, activation='relu')(ffn_output)
    ffn_output = Dropout(0.2)(ffn_output)  # Dropout率
    out2 = LayerNormalization(epsilon=1e-6)(out1 + ffn_output)

    final_output = out2[:, -1, :]
    outputs = Dense(num_labels, activation='softmax')(final_output)

    model = Model(inputs=inputs, outputs=outputs)
    return model

# 模型实例化和编译
vocab_size = len(tokenizer.word_index) + 1
num_labels = len(label_tokenizer.word_index) + 1
# 重新构建模型
model = transformer_model(vocab_size, num_labels)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 设置提前停止
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# 开始训练
history = model.fit(
    train_seqs, train_labels,
    validation_data=(val_seqs, val_labels),
    epochs=10,
    batch_size=32,
    callbacks=[early_stopping]  # 添加回调函数
)

# 保存模型
model.save('emotion_transformer_model.h5')

def save_tokenizer(tokenizer, file_path):
    tokenizer_data = {'word_index': tokenizer.word_index}
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(tokenizer_data, f, ensure_ascii=False)

# 在模型训练脚本中调用此函数以保存分词器
save_tokenizer(tokenizer, 'tokenizer.json')


# 加载模型
loaded_model = tf.keras.models.load_model('transformer_model.h5')

# 使用加载的模型进行测试
test_loss, test_acc = loaded_model.evaluate(test_seqs, test_labels)
print("Test Accuracy: ", test_acc)

# 训练和验证的准确率曲线
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy over epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# 训练和验证的损失曲线
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss over epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# 随机选择几个测试样本
indices = np.random.choice(range(len(test_seqs)), 5)
samples = test_seqs[indices]
labels = test_labels[indices]

# 进行预测
predictions = loaded_model.predict(samples)
predicted_classes = np.argmax(predictions, axis=1)
actual_classes = labels.flatten()

# 可视化预测结果和实际标签
fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(15, 3))
for i, ax in enumerate(axes):
    ax.imshow(samples[i].reshape(1, -1), cmap='viridis', aspect='auto')
    ax.set_title(f'Pred: {predicted_classes[i]}\nActual: {actual_classes[i]}')
    ax.axis('off')

plt.show()


