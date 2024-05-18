#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: yxw
"""

import subprocess

# 在torch_env环境中运行数据增强脚本
def run_data_augmentation():
    subprocess.run(["conda", "run", "--name", "torch_env", "python", "data_augmentation.py"])

# 在tf_env环境中运行模型训练脚本
def run_model_training():
    subprocess.run(["conda", "run", "--name", "tf_env", "python", "train_model.py"])

# 执行数据增强和模型训练
run_data_augmentation()
run_model_training()
