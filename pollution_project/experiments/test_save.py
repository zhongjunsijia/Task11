#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试文件保存功能
"""

import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor

# 创建一个简单的测试模型
print("创建测试模型...")
X = np.random.rand(100, 10)
y = np.random.rand(100)

model = RandomForestRegressor(n_estimators=10, random_state=42)
model.fit(X, y)

# 测试保存到当前目录
print("测试保存到当前目录...")
try:
    joblib.dump(model, "test_model.pkl")
    print("成功保存到当前目录")
    # 检查文件是否存在
    if os.path.exists("test_model.pkl"):
        print("文件存在！")
    else:
        print("文件不存在！")
except Exception as e:
    print(f"保存失败: {e}")

# 测试保存到models目录
print("\n测试保存到models目录...")
try:
    if not os.path.exists('models'):
        os.makedirs('models')
        print("创建了models目录")
    joblib.dump(model, "models/test_model.pkl")
    print("成功保存到models目录")
    # 检查文件是否存在
    if os.path.exists("models/test_model.pkl"):
        print("文件存在！")
    else:
        print("文件不存在！")
except Exception as e:
    print(f"保存失败: {e}")

# 测试加载模型
print("\n测试加载模型...")
try:
    loaded_model = joblib.load("test_model.pkl")
    print("成功加载模型")
except Exception as e:
    print(f"加载失败: {e}")

print("\n测试完成！")
