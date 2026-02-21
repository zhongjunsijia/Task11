#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试模型保存功能
"""

import os
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor

# 确保models目录存在
if not os.path.exists('models'):
    os.makedirs('models')
    print("创建了models目录")

# 创建一个简单的测试模型
X = np.random.rand(100, 5)
y = np.random.rand(100)

model = RandomForestRegressor(n_estimators=10, random_state=42)
model.fit(X, y)

# 测试保存模型
try:
    # 保存到models目录
    model_path = os.path.abspath('models/test_rf_model.pkl')
    print(f"尝试保存到: {model_path}")
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"模型已保存到: {model_path}")
    
    # 检查文件是否存在
    if os.path.exists(model_path):
        print(f"文件存在: {model_path}")
        print(f"文件大小: {os.path.getsize(model_path)} 字节")
    else:
        print(f"文件不存在: {model_path}")
        
    # 测试加载模型
    with open(model_path, 'rb') as f:
        loaded_model = pickle.load(f)
    print("模型加载成功")
    
    # 测试模型预测
    test_X = np.random.rand(5, 5)
    predictions = loaded_model.predict(test_X)
    print(f"模型预测成功: {predictions}")
    
except Exception as e:
    print(f"保存模型时出错: {e}")
    import traceback
    traceback.print_exc()

# 列出models目录中的文件
print("\nmodels目录中的文件:")
for file in os.listdir('models'):
    print(f"- {file}")
