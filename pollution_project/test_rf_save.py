#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试RF模型保存功能
"""

import os
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor

# 确保models目录存在
MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

# 创建一个简单的测试模型
print("创建测试RF模型...")
rf = RandomForestRegressor(
    n_estimators=10,
    max_depth=5,
    random_state=42,
    n_jobs=1  # 使用单线程避免序列化问题
)

# 生成一些随机数据进行训练
X = np.random.rand(100, 5)
y = np.random.rand(100)

print("训练测试模型...")
rf.fit(X, y)

# 测试保存
print("测试模型保存...")
try:
    model_path = os.path.abspath('models/test_rf_model_simple.pkl')
    print(f"保存路径: {model_path}")
    
    with open(model_path, 'wb') as f:
        pickle.dump(rf, f)
    
    print("模型保存成功！")
    
    # 检查文件是否存在
    if os.path.exists(model_path):
        print(f"文件存在: {model_path}")
        print(f"文件大小: {os.path.getsize(model_path)} 字节")
    else:
        print(f"文件不存在: {model_path}")
        
    # 测试加载
    print("测试模型加载...")
    with open(model_path, 'rb') as f:
        loaded_rf = pickle.load(f)
    
    print("模型加载成功！")
    
    # 测试预测
    test_X = np.random.rand(5, 5)
    predictions = loaded_rf.predict(test_X)
    print(f"预测结果: {predictions}")
    print("测试完成，模型保存和加载功能正常！")
    
except Exception as e:
    print(f"保存模型时出错: {e}")
    import traceback
    traceback.print_exc()
