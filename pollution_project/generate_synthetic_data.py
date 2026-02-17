#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成合成训练数据

为了快速开始模型训练和评价，直接生成符合格式要求的合成数据
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# 确保数据目录存在
PROCESSED_DIR = os.path.join('data', 'processed')
os.makedirs(PROCESSED_DIR, exist_ok=True)

# 配置参数
window_length = 24  # 时间窗口长度
n_features = 8      # 特征数量
n_samples = 10000   # 每个预测步长的样本数量
forecast_horizons = [1, 6, 12, 24]  # 预测步长

# 生成合成数据
def generate_synthetic_data(horizon, n_samples=10000):
    """
    生成合成训练数据
    
    Args:
        horizon: 预测步长
        n_samples: 样本数量
    
    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test
    """
    print(f"生成预测步长 {horizon} 小时的合成数据...")
    
    # 生成特征数据
    X = np.random.randn(n_samples, window_length * n_features)
    
    # 生成目标数据（基于特征的线性组合 + 噪声）
    # 模拟不同预测步长的难度
    noise_level = 5 + horizon * 0.5  # 预测步长越长，噪声越大
    weights = np.random.randn(window_length * n_features)
    y = X.dot(weights) + np.random.randn(n_samples) * noise_level
    
    # 确保目标值非负
    y = np.maximum(0, y)
    
    # 划分数据集
    # 训练集: 70%
    # 验证集: 15%
    # 测试集: 15%
    train_size = int(n_samples * 0.7)
    val_size = int(n_samples * 0.15)
    test_size = n_samples - train_size - val_size
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    
    X_val = X[train_size:train_size + val_size]
    y_val = y[train_size:train_size + val_size]
    
    X_test = X[train_size + val_size:]
    y_test = y[train_size + val_size:]
    
    print(f"数据生成完成: 训练集={X_train.shape[0]}, 验证集={X_val.shape[0]}, 测试集={X_test.shape[0]}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test

# 保存数据
def save_data(X_train, y_train, X_val, y_val, X_test, y_test, horizon):
    """
    保存数据到文件
    
    Args:
        X_train, y_train, X_val, y_val, X_test, y_test: 数据集
        horizon: 预测步长
    """
    horizon_dir = os.path.join(PROCESSED_DIR, f'horizon_{horizon}')
    os.makedirs(horizon_dir, exist_ok=True)
    
    np.save(os.path.join(horizon_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(horizon_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(horizon_dir, 'X_val.npy'), X_val)
    np.save(os.path.join(horizon_dir, 'y_val.npy'), y_val)
    np.save(os.path.join(horizon_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(horizon_dir, 'y_test.npy'), y_test)
    
    print(f"数据已保存到: {horizon_dir}")

# 主函数
def main():
    """
    主函数
    """
    print("=" * 80)
    print("生成合成训练数据")
    print("=" * 80)
    
    for horizon in forecast_horizons:
        try:
            # 生成数据
            X_train, y_train, X_val, y_val, X_test, y_test = generate_synthetic_data(horizon, n_samples)
            
            # 保存数据
            save_data(X_train, y_train, X_val, y_val, X_test, y_test, horizon)
            
            print(f"预测步长 {horizon} 小时的数据生成完成")
            print("-" * 80)
        except Exception as e:
            print(f"生成预测步长 {horizon} 小时的数据时出错: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("=" * 80)
    print("所有预测步长的数据生成完成！")
    print("=" * 80)

if __name__ == "__main__":
    main()
