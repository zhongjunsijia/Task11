#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练支持向量回归(SVR)模型

对应论文 3.1.1 节，属于传统机器学习模型，作为基准模型
"""

import os
import sys
import numpy as np
import joblib
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 确保数据目录存在
PROCESSED_DIR = os.path.join('data', 'processed')
MODEL_DIR = os.path.join('models')
os.makedirs(MODEL_DIR, exist_ok=True)

def load_data(horizon=1):
    """
    加载预处理好的数据集
    
    Args:
        horizon: 预测步长
    
    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test
    """
    horizon_dir = os.path.join(PROCESSED_DIR, f'horizon_{horizon}')
    
    try:
        X_train = np.load(os.path.join(horizon_dir, 'X_train.npy'))
        y_train = np.load(os.path.join(horizon_dir, 'y_train.npy'))
        X_val = np.load(os.path.join(horizon_dir, 'X_val.npy'))
        y_val = np.load(os.path.join(horizon_dir, 'y_val.npy'))
        X_test = np.load(os.path.join(horizon_dir, 'X_test.npy'))
        y_test = np.load(os.path.join(horizon_dir, 'y_test.npy'))
        
        print(f"数据加载成功，预测步长: {horizon}小时")
        print(f"训练集: {X_train.shape}, 验证集: {X_val.shape}, 测试集: {X_test.shape}")
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    except FileNotFoundError:
        print(f"错误: 未找到预测步长 {horizon} 的数据，请先运行数据处理脚本")
        sys.exit(1)

def train_svr_model(X_train, y_train, X_val, y_val, horizon=1):
    """
    训练SVR模型
    
    Args:
        X_train: 训练特征
        y_train: 训练标签
        X_val: 验证特征
        y_val: 验证标签
        horizon: 预测步长
    
    Returns:
        best_svr: 最优模型
    """
    print("=" * 80)
    print(f"训练SVR模型 (预测步长: {horizon}小时)")
    print("=" * 80)
    
    # 1. 模型初始化
    print("1. 模型初始化...")
    svr = SVR(
        kernel='rbf',  # 径向基函数核
        C=1.0,         # 正则化参数
        gamma='scale', # 核系数
        epsilon=0.1,   # 不敏感损失函数的参数
        cache_size=200 # 缓存大小
    )
    
    # 2. 超参数优化
    print("2. 超参数优化 (网格搜索 + 5折时间序列交叉验证)...")
    tscv = TimeSeriesSplit(n_splits=5)
    
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
        'epsilon': [0.01, 0.1, 0.2, 0.5]
    }
    
    grid_search = GridSearchCV(
        estimator=svr,
        param_grid=param_grid,
        cv=tscv,
        scoring='neg_mean_squared_error',  # 评价指标（负MSE，越大越好）
        n_jobs=-1,
        verbose=1
    )
    
    # 训练并寻找最优参数
    grid_search.fit(X_train, y_train)
    
    # 获取最优模型
    best_svr = grid_search.best_estimator_
    print(f"SVR最优超参数：{grid_search.best_params_}")
    print(f"最优交叉验证得分：{-grid_search.best_score_:.2f}")
    
    # 3. 模型训练
    print("3. 用最优超参数在完整训练集上训练...")
    best_svr.fit(X_train, y_train)
    
    # 4. 中间评估
    print("4. 在验证集上评估模型...")
    y_val_pred = best_svr.predict(X_val)
    
    mae = mean_absolute_error(y_val, y_val_pred)
    rmse = mean_squared_error(y_val, y_val_pred, squared=False)
    r2 = r2_score(y_val, y_val_pred)
    
    print(f"SVR验证集指标：")
    print(f"MAE = {mae:.2f}")
    print(f"RMSE = {rmse:.2f}")
    print(f"R² = {r2:.2f}")
    
    # 5. 模型保存
    print("5. 保存模型...")
    model_path = os.path.join(MODEL_DIR, f'svr_model_horizon{horizon}.pkl')
    joblib.dump(best_svr, model_path)
    print(f"模型已保存到: {model_path}")
    
    return best_svr

def evaluate_model(model, X_test, y_test, horizon=1):
    """
    在测试集上评估模型
    
    Args:
        model: 训练好的模型
        X_test: 测试特征
        y_test: 测试标签
        horizon: 预测步长
    """
    print(f"\n在测试集上评估SVR模型 (预测步长: {horizon}小时)...")
    
    y_test_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_test_pred)
    rmse = mean_squared_error(y_test, y_test_pred, squared=False)
    r2 = r2_score(y_test, y_test_pred)
    
    print(f"SVR测试集指标：")
    print(f"MAE = {mae:.2f}")
    print(f"RMSE = {rmse:.2f}")
    print(f"R² = {r2:.2f}")
    
    return mae, rmse, r2

def main():
    """
    主函数
    """
    # 测试不同的预测步长
    forecast_horizons = [1, 6, 12, 24]
    
    for horizon in forecast_horizons:
        try:
            # 加载数据
            X_train, y_train, X_val, y_val, X_test, y_test = load_data(horizon)
            
            # 训练模型
            best_svr = train_svr_model(X_train, y_train, X_val, y_val, horizon)
            
            # 评估模型
            evaluate_model(best_svr, X_test, y_test, horizon)
            
            print("\n" + "=" * 80)
            print(f"SVR模型训练完成 (预测步长: {horizon}小时)")
            print("=" * 80)
        except Exception as e:
            print(f"训练SVR模型时出错 (预测步长: {horizon}小时): {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
