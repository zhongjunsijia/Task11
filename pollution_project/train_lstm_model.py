#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练长短期记忆网络(LSTM)模型

对应论文 3.1.3 节，属于深度学习模型，作为基准模型
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import optuna

# 确保数据目录存在
PROCESSED_DIR = os.path.join('data', 'processed')
MODEL_DIR = os.path.join('models')
os.makedirs(MODEL_DIR, exist_ok=True)

# 时间窗口长度
L = 24

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

def create_lstm_input(X, y, window_size):
    """
    将2D特征矩阵转换为LSTM所需的3D张量
    
    Args:
        X: 2D特征矩阵
        y: 目标变量
        window_size: 时间窗口大小
    
    Returns:
        X_lstm: 3D张量 (样本数, 时间步, 特征数)
        y_lstm: 目标变量
    """
    X_lstm, y_lstm = [], []
    for i in range(window_size, len(X)):
        X_lstm.append(X[i-window_size:i, :])  # 历史window_size小时的特征
        y_lstm.append(y[i])  # 未来1小时的PM2.5浓度
    return np.array(X_lstm), np.array(y_lstm)

def build_lstm_model(input_shape, units=128, dropout_rate=0.2, learning_rate=0.001):
    """
    构建LSTM模型
    
    Args:
        input_shape: 输入形状 (时间步, 特征数)
        units: LSTM单元数
        dropout_rate: Dropout率
        learning_rate: 学习率
    
    Returns:
        构建好的LSTM模型
    """
    model = Sequential([
        LSTM(units=units, input_shape=input_shape),
        Dropout(dropout_rate),
        Dense(64, activation='relu'),
        Dropout(dropout_rate),
        Dense(1)
    ])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='mean_squared_error',
        metrics=['mean_absolute_error']
    )
    
    return model

def optimize_hyperparameters(X_train_lstm, y_train_lstm, X_val_lstm, y_val_lstm):
    """
    使用Optuna进行贝叶斯优化
    
    Args:
        X_train_lstm: 训练数据
        y_train_lstm: 训练标签
        X_val_lstm: 验证数据
        y_val_lstm: 验证标签
    
    Returns:
        最优超参数
    """
    print("开始超参数优化...")
    
    def objective(trial):
        # 待优化超参数
        units = trial.suggest_categorical('units', [64, 128, 256])
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.3)
        learning_rate = trial.suggest_categorical('lr', [0.0001, 0.0005, 0.001])
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
        
        # 构建模型
        model = Sequential([
            LSTM(units=units, input_shape=(L, X_train_lstm.shape[2])),
            Dropout(dropout_rate),
            Dense(64, activation='relu'),
            Dropout(dropout_rate),
            Dense(1)
        ])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='mean_squared_error'
        )
        
        # 训练（加入早停机制）
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=5, restore_best_weights=True
        )
        history = model.fit(
            X_train_lstm, y_train_lstm,
            batch_size=batch_size,
            epochs=50,
            validation_data=(X_val_lstm, y_val_lstm),
            callbacks=[early_stop],
            verbose=0
        )
        
        # 返回验证集RMSE
        val_rmse = np.sqrt(min(history.history['val_loss']))
        return val_rmse
    
    # 运行贝叶斯优化（最大评估30次）
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=30)
    
    print(f"LSTM最优超参数：{study.best_params}")
    print(f"最优验证集RMSE：{study.best_value:.2f}")
    
    return study.best_params

def train_lstm_model(X_train, y_train, X_val, y_val, X_test, y_test, horizon=1):
    """
    训练LSTM模型
    
    Args:
        X_train: 训练特征
        y_train: 训练标签
        X_val: 验证特征
        y_val: 验证标签
        X_test: 测试特征
        y_test: 测试标签
        horizon: 预测步长
    
    Returns:
        best_lstm: 最优模型
    """
    print("=" * 80)
    print(f"训练LSTM模型 (预测步长: {horizon}小时)")
    print("=" * 80)
    
    # 1. 数据格式调整
    print("1. 数据格式调整...")
    X_train_lstm, y_train_lstm = create_lstm_input(X_train, y_train, L)
    X_val_lstm, y_val_lstm = create_lstm_input(X_val, y_val, L)
    X_test_lstm, y_test_lstm = create_lstm_input(X_test, y_test, L)
    
    print(f"LSTM输入形状: {X_train_lstm.shape}")
    
    # 2. 超参数优化
    print("2. 超参数优化...")
    best_params = optimize_hyperparameters(X_train_lstm, y_train_lstm, X_val_lstm, y_val_lstm)
    
    # 3. 构建最优模型
    print("3. 构建最优模型...")
    # 最优超参数
    print("LSTM最优超参数：", best_params)
    # 重新构建最优模型
    final_lstm = Sequential([
        LSTM(units=best_params['units'], input_shape=(L, X_train_lstm.shape[2])),
        Dropout(best_params['dropout_rate']),
        Dense(64, activation='relu'),
        Dropout(best_params['dropout_rate']),
        Dense(1)
    ])
    final_lstm.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=best_params['lr']),
        loss='mean_squared_error'
    )
    
    final_lstm.summary()
    
    # 4. 模型训练
    print("4. 模型训练...")
    
    # 回调函数
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        'lstm_best_model.h5', monitor='val_loss', save_best_only=True
    )
    
    # 训练
    history = final_lstm.fit(
        X_train_lstm, y_train_lstm,
        batch_size=best_params['batch_size'],
        epochs=50,
        validation_data=(X_val_lstm, y_val_lstm),
        callbacks=[early_stop, model_checkpoint],
        verbose=1
    )
    
    # 5. 中间评估
    print("5. 在验证集上评估模型...")
    val_loss, val_mae = final_lstm.evaluate(X_val_lstm, y_val_lstm, verbose=0)
    val_rmse = np.sqrt(val_loss)
    
    print(f"LSTM验证集指标：")
    print(f"MAE = {val_mae:.2f}")
    print(f"RMSE = {val_rmse:.2f}")
    
    # 6. 模型保存
    print("6. 保存模型...")
    model_path = 'lstm_best_model.h5'
    final_lstm.save(model_path)
    print(f"模型已保存到: {model_path}")
    
    return final_lstm

def evaluate_model(model, X_test, y_test, horizon=1):
    """
    在测试集上评估模型
    
    Args:
        model: 训练好的模型
        X_test: 测试特征
        y_test: 测试标签
        horizon: 预测步长
    """
    print(f"\n在测试集上评估LSTM模型 (预测步长: {horizon}小时)...")
    
    # 转换测试数据
    X_test_lstm, y_test_lstm = create_lstm_input(X_test, y_test, L)
    
    # 评估
    test_loss, test_mae = model.evaluate(X_test_lstm, y_test_lstm, verbose=0)
    test_rmse = np.sqrt(test_loss)
    
    # 预测
    y_test_pred = model.predict(X_test_lstm).flatten()
    r2 = r2_score(y_test_lstm, y_test_pred)
    
    print(f"LSTM测试集指标：")
    print(f"MAE = {test_mae:.2f}")
    print(f"RMSE = {test_rmse:.2f}")
    print(f"R² = {r2:.2f}")
    
    return test_mae, test_rmse, r2

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
            best_lstm = train_lstm_model(X_train, y_train, X_val, y_val, X_test, y_test, horizon)
            
            # 评估模型
            evaluate_model(best_lstm, X_test, y_test, horizon)
            
            print("\n" + "=" * 80)
            print(f"LSTM模型训练完成 (预测步长: {horizon}小时)")
            print("=" * 80)
        except Exception as e:
            print(f"训练LSTM模型时出错 (预测步长: {horizon}小时): {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
