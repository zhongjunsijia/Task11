#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练RF-LSTM-Attention混合模型

对应论文 3.2 节，核心创新模型，整合了RF特征选择、LSTM时序建模和Attention注意力机制
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Layer
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt

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

class AttentionLayer(Layer):
    """
    自定义Attention层（加性注意力）
    对应论文 3.2.3 节
    """
    def __init__(self):
        super(AttentionLayer, self).__init__()
    
    def build(self, input_shape):
        # 输入形状：(batch_size, time_steps, units)
        self.W = self.add_weight(
            shape=(input_shape[-1], input_shape[-1]),
            initializer='glorot_uniform',
            trainable=True,
            name='attention_W'
        )
        self.b = self.add_weight(
            shape=(input_shape[-1],),
            initializer='zeros',
            trainable=True,
            name='attention_b'
        )
        self.v = self.add_weight(
            shape=(input_shape[-1], 1),
            initializer='glorot_uniform',
            trainable=True,
            name='attention_v'
        )
        super(AttentionLayer, self).build(input_shape)
    
    def call(self, inputs):
        # inputs: (batch_size, time_steps, units)
        e = tf.tanh(tf.matmul(tf.reshape(inputs, (-1, inputs.shape[-1])), self.W) + self.b)
        e = tf.reshape(e, (-1, inputs.shape[1], inputs.shape[-1]))
        alpha = tf.nn.softmax(tf.matmul(e, self.v), axis=1)  # 注意力权重 (batch_size, time_steps, 1)
        context = tf.reduce_sum(inputs * alpha, axis=1)  # 上下文向量 (batch_size, units)
        return context, alpha  # 返回上下文向量和注意力权重

def train_rf_lstm_attention_model(X_train, y_train, X_val, y_val, X_test, y_test, horizon=1):
    """
    训练RF-LSTM-Attention混合模型
    
    Args:
        X_train: 训练特征
        y_train: 训练标签
        X_val: 验证特征
        y_val: 验证标签
        X_test: 测试特征
        y_test: 测试标签
        horizon: 预测步长
    
    Returns:
        best_model: 最优模型
    """
    print("=" * 80)
    print(f"训练RF-LSTM-Attention模型 (预测步长: {horizon}小时)")
    print("=" * 80)
    
    # 第一步：RF特征选择
    print("1. RF特征选择...")
    try:
        # 加载之前训练好的最优RF模型
        best_rf = joblib.load("rf_best_model.pkl")
        feature_importance = best_rf.feature_importances_
        
        # 选取累计贡献率达80%的Top-K特征
        cumulative_importance = np.cumsum(sorted(feature_importance, reverse=True))
        K = np.where(cumulative_importance >= 0.8)[0][0] + 1
        top_feature_indices = np.argsort(feature_importance)[-K:]
        
        print(f"选取的Top-K特征数量: {K}")
        print(f"Top-K特征索引: {top_feature_indices}")
        
        # 筛选Top-K特征，生成降维后的特征矩阵
        X_train_rf = X_train[:, top_feature_indices]
        X_val_rf = X_val[:, top_feature_indices]
        X_test_rf = X_test[:, top_feature_indices]
        
        print(f"降维后特征形状 - 训练集: {X_train_rf.shape}, 验证集: {X_val_rf.shape}")
    except FileNotFoundError:
        print("错误: 未找到RF模型文件 'rf_best_model.pkl'，请先训练RF模型")
        sys.exit(1)
    
    # 第二步：调整数据格式
    print("2. 调整数据格式...")
    X_train_rf_lstm, y_train_rf_lstm = create_lstm_input(X_train_rf, y_train, L)
    X_val_rf_lstm, y_val_rf_lstm = create_lstm_input(X_val_rf, y_val, L)
    X_test_rf_lstm, y_test_rf_lstm = create_lstm_input(X_test_rf, y_test, L)
    
    print(f"LSTM输入形状: {X_train_rf_lstm.shape}")
    
    # 第三步：实现Attention注意力机制
    print("3. 实现Attention注意力机制...")
    # 已在上面定义了AttentionLayer类
    
    # 第四步：搭建混合模型架构
    print("4. 搭建混合模型架构...")
    # 输入层：(batch_size, L, K)，K为Top-K特征数
    inputs = Input(shape=(L, X_train_rf_lstm.shape[2]))
    # LSTM层：返回序列=True（输出所有时间步，供Attention使用）
    lstm1 = LSTM(128, return_sequences=True)(inputs)
    lstm1 = Dropout(0.2)(lstm1)
    lstm2 = LSTM(64, return_sequences=True)(lstm1)
    lstm2 = Dropout(0.2)(lstm2)
    # Attention层：接收LSTM输出，输出上下文向量
    attention_layer = AttentionLayer()
    context_vector, attention_weights = attention_layer(lstm2)
    # 全连接层
    dense1 = Dense(32, activation='relu')(context_vector)
    dense1 = Dropout(0.2)(dense1)
    outputs = Dense(1)(dense1)  # 输出预测值
    # 构建模型（同时输出预测值和注意力权重，便于后续分析）
    rf_lstm_attention = Model(inputs=inputs, outputs=[outputs, attention_weights])
    # 编译模型
    rf_lstm_attention.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss='mean_squared_error',
        metrics=['mean_absolute_error']
    )
    
    rf_lstm_attention.summary()
    
    # 第五步：模型训练
    print("5. 模型训练...")
    
    # 回调函数
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        'rf_lstm_attention_best_model.h5', monitor='val_loss', save_best_only=True
    )
    
    # 训练（批量大小、epochs参考LSTM的最优参数）
    history = rf_lstm_attention.fit(
        X_train_rf_lstm, [y_train_rf_lstm, np.zeros_like(y_train_rf_lstm)],  # 注意力权重无监督，用占位符
        batch_size=64,
        epochs=50,
        validation_data=(X_val_rf_lstm, [y_val_rf_lstm, np.zeros_like(y_val_rf_lstm)]),
        callbacks=[early_stop, model_checkpoint],
        verbose=1
    )
    
    # 第六步：模型评估与注意力权重分析
    print("6. 模型评估与注意力权重分析...")
    # 预测
    y_val_pred, attention_weights_val = rf_lstm_attention.predict(X_val_rf_lstm)
    # 评估指标
    mae = mean_absolute_error(y_val_rf_lstm, y_val_pred)
    rmse = mean_squared_error(y_val_rf_lstm, y_val_pred, squared=False)
    r2 = r2_score(y_val_rf_lstm, y_val_pred)
    
    print(f"RF-LSTM-Attention验证集指标：")
    print(f"MAE = {mae:.2f}")
    print(f"RMSE = {rmse:.2f}")
    print(f"R² = {r2:.2f}")
    
    # 注意力权重可视化（查看某样本的权重分布）
    print("7. 注意力权重可视化...")
    sample_idx = 0
    weights = attention_weights_val[sample_idx].flatten()
    plt.plot(range(L), weights)
    plt.xlabel("时间步（历史24小时）")
    plt.ylabel("注意力权重")
    plt.title("某样本的注意力权重分布")
    plt.savefig(f'attention_weights_sample_{horizon}.png')
    print(f"注意力权重图已保存到: attention_weights_sample_{horizon}.png")
    
    # 模型保存
    print("8. 保存模型...")
    model_path = 'rf_lstm_attention_best_model.h5'
    rf_lstm_attention.save(model_path)
    print(f"模型已保存到: {model_path}")
    
    return rf_lstm_attention

def evaluate_model(model, X_test, y_test, horizon=1):
    """
    在测试集上评估模型
    
    Args:
        model: 训练好的模型
        X_test: 测试特征
        y_test: 测试标签
        horizon: 预测步长
    """
    print(f"\n在测试集上评估RF-LSTM-Attention模型 (预测步长: {horizon}小时)...")
    
    # 加载RF模型获取特征选择信息
    try:
        best_rf = joblib.load("rf_best_model.pkl")
        feature_importance = best_rf.feature_importances_
        cumulative_importance = np.cumsum(sorted(feature_importance, reverse=True))
        K = np.where(cumulative_importance >= 0.8)[0][0] + 1
        top_feature_indices = np.argsort(feature_importance)[-K:]
        
        # 筛选Top-K特征
        X_test_rf = X_test[:, top_feature_indices]
        
        # 转换测试数据
        X_test_rf_lstm, y_test_rf_lstm = create_lstm_input(X_test_rf, y_test, L)
        
        # 预测
        y_test_pred, _ = model.predict(X_test_rf_lstm)
        
        # 评估
        test_mae = mean_absolute_error(y_test_rf_lstm, y_test_pred)
        test_rmse = mean_squared_error(y_test_rf_lstm, y_test_pred, squared=False)
        test_r2 = r2_score(y_test_rf_lstm, y_test_pred)
        
        print(f"RF-LSTM-Attention测试集指标：")
        print(f"MAE = {test_mae:.2f}")
        print(f"RMSE = {test_rmse:.2f}")
        print(f"R² = {test_r2:.2f}")
        
        return test_mae, test_rmse, test_r2
    except FileNotFoundError:
        print("错误: 未找到RF模型文件 'rf_best_model.pkl'")
        return None, None, None

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
            best_model = train_rf_lstm_attention_model(X_train, y_train, X_val, y_val, X_test, y_test, horizon)
            
            # 评估模型
            evaluate_model(best_model, X_test, y_test, horizon)
            
            print("\n" + "=" * 80)
            print(f"RF-LSTM-Attention模型训练完成 (预测步长: {horizon}小时)")
            print("=" * 80)
        except Exception as e:
            print(f"训练RF-LSTM-Attention模型时出错 (预测步长: {horizon}小时): {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
