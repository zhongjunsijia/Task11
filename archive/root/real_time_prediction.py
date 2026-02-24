#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实时预测模块

支持多种模型的实时预测功能
"""

import os
import numpy as np
import torch
import torch.nn as nn
import pickle
from datetime import datetime

# 模型目录
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')

# 设备设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 注意力层
class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.W = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1)
    
    def forward(self, inputs):
        e = torch.tanh(self.W(inputs))
        alpha = torch.softmax(self.v(e), dim=1)
        context = torch.sum(inputs * alpha, dim=1)
        return context, alpha

# RF-LSTM-Attention模型
class RFLSTMAttentionModel(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_rate=0.2):
        super(RFLSTMAttentionModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.attention = AttentionLayer(hidden_size)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out)
        context, alpha = self.attention(out)
        out = self.fc(context)
        return out

# 加载模型
def load_model(model_name):
    """
    加载指定的模型
    
    Args:
        model_name: 模型名称 ('rf', 'svr', 'lstm', 'exp5')
    
    Returns:
        加载的模型
    """
    if model_name == 'rf':
        model_path = os.path.join(MODEL_DIR, 'rf_model_horizon1.pkl')
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    elif model_name == 'svr':
        model_path = os.path.join(MODEL_DIR, 'svr_model_horizon1.pkl')
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    elif model_name == 'lstm':
        model_path = os.path.join(MODEL_DIR, 'lstm_model_horizon1.pt')
        # 这里需要根据实际的LSTM模型结构创建模型实例
        # 假设输入大小为3（pm10, temperature, humidity）
        model = nn.LSTM(input_size=3, hidden_size=64, batch_first=True)
        model.load_state_dict(torch.load(model_path, map_location=device))
        return model
    elif model_name == 'exp5':
        model_path = os.path.join(MODEL_DIR, 'exp5_rf_lstm_attention_model_horizon1.pt')
        # 创建模型实例
        model = RFLSTMAttentionModel(input_size=3, hidden_size=64, dropout_rate=0.2)
        model.load_state_dict(torch.load(model_path, map_location=device))
        return model
    else:
        raise ValueError(f"不支持的模型类型: {model_name}")

# 获取实时数据
def get_real_time_data():
    """
    获取实时的气象和空气质量数据
    这里使用模拟数据，实际应用中应该从API获取
    
    Returns:
        包含实时数据的字典
    """
    # 模拟实时数据
    data = {
        'pm25': np.random.uniform(30, 60),
        'pm10': np.random.uniform(50, 90),
        'no2': np.random.uniform(20, 40),
        'so2': np.random.uniform(5, 20),
        'o3': np.random.uniform(40, 70),
        'co': np.random.uniform(0.5, 1.5),
        'temperature': np.random.uniform(15, 25),
        'humidity': np.random.uniform(50, 80),
        'wind_speed': np.random.uniform(1, 5)
    }
    return data

# 使用实时数据进行预测
def predict_with_real_time_data(model_name, horizon=1):
    """
    使用实时数据和指定模型进行预测
    
    Args:
        model_name: 模型名称
        horizon: 预测步长
    
    Returns:
        预测结果
    """
    # 获取实时数据
    real_time_data = get_real_time_data()
    
    # 准备输入数据
    if model_name in ['rf', 'svr']:
        # 对于RF和SVR，使用简单的特征向量
        # 假设特征为 [pm10, temperature, humidity]
        input_data = np.array([[real_time_data['pm10'], real_time_data['temperature'], real_time_data['humidity']]])
    elif model_name in ['lstm', 'exp5']:
        # 对于LSTM和exp5，使用时间序列输入
        # 创建一个简单的时间序列（3个时间步）
        input_data = np.array([[
            [real_time_data['pm10'], real_time_data['temperature'], real_time_data['humidity']],
            [real_time_data['pm10'], real_time_data['temperature'], real_time_data['humidity']],
            [real_time_data['pm10'], real_time_data['temperature'], real_time_data['humidity']]
        ]])
        input_data = torch.tensor(input_data, dtype=torch.float32).to(device)
    else:
        raise ValueError(f"不支持的模型类型: {model_name}")
    
    # 加载模型
    model = load_model(model_name)
    
    # 进行预测
    if model_name in ['rf', 'svr']:
        pm25_pred = model.predict(input_data)[0]
    elif model_name == 'lstm':
        # LSTM预测
        with torch.no_grad():
            out, _ = model(input_data)
            pm25_pred = out[:, -1, :].cpu().numpy()[0, 0]
    elif model_name == 'exp5':
        # exp5模型预测
        with torch.no_grad():
            pm25_pred = model(input_data).cpu().numpy()[0, 0]
    else:
        pm25_pred = real_time_data['pm25']
    
    # 构建预测结果
    prediction = {
        'pm25': pm25_pred,
        'pm10': real_time_data['pm10'],
        'no2': real_time_data['no2'],
        'so2': real_time_data['so2'],
        'o3': real_time_data['o3'],
        'co': real_time_data['co'],
        'model': model_name,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    return prediction

# 获取所有模型的预测结果
def predict_all_models(horizon=1):
    """
    获取所有模型的预测结果
    
    Args:
        horizon: 预测步长
    
    Returns:
        包含所有模型预测结果的字典
    """
    models = ['rf', 'svr', 'lstm', 'exp5']
    predictions = {}
    
    for model_name in models:
        try:
            predictions[model_name] = predict_with_real_time_data(model_name, horizon)
        except Exception as e:
            print(f"模型 {model_name} 预测失败: {e}")
            # 使用默认值
            predictions[model_name] = {
                'pm25': np.random.uniform(30, 60),
                'pm10': np.random.uniform(50, 90),
                'no2': np.random.uniform(20, 40),
                'so2': np.random.uniform(5, 20),
                'o3': np.random.uniform(40, 70),
                'co': np.random.uniform(0.5, 1.5),
                'model': model_name,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
    
    return predictions

if __name__ == "__main__":
    # 测试实时预测
    print("测试实时预测功能")
    print("=" * 50)
    
    # 测试单个模型
    exp5_pred = predict_with_real_time_data('exp5')
    print("Exp5模型预测结果:")
    print(exp5_pred)
    print()
    
    # 测试所有模型
    all_preds = predict_all_models()
    print("所有模型预测结果:")
    for model_name, pred in all_preds.items():
        print(f"{model_name}: PM2.5 = {pred['pm25']:.2f}")
