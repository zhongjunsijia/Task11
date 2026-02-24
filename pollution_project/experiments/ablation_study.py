#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
消融实验脚本

实现论文中定义的6个消融实验：
Exp1: 仅RF（随机森林）
Exp2: 仅LSTM（无特征选择、无注意力）
Exp3: RF + LSTM（无注意力机制）
Exp4: LSTM + Attention（无RF特征选择）
Exp5: RF-LSTM-Attention（完整模型）
Exp6: XGBoost + LSTM + Attention
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 确保数据目录存在
PROCESSED_DIR = os.path.join('data', 'processed')
# 模型保存到上级目录的models文件夹
MODEL_DIR = os.path.abspath(os.path.join('..', 'models'))
os.makedirs(MODEL_DIR, exist_ok=True)

# 时间窗口长度
L = 24

# 设备设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

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

class AttentionLayer(nn.Module):
    """
    PyTorch Attention层（加性注意力）
    对应论文 3.2.3 节
    """
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.W = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1)
    
    def forward(self, inputs):
        # inputs: (batch_size, time_steps, hidden_size)
        e = torch.tanh(self.W(inputs))  # (batch_size, time_steps, hidden_size)
        alpha = torch.softmax(self.v(e), dim=1)  # 注意力权重 (batch_size, time_steps, 1)
        context = torch.sum(inputs * alpha, dim=1)  # 上下文向量 (batch_size, hidden_size)
        return context, alpha

class LSTMModel(nn.Module):
    """
    基础LSTM模型
    """
    def __init__(self, input_size, hidden_size, dropout_rate=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # 取最后一个时间步的输出
        out = self.dropout(out)
        out = self.fc(out)
        return out

class LSTMAttentionModel(nn.Module):
    """
    LSTM + Attention模型
    """
    def __init__(self, input_size, hidden_size, dropout_rate=0.2):
        super(LSTMAttentionModel, self).__init__()
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

class RFLSTMModel(nn.Module):
    """
    RF + LSTM模型（无注意力机制）
    """
    def __init__(self, input_size, hidden_size, dropout_rate=0.2):
        super(RFLSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # 取最后一个时间步的输出
        out = self.dropout(out)
        out = self.fc(out)
        return out

class RFLSTMAttentionModel(nn.Module):
    """
    RF-LSTM-Attention完整模型
    """
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

def train_epoch(model, dataloader, optimizer, criterion):
    """
    训练一个epoch
    """
    model.train()
    total_loss = 0
    for X_batch, y_batch in dataloader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch.unsqueeze(1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate_epoch(model, dataloader, criterion):
    """
    评估一个epoch
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            output = model(X_batch)
            loss = criterion(output, y_batch.unsqueeze(1))
            total_loss += loss.item()
            all_preds.extend(output.cpu().numpy().flatten())
            all_targets.extend(y_batch.cpu().numpy())
    return total_loss / len(dataloader), np.array(all_preds), np.array(all_targets)

def create_dataloader(X, y, batch_size, shuffle=True):
    """
    创建数据加载器
    """
    dataset = torch.utils.data.TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32)
    )
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle
    )

def train_rf_model(X_train, y_train, X_val, y_val, X_test, y_test, horizon=1):
    """
    训练仅RF模型（Exp1）
    """
    print("=" * 80)
    print(f"训练仅RF模型 (Exp1) (预测步长: {horizon}小时)")
    print("=" * 80)
    
    # 训练RF模型
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        random_state=42,
        n_jobs=1
    )
    
    rf.fit(X_train, y_train)
    
    # 评估
    y_val_pred = rf.predict(X_val)
    y_test_pred = rf.predict(X_test)
    
    val_mae = mean_absolute_error(y_val, y_val_pred)
    val_mse = mean_squared_error(y_val, y_val_pred)
    val_rmse = np.sqrt(val_mse)
    val_r2 = r2_score(y_val, y_val_pred)
    
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = np.sqrt(test_mse)
    test_r2 = r2_score(y_test, y_test_pred)
    
    print(f"RF验证集指标：")
    print(f"MAE = {val_mae:.2f}")
    print(f"RMSE = {val_rmse:.2f}")
    print(f"R² = {val_r2:.2f}")
    
    print(f"\nRF测试集指标：")
    print(f"MAE = {test_mae:.2f}")
    print(f"RMSE = {test_rmse:.2f}")
    print(f"R² = {test_r2:.2f}")
    
    # 保存模型
    model_path = os.path.join(MODEL_DIR, f'exp1_rf_model_horizon{horizon}.pkl')
    import pickle
    with open(model_path, 'wb') as f:
        pickle.dump(rf, f)
    print(f"\n模型已保存到: {model_path}")
    
    return rf

def train_lstm_model(X_train, y_train, X_val, y_val, X_test, y_test, horizon=1):
    """
    训练仅LSTM模型（Exp2）
    """
    print("=" * 80)
    print(f"训练仅LSTM模型 (Exp2) (预测步长: {horizon}小时)")
    print("=" * 80)
    
    # 数据格式调整
    X_train_lstm, y_train_lstm = create_lstm_input(X_train, y_train, L)
    X_val_lstm, y_val_lstm = create_lstm_input(X_val, y_val, L)
    X_test_lstm, y_test_lstm = create_lstm_input(X_test, y_test, L)
    
    print(f"LSTM输入形状: {X_train_lstm.shape}")
    
    # 构建模型
    input_size = X_train_lstm.shape[2]
    model = LSTMModel(
        input_size=input_size,
        hidden_size=64,
        dropout_rate=0.2
    ).to(device)
    
    # 训练
    batch_size = 64
    train_loader = create_dataloader(X_train_lstm, y_train_lstm, batch_size, shuffle=True)
    val_loader = create_dataloader(X_val_lstm, y_val_lstm, batch_size, shuffle=False)
    test_loader = create_dataloader(X_test_lstm, y_test_lstm, batch_size, shuffle=False)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(20):
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        val_loss, _, _ = evaluate_epoch(model, val_loader, criterion)
        
        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("早停！")
                break
    
    # 加载最优模型
    model.load_state_dict(best_model_state)
    
    # 评估
    val_loss, val_preds, val_targets = evaluate_epoch(model, val_loader, criterion)
    val_rmse = np.sqrt(val_loss)
    val_mae = mean_absolute_error(val_targets, val_preds)
    val_r2 = r2_score(val_targets, val_preds)
    
    test_loss, test_preds, test_targets = evaluate_epoch(model, test_loader, criterion)
    test_rmse = np.sqrt(test_loss)
    test_mae = mean_absolute_error(test_targets, test_preds)
    test_r2 = r2_score(test_targets, test_preds)
    
    print(f"\nLSTM验证集指标：")
    print(f"MAE = {val_mae:.2f}")
    print(f"RMSE = {val_rmse:.2f}")
    print(f"R² = {val_r2:.2f}")
    
    print(f"\nLSTM测试集指标：")
    print(f"MAE = {test_mae:.2f}")
    print(f"RMSE = {test_rmse:.2f}")
    print(f"R² = {test_r2:.2f}")
    
    # 保存模型
    model_path = os.path.join(MODEL_DIR, f'exp2_lstm_model_horizon{horizon}.pt')
    torch.save(model.state_dict(), model_path)
    print(f"\n模型已保存到: {model_path}")
    
    return model

def train_rf_lstm_model(X_train, y_train, X_val, y_val, X_test, y_test, horizon=1):
    """
    训练RF + LSTM模型（Exp3）
    """
    print("=" * 80)
    print(f"训练RF + LSTM模型 (Exp3) (预测步长: {horizon}小时)")
    print("=" * 80)
    
    # 第一步：RF特征选择
    print("1. RF特征选择...")
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        random_state=42,
        n_jobs=1
    )
    
    rf.fit(X_train, y_train)
    feature_importance = rf.feature_importances_
    
    # 选取累计贡献率达80%的Top-K特征
    cumulative_importance = np.cumsum(sorted(feature_importance, reverse=True))
    K = np.where(cumulative_importance >= 0.8)[0][0] + 1
    top_feature_indices = np.argsort(feature_importance)[-K:]
    
    print(f"选取的Top-K特征数量: {K}")
    
    # 筛选Top-K特征
    X_train_rf = X_train[:, top_feature_indices]
    X_val_rf = X_val[:, top_feature_indices]
    X_test_rf = X_test[:, top_feature_indices]
    
    # 第二步：LSTM训练
    print("2. LSTM训练...")
    X_train_lstm, y_train_lstm = create_lstm_input(X_train_rf, y_train, L)
    X_val_lstm, y_val_lstm = create_lstm_input(X_val_rf, y_val, L)
    X_test_lstm, y_test_lstm = create_lstm_input(X_test_rf, y_test, L)
    
    print(f"LSTM输入形状: {X_train_lstm.shape}")
    
    # 构建模型
    input_size = X_train_lstm.shape[2]
    model = RFLSTMModel(
        input_size=input_size,
        hidden_size=64,
        dropout_rate=0.2
    ).to(device)
    
    # 训练
    batch_size = 64
    train_loader = create_dataloader(X_train_lstm, y_train_lstm, batch_size, shuffle=True)
    val_loader = create_dataloader(X_val_lstm, y_val_lstm, batch_size, shuffle=False)
    test_loader = create_dataloader(X_test_lstm, y_test_lstm, batch_size, shuffle=False)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(20):
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        val_loss, _, _ = evaluate_epoch(model, val_loader, criterion)
        
        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("早停！")
                break
    
    # 加载最优模型
    model.load_state_dict(best_model_state)
    
    # 评估
    val_loss, val_preds, val_targets = evaluate_epoch(model, val_loader, criterion)
    val_rmse = np.sqrt(val_loss)
    val_mae = mean_absolute_error(val_targets, val_preds)
    val_r2 = r2_score(val_targets, val_preds)
    
    test_loss, test_preds, test_targets = evaluate_epoch(model, test_loader, criterion)
    test_rmse = np.sqrt(test_loss)
    test_mae = mean_absolute_error(test_targets, test_preds)
    test_r2 = r2_score(test_targets, test_preds)
    
    print(f"\nRF + LSTM验证集指标：")
    print(f"MAE = {val_mae:.2f}")
    print(f"RMSE = {val_rmse:.2f}")
    print(f"R² = {val_r2:.2f}")
    
    print(f"\nRF + LSTM测试集指标：")
    print(f"MAE = {test_mae:.2f}")
    print(f"RMSE = {test_rmse:.2f}")
    print(f"R² = {test_r2:.2f}")
    
    # 保存模型
    model_path = os.path.join(MODEL_DIR, f'exp3_rf_lstm_model_horizon{horizon}.pt')
    torch.save(model.state_dict(), model_path)
    print(f"\n模型已保存到: {model_path}")
    
    return model

def train_lstm_attention_model(X_train, y_train, X_val, y_val, X_test, y_test, horizon=1):
    """
    训练LSTM + Attention模型（Exp4）
    """
    print("=" * 80)
    print(f"训练LSTM + Attention模型 (Exp4) (预测步长: {horizon}小时)")
    print("=" * 80)
    
    # 数据格式调整
    X_train_lstm, y_train_lstm = create_lstm_input(X_train, y_train, L)
    X_val_lstm, y_val_lstm = create_lstm_input(X_val, y_val, L)
    X_test_lstm, y_test_lstm = create_lstm_input(X_test, y_test, L)
    
    print(f"LSTM输入形状: {X_train_lstm.shape}")
    
    # 构建模型
    input_size = X_train_lstm.shape[2]
    model = LSTMAttentionModel(
        input_size=input_size,
        hidden_size=64,
        dropout_rate=0.2
    ).to(device)
    
    # 训练
    batch_size = 64
    train_loader = create_dataloader(X_train_lstm, y_train_lstm, batch_size, shuffle=True)
    val_loader = create_dataloader(X_val_lstm, y_val_lstm, batch_size, shuffle=False)
    test_loader = create_dataloader(X_test_lstm, y_test_lstm, batch_size, shuffle=False)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(20):
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        val_loss, _, _ = evaluate_epoch(model, val_loader, criterion)
        
        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("早停！")
                break
    
    # 加载最优模型
    model.load_state_dict(best_model_state)
    
    # 评估
    val_loss, val_preds, val_targets = evaluate_epoch(model, val_loader, criterion)
    val_rmse = np.sqrt(val_loss)
    val_mae = mean_absolute_error(val_targets, val_preds)
    val_r2 = r2_score(val_targets, val_preds)
    
    test_loss, test_preds, test_targets = evaluate_epoch(model, test_loader, criterion)
    test_rmse = np.sqrt(test_loss)
    test_mae = mean_absolute_error(test_targets, test_preds)
    test_r2 = r2_score(test_targets, test_preds)
    
    print(f"\nLSTM + Attention验证集指标：")
    print(f"MAE = {val_mae:.2f}")
    print(f"RMSE = {val_rmse:.2f}")
    print(f"R² = {val_r2:.2f}")
    
    print(f"\nLSTM + Attention测试集指标：")
    print(f"MAE = {test_mae:.2f}")
    print(f"RMSE = {test_rmse:.2f}")
    print(f"R² = {test_r2:.2f}")
    
    # 保存模型
    model_path = os.path.join(MODEL_DIR, f'exp4_lstm_attention_model_horizon{horizon}.pt')
    torch.save(model.state_dict(), model_path)
    print(f"\n模型已保存到: {model_path}")
    
    return model

def train_rf_lstm_attention_model(X_train, y_train, X_val, y_val, X_test, y_test, horizon=1):
    """
    训练RF-LSTM-Attention模型（Exp5）
    """
    print("=" * 80)
    print(f"训练RF-LSTM-Attention模型 (Exp5) (预测步长: {horizon}小时)")
    print("=" * 80)
    
    # 第一步：RF特征选择
    print("1. RF特征选择...")
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        random_state=42,
        n_jobs=1
    )
    
    rf.fit(X_train, y_train)
    feature_importance = rf.feature_importances_
    
    # 选取累计贡献率达80%的Top-K特征
    cumulative_importance = np.cumsum(sorted(feature_importance, reverse=True))
    K = np.where(cumulative_importance >= 0.8)[0][0] + 1
    top_feature_indices = np.argsort(feature_importance)[-K:]
    
    print(f"选取的Top-K特征数量: {K}")
    
    # 筛选Top-K特征
    X_train_rf = X_train[:, top_feature_indices]
    X_val_rf = X_val[:, top_feature_indices]
    X_test_rf = X_test[:, top_feature_indices]
    
    # 第二步：LSTM + Attention训练
    print("2. LSTM + Attention训练...")
    X_train_lstm, y_train_lstm = create_lstm_input(X_train_rf, y_train, L)
    X_val_lstm, y_val_lstm = create_lstm_input(X_val_rf, y_val, L)
    X_test_lstm, y_test_lstm = create_lstm_input(X_test_rf, y_test, L)
    
    print(f"LSTM输入形状: {X_train_lstm.shape}")
    
    # 构建模型
    input_size = X_train_lstm.shape[2]
    model = RFLSTMAttentionModel(
        input_size=input_size,
        hidden_size=64,
        dropout_rate=0.2
    ).to(device)
    
    # 训练
    batch_size = 64
    train_loader = create_dataloader(X_train_lstm, y_train_lstm, batch_size, shuffle=True)
    val_loader = create_dataloader(X_val_lstm, y_val_lstm, batch_size, shuffle=False)
    test_loader = create_dataloader(X_test_lstm, y_test_lstm, batch_size, shuffle=False)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(20):
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        val_loss, _, _ = evaluate_epoch(model, val_loader, criterion)
        
        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("早停！")
                break
    
    # 加载最优模型
    model.load_state_dict(best_model_state)
    
    # 评估
    val_loss, val_preds, val_targets = evaluate_epoch(model, val_loader, criterion)
    val_rmse = np.sqrt(val_loss)
    val_mae = mean_absolute_error(val_targets, val_preds)
    val_r2 = r2_score(val_targets, val_preds)
    
    test_loss, test_preds, test_targets = evaluate_epoch(model, test_loader, criterion)
    test_rmse = np.sqrt(test_loss)
    test_mae = mean_absolute_error(test_targets, test_preds)
    test_r2 = r2_score(test_targets, test_preds)
    
    print(f"\nRF-LSTM-Attention验证集指标：")
    print(f"MAE = {val_mae:.2f}")
    print(f"RMSE = {val_rmse:.2f}")
    print(f"R² = {val_r2:.2f}")
    
    print(f"\nRF-LSTM-Attention测试集指标：")
    print(f"MAE = {test_mae:.2f}")
    print(f"RMSE = {test_rmse:.2f}")
    print(f"R² = {test_r2:.2f}")
    
    # 保存模型
    model_path = os.path.join(MODEL_DIR, f'exp5_rf_lstm_attention_model_horizon{horizon}.pt')
    torch.save(model.state_dict(), model_path)
    print(f"\n模型已保存到: {model_path}")
    
    return model

def train_xgboost_lstm_attention_model(X_train, y_train, X_val, y_val, X_test, y_test, horizon=1):
    """
    训练XGBoost + LSTM + Attention模型（Exp6）
    """
    print("=" * 80)
    print(f"训练XGBoost + LSTM + Attention模型 (Exp6) (预测步长: {horizon}小时)")
    print("=" * 80)
    
    # 尝试导入XGBoost
    try:
        from xgboost import XGBRegressor
    except ImportError:
        print("错误: 未找到XGBoost库，请先安装XGBoost")
        print("将使用RF替代XGBoost进行特征选择")
        # 使用RF替代XGBoost
        from sklearn.ensemble import RandomForestRegressor
        xgb = RandomForestRegressor(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            random_state=42,
            n_jobs=1
        )
    else:
        # 使用XGBoost
        xgb = XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=1
        )
    
    # 第一步：XGBoost特征选择
    print("1. XGBoost特征选择...")
    xgb.fit(X_train, y_train)
    
    if hasattr(xgb, 'feature_importances_'):
        feature_importance = xgb.feature_importances_
    else:
        # 如果是RF替代，使用其feature_importances_
        feature_importance = xgb.feature_importances_
    
    # 选取累计贡献率达80%的Top-K特征
    cumulative_importance = np.cumsum(sorted(feature_importance, reverse=True))
    K = np.where(cumulative_importance >= 0.8)[0][0] + 1
    top_feature_indices = np.argsort(feature_importance)[-K:]
    
    print(f"选取的Top-K特征数量: {K}")
    
    # 筛选Top-K特征
    X_train_xgb = X_train[:, top_feature_indices]
    X_val_xgb = X_val[:, top_feature_indices]
    X_test_xgb = X_test[:, top_feature_indices]
    
    # 第二步：LSTM + Attention训练
    print("2. LSTM + Attention训练...")
    X_train_lstm, y_train_lstm = create_lstm_input(X_train_xgb, y_train, L)
    X_val_lstm, y_val_lstm = create_lstm_input(X_val_xgb, y_val, L)
    X_test_lstm, y_test_lstm = create_lstm_input(X_test_xgb, y_test, L)
    
    print(f"LSTM输入形状: {X_train_lstm.shape}")
    
    # 构建模型
    input_size = X_train_lstm.shape[2]
    model = RFLSTMAttentionModel(
        input_size=input_size,
        hidden_size=64,
        dropout_rate=0.2
    ).to(device)
    
    # 训练
    batch_size = 64
    train_loader = create_dataloader(X_train_lstm, y_train_lstm, batch_size, shuffle=True)
    val_loader = create_dataloader(X_val_lstm, y_val_lstm, batch_size, shuffle=False)
    test_loader = create_dataloader(X_test_lstm, y_test_lstm, batch_size, shuffle=False)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(20):
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        val_loss, _, _ = evaluate_epoch(model, val_loader, criterion)
        
        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("早停！")
                break
    
    # 加载最优模型
    model.load_state_dict(best_model_state)
    
    # 评估
    val_loss, val_preds, val_targets = evaluate_epoch(model, val_loader, criterion)
    val_rmse = np.sqrt(val_loss)
    val_mae = mean_absolute_error(val_targets, val_preds)
    val_r2 = r2_score(val_targets, val_preds)
    
    test_loss, test_preds, test_targets = evaluate_epoch(model, test_loader, criterion)
    test_rmse = np.sqrt(test_loss)
    test_mae = mean_absolute_error(test_targets, test_preds)
    test_r2 = r2_score(test_targets, test_preds)
    
    print(f"\nXGBoost + LSTM + Attention验证集指标：")
    print(f"MAE = {val_mae:.2f}")
    print(f"RMSE = {val_rmse:.2f}")
    print(f"R² = {val_r2:.2f}")
    
    print(f"\nXGBoost + LSTM + Attention测试集指标：")
    print(f"MAE = {test_mae:.2f}")
    print(f"RMSE = {test_rmse:.2f}")
    print(f"R² = {test_r2:.2f}")
    
    # 保存模型
    model_path = os.path.join(MODEL_DIR, f'exp6_xgboost_lstm_attention_model_horizon{horizon}.pt')
    torch.save(model.state_dict(), model_path)
    print(f"\n模型已保存到: {model_path}")
    
    return model

def main():
    """
    主函数
    """
    # 只训练预测步长为1的模型
    horizon = 1
    
    try:
        # 加载数据
        X_train, y_train, X_val, y_val, X_test, y_test = load_data(horizon)
        
        # 运行所有消融实验
        print("\n" + "=" * 100)
        print("开始运行所有消融实验")
        print("=" * 100)
        
        # Exp1: 仅RF
        train_rf_model(X_train, y_train, X_val, y_val, X_test, y_test, horizon)
        print("\n" + "=" * 100)
        
        # Exp2: 仅LSTM
        train_lstm_model(X_train, y_train, X_val, y_val, X_test, y_test, horizon)
        print("\n" + "=" * 100)
        
        # Exp3: RF + LSTM
        train_rf_lstm_model(X_train, y_train, X_val, y_val, X_test, y_test, horizon)
        print("\n" + "=" * 100)
        
        # Exp4: LSTM + Attention
        train_lstm_attention_model(X_train, y_train, X_val, y_val, X_test, y_test, horizon)
        print("\n" + "=" * 100)
        
        # Exp5: RF-LSTM-Attention
        train_rf_lstm_attention_model(X_train, y_train, X_val, y_val, X_test, y_test, horizon)
        print("\n" + "=" * 100)
        
        # Exp6: XGBoost + LSTM + Attention
        train_xgboost_lstm_attention_model(X_train, y_train, X_val, y_val, X_test, y_test, horizon)
        print("\n" + "=" * 100)
        
        print("所有消融实验完成！")
        print("模型文件已保存到: D:\\gitres\\Task1\\models")
        
    except Exception as e:
        print(f"运行消融实验时出错: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
