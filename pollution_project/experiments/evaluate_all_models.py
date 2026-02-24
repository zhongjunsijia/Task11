#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
评估所有训练好的模型

对RF、SVR、LSTM和RF-LSTM-Attention模型进行综合评估，生成评估报告和可视化结果
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 确保数据目录存在
PROCESSED_DIR = os.path.join('data', 'processed')
EVALUATION_DIR = os.path.join('evaluation')
os.makedirs(EVALUATION_DIR, exist_ok=True)

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

def evaluate_rf_model(X_test, y_test, horizon=1):
    """
    评估RF模型
    
    Args:
        X_test: 测试特征
        y_test: 测试标签
        horizon: 预测步长
    
    Returns:
        评估指标 (mae, rmse, r2)
    """
    try:
        # 加载RF模型
        model_path = os.path.join('models', f'rf_model_horizon{horizon}.pkl')
        best_rf = joblib.load(model_path)
        
        # 预测
        y_test_pred = best_rf.predict(X_test)
        
        # 评估
        mae = mean_absolute_error(y_test, y_test_pred)
        rmse = mean_squared_error(y_test, y_test_pred, squared=False)
        r2 = r2_score(y_test, y_test_pred)
        
        print(f"\nRF模型测试集指标：")
        print(f"MAE = {mae:.2f}")
        print(f"RMSE = {rmse:.2f}")
        print(f"R² = {r2:.2f}")
        
        return mae, rmse, r2, y_test_pred, best_rf
    except FileNotFoundError:
        print(f"错误: 未找到RF模型文件 '{model_path}'")
        return None, None, None, None, None

def evaluate_svr_model(X_test, y_test, horizon=1):
    """
    评估SVR模型
    
    Args:
        X_test: 测试特征
        y_test: 测试标签
        horizon: 预测步长
    
    Returns:
        评估指标 (mae, rmse, r2)
    """
    try:
        # 加载SVR模型
        model_path = os.path.join('models', f'svr_model_horizon{horizon}.pkl')
        best_svr = joblib.load(model_path)
        
        # 预测
        y_test_pred = best_svr.predict(X_test)
        
        # 评估
        mae = mean_absolute_error(y_test, y_test_pred)
        rmse = mean_squared_error(y_test, y_test_pred, squared=False)
        r2 = r2_score(y_test, y_test_pred)
        
        print(f"\nSVR模型测试集指标：")
        print(f"MAE = {mae:.2f}")
        print(f"RMSE = {rmse:.2f}")
        print(f"R² = {r2:.2f}")
        
        return mae, rmse, r2, y_test_pred
    except FileNotFoundError:
        print(f"错误: 未找到SVR模型文件 '{model_path}'")
        return None, None, None, None

def evaluate_lstm_model(X_test, y_test):
    """
    评估LSTM模型
    
    Args:
        X_test: 测试特征
        y_test: 测试标签
    
    Returns:
        评估指标 (mae, rmse, r2)
    """
    print("警告: 跳过LSTM模型评估，因为未安装TensorFlow")
    return None, None, None, None, None

def evaluate_rf_lstm_attention_model(X_test, y_test):
    """
    评估RF-LSTM-Attention模型
    
    Args:
        X_test: 测试特征
        y_test: 测试标签
    
    Returns:
        评估指标 (mae, rmse, r2)
    """
    print("警告: 跳过RF-LSTM-Attention模型评估，因为未安装TensorFlow")
    return None, None, None, None, None, None

def plot_model_comparison(metrics, horizon=1):
    """
    绘制模型性能对比图
    
    Args:
        metrics: 各模型的评估指标
        horizon: 预测步长
    """
    models = list(metrics.keys())
    maes = [metrics[model]['mae'] for model in models]
    rmses = [metrics[model]['rmse'] for model in models]
    r2s = [metrics[model]['r2'] for model in models]
    
    # 创建子图
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 绘制MAE对比
    sns.barplot(x=models, y=maes, ax=axes[0])
    axes[0].set_title(f'MAE对比 (预测步长: {horizon}小时)')
    axes[0].set_ylabel('MAE')
    for i, v in enumerate(maes):
        axes[0].text(i, v + 0.01, f'{v:.2f}', ha='center')
    
    # 绘制RMSE对比
    sns.barplot(x=models, y=rmses, ax=axes[1])
    axes[1].set_title(f'RMSE对比 (预测步长: {horizon}小时)')
    axes[1].set_ylabel('RMSE')
    for i, v in enumerate(rmses):
        axes[1].text(i, v + 0.01, f'{v:.2f}', ha='center')
    
    # 绘制R²对比
    sns.barplot(x=models, y=r2s, ax=axes[2])
    axes[2].set_title(f'R²对比 (预测步长: {horizon}小时)')
    axes[2].set_ylabel('R²')
    for i, v in enumerate(r2s):
        axes[2].text(i, v + 0.01, f'{v:.2f}', ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(EVALUATION_DIR, f'model_comparison_horizon{horizon}.png'))
    print(f"模型对比图已保存到: {os.path.join(EVALUATION_DIR, f'model_comparison_horizon{horizon}.png')}")

def plot_predicted_vs_actual(y_test, predictions, horizon=1):
    """
    绘制预测值 vs 真实值图
    
    Args:
        y_test: 真实值
        predictions: 各模型的预测值
        horizon: 预测步长
    """
    # 只取前200个样本进行可视化，避免图形过于拥挤
    sample_size = min(200, len(y_test))
    indices = np.arange(sample_size)
    
    plt.figure(figsize=(15, 8))
    
    # 绘制真实值
    plt.plot(indices, y_test[:sample_size], label='真实值', linewidth=2)
    
    # 绘制各模型预测值
    colors = ['red', 'green', 'purple', 'orange']
    for i, (model_name, y_pred) in enumerate(predictions.items()):
        if y_pred is not None:
            # 确保预测值长度与真实值一致
            if len(y_pred) >= sample_size:
                plt.plot(indices, y_pred[:sample_size], label=model_name, linewidth=1.5, alpha=0.7, color=colors[i])
            else:
                # 对于LSTM和RF-LSTM-Attention，需要调整索引
                start_idx = len(y_test) - len(y_pred)
                if start_idx < sample_size:
                    plt.plot(indices[start_idx:start_idx+len(y_pred[:sample_size-start_idx])], 
                             y_pred[:sample_size-start_idx], label=model_name, linewidth=1.5, alpha=0.7, color=colors[i])
    
    plt.title(f'预测值 vs 真实值 (预测步长: {horizon}小时)')
    plt.xlabel('样本索引')
    plt.ylabel('PM2.5浓度')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(EVALUATION_DIR, f'predicted_vs_actual_horizon{horizon}.png'))
    print(f"预测值 vs 真实值图已保存到: {os.path.join(EVALUATION_DIR, f'predicted_vs_actual_horizon{horizon}.png')}")

def plot_attention_weights(attention_weights, horizon=1):
    """
    绘制注意力权重图
    
    Args:
        attention_weights: 注意力权重
        horizon: 预测步长
    """
    if attention_weights is not None:
        # 取第一个样本的注意力权重进行可视化
        sample_idx = 0
        weights = attention_weights[sample_idx].flatten()
        
        plt.figure(figsize=(12, 6))
        plt.plot(range(L), weights)
        plt.xlabel('时间步（历史24小时）')
        plt.ylabel('注意力权重')
        plt.title(f'注意力权重分布 (预测步长: {horizon}小时)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(EVALUATION_DIR, f'attention_weights_horizon{horizon}.png'))
        print(f"注意力权重图已保存到: {os.path.join(EVALUATION_DIR, f'attention_weights_horizon{horizon}.png')}")
    else:
        print("警告: 跳过注意力权重图绘制，因为未评估RF-LSTM-Attention模型")

def generate_evaluation_report(metrics, horizon=1):
    """
    生成评估报告
    
    Args:
        metrics: 各模型的评估指标
        horizon: 预测步长
    """
    report_path = os.path.join(EVALUATION_DIR, f'evaluation_report_horizon{horizon}.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"模型评估报告 (预测步长: {horizon}小时)\n")
        f.write("=" * 80 + "\n\n")
        
        # 写入各模型评估指标
        for model_name, model_metrics in metrics.items():
            f.write(f"{model_name}模型评估指标：\n")
            f.write(f"MAE = {model_metrics['mae']:.2f}\n")
            f.write(f"RMSE = {model_metrics['rmse']:.2f}\n")
            f.write(f"R² = {model_metrics['r2']:.2f}\n")
            f.write("-" * 40 + "\n\n")
        
        # 确定最佳模型
        if metrics:
            # 根据RMSE排序
            sorted_models = sorted(metrics.items(), key=lambda x: x[1]['rmse'])
            best_model = sorted_models[0]
            
            f.write("最佳模型：\n")
            f.write(f"{best_model[0]}\n")
            f.write(f"最佳RMSE: {best_model[1]['rmse']:.2f}\n")
            f.write(f"对应的MAE: {best_model[1]['mae']:.2f}\n")
            f.write(f"对应的R²: {best_model[1]['r2']:.2f}\n")
    
    print(f"评估报告已生成：{report_path}")

def main():
    """
    主函数
    """
    # 测试不同的预测步长
    forecast_horizons = [1, 6, 12, 24]
    
    for horizon in forecast_horizons:
        print("=" * 80)
        print(f"评估所有模型 (预测步长: {horizon}小时)")
        print("=" * 80)
        
        try:
            # 加载数据
            X_train, y_train, X_val, y_val, X_test, y_test = load_data(horizon)
            
            # 评估各模型
            metrics = {}
            predictions = {}
            
            # 评估RF模型
            rf_mae, rf_rmse, rf_r2, rf_pred, best_rf = evaluate_rf_model(X_test, y_test, horizon)
            if rf_mae is not None:
                metrics['RF'] = {'mae': rf_mae, 'rmse': rf_rmse, 'r2': rf_r2}
                predictions['RF'] = rf_pred
            
            # 评估SVR模型
            svr_mae, svr_rmse, svr_r2, svr_pred = evaluate_svr_model(X_test, y_test, horizon)
            if svr_mae is not None:
                metrics['SVR'] = {'mae': svr_mae, 'rmse': svr_rmse, 'r2': svr_r2}
                predictions['SVR'] = svr_pred
            
            # 评估LSTM模型
            lstm_mae, lstm_rmse, lstm_r2, lstm_pred, lstm_y_test = evaluate_lstm_model(X_test, y_test)
            if lstm_mae is not None:
                metrics['LSTM'] = {'mae': lstm_mae, 'rmse': lstm_rmse, 'r2': lstm_r2}
                predictions['LSTM'] = lstm_pred
            
            # 评估RF-LSTM-Attention模型
            rf_lstm_attention_mae, rf_lstm_attention_rmse, rf_lstm_attention_r2, rf_lstm_attention_pred, rf_lstm_attention_y_test, attention_weights = evaluate_rf_lstm_attention_model(X_test, y_test)
            if rf_lstm_attention_mae is not None:
                metrics['RF-LSTM-Attention'] = {'mae': rf_lstm_attention_mae, 
                                               'rmse': rf_lstm_attention_rmse, 
                                               'r2': rf_lstm_attention_r2}
                predictions['RF-LSTM-Attention'] = rf_lstm_attention_pred
            
            # 绘制模型对比图
            if metrics:
                plot_model_comparison(metrics, horizon)
            
            # 绘制预测值vs真实值
            plot_predicted_vs_actual(y_test, predictions, horizon)
            
            # 绘制注意力权重图
            if attention_weights is not None:
                plot_attention_weights(attention_weights, horizon)
            
            # 生成评估报告
            if metrics:
                generate_evaluation_report(metrics, horizon)
            
            print("\n" + "=" * 80)
            print(f"模型评估完成 (预测步长: {horizon}小时)")
            print("=" * 80)
            
        except Exception as e:
            print(f"评估模型时出错 (预测步长: {horizon}小时): {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
