#!/usr/bin/env python3
"""
独立的Lasso回归系数图生成脚本

该脚本不依赖Django，直接生成数据和Lasso回归系数组合图表。
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from datetime import datetime


def generate_data():
    """
    生成数据
    """
    print("生成数据...")
    
    # 生成样本
    n_samples = 1000
    np.random.seed(42)
    
    # 生成特征
    temperature = np.random.normal(20, 5, n_samples)
    humidity = np.random.normal(60, 15, n_samples)
    wind_speed = np.random.normal(5, 2, n_samples)
    pressure = np.random.normal(1000, 10, n_samples)
    month = np.random.randint(1, 13, n_samples)
    day_of_week = np.random.randint(0, 7, n_samples)
    
    # 生成污染物数据
    pollutants = {
        'pm25': 2.0 * temperature - 1.5 * humidity - 3.0 * wind_speed - 0.5 * pressure + 0.1 * month + np.random.normal(0, 10, n_samples),
        'pm10': 1.5 * temperature - 1.0 * humidity - 2.5 * wind_speed - 0.4 * pressure + 0.08 * month + np.random.normal(0, 8, n_samples),
        'so2': 1.0 * temperature - 0.8 * humidity - 2.0 * wind_speed - 0.3 * pressure + 0.05 * month + np.random.normal(0, 5, n_samples),
        'no2': 1.2 * temperature - 0.9 * humidity - 2.2 * wind_speed - 0.35 * pressure + 0.06 * month + np.random.normal(0, 6, n_samples),
        'o3': 1.3 * temperature - 1.1 * humidity - 1.8 * wind_speed - 0.4 * pressure + 0.07 * month + np.random.normal(0, 7, n_samples),
        'co': 1.8 * temperature - 1.2 * humidity - 2.8 * wind_speed - 0.5 * pressure + 0.09 * month + np.random.normal(0, 9, n_samples)
    }
    
    # 确保所有值为正
    for key in pollutants:
        pollutants[key] = np.maximum(0, pollutants[key])
    
    # 生成数据
    data = {
        'temperature': temperature,
        'humidity': humidity,
        'wind_speed': wind_speed,
        'pressure': pressure,
        'month': month,
        'day_of_week': day_of_week
    }
    
    # 添加污染物数据
    for pollutant, values in pollutants.items():
        data[pollutant] = values
    
    # 转换为DataFrame
    df = pd.DataFrame(data)
    
    print(f"生成的数据形状: {df.shape}")
    print(f"包含的污染物: {list(pollutants.keys())}")
    
    # 打印相关性信息
    print("\n特征与污染物相关性:")
    for pollutant in list(pollutants.keys())[:2]:
        print(f"\n{pollutant}:")
        for feature in ['temperature', 'humidity', 'wind_speed']:
            corr = df[feature].corr(df[pollutant])
            print(f"{feature}: {corr:.4f}")
    
    return df


def train_lasso(X, y, alpha=0.01):
    """
    训练Lasso模型
    """
    # 标准化特征
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 训练Lasso模型
    lasso = Lasso(alpha=alpha, random_state=42, max_iter=10000)
    lasso.fit(X_scaled, y)
    
    # 获取系数
    coefficients = lasso.coef_
    
    print(f"Lasso系数: {coefficients}")
    
    return coefficients


def plot_combined_coefficients(all_coefficients, features, pollutants):
    """
    生成组合系数图
    """
    print("生成组合Lasso回归系数图...")
    
    # 创建图表
    plt.figure(figsize=(16, 10))
    
    # 设置条形图参数
    bar_width = 0.12
    index = np.arange(len(features))
    
    # 颜色列表
    colors = ['skyblue', 'lightgreen', 'salmon', 'gold', 'purple', 'cyan']
    
    # 绘制每个污染物的系数
    for i, (pollutant, coefficients) in enumerate(all_coefficients.items()):
        plt.bar(
            index + i * bar_width,
            coefficients,
            bar_width,
            label=pollutant,
            color=colors[i % len(colors)]
        )
    
    # 设置x轴标签
    plt.xticks(index + bar_width * (len(pollutants) - 1) / 2, features, rotation=45, ha='right')
    
    # 添加标题和标签
    plt.title('Combined Lasso Regression Coefficients for All Pollutants', fontsize=18)
    plt.xlabel('Features', fontsize=14)
    plt.ylabel('Coefficient Value', fontsize=14)
    plt.legend(fontsize=12)
    
    # 添加水平参考线
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图表
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pollution_project', 'pollution_app', 'static', 'images')
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'standalone_combined_lasso_coefficients_{timestamp}.png'
    filepath = os.path.join(output_dir, filename)
    
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"组合Lasso回归系数图已保存到: {filepath}")
    
    return filepath


def main():
    """
    主函数
    """
    print("=" * 80)
    print("生成组合Lasso回归系数图")
    print("=" * 80)
    
    try:
        # 生成数据
        df = generate_data()
        
        # 定义特征和目标变量
        features = ['temperature', 'humidity', 'wind_speed', 'pressure', 'month', 'day_of_week']
        pollutants = ['pm25', 'pm10', 'so2', 'no2', 'o3', 'co']
        
        # 存储所有污染物的系数
        all_coefficients = {}
        
        # 对每个污染物执行Lasso回归
        for pollutant in pollutants:
            print(f"\n处理{pollutant}...")
            # 准备数据
            X = df[features]
            y = df[pollutant]
            
            # 训练模型
            coefficients = train_lasso(X, y, alpha=0.01)
            all_coefficients[pollutant] = coefficients
        
        # 生成组合图
        plot_path = plot_combined_coefficients(all_coefficients, features, pollutants)
        
        print("\n" + "=" * 80)
        print("任务完成！")
        print("=" * 80)
        
        print("生成的组合Lasso回归系数图：")
        print(f"- {plot_path}")
        print("\n该图表包含所有污染物的Lasso回归系数，便于比较不同污染物的特征重要性。")
        
    except Exception as e:
        print(f"执行过程中出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()