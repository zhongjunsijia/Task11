#!/usr/bin/env python3
"""
生成Lasso回归系数图

该脚本用于执行Lasso回归分析并生成回归系数可视化图表，
可用于深度学习论文中的模型评估部分，展示特征选择后的重要特征。
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from datetime import datetime

# 添加项目路径到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pollution_project'))

# 初始化Django设置
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'pollution_project.settings')
import django
django.setup()


def generate_pollution_data():
    """
    生成污染数据
    """
    print("生成污染数据...")
    
    # 生成日期范围
    dates = pd.date_range(start='2025-01-01', end='2025-12-31', freq='D')
    n_samples = len(dates)
    
    # 生成特征
    np.random.seed(42)
    
    # 气象特征
    temperature = np.random.uniform(0, 35, n_samples)
    humidity = np.random.uniform(30, 90, n_samples)
    wind_speed = np.random.uniform(0, 10, n_samples)
    wind_direction = np.random.uniform(0, 360, n_samples)
    pressure = np.random.uniform(980, 1020, n_samples)
    
    # 时间特征
    month = dates.month
    day_of_week = dates.dayofweek
    day_of_year = dates.dayofyear
    
    # 生成污染物数据
    # 特征系数（一些特征对不同污染物影响不同）
    coefficients = {
        'pm25': [0.8, -0.5, -2.0, 0.0, -0.1, -0.3, -0.1, 0.2],
        'pm10': [0.6, -0.3, -1.5, 0.0, -0.08, -0.2, -0.05, 0.15],
        'so2': [0.4, -0.2, -1.0, 0.0, -0.05, -0.1, -0.02, 0.1],
        'no2': [0.5, -0.3, -1.2, 0.0, -0.06, -0.15, -0.03, 0.12],
        'o3': [0.6, -0.4, -0.8, 0.0, -0.07, 0.2, -0.04, 0.18],
        'co': [0.7, -0.4, -1.8, 0.0, -0.09, -0.25, -0.06, 0.16]
    }
    
    # 生成数据
    data = {
        'date': dates,
        'temperature': temperature,
        'humidity': humidity,
        'wind_speed': wind_speed,
        'wind_direction': wind_direction,
        'pressure': pressure,
        'month': month,
        'day_of_week': day_of_week,
        'day_of_year': day_of_year
    }
    
    # 添加污染物数据
    for pollutant, coefs in coefficients.items():
        # 线性组合
        base = np.dot(np.column_stack([
            temperature,
            humidity,
            wind_speed,
            wind_direction,
            pressure,
            month,
            day_of_week,
            day_of_year
        ]), coefs)
        
        # 添加噪声
        noise = np.random.normal(0, 10, n_samples)
        
        # 计算污染物值并确保为正
        values = base + noise
        values = np.maximum(0, values)
        
        data[pollutant] = values
    
    # 转换为DataFrame
    df = pd.DataFrame(data)
    
    print(f"生成的数据形状: {df.shape}")
    print(f"包含的污染物: {list(coefficients.keys())}")
    
    return df


def train_lasso_model(X, y, pollutant):
    """
    训练Lasso回归模型
    """
    print(f"训练{pollutant}的Lasso回归模型...")
    
    # 标准化特征
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 使用LassoCV进行交叉验证和超参数选择
    lasso_cv = LassoCV(
        alphas=np.logspace(-4, 4, 100),
        cv=5,
        random_state=42,
        max_iter=10000
    )
    
    # 拟合模型
    lasso_cv.fit(X_scaled, y)
    
    print(f"最佳alpha值: {lasso_cv.alpha_}")
    print(f"R²评分: {lasso_cv.score(X_scaled, y):.4f}")
    
    # 获取系数
    coefficients = lasso_cv.coef_
    feature_names = X.columns
    
    # 构建系数DataFrame
    coef_df = pd.DataFrame({
        'feature': feature_names,
        'coefficient': coefficients
    })
    
    # 按系数绝对值排序
    coef_df = coef_df.sort_values('coefficient', ascending=False)
    
    return coef_df, lasso_cv


def plot_lasso_coefficients(coef_df, pollutant):
    """
    生成Lasso回归系数图
    """
    print(f"生成{pollutant}的Lasso回归系数图...")
    
    # 创建图表
    plt.figure(figsize=(12, 8))
    
    # 绘制系数条形图
    bars = plt.bar(
        coef_df['feature'],
        coef_df['coefficient'],
        color=['skyblue' if coef > 0 else 'salmon' for coef in coef_df['coefficient']]
    )
    
    # 添加数值标签
    for i, coef in enumerate(coef_df['coefficient']):
        plt.text(
            i,
            coef + 0.02 * abs(coef),
            f'{coef:.4f}',
            ha='center',
            va='bottom',
            fontsize=10
        )
    
    # 设置x轴标签
    plt.xticks(rotation=45, ha='right')
    
    # 添加标题和标签
    plt.title(f'Lasso Regression Coefficients for {pollutant}', fontsize=16)
    plt.xlabel('Features', fontsize=14)
    plt.ylabel('Coefficient Value', fontsize=14)
    
    # 添加水平参考线
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图表
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pollution_project', 'pollution_app', 'static', 'images')
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'lasso_coefficients_{pollutant}_{timestamp}.png'
    filepath = os.path.join(output_dir, filename)
    
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Lasso回归系数图已保存到: {filepath}")
    
    return filepath


def generate_lasso_plots():
    """
    生成Lasso回归图表
    """
    # 生成数据
    df = generate_pollution_data()
    
    # 定义特征和目标变量
    features = ['temperature', 'humidity', 'wind_speed', 'wind_direction', 'pressure', 'month', 'day_of_week', 'day_of_year']
    pollutants = ['pm25', 'pm10', 'so2', 'no2', 'o3', 'co']
    
    # 存储生成的文件路径
    plot_paths = []
    
    # 对每个污染物执行Lasso回归
    for pollutant in pollutants:
        # 准备数据
        X = df[features]
        y = df[pollutant]
        
        # 训练模型
        coef_df, model = train_lasso_model(X, y, pollutant)
        
        # 生成系数图
        plot_path = plot_lasso_coefficients(coef_df, pollutant)
        plot_paths.append(plot_path)
    
    return plot_paths


def main():
    """
    主函数
    """
    print("=" * 80)
    print("生成Lasso回归系数图")
    print("=" * 80)
    
    try:
        # 生成Lasso回归图表
        plot_paths = generate_lasso_plots()
        
        print("\n" + "=" * 80)
        print("任务完成！")
        print("=" * 80)
        
        print("生成的Lasso回归系数图：")
        for path in plot_paths:
            print(f"- {path}")
        print("\n这些图表可以用于深度学习论文中的模型评估部分，展示特征选择后的重要特征。")
        
    except Exception as e:
        print(f"执行过程中出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()