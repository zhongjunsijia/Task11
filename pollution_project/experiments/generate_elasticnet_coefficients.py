#!/usr/bin/env python3
"""
生成ElasticNet回归系数图

该脚本用于训练ElasticNet模型并生成回归系数的可视化图表，
可用于深度学习论文中的模型评估部分。
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# 添加项目路径到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pollution_project'))

# 初始化Django设置
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'pollution_project.settings')
import django
django.setup()

# 导入模型管理器
from pollution_app.utils.model_manager import model_manager


def generate_training_data():
    """
    生成训练数据
    """
    print("生成训练数据...")
    
    # 尝试加载真实数据
    data_files = [
        'uploads/air_pollution_china.csv',
        'uploads/pollution_test_data.csv',
        'uploads/updated_pollution_dataset.csv'
    ]
    
    for file_path in data_files:
        full_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_path)
        if os.path.exists(full_path):
            print(f"使用真实数据文件: {file_path}")
            try:
                df = pd.read_csv(full_path)
                # 检查数据格式
                if 'date' in df.columns and any(col in df.columns for col in ['pm25', 'pm10', 'no2', 'so2', 'o3', 'co']):
                    # 处理日期列
                    df['date'] = pd.to_datetime(df['date'], errors='coerce')
                    # 提取特征
                    df['day_of_week'] = df['date'].dt.dayofweek
                    df['month'] = df['date'].dt.month
                    return df
            except Exception as e:
                print(f"读取文件失败: {e}")
                continue
    
    # 如果没有真实数据，使用模拟数据
    print("使用模拟数据")
    data = {
        'date': pd.date_range(start='2025-01-01', periods=365),
        'pm25': np.random.randint(20, 150, 365),
        'pm10': np.random.randint(30, 200, 365),
        'no2': np.random.randint(10, 100, 365),
        'so2': np.random.randint(5, 80, 365),
        'o3': np.random.randint(20, 180, 365),
        'co': np.random.uniform(0.5, 7, 365),
        'temperature': np.random.uniform(10, 35, 365),
        'humidity': np.random.uniform(30, 90, 365),
        'wind_speed': np.random.uniform(1, 8, 365)
    }
    df = pd.DataFrame(data)
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    return df


def train_elasticnet_model(data, pollutant='pm25'):
    """
    训练ElasticNet模型
    """
    print(f"训练ElasticNet模型预测{pollutant}...")
    
    # 定义特征
    features = ['day_of_week', 'month', 'temperature', 'humidity', 'wind_speed']
    X = data[features]
    y = data[pollutant]
    
    # 移除缺失值
    mask = ~np.isnan(y)
    X_clean = X[mask]
    y_clean = y[mask]
    
    if len(y_clean) == 0:
        print(f"没有有效的{pollutant}数据，跳过...")
        return None, None
    
    # 训练模型
    model, scaler, metrics, version = model_manager.train_model(
        X_clean, y_clean,
        model_type='elasticnet',
        model_name='elasticnet_predictor',
        pollutant=pollutant,
        city='全国',
        activate=True
    )
    
    print(f"ElasticNet模型训练完成！版本: {version}")
    print(f"模型评估指标: {metrics}")
    
    return model, features


def plot_coefficients(model, features, pollutant='pm25'):
    """
    生成回归系数图
    """
    if model is None:
        print("模型为空，无法生成系数图")
        return None
    
    print(f"生成{pollutant}的ElasticNet回归系数图...")
    
    # 获取回归系数
    coefficients = model.coef_
    
    # 创建图表
    plt.figure(figsize=(10, 6))
    
    # 绘制系数条形图
    bars = plt.bar(range(len(coefficients)), coefficients, color='skyblue')
    
    # 添加数值标签
    for bar, coeff in zip(bars, coefficients):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height, 
                 f'{coeff:.4f}',
                 ha='center', va='bottom')
    
    # 设置x轴标签
    plt.xticks(range(len(features)), features, rotation=45, ha='right')
    
    # 添加标题和标签
    plt.title(f'ElasticNet Regression Coefficients for {pollutant}', fontsize=14)
    plt.xlabel('Features', fontsize=12)
    plt.ylabel('Coefficient Value', fontsize=12)
    
    # 添加水平参考线
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图表
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pollution_project', 'pollution_app', 'static', 'images')
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'elasticnet_coefficients_{pollutant}_{timestamp}.png'
    filepath = os.path.join(output_dir, filename)
    
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"回归系数图已保存到: {filepath}")
    
    return filepath


def generate_all_pollutants_coefficients():
    """
    为所有污染物生成回归系数图
    """
    # 生成训练数据
    data = generate_training_data()
    
    # 定义污染物列表
    pollutants = ['pm25', 'pm10', 'no2', 'so2', 'o3', 'co']
    
    # 为每个污染物生成系数图
    output_files = []
    for pollutant in pollutants:
        print(f"\n处理{pollutant}...")
        model, features = train_elasticnet_model(data, pollutant)
        if model:
            filepath = plot_coefficients(model, features, pollutant)
            if filepath:
                output_files.append(filepath)
    
    return output_files


def main():
    """
    主函数
    """
    print("=" * 80)
    print("生成ElasticNet回归系数图")
    print("=" * 80)
    
    try:
        # 为所有污染物生成系数图
        output_files = generate_all_pollutants_coefficients()
        
        print("\n" + "=" * 80)
        print("任务完成！")
        print("=" * 80)
        print("生成的回归系数图：")
        for file in output_files:
            print(f"- {file}")
        
        print("\n这些图表可以用于深度学习论文中的模型评估部分。")
        
    except Exception as e:
        print(f"执行过程中出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
