#!/usr/bin/env python3
"""
生成组合ElasticNet回归系数图

该脚本用于训练ElasticNet模型并生成包含所有污染物回归系数的组合图表，
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


def plot_combined_coefficients(models_dict, features):
    """
    生成组合回归系数图
    """
    if not models_dict:
        print("模型字典为空，无法生成系数图")
        return None
    
    print("生成所有污染物的组合ElasticNet回归系数图...")
    
    # 污染物名称映射
    pollutant_names = {
        'pm25': 'PM2.5',
        'pm10': 'PM10',
        'no2': 'NO₂',
        'so2': 'SO₂',
        'o3': 'O₃',
        'co': 'CO'
    }
    
    # 颜色映射
    colors = {
        'pm25': '#FF6384',  # 红色
        'pm10': '#36A2EB',  # 蓝色
        'no2': '#FFCE56',   # 黄色
        'so2': '#4BC0C0',   # 青色
        'o3': '#9966FF',    # 紫色
        'co': '#FF9F40'     # 橙色
    }
    
    # 收集所有系数数据
    coefficients_data = {}
    for pollutant, model in models_dict.items():
        if model is not None:
            coefficients_data[pollutant] = model.coef_
    
    # 特征数量
    n_features = len(features)
    # 污染物数量
    n_pollutants = len(coefficients_data)
    
    # 创建图表
    plt.figure(figsize=(12, 8))
    
    # 计算条形图的宽度
    bar_width = 0.8 / n_pollutants
    
    # 绘制每个特征的系数
    for i, feature in enumerate(features):
        # 每个特征的x位置
        x_positions = [i + j * bar_width for j in range(n_pollutants)]
        
        # 为每个污染物绘制系数
        for j, (pollutant, coefficients) in enumerate(coefficients_data.items()):
            coefficient = coefficients[i]
            plt.bar(
                x_positions[j],
                coefficient,
                width=bar_width,
                color=colors[pollutant],
                label=pollutant_names[pollutant] if i == 0 else "",
                alpha=0.7
            )
            
            # 添加数值标签
            plt.text(
                x_positions[j],
                coefficient,
                f'{coefficient:.4f}',
                ha='center',
                va='bottom' if coefficient > 0 else 'top',
                fontsize=8
            )
    
    # 设置x轴标签
    plt.xticks(
        [i + (n_pollutants - 1) * bar_width / 2 for i in range(n_features)],
        features,
        rotation=45,
        ha='right'
    )
    
    # 添加标题和标签
    plt.title('ElasticNet Regression Coefficients for All Pollutants', fontsize=16)
    plt.xlabel('Features', fontsize=14)
    plt.ylabel('Coefficient Value', fontsize=14)
    
    # 添加水平参考线
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # 添加图例
    plt.legend(title='Pollutants', fontsize=10)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图表
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pollution_project', 'pollution_app', 'static', 'images')
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'combined_elasticnet_coefficients_{timestamp}.png'
    filepath = os.path.join(output_dir, filename)
    
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"组合回归系数图已保存到: {filepath}")
    
    return filepath


def generate_combined_coefficients_plot():
    """
    生成组合系数图
    """
    # 生成训练数据
    data = generate_training_data()
    
    # 定义污染物列表
    pollutants = ['pm25', 'pm10', 'no2', 'so2', 'o3', 'co']
    
    # 训练所有模型
    models_dict = {}
    features = None
    
    for pollutant in pollutants:
        print(f"\n处理{pollutant}...")
        model, feat = train_elasticnet_model(data, pollutant)
        if model:
            models_dict[pollutant] = model
            if features is None:
                features = feat
    
    # 生成组合系数图
    if models_dict and features:
        filepath = plot_combined_coefficients(models_dict, features)
        return filepath
    else:
        print("无法生成组合系数图，因为没有训练好的模型")
        return None


def main():
    """
    主函数
    """
    print("=" * 80)
    print("生成组合ElasticNet回归系数图")
    print("=" * 80)
    
    try:
        # 生成组合系数图
        output_file = generate_combined_coefficients_plot()
        
        print("\n" + "=" * 80)
        print("任务完成！")
        print("=" * 80)
        
        if output_file:
            print(f"生成的组合回归系数图：")
            print(f"- {output_file}")
            print("\n该图表可以用于深度学习论文中的模型评估部分，展示所有污染物的特征重要性比较。")
        else:
            print("无法生成组合回归系数图。")
        
    except Exception as e:
        print(f"执行过程中出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
