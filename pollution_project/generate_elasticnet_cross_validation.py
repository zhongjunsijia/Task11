#!/usr/bin/env python3
"""
生成ElasticNet交叉验证图

该脚本用于训练ElasticNet模型并生成交叉验证可视化图表，
可用于深度学习论文中的模型评估部分，展示不同超参数对模型性能的影响。
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler

# 添加项目路径到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pollution_project'))

# 初始化Django设置
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'pollution_project.settings')
import django
django.setup()


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


def perform_cross_validation(X, y, alphas, l1_ratios):
    """
    执行交叉验证，评估不同超参数组合的模型性能
    """
    print("执行ElasticNet交叉验证...")
    
    # 标准化数据
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 交叉验证设置
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # 存储结果
    results = []
    
    # 对每个alpha和l1_ratio组合进行交叉验证
    for alpha in alphas:
        for l1_ratio in l1_ratios:
            print(f"评估 alpha={alpha:.4f}, l1_ratio={l1_ratio:.4f}...")
            
            # 创建模型
            model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000)
            
            # 交叉验证
            scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='r2')
            
            # 计算平均得分和标准差
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            
            results.append({
                'alpha': alpha,
                'l1_ratio': l1_ratio,
                'mean_r2': mean_score,
                'std_r2': std_score
            })
    
    return results


def plot_cross_validation_heatmap(results, pollutant='pm25'):
    """
    生成交叉验证热图
    """
    print(f"生成{pollutant}的ElasticNet交叉验证热图...")
    
    # 转换结果为DataFrame
    df = pd.DataFrame(results)
    
    # 创建透视表
    pivot_table = df.pivot(index='alpha', columns='l1_ratio', values='mean_r2')
    
    # 创建图表
    plt.figure(figsize=(12, 8))
    
    # 绘制热图
    heatmap = plt.imshow(pivot_table, cmap='RdYlBu_r', aspect='auto')
    
    # 添加颜色条
    cbar = plt.colorbar(heatmap)
    cbar.set_label('Mean R² Score', fontsize=12)
    
    # 设置x轴和y轴标签
    plt.xticks(np.arange(len(pivot_table.columns)), [f'{x:.2f}' for x in pivot_table.columns], rotation=45)
    plt.yticks(np.arange(len(pivot_table.index)), [f'{y:.4f}' for y in pivot_table.index])
    
    # 添加标题和标签
    plt.title(f'ElasticNet Cross-Validation Heatmap for {pollutant}', fontsize=16)
    plt.xlabel('l1_ratio', fontsize=14)
    plt.ylabel('alpha', fontsize=14)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图表
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pollution_project', 'pollution_app', 'static', 'images')
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'elasticnet_cv_heatmap_{pollutant}_{timestamp}.png'
    filepath = os.path.join(output_dir, filename)
    
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"交叉验证热图已保存到: {filepath}")
    
    return filepath


def plot_cross_validation_curves(results, pollutant='pm25'):
    """
    生成交叉验证曲线
    """
    print(f"生成{pollutant}的ElasticNet交叉验证曲线...")
    
    # 转换结果为DataFrame
    df = pd.DataFrame(results)
    
    # 按l1_ratio分组
    l1_ratios = df['l1_ratio'].unique()
    
    # 创建图表
    plt.figure(figsize=(12, 8))
    
    # 对每个l1_ratio绘制一条曲线
    for l1_ratio in l1_ratios:
        ratio_data = df[df['l1_ratio'] == l1_ratio]
        ratio_data = ratio_data.sort_values('alpha')
        
        # 绘制平均R²得分
        plt.plot(ratio_data['alpha'], ratio_data['mean_r2'], marker='o', label=f'l1_ratio={l1_ratio:.2f}')
        
        # 添加误差线
        plt.fill_between(
            ratio_data['alpha'],
            ratio_data['mean_r2'] - ratio_data['std_r2'],
            ratio_data['mean_r2'] + ratio_data['std_r2'],
            alpha=0.1
        )
    
    # 添加标题和标签
    plt.title(f'ElasticNet Cross-Validation Curves for {pollutant}', fontsize=16)
    plt.xlabel('alpha', fontsize=14)
    plt.ylabel('Mean R² Score', fontsize=14)
    plt.xscale('log')  # 使用对数刻度显示alpha
    
    # 添加图例
    plt.legend(title='l1_ratio', fontsize=10)
    
    # 添加网格
    plt.grid(alpha=0.3)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图表
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pollution_project', 'pollution_app', 'static', 'images')
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'elasticnet_cv_curves_{pollutant}_{timestamp}.png'
    filepath = os.path.join(output_dir, filename)
    
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"交叉验证曲线已保存到: {filepath}")
    
    return filepath


def generate_cross_validation_plots(pollutant='pm25'):
    """
    生成交叉验证图表
    """
    # 生成训练数据
    data = generate_training_data()
    
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
    
    # 定义超参数范围
    alphas = np.logspace(-4, 2, 10)  # 从0.0001到100的10个对数间隔值
    l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.9]  # 不同的l1_ratio值
    
    # 执行交叉验证
    results = perform_cross_validation(X_clean, y_clean, alphas, l1_ratios)
    
    # 生成热图
    heatmap_path = plot_cross_validation_heatmap(results, pollutant)
    
    # 生成曲线
    curves_path = plot_cross_validation_curves(results, pollutant)
    
    return heatmap_path, curves_path


def generate_all_pollutants_cv_plots():
    """
    为所有污染物生成交叉验证图表
    """
    # 定义污染物列表
    pollutants = ['pm25', 'pm10', 'no2', 'so2', 'o3', 'co']
    
    # 为每个污染物生成交叉验证图表
    output_files = {}
    for pollutant in pollutants:
        print(f"\n处理{pollutant}...")
        heatmap_path, curves_path = generate_cross_validation_plots(pollutant)
        if heatmap_path and curves_path:
            output_files[pollutant] = {
                'heatmap': heatmap_path,
                'curves': curves_path
            }
    
    return output_files


def main():
    """
    主函数
    """
    print("=" * 80)
    print("生成ElasticNet交叉验证图")
    print("=" * 80)
    
    try:
        # 为所有污染物生成交叉验证图表
        output_files = generate_all_pollutants_cv_plots()
        
        print("\n" + "=" * 80)
        print("任务完成！")
        print("=" * 80)
        
        if output_files:
            print("生成的交叉验证图表：")
            for pollutant, files in output_files.items():
                print(f"\n{pollutant}:")
                print(f"- 热图: {files['heatmap']}")
                print(f"- 曲线: {files['curves']}")
            
            print("\n这些图表可以用于深度学习论文中的模型评估部分，展示ElasticNet模型在不同超参数下的性能。")
        else:
            print("无法生成交叉验证图表，因为没有有效的训练数据。")
        
    except Exception as e:
        print(f"执行过程中出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
