#!/usr/bin/env python3
"""
生成组合ElasticNet交叉验证图

该脚本用于训练ElasticNet模型并生成包含所有污染物交叉验证结果的组合图表，
可用于深度学习论文中的模型评估部分，展示不同污染物的超参数敏感性比较。
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


def plot_combined_cv_curves(all_results, alphas, l1_ratios):
    """
    生成所有污染物的组合交叉验证曲线图
    """
    print("生成所有污染物的组合ElasticNet交叉验证曲线图...")
    
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
    
    # 创建图表
    plt.figure(figsize=(14, 10))
    
    # 对每个l1_ratio创建一个子图
    for i, l1_ratio in enumerate(l1_ratios, 1):
        plt.subplot(2, 3, i)
        
        # 对每个污染物绘制曲线
        for pollutant, results in all_results.items():
            # 转换结果为DataFrame
            df = pd.DataFrame(results)
            # 筛选当前l1_ratio的数据
            ratio_data = df[df['l1_ratio'] == l1_ratio]
            # 按alpha排序
            ratio_data = ratio_data.sort_values('alpha')
            
            # 绘制平均R²得分
            plt.plot(
                ratio_data['alpha'], 
                ratio_data['mean_r2'], 
                marker='o', 
                label=pollutant_names[pollutant],
                color=colors[pollutant]
            )
            
            # 添加误差线
            plt.fill_between(
                ratio_data['alpha'],
                ratio_data['mean_r2'] - ratio_data['std_r2'],
                ratio_data['mean_r2'] + ratio_data['std_r2'],
                alpha=0.1,
                color=colors[pollutant]
            )
        
        # 添加标题和标签
        plt.title(f'l1_ratio = {l1_ratio}', fontsize=12)
        plt.xlabel('alpha', fontsize=10)
        plt.ylabel('Mean R² Score', fontsize=10)
        plt.xscale('log')  # 使用对数刻度显示alpha
        plt.grid(alpha=0.3)
        
        # 在第一个子图添加图例
        if i == 1:
            plt.legend(title='Pollutants', fontsize=8, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 调整布局
    plt.tight_layout()
    
    # 添加主标题
    plt.suptitle('ElasticNet Cross-Validation Curves for All Pollutants', fontsize=16, y=1.02)
    
    # 保存图表
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pollution_project', 'pollution_app', 'static', 'images')
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'combined_elasticnet_cv_curves_{timestamp}.png'
    filepath = os.path.join(output_dir, filename)
    
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"组合交叉验证曲线图已保存到: {filepath}")
    
    return filepath


def plot_combined_cv_heatmaps(all_results, alphas, l1_ratios):
    """
    生成所有污染物的组合交叉验证热图
    """
    print("生成所有污染物的组合ElasticNet交叉验证热图...")
    
    # 污染物名称映射
    pollutant_names = {
        'pm25': 'PM2.5',
        'pm10': 'PM10',
        'no2': 'NO₂',
        'so2': 'SO₂',
        'o3': 'O₃',
        'co': 'CO'
    }
    
    # 创建图表
    fig, axes = plt.subplots(2, 3, figsize=(14, 10))
    axes = axes.flatten()
    
    # 对每个污染物创建一个热图
    for i, (pollutant, results) in enumerate(all_results.items()):
        ax = axes[i]
        
        # 转换结果为DataFrame
        df = pd.DataFrame(results)
        
        # 创建透视表
        pivot_table = df.pivot(index='alpha', columns='l1_ratio', values='mean_r2')
        
        # 绘制热图
        im = ax.imshow(pivot_table, cmap='RdYlBu_r', aspect='auto')
        
        # 设置x轴和y轴标签
        ax.set_xticks(np.arange(len(pivot_table.columns)))
        ax.set_xticklabels([f'{x:.2f}' for x in pivot_table.columns], rotation=45, ha='right', fontsize=8)
        
        ax.set_yticks(np.arange(len(pivot_table.index)))
        ax.set_yticklabels([f'{y:.4f}' for y in pivot_table.index], fontsize=8)
        
        # 添加标题
        ax.set_title(pollutant_names[pollutant], fontsize=12)
        ax.set_xlabel('l1_ratio', fontsize=10)
        ax.set_ylabel('alpha', fontsize=10)
    
    # 添加颜色条
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Mean R² Score', fontsize=12)
    
    # 调整布局
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    
    # 添加主标题
    plt.suptitle('ElasticNet Cross-Validation Heatmaps for All Pollutants', fontsize=16, y=1.02)
    
    # 保存图表
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pollution_project', 'pollution_app', 'static', 'images')
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'combined_elasticnet_cv_heatmaps_{timestamp}.png'
    filepath = os.path.join(output_dir, filename)
    
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"组合交叉验证热图已保存到: {filepath}")
    
    return filepath


def generate_combined_cv_plots():
    """
    生成所有污染物的组合交叉验证图表
    """
    # 生成训练数据
    data = generate_training_data()
    
    # 定义污染物列表
    pollutants = ['pm25', 'pm10', 'no2', 'so2', 'o3', 'co']
    
    # 定义特征
    features = ['day_of_week', 'month', 'temperature', 'humidity', 'wind_speed']
    
    # 定义超参数范围
    alphas = np.logspace(-4, 2, 10)  # 从0.0001到100的10个对数间隔值
    l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.9]  # 不同的l1_ratio值
    
    # 存储所有污染物的结果
    all_results = {}
    
    # 为每个污染物执行交叉验证
    for pollutant in pollutants:
        print(f"\n处理{pollutant}...")
        
        X = data[features]
        y = data[pollutant]
        
        # 移除缺失值
        mask = ~np.isnan(y)
        X_clean = X[mask]
        y_clean = y[mask]
        
        if len(y_clean) == 0:
            print(f"没有有效的{pollutant}数据，跳过...")
            continue
        
        # 执行交叉验证
        results = perform_cross_validation(X_clean, y_clean, alphas, l1_ratios)
        all_results[pollutant] = results
    
    # 生成组合曲线图
    curves_path = plot_combined_cv_curves(all_results, alphas, l1_ratios)
    
    # 生成组合热图
    heatmaps_path = plot_combined_cv_heatmaps(all_results, alphas, l1_ratios)
    
    return curves_path, heatmaps_path


def main():
    """
    主函数
    """
    print("=" * 80)
    print("生成组合ElasticNet交叉验证图")
    print("=" * 80)
    
    try:
        # 生成组合交叉验证图表
        curves_path, heatmaps_path = generate_combined_cv_plots()
        
        print("\n" + "=" * 80)
        print("任务完成！")
        print("=" * 80)
        
        print("生成的组合交叉验证图表：")
        print(f"- 组合曲线图: {curves_path}")
        print(f"- 组合热图: {heatmaps_path}")
        print("\n这些图表可以用于深度学习论文中的模型评估部分，展示所有污染物的超参数敏感性比较。")
        
    except Exception as e:
        print(f"执行过程中出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
