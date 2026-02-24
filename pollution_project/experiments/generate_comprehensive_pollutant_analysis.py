#!/usr/bin/env python3
"""
生成全面的污染物分析组合图

该脚本用于生成一张包含所有污染物综合分析的组合图表，
可用于深度学习论文中的模型评估部分，展示所有污染物的统计信息、
模型性能和特征重要性分析。
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LassoCV, ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
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


def train_models(X, y):
    """
    训练多个回归模型
    """
    # 标准化特征
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 训练Lasso模型
    lasso_cv = LassoCV(
        alphas=np.logspace(-4, 4, 100),
        cv=5,
        random_state=42,
        max_iter=10000
    )
    lasso_cv.fit(X_scaled, y)
    
    # 训练ElasticNet模型
    elasticnet_cv = ElasticNetCV(
        alphas=np.logspace(-4, 4, 100),
        l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
        cv=5,
        random_state=42,
        max_iter=10000
    )
    elasticnet_cv.fit(X_scaled, y)
    
    return lasso_cv, elasticnet_cv


def calculate_metrics(model, X, y):
    """
    计算模型性能指标
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_pred = model.predict(X_scaled)
    
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    
    return r2, rmse


def plot_comprehensive_pollutant_analysis(df, features, pollutants):
    """
    生成全面的污染物分析组合图
    """
    print("生成全面的污染物分析组合图...")
    
    # 创建大图
    fig = plt.figure(figsize=(20, 16))
    
    # 1. 污染物统计信息（箱线图）
    ax1 = plt.subplot(3, 2, 1)
    pollutant_data = [df[pollutant] for pollutant in pollutants]
    box = ax1.boxplot(pollutant_data, tick_labels=pollutants, patch_artist=True)
    ax1.set_title('Pollutant Distribution (Box Plot)', fontsize=14)
    ax1.set_ylabel('Concentration', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # 设置不同颜色
    colors = ['skyblue', 'lightgreen', 'salmon', 'gold', 'purple', 'cyan']
    
    # 设置箱体颜色
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    
    # 设置 whiskers 和 caps 颜色
    for i, whisker in enumerate(box['whiskers']):
        color = colors[i // 2 % len(colors)]
        whisker.set_color(color)
    
    for i, cap in enumerate(box['caps']):
        color = colors[i // 2 % len(colors)]
        cap.set_color(color)
    
    # 设置 medians 颜色
    for median in box['medians']:
        median.set_color('black')
    
    # 2. 模型性能比较（条形图）
    ax2 = plt.subplot(3, 2, 2)
    
    # 存储模型性能
    model_performance = {}
    for pollutant in pollutants:
        X = df[features]
        y = df[pollutant]
        
        lasso_model, elasticnet_model = train_models(X, y)
        lasso_r2, _ = calculate_metrics(lasso_model, X, y)
        elasticnet_r2, _ = calculate_metrics(elasticnet_model, X, y)
        
        model_performance[pollutant] = {
            'Lasso': lasso_r2,
            'ElasticNet': elasticnet_r2
        }
    
    # 转换为DataFrame
    performance_df = pd.DataFrame(model_performance).T
    
    # 绘制条形图
    bar_width = 0.35
    index = np.arange(len(pollutants))
    
    bars1 = ax2.bar(index - bar_width/2, performance_df['Lasso'], bar_width, label='Lasso', color='skyblue')
    bars2 = ax2.bar(index + bar_width/2, performance_df['ElasticNet'], bar_width, label='ElasticNet', color='salmon')
    
    ax2.set_title('Model Performance Comparison (R² Score)', fontsize=14)
    ax2.set_ylabel('R² Score', fontsize=12)
    ax2.set_xticks(index)
    ax2.set_xticklabels(pollutants, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.1)
    
    # 3. 特征重要性热图（Lasso）
    ax3 = plt.subplot(3, 2, 3)
    
    # 存储Lasso系数
    lasso_coefficients = {}
    for pollutant in pollutants:
        X = df[features]
        y = df[pollutant]
        lasso_model, _ = train_models(X, y)
        lasso_coefficients[pollutant] = lasso_model.coef_
    
    lasso_coef_df = pd.DataFrame(lasso_coefficients, index=features)
    
    sns.heatmap(lasso_coef_df, annot=True, fmt='.2f', cmap='RdBu', center=0, ax=ax3)
    ax3.set_title('Lasso Regression Coefficients (Feature Importance)', fontsize=14)
    ax3.set_ylabel('Features', fontsize=12)
    ax3.set_xlabel('Pollutants', fontsize=12)
    
    # 4. 污染物相关性热图
    ax4 = plt.subplot(3, 2, 4)
    
    pollutant_corr = df[pollutants].corr()
    sns.heatmap(pollutant_corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax4)
    ax4.set_title('Pollutant Correlation Matrix', fontsize=14)
    
    # 5. 特征重要性比较（综合）
    ax5 = plt.subplot(3, 2, 5)
    
    # 计算每个特征对所有污染物的平均重要性（绝对值）
    feature_importance = {}
    for feature in features:
        importance = np.mean([abs(lasso_coefficients[pollutant][i]) for i, feat in enumerate(features) if feat == feature for pollutant in pollutants])
        feature_importance[feature] = importance
    
    # 排序
    sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    sorted_features = [item[0] for item in sorted_importance]
    sorted_values = [item[1] for item in sorted_importance]
    
    ax5.bar(range(len(sorted_features)), sorted_values, color='lightgreen')
    ax5.set_title('Average Feature Importance Across All Pollutants', fontsize=14)
    ax5.set_ylabel('Average Absolute Coefficient', fontsize=12)
    ax5.set_xticks(range(len(sorted_features)))
    ax5.set_xticklabels(sorted_features, rotation=45, ha='right')
    ax5.grid(True, alpha=0.3)
    
    # 6. 月度污染趋势
    ax6 = plt.subplot(3, 2, 6)
    
    # 计算月度平均值
    monthly_data = df.copy()
    monthly_data['month'] = df['date'].dt.month
    monthly_avg = monthly_data.groupby('month')[pollutants].mean()
    
    for i, pollutant in enumerate(pollutants):
        ax6.plot(monthly_avg.index, monthly_avg[pollutant], marker='o', label=pollutant, color=colors[i])
    
    ax6.set_title('Monthly Pollutant Concentration Trends', fontsize=14)
    ax6.set_xlabel('Month', fontsize=12)
    ax6.set_ylabel('Average Concentration', fontsize=12)
    ax6.set_xticks(range(1, 13))
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图表
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pollution_project', 'pollution_app', 'static', 'images')
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'comprehensive_pollutant_analysis_{timestamp}.png'
    filepath = os.path.join(output_dir, filename)
    
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"全面的污染物分析组合图已保存到: {filepath}")
    
    return filepath


def main():
    """
    主函数
    """
    print("=" * 80)
    print("生成全面的污染物分析组合图")
    print("=" * 80)
    
    try:
        # 生成数据
        df = generate_pollution_data()
        
        # 定义特征和污染物
        features = ['temperature', 'humidity', 'wind_speed', 'wind_direction', 'pressure', 'month', 'day_of_week', 'day_of_year']
        pollutants = ['pm25', 'pm10', 'so2', 'no2', 'o3', 'co']
        
        # 生成全面的污染物分析组合图
        plot_path = plot_comprehensive_pollutant_analysis(df, features, pollutants)
        
        print("\n" + "=" * 80)
        print("任务完成！")
        print("=" * 80)
        
        print("生成的全面污染物分析组合图：")
        print(f"- {plot_path}")
        print("\n该图表包含：")
        print("1. 污染物分布箱线图")
        print("2. 模型性能比较（Lasso vs ElasticNet）")
        print("3. Lasso回归特征重要性热图")
        print("4. 污染物相关性矩阵")
        print("5. 所有污染物的平均特征重要性")
        print("6. 月度污染趋势图")
        print("\n该图表可以用于深度学习论文中的模型评估部分，全面展示所有污染物的分析结果。")
        
    except Exception as e:
        print(f"执行过程中出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()