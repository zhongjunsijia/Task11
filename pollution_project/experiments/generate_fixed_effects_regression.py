#!/usr/bin/env python3
"""
生成Fixed Effects回归系数图

该脚本用于执行固定效应回归分析并生成回归系数可视化图表，
可用于深度学习论文中的模型评估部分，展示在控制固定效应后不同特征的影响。
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
from datetime import datetime

# 添加项目路径到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pollution_project'))

# 初始化Django设置
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'pollution_project.settings')
import django
django.setup()


def generate_panel_data():
    """
    生成面板数据（按城市和时间）
    """
    print("生成面板数据...")
    
    # 城市列表
    cities = ['北京', '上海', '广州', '深圳', '杭州', '成都', '武汉', '西安', '南京', '重庆']
    
    # 时间范围
    dates = pd.date_range(start='2025-01-01', end='2025-12-31', freq='D')
    
    # 生成面板数据
    data = []
    for city in cities:
        for date in dates:
            # 基础值
            base_pm25 = np.random.randint(50, 150)
            
            # 时间效应
            month_effect = np.sin(2 * np.pi * date.month / 12) * 20
            day_effect = np.cos(2 * np.pi * date.dayofweek / 7) * 10
            
            # 城市效应（不同城市有不同的基础污染水平）
            city_effect = {'北京': 30, '上海': 20, '广州': 10, '深圳': 5, '杭州': 15, '成都': 25, '武汉': 20, '西安': 35, '南京': 18, '重庆': 22}[city]
            
            # 气象因素影响
            temperature = np.random.uniform(0, 35)
            humidity = np.random.uniform(30, 90)
            wind_speed = np.random.uniform(0, 10)
            
            # 计算PM2.5值
            pm25 = base_pm25 + month_effect + day_effect + city_effect + \
                   0.5 * temperature - 0.3 * humidity - 2 * wind_speed + \
                   np.random.normal(0, 10)
            
            # 确保值为正
            pm25 = max(0, pm25)
            
            # 添加到数据
            data.append({
                'city': city,
                'date': date,
                'pm25': pm25,
                'temperature': temperature,
                'humidity': humidity,
                'wind_speed': wind_speed,
                'month': date.month,
                'day_of_week': date.dayofweek
            })
    
    # 转换为DataFrame
    df = pd.DataFrame(data)
    
    print(f"生成的面板数据形状: {df.shape}")
    print(f"城市数量: {df['city'].nunique()}")
    print(f"时间跨度: {df['date'].min()} 到 {df['date'].max()}")
    
    return df


def run_fixed_effects_regression(df, dependent_var='pm25', fixed_effect_var='city', features=None):
    """
    执行固定效应回归
    使用虚拟变量法实现固定效应
    """
    print("执行Fixed Effects回归...")
    
    # 默认特征
    if features is None:
        features = ['temperature', 'humidity', 'wind_speed', 'month', 'day_of_week']
    
    # 创建城市虚拟变量（固定效应）
    city_dummies = pd.get_dummies(df[fixed_effect_var], drop_first=True, prefix='city')  # 避免多重共线性
    
    # 构建回归数据
    X = df[features].copy()
    X = pd.concat([X, city_dummies], axis=1)
    y = df[dependent_var]
    
    # 确保所有数据都是数值类型
    X = X.astype(float)
    y = y.astype(float)
    
    # 检查并处理缺失值
    if X.isnull().any().any():
        print("注意：数据中存在缺失值，正在处理...")
        X = X.dropna()
        y = y.loc[X.index]
    
    # 添加常数项
    X = sm.add_constant(X)
    
    # 拟合模型
    model = OLS(y, X).fit()
    
    print("回归结果摘要:")
    print(model.summary())
    
    return model, features


def plot_fixed_effects_coefficients(model, features, pollutant='pm25'):
    """
    生成固定效应回归系数图
    """
    print(f"生成{pollutant}的Fixed Effects回归系数图...")
    
    # 提取特征系数（排除固定效应虚拟变量）
    coefficients = model.params
    std_errors = model.bse
    
    # 筛选出我们关心的特征系数
    feature_coefficients = {}
    feature_std_errors = {}
    
    for feature in features:
        if feature in coefficients:
            feature_coefficients[feature] = coefficients[feature]
            feature_std_errors[feature] = std_errors[feature]
    
    # 转换为DataFrame并排序
    coef_df = pd.DataFrame({
        'coefficient': feature_coefficients,
        'std_error': feature_std_errors
    }).sort_values('coefficient', ascending=False)
    
    # 创建图表
    plt.figure(figsize=(12, 8))
    
    # 绘制系数条形图
    bars = plt.bar(
        range(len(coef_df)), 
        coef_df['coefficient'], 
        yerr=1.96 * coef_df['std_error'],  # 95% 置信区间
        color='skyblue',
        capsize=5
    )
    
    # 添加数值标签
    for i, (coef, std_err) in enumerate(zip(coef_df['coefficient'], coef_df['std_error'])):
        plt.text(
            i, 
            coef + 0.02 * abs(coef), 
            f'{coef:.4f}',
            ha='center', 
            va='bottom',
            fontsize=10
        )
    
    # 设置x轴标签
    plt.xticks(
        range(len(coef_df)), 
        coef_df.index, 
        rotation=45, 
        ha='right'
    )
    
    # 添加标题和标签
    plt.title(f'Fixed Effects Regression Coefficients for {pollutant}', fontsize=16)
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
    filename = f'fixed_effects_coefficients_{pollutant}_{timestamp}.png'
    filepath = os.path.join(output_dir, filename)
    
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Fixed Effects回归系数图已保存到: {filepath}")
    
    return filepath


def plot_fixed_effects_comparison(df, dependent_var='pm25', features=None):
    """
    生成固定效应与普通OLS回归的系数比较图
    """
    print("生成Fixed Effects与OLS回归系数比较图...")
    
    # 默认特征
    if features is None:
        features = ['temperature', 'humidity', 'wind_speed', 'month', 'day_of_week']
    
    # 运行固定效应回归
    fe_model, _ = run_fixed_effects_regression(df, dependent_var, 'city', features)
    
    # 运行普通OLS回归
    print("执行普通OLS回归...")
    X_ols = df[features]
    X_ols = sm.add_constant(X_ols)
    y = df[dependent_var]
    ols_model = OLS(y, X_ols).fit()
    
    # 提取系数
    fe_coefficients = {}
    ols_coefficients = {}
    fe_std_errors = {}
    ols_std_errors = {}
    
    for feature in features:
        if feature in fe_model.params:
            fe_coefficients[feature] = fe_model.params[feature]
            fe_std_errors[feature] = fe_model.bse[feature]
        if feature in ols_model.params:
            ols_coefficients[feature] = ols_model.params[feature]
            ols_std_errors[feature] = ols_model.bse[feature]
    
    # 转换为DataFrame
    coef_df = pd.DataFrame({
        'FE_Coefficient': fe_coefficients,
        'FE_Std_Error': fe_std_errors,
        'OLS_Coefficient': ols_coefficients,
        'OLS_Std_Error': ols_std_errors
    }).sort_values('FE_Coefficient', ascending=False)
    
    # 创建图表
    plt.figure(figsize=(14, 8))
    
    # 绘制系数条形图
    bar_width = 0.35
    index = np.arange(len(coef_df))
    
    bars1 = plt.bar(
        index - bar_width/2, 
        coef_df['FE_Coefficient'], 
        bar_width, 
        yerr=1.96 * coef_df['FE_Std_Error'],
        label='Fixed Effects',
        color='skyblue',
        capsize=5
    )
    
    bars2 = plt.bar(
        index + bar_width/2, 
        coef_df['OLS_Coefficient'], 
        bar_width, 
        yerr=1.96 * coef_df['OLS_Std_Error'],
        label='OLS',
        color='coral',
        capsize=5
    )
    
    # 添加数值标签
    for i, (fe_coef, ols_coef) in enumerate(zip(coef_df['FE_Coefficient'], coef_df['OLS_Coefficient'])):
        plt.text(
            i - bar_width/2, 
            fe_coef + 0.02 * abs(fe_coef), 
            f'{fe_coef:.4f}',
            ha='center', 
            va='bottom',
            fontsize=9
        )
        plt.text(
            i + bar_width/2, 
            ols_coef + 0.02 * abs(ols_coef), 
            f'{ols_coef:.4f}',
            ha='center', 
            va='bottom',
            fontsize=9
        )
    
    # 设置x轴标签
    plt.xticks(index, coef_df.index, rotation=45, ha='right')
    
    # 添加标题和标签
    plt.title(f'Fixed Effects vs OLS Regression Coefficients for {dependent_var}', fontsize=16)
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
    filename = f'fixed_effects_vs_ols_{dependent_var}_{timestamp}.png'
    filepath = os.path.join(output_dir, filename)
    
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Fixed Effects与OLS回归系数比较图已保存到: {filepath}")
    
    return filepath


def generate_fixed_effects_plots():
    """
    生成固定效应回归图表
    """
    # 生成面板数据
    df = generate_panel_data()
    
    # 定义特征
    features = ['temperature', 'humidity', 'wind_speed', 'month', 'day_of_week']
    
    # 运行固定效应回归
    model, _ = run_fixed_effects_regression(df, 'pm25', 'city', features)
    
    # 生成系数图
    coefficients_path = plot_fixed_effects_coefficients(model, features, 'pm25')
    
    # 生成FE vs OLS比较图
    comparison_path = plot_fixed_effects_comparison(df, 'pm25', features)
    
    return coefficients_path, comparison_path


def main():
    """
    主函数
    """
    print("=" * 80)
    print("生成Fixed Effects回归系数图")
    print("=" * 80)
    
    try:
        # 生成固定效应回归图表
        coefficients_path, comparison_path = generate_fixed_effects_plots()
        
        print("\n" + "=" * 80)
        print("任务完成！")
        print("=" * 80)
        
        print("生成的固定效应回归图表：")
        print(f"- 固定效应回归系数图: {coefficients_path}")
        print(f"- FE vs OLS比较图: {comparison_path}")
        print("\n这些图表可以用于深度学习论文中的模型评估部分，展示在控制固定效应后不同特征的影响。")
        
    except Exception as e:
        print(f"执行过程中出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
