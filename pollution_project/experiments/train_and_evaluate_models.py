#!/usr/bin/env python3
"""
模型训练和评估脚本

该脚本用于训练多种机器学习模型，评估模型效果，并生成详细的评估报告。
支持的模型类型：
- linear: 线性回归
- ridge: 岭回归
- lasso: Lasso回归
- rf: 随机森林
- gb: 梯度提升树
- svr: 支持向量回归
- mlp: 多层感知器（神经网络）
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime

# 添加项目路径到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pollution_project'))

# 初始化Django设置
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'pollution_project.settings')
import django
django.setup()

# 现在导入Django相关模块
from django.conf import settings

# 导入模型管理器
from pollution_app.utils.model_manager import model_manager
from pollution_app.prediction_model import train_and_save_model


def generate_training_data():
    """
    生成训练数据
    如果有真实数据，优先使用真实数据；否则使用模拟数据
    """
    print("生成训练数据...")
    
    # 尝试加载真实数据
    data_files = [
        'uploads/air_pollution_china.csv',
        'uploads/pollution_test_data.csv',
        'uploads/updated_pollution_dataset.csv'
    ]
    
    for file_path in data_files:
        full_path = os.path.join(settings.BASE_DIR, file_path)
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


def train_models():
    """
    训练所有模型
    """
    print("=" * 80)
    print("开始训练模型...")
    print("=" * 80)
    
    # 生成训练数据
    data = generate_training_data()
    
    # 定义特征
    features = ['day_of_week', 'month', 'temperature', 'humidity', 'wind_speed']
    
    # 定义要训练的模型类型
    model_types = ['linear', 'ridge', 'lasso', 'rf', 'gb', 'svr', 'mlp']
    
    # 训练所有模型
    results = model_manager.train_all_pollutants(
        data=data,
        features=features,
        model_types=model_types,
        model_name='pollution_predictor',
        city='全国',
        activate=True
    )
    
    print("=" * 80)
    print("模型训练完成！")
    print("=" * 80)
    
    return results


def evaluate_models():
    """
    评估模型效果
    """
    print("=" * 80)
    print("开始评估模型...")
    print("=" * 80)
    
    # 获取所有模型版本
    pollutants = ['pm25', 'pm10', 'no2', 'so2', 'o3', 'co']
    evaluation_results = {}
    
    for pollutant in pollutants:
        print(f"评估 {pollutant} 的模型...")
        versions = model_manager.get_model_versions('pollution_predictor', pollutant)
        if versions:
            evaluation_results[pollutant] = versions
            print(f"找到 {len(versions)} 个模型版本")
        else:
            print(f"没有找到 {pollutant} 的模型版本")
    
    # 比较不同模型类型的性能
    print("\n比较不同模型类型的性能...")
    model_comparison = {}
    
    for pollutant in pollutants:
        comparison = model_manager.compare_model_versions('pollution_predictor', pollutant)
        if comparison:
            model_comparison[pollutant] = comparison
            # 找出最佳模型
            best_model = max(comparison, key=lambda x: x['test_r2'])
            print(f"{pollutant} 的最佳模型: {best_model['model_type']} v{best_model['version']} (R²: {best_model['test_r2']:.4f})")
    
    print("=" * 80)
    print("模型评估完成！")
    print("=" * 80)
    
    return evaluation_results, model_comparison


def serialize_datetime(obj):
    """
    序列化datetime对象
    """
    if isinstance(obj, datetime):
        return obj.strftime('%Y-%m-%d %H:%M:%S')
    raise TypeError(f"Type {type(obj)} not serializable")

def generate_evaluation_report(results, comparison):
    """
    生成评估报告
    """
    print("=" * 80)
    print("生成评估报告...")
    print("=" * 80)
    
    # 处理datetime对象
    processed_comparison = {}
    for pollutant, models in comparison.items():
        processed_models = []
        for model in models:
            processed_model = model.copy()
            if 'created_at' in processed_model:
                processed_model['created_at'] = serialize_datetime(processed_model['created_at'])
            processed_models.append(processed_model)
        processed_comparison[pollutant] = processed_models
    
    report = {
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model_comparison': processed_comparison,
        'detailed_results': results
    }
    
    # 保存报告
    report_path = os.path.join(settings.BASE_DIR, 'pollution_app', 'models', 'model_evaluation_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=serialize_datetime)
    
    print(f"评估报告已保存到: {report_path}")
    
    # 生成可读性更好的文本报告
    text_report = "# 模型评估报告\n\n"
    text_report += f"生成时间: {report['generated_at']}\n\n"
    
    for pollutant, models in processed_comparison.items():
        text_report += f"## {pollutant} 模型评估\n\n"
        text_report += "| 模型类型 | 版本 | R² 评分 | RMSE | MAE | 训练时间(秒) | 是否激活 |\n"
        text_report += "|---------|------|---------|------|-----|--------------|----------|\n"
        
        for model in models:
            text_report += f"| {model['model_type']} | {model['version']} | {model['test_r2']:.4f} | {model['test_rmse']:.4f} | {model['test_mae']:.4f} | {model['training_time']:.2f} | {'✓' if model['is_active'] else '✗'} |\n"
        
        text_report += "\n"
    
    text_report_path = os.path.join(settings.BASE_DIR, 'pollution_app', 'models', 'model_evaluation_report.md')
    with open(text_report_path, 'w', encoding='utf-8') as f:
        f.write(text_report)
    
    print(f"文本评估报告已保存到: {text_report_path}")
    print("=" * 80)
    print("评估报告生成完成！")
    print("=" * 80)
    
    return report


def main():
    """
    主函数
    """
    # 初始化Django设置
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'pollution_project.settings')
    import django
    django.setup()
    
    try:
        # 训练模型
        training_results = train_models()
        
        # 评估模型
        evaluation_results, model_comparison = evaluate_models()
        
        # 生成评估报告
        report = generate_evaluation_report(evaluation_results, model_comparison)
        
        print("\n任务完成！")
        print("您可以查看以下文件获取详细信息：")
        print("1. model_evaluation_report.json - 详细评估数据")
        print("2. model_evaluation_report.md - 可读性更好的评估报告")
        print("3. model_training_results.json - 训练结果详细数据")
        
    except Exception as e:
        print(f"执行过程中出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
