#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试真实实时数据预测功能
"""

import sys
import os
from datetime import date, datetime

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'pollution_project'))

# 配置Django环境
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'pollution_project.settings')
import django
django.setup()

# 导入预测模块
from pollution_app.prediction_model import predict_pollution

if __name__ == "__main__":
    print("测试真实实时数据预测功能")
    print("=" * 60)
    
    # 测试RF模型
    print("测试RF模型（真实数据）")
    try:
        rf_result = predict_pollution(
            date=date.today(),
            temperature=25.0,
            humidity=60.0,
            wind_speed=3.0,
            pm25=50.0,
            pm10=75.0,
            no2=30.0,
            so2=10.0,
            o3=40.0,
            co=1.0,
            model_type='rf',
            use_real_time_data=True
        )
        print(f"RF模型预测结果: {rf_result}")
        print(f"使用的模型: {rf_result.get('model', '未知')}")
        print(f"时间戳: {rf_result.get('timestamp', '未知')}")
    except Exception as e:
        print(f"RF模型测试失败: {e}")
    
    print("=" * 60)
    
    # 测试SVR模型
    print("测试SVR模型（真实数据）")
    try:
        svr_result = predict_pollution(
            date=date.today(),
            temperature=25.0,
            humidity=60.0,
            wind_speed=3.0,
            pm25=50.0,
            pm10=75.0,
            no2=30.0,
            so2=10.0,
            o3=40.0,
            co=1.0,
            model_type='svr',
            use_real_time_data=True
        )
        print(f"SVR模型预测结果: {svr_result}")
        print(f"使用的模型: {svr_result.get('model', '未知')}")
        print(f"时间戳: {svr_result.get('timestamp', '未知')}")
    except Exception as e:
        print(f"SVR模型测试失败: {e}")
    
    print("=" * 60)
    
    # 测试LSTM模型
    print("测试LSTM模型（真实数据）")
    try:
        lstm_result = predict_pollution(
            date=date.today(),
            temperature=25.0,
            humidity=60.0,
            wind_speed=3.0,
            pm25=50.0,
            pm10=75.0,
            no2=30.0,
            so2=10.0,
            o3=40.0,
            co=1.0,
            model_type='lstm',
            use_real_time_data=True
        )
        print(f"LSTM模型预测结果: {lstm_result}")
        print(f"使用的模型: {lstm_result.get('model', '未知')}")
        print(f"时间戳: {lstm_result.get('timestamp', '未知')}")
    except Exception as e:
        print(f"LSTM模型测试失败: {e}")
    
    print("=" * 60)
    print("真实数据测试完成！")
