#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成样本数据并插入到数据库中
"""

import os
import django
import numpy as np
from datetime import datetime, timedelta

# 设置Django环境
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'pollution_project.settings')
django.setup()

from pollution_app.models import PollutionData

def generate_sample_data():
    """
    生成样本数据并插入到数据库中
    """
    print("开始生成样本数据...")
    
    # 清除现有的数据
    PollutionData.objects.all().delete()
    print("已清除现有数据")
    
    # 生成最近7天的数据
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    # 生成每小时的数据
    current_date = start_date
    records = []
    
    while current_date <= end_date:
        # 生成基础值
        base_pm25 = 30
        base_pm10 = 70
        base_o3 = 50
        
        # 添加时间变化（早上和晚上较高）
        hour = current_date.hour
        time_factor = 1.5 if (hour >= 7 and hour <= 9) or (hour >= 17 and hour <= 19) else 1.0
        
        # 添加随机波动
        pm25 = max(0, base_pm25 * time_factor * np.random.normal(1, 0.3))
        pm10 = max(0, base_pm10 * time_factor * np.random.normal(1, 0.3))
        o3 = max(0, base_o3 * np.random.normal(1, 0.3))
        no2 = max(0, 40 * np.random.normal(1, 0.3))
        so2 = max(0, 15 * np.random.normal(1, 0.3))
        co = max(0, 1.0 * np.random.normal(1, 0.3))
        temperature = 20 + np.random.normal(0, 5)
        humidity = 60 + np.random.normal(0, 10)
        wind_speed = 3 + np.random.normal(0, 1)
        
        # 创建记录
        record = PollutionData(
            date=current_date,
            pm25=pm25,
            pm10=pm10,
            o3=o3,
            no2=no2,
            so2=so2,
            co=co,
            temperature=temperature,
            humidity=humidity,
            wind_speed=wind_speed,
            wind_direction=np.random.normal(0, 180),
            pressure=1013 + np.random.normal(0, 5),
            precipitation=np.random.normal(0, 2)  # 添加降水量字段
        )
        records.append(record)
        
        # 增加一小时
        current_date += timedelta(hours=1)
    
    # 批量插入数据
    PollutionData.objects.bulk_create(records)
    print(f"成功生成并插入 {len(records)} 条样本数据")

if __name__ == "__main__":
    generate_sample_data()
