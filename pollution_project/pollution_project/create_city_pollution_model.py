import os
import django

# 设置Django环境
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'pollution_project.settings')
django.setup()

# 初始化城市数据
def initialize_city_data():
    from django.utils import timezone
    from pollution_app.models import CityPollutionData
    
    # 中国主要城市AQI数据（模拟数据）
    cities_data = [
        {'city': '贺州市', 'aqi': 56, 'pm25': 35, 'pm10': 58, 'no2': 22, 'so2': 8, 'o3': 65, 'co': 0.8, 'quality': '良好'},
        {'city': '桂林市', 'aqi': 68, 'pm25': 42, 'pm10': 65, 'no2': 28, 'so2': 10, 'o3': 72, 'co': 0.9, 'quality': '良'},
        {'city': '南宁市', 'aqi': 82, 'pm25': 52, 'pm10': 78, 'no2': 35, 'so2': 12, 'o3': 80, 'co': 1.1, 'quality': '良'},
        {'city': '柳州市', 'aqi': 105, 'pm25': 68, 'pm10': 95, 'no2': 42, 'so2': 15, 'o3': 85, 'co': 1.3, 'quality': '轻度污染'},
        {'city': '梧州市', 'aqi': 112, 'pm25': 72, 'pm10': 102, 'no2': 45, 'so2': 18, 'o3': 88, 'co': 1.4, 'quality': '轻度污染'},
        {'city': '北京市', 'aqi': 95, 'pm25': 60, 'pm10': 85, 'no2': 38, 'so2': 14, 'o3': 75, 'co': 1.2, 'quality': '良'},
        {'city': '上海市', 'aqi': 75, 'pm25': 45, 'pm10': 68, 'no2': 30, 'so2': 11, 'o3': 70, 'co': 1.0, 'quality': '良'},
        {'city': '广州市', 'aqi': 88, 'pm25': 55, 'pm10': 82, 'no2': 36, 'so2': 13, 'o3': 78, 'co': 1.1, 'quality': '良'},
        {'city': '深圳市', 'aqi': 72, 'pm25': 43, 'pm10': 65, 'no2': 29, 'so2': 10, 'o3': 68, 'co': 0.9, 'quality': '良'},
        {'city': '成都市', 'aqi': 120, 'pm25': 78, 'pm10': 108, 'no2': 48, 'so2': 20, 'o3': 90, 'co': 1.5, 'quality': '轻度污染'},
        {'city': '重庆市', 'aqi': 115, 'pm25': 75, 'pm10': 105, 'no2': 46, 'so2': 19, 'o3': 89, 'co': 1.4, 'quality': '轻度污染'},
        {'city': '武汉市', 'aqi': 98, 'pm25': 62, 'pm10': 88, 'no2': 39, 'so2': 15, 'o3': 76, 'co': 1.2, 'quality': '良'},
        {'city': '西安市', 'aqi': 135, 'pm25': 85, 'pm10': 115, 'no2': 52, 'so2': 22, 'o3': 95, 'co': 1.6, 'quality': '轻度污染'},
        {'city': '杭州市', 'aqi': 78, 'pm25': 48, 'pm10': 72, 'no2': 32, 'so2': 12, 'o3': 74, 'co': 1.0, 'quality': '良'},
        {'city': '南京市', 'aqi': 85, 'pm25': 53, 'pm10': 80, 'no2': 37, 'so2': 14, 'o3': 79, 'co': 1.1, 'quality': '良'},
        {'city': '天津市', 'aqi': 102, 'pm25': 66, 'pm10': 92, 'no2': 40, 'so2': 16, 'o3': 83, 'co': 1.2, 'quality': '轻度污染'},
        {'city': '苏州市', 'aqi': 70, 'pm25': 40, 'pm10': 62, 'no2': 26, 'so2': 9, 'o3': 68, 'co': 0.9, 'quality': '良'},
        {'city': '郑州市', 'aqi': 125, 'pm25': 80, 'pm10': 110, 'no2': 49, 'so2': 21, 'o3': 92, 'co': 1.5, 'quality': '轻度污染'},
        {'city': '长沙市', 'aqi': 90, 'pm25': 58, 'pm10': 85, 'no2': 36, 'so2': 13, 'o3': 77, 'co': 1.1, 'quality': '良'},
        {'city': '青岛市', 'aqi': 65, 'pm25': 38, 'pm10': 55, 'no2': 24, 'so2': 9, 'o3': 66, 'co': 0.8, 'quality': '良'},
    ]
    
    for city_data in cities_data:
        city, created = CityPollutionData.objects.get_or_create(
            city=city_data['city'],
            defaults={
                'aqi': city_data['aqi'],
                'pm25': city_data['pm25'],
                'pm10': city_data['pm10'],
                'no2': city_data['no2'],
                'so2': city_data['so2'],
                'o3': city_data['o3'],
                'co': city_data['co'],
                'quality': city_data['quality'],
                'update_time': timezone.now()
            }
        )
        if created:
            print(f"创建城市数据: {city_data['city']} - AQI: {city_data['aqi']}")
        else:
            # 更新现有城市数据
            for key, value in city_data.items():
                setattr(city, key, value)
            city.update_time = timezone.now()
            city.save()
            print(f"更新城市数据: {city_data['city']} - AQI: {city_data['aqi']}")

if __name__ == '__main__':
    print("开始初始化城市污染数据...")
    
    # 检查模型是否已存在
    from django.apps import apps
    try:
        CityPollutionData = apps.get_model('pollution_app', 'CityPollutionData')
        print("城市污染数据模型已存在")
    except LookupError:
        print("城市污染数据模型不存在，需要先在models.py中添加")
        exit(1)
    
    # 初始化城市数据
    print("初始化城市污染数据...")
    initialize_city_data()
    print("城市污染数据初始化完成！")
