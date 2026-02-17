import pandas as pd
import numpy as np

def prepare_data(input_path, output_path):
    # 1. 读取数据
    data = pd.read_csv(input_path)

    # 2. 数据清洗
    # 处理缺失值
    data = data.dropna()

    # 处理异常值（示例：移除PM2.5>1000的值）
    if 'PM2.5' in data.columns:
        data = data[data['PM2.5'] < 1000]

    # 3. 特征选择和重命名（确保列名符合后续处理要求）
    # 打印当前列名
    print("当前数据列名:")
    print(data.columns.tolist())

    # 根据实际列名进行重命名
    rename_map = {
        'PM2.5': 'pm2.5',
        'PM10': 'pm10',
        'NO2': 'no2',
        'SO2': 'so2',
        'CO': 'co',
        'Temperature': 't2m',
        'Humidity': 'rh',
        'Proximity': 'proximity',
        'Population': 'population',
        'Air Quality': 'air_quality'
    }

    # 执行重命名
    data.rename(columns=rename_map, inplace=True)

    # 4. 添加时间特征（假设数据按时间顺序排列）
    data['time'] = pd.date_range(start='2020-01-01', periods=len(data), freq='H')
    data['hour'] = data['time'].dt.hour
    data['day'] = data['time'].dt.day
    data['month'] = data['time'].dt.month
    data['year'] = data['time'].dt.year
    data['season'] = (data['month'] % 12 + 3) // 3  # 1-4 对应春夏秋冬

    # 5. 添加站点ID（假设为单个站点）
    data['station_id'] = 'station1'

    # 6. 保存处理后的数据
    data.to_csv(output_path, index=False)
    print(f"\n数据准备完成，已保存至 {output_path}")
    print(f"处理后的数据形状: {data.shape}")
    print("处理后的数据列名:")
    print(data.columns.tolist())

if __name__ == "__main__":
    # 示例调用
    input_path = "air_quality/raw/pollution_test_data.csv"
    output_path = "air_quality/processed/aligned_data.csv"
    prepare_data(input_path, output_path)